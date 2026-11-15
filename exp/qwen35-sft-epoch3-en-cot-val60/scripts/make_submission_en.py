from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "quan-continuation-cot-v1" / "scripts"))

from lifemate import pipeline as pipeline_module  # noqa: E402
from lifemate.agent import Agent  # noqa: E402
from lifemate.asr import PhoASRModel  # noqa: E402
from lifemate.pipeline import LifeMatePipeline  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl, seed_everything  # noqa: E402
from run_downstream import COT_TEXT, canonical_prompt_hash, inject_cot  # noqa: E402

EXP = ROOT / "exp" / "qwen35-sft-epoch3-en-cot-val60"
OUT = EXP / "output" / "submission"
INFERENCE = EXP / "output" / "inference"
PROVENANCE = EXP / "output" / "model-provenance"
CONFIG_PATH = INFERENCE / "config_manifest.json"
LOCK_PATH = INFERENCE / "run_lock.json"
INFERENCE_RUN_PATH = INFERENCE / "run-COT.json"
MODEL_MANIFEST_PATH = PROVENANCE / "ollama-models.json"
REFERENCE_MANIFEST_PATH = ROOT / "outputs" / "submissions" / (
    "qwen35-track-c-quan-ft-beam32-num-predict-768-num-batch-512-public-test_manifest.json")
EXPECTED_OLLAMA_VERSION = "0.40.15"
MODEL_TAG = "lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
TRANSCRIPT_SOURCE = "exp/qwen35-track-c-quan-vs-phoasr-val309/output/submission-artifacts/" + (
    "qwen35-track-c-quan-ft-beam32-num-predict-768-num-batch-512-public-test_transcripts.jsonl")
TRANSCRIPT_MANIFEST_SOURCE = TRANSCRIPT_SOURCE.removesuffix(".jsonl") + "_manifest.json"
SUBMISSION_STEM = "qwen35-sft-epoch3-en-cot-quan-beam32-num-predict-768-num-batch-512-public-test"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"
SUBMISSION_PATH = SUBMISSION_DIR / f"{SUBMISSION_STEM}.jsonl"
PREDICTIONS_PATH = SUBMISSION_DIR / f"{SUBMISSION_STEM}_predictions.jsonl"
MANIFEST_PATH = SUBMISSION_DIR / f"{SUBMISSION_STEM}_manifest.json"
WORK_SUBMISSION = OUT / "submission-working.jsonl"
WORK_PREDICTIONS = OUT / "predictions-working.jsonl"
WORK_REQUESTS = OUT / "requests-telemetry.jsonl"
RUN_LOG_PATH = OUT / "submission-run.log"
SEED = 17
CAP = 768
CONTEXT = 8192
BATCH = 512
MAX_STEPS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def api_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request(OLLAMA_HOST + path)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def build_locked_inputs() -> dict[str, Any]:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run with system Python, outside .venv, to reach the local Ollama server")
    config = read_json(CONFIG_PATH)
    lock = read_json(LOCK_PATH)
    run = read_json(INFERENCE_RUN_PATH)
    if config.get("status") != "epoch3_en_cot_config_locked" or lock.get("status") != "locked" or run.get("status") != "complete":
        raise RuntimeError("The completed English epoch-3 + CoT inference lock is required before submission generation")
    if lock.get("config_manifest_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError("English epoch-3 inference config does not match its frozen run lock")
    model_manifest = read_json(MODEL_MANIFEST_PATH)
    if (model_manifest.get("status") != "registered" or model_manifest.get("tag") != MODEL_TAG
            or config["branches"]["epoch3_en_COT"]["model"] != MODEL_TAG
            or config["branches"]["epoch3_en_COT"]["model_digest"] != model_manifest.get("ollama_digest")
            or lock.get("epoch3_en_model_digest") != model_manifest.get("ollama_digest")
            or run.get("model_digest") != model_manifest.get("ollama_digest")):
        raise RuntimeError("English epoch-3 model tag/digest differs across the run, config and registration manifests")
    parameter_count = model_manifest.get("model_info", {}).get("general.parameter_count")
    if parameter_count is None or float(parameter_count) >= 5_000_000_000:
        raise RuntimeError(f"Registered LLM parameter count is missing or at/above 5B: {parameter_count}")
    if CONTEXT > 16_384:
        raise RuntimeError(f"Configured context exceeds the competition limit: {CONTEXT}")
    gguf_path = ROOT / model_manifest["gguf"]["path"]
    if not gguf_path.is_file() or sha256(gguf_path) != model_manifest["gguf"]["sha256"]:
        raise RuntimeError("Registered epoch-3 GGUF is missing or no longer matches its recorded SHA-256")

    reference_manifest = read_json(REFERENCE_MANIFEST_PATH)
    transcript_path = ROOT / reference_manifest["transcripts"]
    transcript_manifest_path = ROOT / reference_manifest["transcript_manifest"]
    transcript_manifest = read_json(transcript_manifest_path)
    asr = config["asr"]
    asr_checkpoint = ROOT / asr["selected_checkpoint"] / "model.safetensors"
    lm_config_path = ROOT / "models" / "quan-transcribe" / "lm" / "lm_config.json"
    if sha256(asr_checkpoint) != asr["selected_checkpoint_sha256"]:
        raise RuntimeError("The locked QUAN checkpoint changed since the benchmark")
    if sha256(lm_config_path) != asr["kenlm_config_sha256"]:
        raise RuntimeError("The locked KenLM config changed since the benchmark")
    if (transcript_manifest.get("split") != "public_test"
            or transcript_manifest.get("samples") != reference_manifest.get("rows")
            or transcript_manifest.get("checkpoint_weights_sha256") != asr["selected_checkpoint_sha256"]
            or transcript_manifest.get("beam_width") != 32
            or transcript_manifest.get("lm_config_sha256") != asr["kenlm_config_sha256"]
            or sha256(transcript_path) != reference_manifest.get("transcript_sha256")
            or sha256(transcript_path) != transcript_manifest.get("transcript_sha256")):
        raise RuntimeError("Cached public-test transcripts do not match the locked QUAN+KenLM configuration")

    data_root = find_data_root("train+public_test/train+public_test")
    split_dir = data_root / "public_test"
    items = list(load_jsonl(split_dir / "public_test.jsonl"))
    ids = [item["id"] for item in items]
    if len(ids) != 634 or len(set(ids)) != len(ids):
        raise RuntimeError(f"Expected 634 unique public-test IDs, found {len(ids)} rows / {len(set(ids))} unique")
    transcript_rows = read_jsonl(transcript_path)
    if [row["id"] for row in transcript_rows] != ids or any(
            not isinstance(row.get("transcript"), str) or not row["transcript"].strip()
            for row in transcript_rows):
        raise RuntimeError("QUAN transcript cache IDs/order/content do not match the public-test manifest")

    cot_hash = canonical_prompt_hash(cot=True)
    if cot_hash != config["agent"]["cot_prompt_sha256_utf8"]:
        raise RuntimeError("Canonical epoch-3 CoT prompt differs from the completed cohort run")

    version = api_json("/api/version").get("version")
    tags = api_json("/api/tags").get("models", [])
    model = next((row for row in tags if row.get("name") == MODEL_TAG), None)
    if version != EXPECTED_OLLAMA_VERSION or not model or model.get("digest") != model_manifest["ollama_digest"]:
        raise RuntimeError(f"Ollama epoch-3 fingerprint mismatch: version={version}, model={model}")

    source_paths = [
        ROOT / "COMPETITION.md", ROOT / "TASK.md",
        ROOT / "src" / "lifemate" / "agent.py", ROOT / "src" / "lifemate" / "pipeline.py",
        ROOT / "src" / "lifemate" / "asr.py", ROOT / "src" / "lifemate" / "tool_executor.py",
        ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "run_downstream.py",
        Path(__file__).resolve(), split_dir / "public_test.jsonl",
        split_dir / "public_test_tools_catalog.json", transcript_path, transcript_manifest_path,
        REFERENCE_MANIFEST_PATH, CONFIG_PATH, LOCK_PATH, INFERENCE_RUN_PATH,
        INFERENCE / "cohort_manifest.json", INFERENCE / "predictions-COT.jsonl",
        MODEL_MANIFEST_PATH, PROVENANCE / "local-audit.json", PROVENANCE / "merge-manifest.json",
        PROVENANCE / "conversion-manifest.json", PROVENANCE / "kaggle-version1-sha256.json",
        asr_checkpoint, lm_config_path,
    ]
    source_hashes = {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in source_paths}
    for relative, expected in config["input_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"English epoch-3 locked inference source changed after its run: {relative}")

    return {
        "status": "preflight_locked",
        "split": "public_test",
        "samples": len(items),
        "public_ids_sha256_newline_joined": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "model": {"tag": MODEL_TAG, "digest": model["digest"], "ollama_version": version,
                  "gguf_sha256": model_manifest["gguf"]["sha256"], "branch": "English-prompt epoch3 + CoT",
                  "parameters": model_manifest.get("model_info", {}).get("general.parameter_count"),
                  "context_limit_used": CONTEXT},
        "asr": {"kind": transcript_manifest.get("decoder"), "checkpoint_sha256": sha256(asr_checkpoint),
                "beam_width": transcript_manifest["beam_width"], "lm_config_sha256": sha256(lm_config_path),
                "transcript_sha256": sha256(transcript_path),
                "transcript_manifest_sha256": sha256(transcript_manifest_path),
                "transcript_count": len(transcript_rows)},
        "prompt": {"language": "en", "variant": "observation-grounded-v2 + exact TASK.md CoT checklist",
                   "cot_prompt_sha256_utf8": cot_hash, "cot_text_sha256_utf8": hashlib.sha256(
                       COT_TEXT.encode("utf-8")).hexdigest(), "raw_chain_of_thought_saved": False},
        "settings": {"thinking": True, "temperature": 0, "protocol": "JSON", "num_predict": CAP,
                     "num_ctx": CONTEXT, "num_batch": BATCH, "max_action_turns": MAX_STEPS,
                     "seed": SEED, "history_asr": "PhoASRModel", "retrieval": False},
        "toolenv": "one official ToolEnv per public-test sample; all executed calls taken from ToolEnv",
        "public_private_test_labels_used": False,
        "inference_lock": {"config_manifest_sha256": sha256(CONFIG_PATH),
                            "run_lock_sha256": sha256(LOCK_PATH),
                            "run_manifest_sha256": sha256(INFERENCE_RUN_PATH),
                            "all_locked_inputs_match": True},
        "source_sha256": source_hashes,
        "paths": {"transcripts": str(transcript_path.relative_to(ROOT)),
                  "transcript_manifest": str(transcript_manifest_path.relative_to(ROOT)),
                  "submission": str(SUBMISSION_PATH.relative_to(ROOT)),
                  "predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
                  "manifest": str(MANIFEST_PATH.relative_to(ROOT))},
    }


def prepare() -> None:
    if any(path.exists() for path in (SUBMISSION_PATH, PREDICTIONS_PATH, MANIFEST_PATH)):
        raise FileExistsError("A target output already exists; refusing to overwrite a submission")
    if any(path.exists() for path in (WORK_SUBMISSION, WORK_PREDICTIONS, WORK_REQUESTS, RUN_LOG_PATH, OUT / "submission-lock.json")):
        raise FileExistsError(f"Submission working artifacts already exist under {OUT}; inspect them before continuing")
    locked_inputs = build_locked_inputs()
    write_json(OUT / "submission-lock.json", {"created_epoch": time.time(), "locked_inputs": locked_inputs})
    print(json.dumps({"status": "preflight_locked", "samples": locked_inputs["samples"],
                      "model": locked_inputs["model"], "prompt": locked_inputs["prompt"],
                      "transcript_sha256": locked_inputs["asr"]["transcript_sha256"]},
                     ensure_ascii=False, indent=2), flush=True)


def verify_resume_prefix(items: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = (WORK_SUBMISSION, WORK_PREDICTIONS, WORK_REQUESTS, RUN_LOG_PATH)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot resume: missing working artifacts: {missing}")

    submission_rows = read_jsonl(WORK_SUBMISSION)
    detail_rows = read_jsonl(WORK_PREDICTIONS)
    request_rows = read_jsonl(WORK_REQUESTS)
    log_lines = RUN_LOG_PATH.read_text(encoding="utf-8-sig").splitlines()
    counts = {len(submission_rows), len(detail_rows), len(log_lines)}
    if len(counts) != 1:
        raise RuntimeError(
            "Cannot resume: submission, prediction and run-log checkpoints have different row counts; "
            "artifacts were left untouched for inspection")

    completed = len(submission_rows)
    if completed > len(items):
        raise RuntimeError(f"Cannot resume: checkpoint has {completed} rows for only {len(items)} public-test samples")
    expected_ids = [item["id"] for item in items[:completed]]
    if [row.get("id") for row in submission_rows] != expected_ids or [
            row.get("id") for row in detail_rows] != expected_ids:
        raise RuntimeError("Cannot resume: saved rows are not an exact prefix of public-test IDs/order")

    request_counts: dict[str, int] = {}
    for request in request_rows:
        sid = request.get("sample_id")
        if not isinstance(sid, str) or sid not in set(expected_ids):
            raise RuntimeError(f"Cannot resume: telemetry contains an unknown or unfinished sample ID: {sid}")
        request_counts[sid] = request_counts.get(sid, 0) + 1

    for index, (row, detail, line, sid) in enumerate(
            zip(submission_rows, detail_rows, log_lines, expected_ids), 1):
        if (row.get("tool_calls") != detail.get("tool_calls")
                or row.get("state_hash") != detail.get("state_hash")
                or not isinstance(row.get("state_hash"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["state_hash"])
                or detail.get("toolenv_count") != 1
                or not isinstance(detail.get("latency_seconds"), (int, float))):
            raise RuntimeError(f"Cannot resume: submission/prediction checkpoint mismatch at {sid}")
        match = re.fullmatch(
            r"\[(\d+)/(\d+)\] ([^ ]+) calls=(\d+) latency=([0-9.]+)s requests=(\d+) error=(.*)", line)
        if (not match or int(match.group(1)) != index or int(match.group(2)) != len(items)
                or match.group(3) != sid or int(match.group(4)) != len(row["tool_calls"])
                or int(match.group(6)) != request_counts.get(sid, 0)
                or match.group(7) != str(detail.get("runtime_error"))):
            raise RuntimeError(f"Cannot resume: run log/telemetry does not confirm a complete checkpoint for {sid}")
        expected_request_ids = [f"C:{sid}:r{request_index}" for request_index in range(1, request_counts[sid] + 1)]
        actual_request_ids = [request.get("request_id") for request in request_rows if request.get("sample_id") == sid]
        if actual_request_ids != expected_request_ids:
            raise RuntimeError(f"Cannot resume: telemetry request IDs are incomplete or out of order for {sid}")

    return completed, submission_rows, detail_rows, request_rows


def run_submission(resume: bool = False) -> None:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run with system Python, outside .venv, to reach the local Ollama server")
    lock_path = OUT / "submission-lock.json"
    locked = read_json(lock_path)
    current = build_locked_inputs()
    final_outputs = (SUBMISSION_PATH, PREDICTIONS_PATH, MANIFEST_PATH)
    if any(path.exists() for path in final_outputs):
        raise FileExistsError("A final submission output already exists; refusing to overwrite it")

    working_paths = (WORK_SUBMISSION, WORK_PREDICTIONS, WORK_REQUESTS, RUN_LOG_PATH)
    has_working_files = any(path.exists() for path in working_paths)
    start_index = 0
    prior_details: list[dict[str, Any]] = []
    prior_requests: list[dict[str, Any]] = []
    if resume:
        if not has_working_files:
            raise FileNotFoundError("--resume requested, but no submission checkpoint exists")
        prior_inputs = locked.get("locked_inputs")
        if not isinstance(prior_inputs, dict):
            raise RuntimeError("Submission lock has no locked_inputs object")
        # The only permitted lock change during continuation is this generator's own
        # source hash, because resume support is added after the original checkpoint.
        own_source = Path(__file__).resolve().relative_to(ROOT).as_posix()
        old_comparable = json.loads(json.dumps(prior_inputs))
        new_comparable = json.loads(json.dumps(current))
        old_hashes = old_comparable.get("source_sha256", {})
        new_hashes = new_comparable.get("source_sha256", {})
        if own_source not in old_hashes or own_source not in new_hashes:
            raise RuntimeError("Cannot resume: generator source hash is missing from the preflight lock")
        old_hashes.pop(own_source)
        new_hashes.pop(own_source)
        if old_comparable != new_comparable:
            raise RuntimeError("Submission source/model/transcript fingerprints changed beyond the resume code")
        data_root = find_data_root("train+public_test/train+public_test")
        items = list(load_jsonl(data_root / "public_test" / "public_test.jsonl"))
        start_index, _, prior_details, prior_requests = verify_resume_prefix(items)
        history = locked.get("resume_history", [])
        if not isinstance(history, list):
            raise RuntimeError("Submission lock has malformed resume_history")
        history.append({"resumed_epoch": time.time(), "completed_samples_before_resume": start_index,
                        "previous_generator_sha256": prior_inputs["source_sha256"][own_source],
                        "generator_sha256": current["source_sha256"][own_source]})
        write_json(lock_path, {"created_epoch": locked.get("created_epoch"),
                               "locked_inputs": current, "resume_history": history})
    else:
        if locked.get("locked_inputs") != current:
            raise RuntimeError("Submission source/model/transcript fingerprints changed after preflight")
        if has_working_files:
            raise FileExistsError("Submission working artifacts exist; pass --resume to verify and continue them")

    seed_everything(SEED)
    data_root = find_data_root("train+public_test/train+public_test")
    split_dir = data_root / "public_test"
    items = list(load_jsonl(split_dir / "public_test.jsonl"))
    transcript_path = ROOT / locked["locked_inputs"]["paths"]["transcripts"]
    transcript_map = {row["id"]: row["transcript"] for row in read_jsonl(transcript_path)}

    os.environ.update({
        "OLLAMA_HOST": OLLAMA_HOST,
        "LIFEMATE_LLM_BACKEND": "ollama",
        "LIFEMATE_OLLAMA_MODEL": MODEL_TAG,
        "LIFEMATE_OLLAMA_CONTEXT": str(CONTEXT),
        "LIFEMATE_OLLAMA_NUM_PREDICT": str(CAP),
        "LIFEMATE_OLLAMA_NUM_BATCH": str(BATCH),
        "LIFEMATE_OLLAMA_THINK": "true",
        "LIFEMATE_OLLAMA_PROTOCOL": "json",
        "LIFEMATE_OLLAMA_TELEMETRY": "1",
        "LIFEMATE_PROMPT_LANGUAGE": "en",
    })
    agent = Agent(max_new_tokens=CAP, context_limit=CONTEXT,
                  ollama_thinking=True, ollama_native_tools=False)
    original_generate = agent._generate

    def generate_with_cot(messages):
        return original_generate(inject_cot(messages))

    agent._generate = generate_with_cot
    pipeline = LifeMatePipeline(data_root, "public_test", asr=PhoASRModel(), agent=agent,
                                max_steps=MAX_STEPS)

    original_create = pipeline_module.create_tool_env
    active_session: dict[str, Any] = {"count": 0, "env": None}

    def create_one_tool_env(*args, **kwargs):
        active_session["count"] += 1
        if active_session["count"] != 1:
            raise RuntimeError("More than one ToolEnv was created for a sample")
        active_session["env"] = original_create(*args, **kwargs)
        return active_session["env"]

    pipeline_module.create_tool_env = create_one_tool_env
    OUT.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = [float(detail["latency_seconds"]) for detail in prior_details]
    runtime_errors = sum(detail.get("runtime_error") is not None for detail in prior_details)
    request_count = len(prior_requests)
    cap_hits = sum(request.get("eval_count", 0) >= CAP for request in prior_requests)
    started = time.perf_counter()
    try:
        mode = "a" if resume else "w"
        with WORK_SUBMISSION.open(mode, encoding="utf-8", newline="\n") as submission, \
                WORK_PREDICTIONS.open(mode, encoding="utf-8", newline="\n") as predictions, \
                WORK_REQUESTS.open(mode, encoding="utf-8", newline="\n") as requests, \
                RUN_LOG_PATH.open(mode, encoding="utf-8", newline="\n") as run_log:
            for index, item in enumerate(items[start_index:], start_index + 1):
                sid = item["id"]
                if sid not in transcript_map:
                    raise RuntimeError(f"Missing locked QUAN transcript for {sid}")
                active_session = {"count": 0, "env": None}
                agent._telemetry_sample_id = sid
                request_start = len(agent.request_telemetry)
                action_turn = [0]
                original_next = agent.next_action

                def tracked_next(*args, **kwargs):
                    action_turn[0] += 1
                    agent._telemetry_action_turn = action_turn[0]
                    return original_next(*args, **kwargs)

                agent.next_action = tracked_next
                sample_started = time.perf_counter()
                runtime_error = None
                result = None
                try:
                    result = pipeline.run(item, transcript_override=transcript_map[sid])
                except Exception as exc:
                    runtime_errors += 1
                    runtime_error = f"{type(exc).__name__}: {exc}"
                finally:
                    agent.next_action = original_next

                env = active_session["env"]
                if active_session["count"] != 1 or env is None:
                    raise RuntimeError(f"The sample {sid} did not receive exactly one usable ToolEnv; stopping without finalizing")
                calls = env.get_tool_calls()
                state_hash = env.get_state_hash()
                if result is not None and result.get("tool_calls") != calls:
                    raise RuntimeError(f"Pipeline calls differ from ToolEnv calls for {sid}; refusing to omit executed actions")
                if not isinstance(state_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", state_hash):
                    raise RuntimeError(f"ToolEnv returned an invalid state hash for {sid}")
                elapsed = result["latency_seconds"] if result is not None else time.perf_counter() - sample_started
                latencies.append(float(elapsed))
                row = {"id": sid, "tool_calls": calls, "state_hash": state_hash}
                detail = {"id": sid, "transcript_used": transcript_map[sid], "tool_calls": calls,
                          "state_hash": state_hash,
                          "observations": result["observations"] if result is not None else [],
                          "model_output": result["model_output"] if result is not None else [],
                          "latency_seconds": elapsed, "runtime_error": runtime_error,
                          "toolenv_count": active_session["count"]}
                submission.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                predictions.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
                submission.flush()
                predictions.flush()
                request_rows = agent.request_telemetry[request_start:]
                for request_index, request_row in enumerate(request_rows, 1):
                    request_row["request_id"] = f"C:{sid}:r{request_index}"
                    requests.write(json.dumps(request_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if request_row.get("eval_count", 0) >= CAP:
                        cap_hits += 1
                requests.flush()
                request_count += len(request_rows)
                agent._telemetry_action_turn = None
                line = (f"[{index}/{len(items)}] {sid} calls={len(calls)} latency={elapsed:.2f}s "
                        f"requests={len(request_rows)} error={runtime_error}")
                run_log.write(line + "\n")
                run_log.flush()
                print(line, flush=True)
    finally:
        pipeline_module.create_tool_env = original_create

    elapsed_run = time.perf_counter() - started
    ids = [item["id"] for item in items]
    submission_rows = read_jsonl(WORK_SUBMISSION)
    detail_rows = read_jsonl(WORK_PREDICTIONS)
    if ([row["id"] for row in submission_rows] != ids
            or [row["id"] for row in detail_rows] != ids
            or any(row["toolenv_count"] != 1 for row in detail_rows)):
        raise RuntimeError("Generated submission does not have exact public-test ID coverage or one ToolEnv/sample")

    validator = ROOT / "exp" / "lifemate-pipeline" / "scripts" / "validate_submission.py"
    check = subprocess.run([sys.executable, str(validator), "--split", "public_test",
                            "--data-root", str(data_root), "--submission", str(WORK_SUBMISSION)],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (OUT / "official-schema-validation.log").write_text(
        check.stdout + ("\nSTDERR:\n" + check.stderr if check.stderr else ""), encoding="utf-8")
    if check.returncode:
        raise RuntimeError(f"Official schema/order validation failed; see {OUT / 'official-schema-validation.log'}")
    validation = json.loads(check.stdout)
    if validation.get("submission_rows") != len(items) or validation.get("errors"):
        raise RuntimeError("Official validator did not accept all generated rows")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(WORK_SUBMISSION), str(SUBMISSION_PATH))
    shutil.move(str(WORK_PREDICTIONS), str(PREDICTIONS_PATH))
    values = latencies
    manifest = {
        "status": "generated_and_schema_validated",
        "submission": str(SUBMISSION_PATH.relative_to(ROOT)),
        "predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "split": "public_test", "rows": len(items), "unique_ids": len(set(ids)),
        "submission_sha256": sha256(SUBMISSION_PATH), "predictions_sha256": sha256(PREDICTIONS_PATH),
        "source_lock": str((OUT / "submission-lock.json").relative_to(ROOT)),
        "run_log": str((OUT / "submission-run.log").relative_to(ROOT)),
        "requests_telemetry": str(WORK_REQUESTS.relative_to(ROOT)),
        "model": locked["locked_inputs"]["model"],
        "asr": locked["locked_inputs"]["asr"], "prompt": locked["locked_inputs"]["prompt"],
        "settings": locked["locked_inputs"]["settings"],
        "toolenv_sessions": {"expected": len(items), "created": len(detail_rows), "exactly_one_per_sample": True},
        "schema_validation": validation,
        "runtime_errors": runtime_errors,
        "latency_seconds_local_diagnostic": {
            "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
            "samples_over_30s": sum(value > 30 for value in values),
            "note": "Local runtime diagnostic only; RTX 4050 timings are not NVIDIA A30 measurements and do not change the JSONL output."},
        "ollama_requests": request_count, "requests_at_num_predict_cap": cap_hits,
        "elapsed_seconds": elapsed_run,
        "elapsed_seconds_note": "Wall time for the final generation process; earlier pre-pause process time is not captured.",
        "resumed_from_samples": start_index, "public_lb_submitted": False,
        "public_private_test_labels_used": False, "raw_chain_of_thought_saved": False,
        "limitations": [
            "This is a public_test JSONL candidate only; no Public LB submission was made.",
            "The enriched cohort60 result is a screening result, not a public-test score estimate.",
            "The training report has no input-weight hash; local provenance cannot prove the remote job consumed identical base bytes.",
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    write_json(OUT / "summary.json", manifest)
    print(json.dumps({"status": manifest["status"], "submission": manifest["submission"],
                      "rows": manifest["rows"], "schema_errors": validation.get("errors"),
                      "runtime_errors": runtime_errors,
                      "mean_latency_seconds_local": manifest["latency_seconds_local_diagnostic"]["mean"],
                      "samples_over_30s_local": manifest["latency_seconds_local_diagnostic"]["samples_over_30s"],
                      "public_lb_submitted": False}, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the English-prompt epoch-3 Qwen SFT + CoT public-test submission")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run_submission(resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
