from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "btc-scoring" / "scripts"))
from lifemate.utils import find_data_root, load_jsonl, seed_everything  # noqa: E402

EXP = ROOT / "exp" / "quan-continuation-cot-v1"
OUT = EXP / "output" / "downstream"
ASR_PREDICTIONS = EXP / "output" / "asr" / "checkpoint-verification-rerun-val309" / "predictions.jsonl"
COHORT_SOURCE = ROOT / "exp" / "inference-latency-optimization-val60" / "output" / "cohort_manifest.json"
MODEL = "qwen3.5:4b-q4_K_M"
HOST = "http://127.0.0.1:11434"
EXPECTED_DIGEST = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
CAP = 768
CONTEXT = 8192
BATCH = 512
SEED = 17
MAX_STEPS = 5
COT_TEXT = (
    "Before choosing the next action, silently reason through this checklist:\n\n"
    "1. Break the user's request into the outcomes that must be achieved.\n"
    "2. For each outcome, identify the supporting evidence in the transcript, context/profile, history, available tool schemas, and latest tool observation. Treat tool-returned values as authoritative.\n"
    "3. Mark outcomes already completed and select exactly one next action that advances an unfinished outcome.\n"
    "4. Check the selected tool and every argument against its schema and the evidence. Do not fill gaps by guessing.\n"
    "5. If a required value is missing and cannot be safely inferred, ask only for the missing information. Otherwise, execute the selected action.\n"
    "6. After the tool result, update which outcomes are complete and repeat this checklist for the next turn. Stop only when all outcomes are complete or cannot be carried out safely.\n\n"
    "Keep this reasoning internal. Do not include the checklist, rationale, or any explanation in the response; return only the required action JSON."
)
OUTPUT_ANCHOR = "Return exactly one JSON object per turn, with no Markdown or explanation: "
SOURCE_FILES = [
    "src/lifemate/agent.py", "src/lifemate/pipeline.py", "src/lifemate/asr.py",
    "src/lifemate/tool_executor.py", "src/lifemate/schemas.py", "src/lifemate/utils.py",
    "src/lifemate/retrieval.py",
    "exp/btc-scoring/scripts/score_btc.py", "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_json(path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(HOST + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))


def inject_cot(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = copy.deepcopy(messages)
    systems = [message for message in copied if message.get("role") == "system"]
    if len(systems) != 1 or OUTPUT_ANCHOR not in systems[0].get("content", ""):
        raise RuntimeError("Cannot place the CoT checklist immediately before the system output-format instruction")
    systems[0]["content"] = systems[0]["content"].replace(OUTPUT_ANCHOR, COT_TEXT + "\n\n" + OUTPUT_ANCHOR, 1)
    return copied


def canonical_prompt_hash(cot: bool) -> str:
    from lifemate.agent import Agent

    os.environ.update({"LIFEMATE_LLM_BACKEND": "ollama", "LIFEMATE_OLLAMA_THINK": "true",
                       "LIFEMATE_OLLAMA_PROTOCOL": "json", "LIFEMATE_PROMPT_LANGUAGE": "en"})
    agent = Agent(max_new_tokens=CAP, context_limit=CONTEXT, ollama_thinking=True, ollama_native_tools=False)
    captured: list[str] = []

    def capture(messages: list[dict[str, Any]]) -> str:
        prepared = inject_cot(messages) if cot else messages
        captured.append(next(m["content"] for m in prepared if m.get("role") == "system"))
        return '{"action":"stop"}'

    agent._generate = capture
    action = agent.next_action("prompt hash probe", {}, [], [], [])
    if action.get("action") != "stop" or len(captured) != 1:
        raise RuntimeError("Canonical prompt capture did not complete locally")
    return hashlib.sha256(captured[0].encode("utf-8")).hexdigest()


def model_fingerprint() -> tuple[str, str]:
    version = api_json("/api/version").get("version")
    models = api_json("/api/tags").get("models", [])
    found = next((model for model in models if model.get("name") == MODEL), None)
    if not found or found.get("digest") != EXPECTED_DIGEST:
        raise RuntimeError(f"Ollama model fingerprint mismatch: {found}")
    if not version:
        raise RuntimeError("Ollama version API returned no version")
    return version, found["digest"]


def verify_locked_inputs(config: dict) -> None:
    for relative, expected in config["input_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Locked downstream input changed after preflight: {relative}")


def prepare() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_source = read_json(COHORT_SOURCE)
    ids = cohort_source["ids"]
    val_ids = json.loads((ROOT / "data" / "splits" / "val_ids.json").read_text(encoding="utf-8-sig"))
    train_ids = json.loads((ROOT / "data" / "splits" / "train_ids.json").read_text(encoding="utf-8-sig"))
    asr_rows = read_jsonl(ASR_PREDICTIONS)
    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train" / "train.jsonl")}
    if len(ids) != 60 or len(set(ids)) != 60 or set(ids) - set(val_ids):
        raise RuntimeError("Cohort60 is not an exact, unique subset of the locked held-out val309 split")
    if set(ids) & set(train_ids):
        raise RuntimeError("Validation cohort unexpectedly overlaps the fine-tuning train split")
    transcripts = {row["id"]: row.get("prediction", "") for row in asr_rows}
    if [row["id"] for row in asr_rows] != val_ids or set(ids) - transcripts.keys() or set(ids) - items.keys():
        raise RuntimeError("Exact validation transcript, ID order, or cohort item coverage failed")
    version, digest = model_fingerprint()
    expected_ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if expected_ids_hash != cohort_source.get("ids_sha256_newline_joined"):
        raise RuntimeError("Cohort60 ID hash differs from its locked manifest")
    current_ckpt = ROOT / "exp" / "quan-transcribe-lifemate-domain-v1" / "output" / "model" / "model.safetensors"
    pho_asr_dir = ROOT / "models" / "pho-asr-whisper-small"
    if not pho_asr_dir.is_dir():
        raise FileNotFoundError(f"PhoASR history model is missing: {pho_asr_dir}")
    pho_asr_files = sorted(path for path in pho_asr_dir.rglob("*") if path.is_file())
    if not pho_asr_files:
        raise FileNotFoundError(f"PhoASR history model has no files: {pho_asr_dir}")
    prompt_hash = canonical_prompt_hash(False)
    cot_hash = canonical_prompt_hash(True)
    if prompt_hash == cot_hash:
        raise RuntimeError("CoT prompt variant did not alter the canonical system prompt")
    input_paths = [ROOT / p for p in SOURCE_FILES]
    input_paths += [COHORT_SOURCE, ASR_PREDICTIONS, ROOT / "exp" / "quan-continuation-cot-v1" / "output" / "checkpoint-selection.json",
                    current_ckpt, ROOT / "models" / "quan-transcribe" / "lm" / "lm_config.json",
                    data_root / "train" / "train.jsonl", data_root / "train" / "train_tools_catalog.json"]
    input_paths += pho_asr_files
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    cohort_out = {"source": "exp/inference-latency-optimization-val60/output/cohort_manifest.json",
                  "source_manifest_sha256": sha256(COHORT_SOURCE), "ids": ids,
                  "ids_sha256_newline_joined": expected_ids_hash, "samples": 60,
                  "subset_of_heldout_val309": True, "enriched_screening_cohort": True,
                  "used_for_training": False}
    config = {
        "experiment": "quan-continuation-cot-v1", "status": "downstream_config_locked",
        "ollama": {"host": HOST, "version": version, "model": MODEL, "model_digest": digest,
                   "thinking": True, "temperature": 0, "protocol": "JSON", "num_predict": CAP,
                   "num_ctx": CONTEXT, "num_batch": BATCH, "seed_python": SEED,
                   "ollama_seed_option": "not sent; the existing Agent Ollama request exposes temperature/context/token cap/batch only"},
        "agent": {"current_prompt": "observation-grounded-v2 English", "current_prompt_sha256_utf8": prompt_hash,
                  "cot_prompt_sha256_utf8": cot_hash, "cot_delta": "Append the TASK checklist immediately before output-format instructions; no other system text changes.",
                  "raw_chain_of_thought_saved": False, "max_action_turns": MAX_STEPS,
                  "history_asr": "PhoASRModel, unchanged across branches", "retrieval": False},
        "asr": {"selected_checkpoint": "exp/quan-transcribe-lifemate-domain-v1/output/model",
                "selected_checkpoint_sha256": sha256(current_ckpt),
                "selection_reason_file": "exp/quan-continuation-cot-v1/output/checkpoint-selection.json",
                "transcript_source": "redecoded current-checkpoint val309 artifact, exact match to archived current checkpoint 309/309",
                "transcript_file_sha256": sha256(ASR_PREDICTIONS), "beam_width": 32,
                "kenlm_config_sha256": sha256(ROOT / "models" / "quan-transcribe" / "lm" / "lm_config.json"),
                "history_asr_model": "models/pho-asr-whisper-small",
                "history_asr_model_manifest_sha256": hashlib.sha256(json.dumps(
                    {str(path.relative_to(ROOT)): sha256(path) for path in pho_asr_files},
                    sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()},
        "toolenv": {"prediction_envs_per_sample_per_branch": 1,
                    "scorer_gold_replay": "official ToolEnv via exp/btc-scoring/scripts/score_btc.py"},
        "cohort": cohort_out, "input_sha256": input_hashes,
        "branches": {"A": {"asr": "selected current checkpoint", "prompt": "observation-grounded-v2 English"},
                     "B": {"asr": "same selected current checkpoint (B=A fallback control)", "prompt": "observation-grounded-v2 English"},
                     "COT": {"asr": "same selected current checkpoint", "prompt": "observation-grounded-v2 English + TASK checklist"}},
        "branch_order": [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]],
        "step2_selected_checkpoint_equals_control": True,
        "public_private_test_used": False,
        "execution_environment": "system Python with native Ollama localhost; not .venv",
    }
    if len(items) < len(cohort_source["ids"]):
        raise RuntimeError("Cohort input check unexpectedly failed")
    if (OUT / "config_manifest.json").exists() or (OUT / "cohort_manifest.json").exists():
        raise FileExistsError("Downstream config artifacts already exist; refusing to overwrite locked setup")
    write_json(OUT / "cohort_manifest.json", cohort_out)
    write_json(OUT / "config_manifest.json", config)
    print(json.dumps({"cohort": 60, "val_ids": len(val_ids), "train_ids": len(train_ids),
                      "ollama_version": version, "model_digest": digest,
                      "base_prompt_sha256": prompt_hash, "cot_prompt_sha256": cot_hash,
                      "current_checkpoint_sha256": sha256(current_ckpt)}, indent=2))
    return 0


def warmup(branch: str, round_index: int) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Return {} as JSON."}],
            "stream": False, "think": True, "format": "json",
            "options": {"temperature": 0, "num_predict": 8, "num_ctx": CONTEXT, "num_batch": BATCH}}
    started = time.perf_counter()
    response = api_json("/api/chat", body)
    result = {"branch": branch, "round": round_index, "excluded_from_measurements": True,
              "wall_seconds": time.perf_counter() - started,
              "prompt_eval_count": response.get("prompt_eval_count"),
              "eval_count": response.get("eval_count"), "load_duration_ns": response.get("load_duration"),
              "total_duration_ns": response.get("total_duration"), "timestamp_epoch": time.time()}
    write_json(OUT / f"warmup-{branch}-round{round_index}.json", result)
    return result


def run_block(branch: str, round_index: int, ids: list[str], transcripts: dict[str, str],
              items: dict[str, dict], cot: bool = False) -> None:
    from lifemate.agent import Agent
    from lifemate.asr import PhoASRModel
    from lifemate.pipeline import LifeMatePipeline

    if cot and branch != "COT":
        raise ValueError("The CoT prompt is reserved for the COT branch")
    pred_path = OUT / f"predictions-{branch}.jsonl"
    trace_path = OUT / f"step-traces-{branch}.jsonl"
    req_path = OUT / f"requests-{branch}.jsonl"
    done = read_jsonl(pred_path)
    done_ids = [row["id"] for row in done]
    if set(done_ids) & set(ids):
        raise RuntimeError(f"Refusing to duplicate already completed IDs for branch {branch}")

    os.environ.update({"LIFEMATE_LLM_BACKEND": "ollama", "LIFEMATE_OLLAMA_MODEL": MODEL,
        "LIFEMATE_OLLAMA_CONTEXT": str(CONTEXT), "LIFEMATE_OLLAMA_NUM_PREDICT": str(CAP),
        "LIFEMATE_OLLAMA_NUM_BATCH": str(BATCH), "LIFEMATE_OLLAMA_THINK": "true",
        "LIFEMATE_OLLAMA_PROTOCOL": "json", "LIFEMATE_PROMPT_LANGUAGE": "en",
        "LIFEMATE_OLLAMA_TELEMETRY": "1", "OLLAMA_HOST": HOST})
    seed_everything(SEED)
    agent = Agent(max_new_tokens=CAP, context_limit=CONTEXT, ollama_thinking=True, ollama_native_tools=False)
    agent._telemetry_sample_id = None
    if cot:
        original_generate = agent._generate

        def generate_with_cot(messages):
            return original_generate(inject_cot(messages))

        agent._generate = generate_with_cot
    original_request = agent._ollama_request

    def request_with_prompt_hash(messages, tools=None):
        start = len(agent.request_telemetry)
        system_message = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        prompt_hash = hashlib.sha256(system_message.encode("utf-8")).hexdigest()
        response = original_request(messages, tools)
        for row in agent.request_telemetry[start:]:
            row["system_prompt_sha256_utf8"] = prompt_hash
            row["prompt_variant"] = "cot" if cot else "observation-grounded-v2"
            row["model_digest"] = EXPECTED_DIGEST
            row["round"] = round_index
        return response

    agent._ollama_request = request_with_prompt_hash
    data_root = find_data_root("train+public_test/train+public_test")
    pipeline = LifeMatePipeline(data_root, "train", asr=PhoASRModel(), agent=agent, max_steps=MAX_STEPS)
    prior_traces = read_jsonl(trace_path)
    prior_requests = read_jsonl(req_path)
    start_block = time.time()
    preds, traces, requests = list(done), list(prior_traces), list(prior_requests)
    for offset, sid in enumerate(ids, 1):
        agent._telemetry_sample_id = sid
        telemetry_start = len(agent.request_telemetry)
        action_turn = [0]
        original_next = agent.next_action

        def tracked_next(*args, **kwargs):
            action_turn[0] += 1
            agent._telemetry_action_turn = action_turn[0]
            return original_next(*args, **kwargs)

        agent.next_action = tracked_next
        started = time.perf_counter()
        error = None
        try:
            result = pipeline.run(items[sid], transcript_override=transcripts[sid])
            pred = {"id": sid, "asr_text": transcripts[sid], "transcript_used": transcripts[sid],
                    "tool_calls": [{"name": c.get("name"), "arguments": c.get("arguments", {})}
                                   for c in result.get("tool_calls", [])],
                    "state_hash": result.get("state_hash"), "observations": result.get("observations", []),
                    "final_state": result.get("final_state", {}), "model_output": result.get("model_output", []),
                    "latency_seconds": result.get("latency_seconds", time.perf_counter() - started)}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            pred = {"id": sid, "asr_text": transcripts[sid], "transcript_used": transcripts[sid],
                    "tool_calls": [], "state_hash": None, "observations": [], "final_state": {},
                    "model_output": [], "latency_seconds": time.perf_counter() - started, "error": error}
        finally:
            agent.next_action = original_next
        usage = agent.request_telemetry[telemetry_start:]
        for i, row in enumerate(usage, 1):
            row["request_id"] = f"{branch}:{sid}:r{i}"
            row["sample_request_index"] = i
        trace = {"id": sid, "branch": branch, "round": round_index,
                 "latency_seconds": pred["latency_seconds"], "runtime_error": error,
                 "turns": pred["model_output"], "request_count": len(usage),
                 "action_turns": len(pred["model_output"]),
                 "repair_request_count": sum(row.get("request_kind") == "repair" for row in usage)}
        preds.append(pred)
        traces.append(trace)
        requests.extend(usage)
        write_jsonl(pred_path, preds)
        write_jsonl(trace_path, traces)
        write_jsonl(req_path, requests)
        print(f"branch={branch} round={round_index} [{offset}/{len(ids)}] {sid} "
              f"{pred['latency_seconds']:.2f}s requests={len(usage)} error={error}", flush=True)
        if error:
            write_json(OUT / f"run-{branch}.json", {"branch": branch, "status": "stopped_on_runtime_error",
                "samples_completed": len(preds), "error": error, "round": round_index,
                "started_epoch": start_block, "ended_epoch": time.time()})
            raise RuntimeError(f"Stopping after runtime exception for branch={branch}, sample={sid}: {error}")
    meta_path = OUT / f"run-{branch}.json"
    meta = read_json(meta_path) if meta_path.exists() else {"branch": branch, "blocks": []}
    meta.setdefault("blocks", []).append({"round": round_index, "ids": ids,
        "started_epoch": start_block, "ended_epoch": time.time(), "warmup": f"warmup-{branch}-round{round_index}.json"})
    meta.update({"status": "in_progress", "model": MODEL, "model_digest": EXPECTED_DIGEST,
        "num_predict": CAP, "num_ctx": CONTEXT, "num_batch": BATCH,
        "temperature": 0, "thinking": True, "protocol": "JSON", "prompt_variant": "cot" if cot else "observation-grounded-v2",
        "asr_checkpoint": "current selected checkpoint", "max_action_turns": MAX_STEPS,
        "toolenv_per_sample": 1, "history_asr": "PhoASRModel", "seed": SEED,
        "samples_completed": len(preds), "execution_order": "paired 15-ID blocks"})
    write_json(meta_path, meta)


def run_step2() -> int:
    if not (OUT / "config_manifest.json").is_file():
        raise FileNotFoundError("Run --prepare first to lock cohort, source and server config")
    config = read_json(OUT / "config_manifest.json")
    verify_locked_inputs(config)
    version, digest = model_fingerprint()
    if version != config["ollama"]["version"] or digest != config["ollama"]["model_digest"]:
        raise RuntimeError("Ollama fingerprint changed after preflight")
    for branch in ("A", "B"):
        if any((OUT / f"{name}-{branch}.jsonl").exists() for name in ("predictions", "step-traces", "requests")):
            raise FileExistsError(f"Branch {branch} artifacts exist; this runner refuses to overwrite or silently resume")
    ids = config["cohort"]["ids"]
    val_ids = json.loads((ROOT / "data" / "splits" / "val_ids.json").read_text(encoding="utf-8-sig"))
    transcript_rows = read_jsonl(ASR_PREDICTIONS)
    transcript_map = {row["id"]: row["prediction"] for row in transcript_rows}
    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train" / "train.jsonl")}
    if [row["id"] for row in transcript_rows] != val_ids or set(ids) - transcript_map.keys():
        raise RuntimeError("Transcript IDs changed after downstream preflight")
    rounds = [ids[i * 15:(i + 1) * 15] for i in range(4)]
    order = [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]]
    write_json(OUT / "step2-execution-order.json", {"rounds": [{"round": i + 1, "ids": block, "branch_order": order[i]}
        for i, block in enumerate(rounds)], "paired_by_sample": True, "num_predict": CAP, "num_batch": BATCH})
    for round_index, block in enumerate(rounds, 1):
        for branch in order[round_index - 1]:
            warmup(branch, round_index)
            run_block(branch, round_index, block, transcript_map, items, cot=False)
    for branch in ("A", "B"):
        predictions = read_jsonl(OUT / f"predictions-{branch}.jsonl")
        traces = read_jsonl(OUT / f"step-traces-{branch}.jsonl")
        if [row["id"] for row in predictions] != ids or [row["id"] for row in traces] != ids:
            raise RuntimeError(f"Branch {branch} does not exactly cover the ordered fixed cohort")
        meta = read_json(OUT / f"run-{branch}.json")
        meta.update({"status": "complete", "ended_epoch": time.time()})
        write_json(OUT / f"run-{branch}.json", meta)
    return 0


def run_cot() -> int:
    config = read_json(OUT / "config_manifest.json")
    verify_locked_inputs(config)
    version, digest = model_fingerprint()
    if version != config["ollama"]["version"] or digest != config["ollama"]["model_digest"]:
        raise RuntimeError("Ollama fingerprint changed after preflight")
    a_meta = read_json(OUT / "run-A.json")
    if a_meta.get("status") != "complete":
        raise RuntimeError("Step 3 requires the completed step-2 prompt-v2 control branch A")
    if any((OUT / f"{name}-COT.jsonl").exists() for name in ("predictions", "step-traces", "requests")):
        raise FileExistsError("COT artifacts exist; this runner refuses to overwrite or silently resume")
    ids = config["cohort"]["ids"]
    transcript_rows = read_jsonl(ASR_PREDICTIONS)
    transcript_map = {row["id"]: row["prediction"] for row in transcript_rows}
    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train" / "train.jsonl")}
    write_json(OUT / "step3-execution-order.json", {
        "branch_A_reused_from": "step2 branch A", "branch_A_status": a_meta["status"],
        "branch_COT_ids": ids, "paired_by_id": True,
        "reuse_requirements_checked": {"same_checkpoint": True, "same_model_digest": digest,
            "same_ollama_version": version, "same_num_predict_ctx_batch_temperature_protocol": True,
            "same_ASR_transcripts": True, "same_source_prompt_v2_hash": config["agent"]["current_prompt_sha256_utf8"]},
        "only_change_B": "Insert TASK CoT checklist before the output-format instruction; all other source/runtime settings are fixed."})
    for round_index in range(1, 5):
        block = ids[(round_index - 1) * 15:round_index * 15]
        warmup("COT", round_index)
        run_block("COT", round_index, block, transcript_map, items, cot=True)
    predictions = read_jsonl(OUT / "predictions-COT.jsonl")
    if [row["id"] for row in predictions] != ids:
        raise RuntimeError("COT branch does not exactly cover the ordered fixed cohort")
    meta = read_json(OUT / "run-COT.json")
    meta.update({"status": "complete", "ended_epoch": time.time()})
    write_json(OUT / "run-COT.json", meta)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--step2", action="store_true")
    group.add_argument("--cot", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        return prepare()
    if args.step2:
        return run_step2()
    return run_cot()


if __name__ == "__main__":
    raise SystemExit(main())
