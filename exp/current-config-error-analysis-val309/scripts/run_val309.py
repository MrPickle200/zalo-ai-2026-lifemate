from __future__ import annotations

import hashlib
import ast
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.asr import PhoASRModel  # noqa: E402
from lifemate.pipeline import LifeMatePipeline  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl, seed_everything  # noqa: E402


OUT = ROOT / "exp/current-config-error-analysis-val309/output"
VAL_IDS_PATH = ROOT / "data/splits/val_ids.json"
TRAIN_IDS_PATH = ROOT / "data/splits/train_ids.json"
QUAN_PREDICTIONS = ROOT / "exp/quan-transcribe-lifemate-domain-v1-val309/output/predictions.jsonl"
QUAN_MODEL = ROOT / "exp/quan-transcribe-lifemate-domain-v1/output/model"
LM_CONFIG = ROOT / "models/quan-transcribe/lm/lm_config.json"
LM_MODEL = ROOT / "models/quan-transcribe/lm"
OLLAMA_MODEL = "qwen3.5:4b-q4_K_M"
OLLAMA_VERSION = "0.40.14"
SEED = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def verify_ollama() -> dict[str, Any]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    with urllib.request.urlopen(host + "/api/version", timeout=10) as response:
        version = json.loads(response.read().decode("utf-8"))["version"]
    with urllib.request.urlopen(host + "/api/tags", timeout=10) as response:
        tags = json.loads(response.read().decode("utf-8")).get("models", [])
    names = [row.get("name") for row in tags]
    if version != OLLAMA_VERSION or OLLAMA_MODEL not in names:
        raise RuntimeError(f"Ollama mismatch: version={version}, model_present={OLLAMA_MODEL in names}")
    return {"host": host, "version": version, "model": OLLAMA_MODEL, "model_present": True}


def extract_english_prompt(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or "prompt_language" not in ast.unparse(node.test) or "en" not in ast.unparse(node.test):
            continue
        for statement in node.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "system" for target in statement.targets
            ):
                prompt = ast.literal_eval(statement.value)
                if not isinstance(prompt, str):
                    raise TypeError("English system prompt is not a string constant")
                return prompt.replace(" /no_think", "")
    raise RuntimeError("Could not locate English prompt literal in agent.py")


def main() -> int:
    out_predictions = OUT / "predictions.jsonl"
    out_traces = OUT / "step-traces.jsonl"
    has_predictions = out_predictions.exists()
    has_traces = out_traces.exists()
    if has_predictions != has_traces:
        raise RuntimeError("Cannot resume: predictions and traces must both exist")

    started_all = time.perf_counter()
    ids = json.loads(VAL_IDS_PATH.read_text(encoding="utf-8"))
    train_ids = json.loads(TRAIN_IDS_PATH.read_text(encoding="utf-8"))
    if len(ids) != 309 or len(set(ids)) != 309 or set(ids) & set(train_ids):
        raise RuntimeError("Split audit failed: expected 309 unique val IDs with no train overlap")

    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train/train.jsonl")}
    if not set(ids).issubset(items):
        raise RuntimeError("Some fixed validation IDs are absent from labeled train data")

    asr_rows = read_jsonl(QUAN_PREDICTIONS)
    asr_by_id = {row["id"]: row for row in asr_rows}
    if len(asr_by_id) != len(asr_rows) or set(asr_by_id) != set(ids):
        raise RuntimeError("QUAN transcript cache does not exactly cover the fixed val309 IDs")
    transcript_rows = [{"id": sid,
                        "reference": items[sid].get("audio_text", ""),
                        "transcript": asr_by_id[sid].get("prediction") or asr_by_id[sid].get("transcript") or ""}
                       for sid in ids]
    if any(not row["transcript"] for row in transcript_rows):
        raise RuntimeError("QUAN transcript cache contains an empty prediction")
    local_transcripts = OUT / "transcript_cache_val309.jsonl"
    write_jsonl(local_transcripts, transcript_rows)

    existing_predictions = read_jsonl(out_predictions) if has_predictions else []
    existing_traces = read_jsonl(out_traces) if has_traces else []
    completed = len(existing_predictions)
    if completed > len(ids) or len(existing_traces) != completed:
        raise RuntimeError("Cannot resume: prediction/trace counts are inconsistent")
    if [row.get("id") for row in existing_predictions] != ids[:completed]:
        raise RuntimeError("Cannot resume: predictions are not an exact prefix of val309 IDs")
    if [row.get("id") for row in existing_traces] != ids[:completed]:
        raise RuntimeError("Cannot resume: traces are not an exact prefix of val309 IDs")

    checkpoint = QUAN_MODEL / "model.safetensors"
    checkpoint_hash = sha256(checkpoint)
    lm_config = json.loads(LM_CONFIG.read_text(encoding="utf-8"))
    if checkpoint_hash != "e31be8516152587134fb62e0d1690954221a0f02c53b6549cf79db58c8eb18ae":
        raise RuntimeError("QUAN checkpoint hash differs from the submitted configuration")
    if sha256(LM_CONFIG) != "6bd543b3e00328b22d927ad06082d3d0bae21f3710a8f83e5c3e6796cd5fc579":
        raise RuntimeError("KenLM configuration hash differs from the submitted configuration")
    if float(lm_config.get("alpha", -1)) != 0.3 or float(lm_config.get("beta", -1)) != 1.0:
        raise RuntimeError("KenLM alpha/beta differ from the submitted configuration")
    audit = json.loads((QUAN_MODEL / "dataset-audit.json").read_text(encoding="utf-8"))
    if audit.get("train_ids") != 1236 or audit.get("train_validation_overlap") != 0:
        raise RuntimeError("QUAN checkpoint audit does not certify the train-only fit")

    ollama = verify_ollama()
    agent_source = ROOT / "src/lifemate/agent.py"
    pipeline_source = ROOT / "src/lifemate/pipeline.py"
    submission_runner = ROOT / "exp/qwen35-track-c-quan-vs-phoasr-val309/scripts/make_qwen35_quan_submission.py"
    source_hashes = {str(path.relative_to(ROOT)): sha256(path)
                     for path in (agent_source, pipeline_source, submission_runner)}
    prompt_text = extract_english_prompt(agent_source)
    manifest = {
        "experiment": "current-config-error-analysis-val309",
        "status": "inference_running",
        "submission_reference": "qwen35-track-c-quan-ft-beam32-public-test",
        "split": "fixed train-only val309",
        "cohort_ids": str(VAL_IDS_PATH.relative_to(ROOT)),
        "cohort_count": len(ids),
        "cohort_ids_sha256_newline_joined": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "train_count": len(train_ids),
        "train_validation_overlap": 0,
        "transcript_cache": {
            "source": str(QUAN_PREDICTIONS.relative_to(ROOT)),
            "source_sha256": sha256(QUAN_PREDICTIONS),
            "source_rows": len(asr_rows),
            "selected_val309_cache": str(local_transcripts.relative_to(ROOT)),
            "selected_val309_cache_sha256": sha256(local_transcripts),
            "asr": "QUAN LifeMate fine-tune; KenLM CTC beam32; alpha=0.3; beta=1.0",
            "checkpoint": str(QUAN_MODEL.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_hash,
            "lm_config": str(LM_CONFIG.relative_to(ROOT)),
            "lm_config_sha256": sha256(LM_CONFIG),
        },
        "agent": {
            "backend": "Ollama",
            "model": OLLAMA_MODEL,
            "ollama": ollama,
            "thinking": True,
            "protocol": "JSON; native tool calling disabled",
            "temperature": 0,
            "num_predict": 512,
            "num_ctx": 8192,
            "prompt_version": "observation-grounded-v2",
            "prompt_language": "en",
            "prompt_source": "src/lifemate/agent.py English system prompt confirmed by user",
            "prompt_sha256_utf8": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "prompt_characters": len(prompt_text),
            "prompt_file_sha256": sha256(agent_source),
        },
        "pipeline": {
            "max_action_turns": 5,
            "toolenv_sessions": "exactly one prediction ToolEnv per sample",
            "history_audio_asr": "PhoASRModel, same as submission runner",
            "runtime_gate_seconds": 30,
            "code_sha256": source_hashes,
            "source_revision_limitation": "submission artifact omitted code hashes; user confirmed the English prompt source; current sources and submission runner are fingerprinted here",
        },
        "prediction_run": {"started_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                           "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU 6 GiB; local timing is not A30 timing",
                           "estimated_seconds": 309 * 32.35,
                           "estimated_hours": 309 * 32.35 / 3600,
                           "sequential": True},
    }
    write_json(OUT / "config_manifest.json", manifest)

    seed_everything(SEED)
    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = OLLAMA_MODEL
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "8192"
    os.environ["LIFEMATE_OLLAMA_NUM_PREDICT"] = "512"
    os.environ["LIFEMATE_OLLAMA_THINK"] = "true"
    os.environ["LIFEMATE_OLLAMA_PROTOCOL"] = "json"
    os.environ["LIFEMATE_PROMPT_LANGUAGE"] = "en"
    agent = Agent(max_new_tokens=512, context_limit=8192, ollama_thinking=True, ollama_native_tools=False)
    history_asr = PhoASRModel()
    pipeline = LifeMatePipeline(data_root, "train", asr=history_asr, agent=agent, max_steps=5)
    write_json(OUT / "run_config.json", {
        "model": OLLAMA_MODEL, "thinking": True, "protocol": "JSON", "temperature": 0,
        "num_predict": 512, "num_ctx": 8192, "prompt": "observation-grounded-v2 English",
        "asr": "QUAN FT + KenLM beam32; transcript_override from fixed val cache",
        "history_asr": "PhoASRModel", "max_action_turns": 5,
        "toolenv_per_sample": 1, "cohort": "fixed train-only val309", "seed": SEED,
    })
    print(f"Starting sequential prediction run: {len(ids)} samples; estimate {manifest['prediction_run']['estimated_hours']:.2f} h before scoring.", flush=True)
    pred_f = out_predictions.open("a" if completed else "w", encoding="utf-8", newline="\n")
    trace_f = out_traces.open("a" if completed else "w", encoding="utf-8", newline="\n")
    try:
        print(f"Resuming sequential prediction run at {completed}/{len(ids)} completed rows.", flush=True)
        for index, sid in enumerate(ids[completed:], start=completed + 1):
            transcript = transcript_rows[index - 1]["transcript"]
            item = items[sid]
            started = time.perf_counter()
            error = None
            try:
                result = pipeline.run(item, transcript_override=transcript)
                calls = [{"name": call.get("name"), "arguments": call.get("arguments", {})}
                         for call in result.get("tool_calls", [])]
                prediction = {
                    "id": sid, "asr_text": transcript, "transcript_used": transcript,
                    "tool_calls": calls, "state_hash": result["state_hash"],
                    "observations": result.get("observations", []),
                    "final_state": result.get("final_state", {}),
                    "model_output": result.get("model_output", []),
                    "latency_seconds": result.get("latency_seconds", time.perf_counter() - started),
                }
            except Exception as exc:  # retain one row for every attempted sample
                error = f"{type(exc).__name__}: {exc}"
                prediction = {"id": sid, "asr_text": transcript, "transcript_used": transcript,
                              "tool_calls": [], "state_hash": None, "observations": [],
                              "final_state": {}, "model_output": [],
                              "latency_seconds": time.perf_counter() - started, "error": error}
            trace = {
                "id": sid,
                "latency_seconds": prediction["latency_seconds"],
                "runtime_error": error,
                "turns": prediction.get("model_output", []),
                "observations": prediction.get("observations", []),
                "schema_rejections": sum(obs.get("result", {}).get("ok") is False
                                          for obs in prediction.get("observations", [])),
                "repair_attempts": sum(bool(turn.get("_diagnostics", {}).get("repair_attempted"))
                                        for turn in prediction.get("model_output", [])),
                "repair_failures": sum(bool(turn.get("_diagnostics", {}).get("repair_attempted")) and
                                        not turn.get("_diagnostics", {}).get("repair_succeeded", False)
                                        for turn in prediction.get("model_output", [])),
                "action_turns": len(prediction.get("model_output", [])),
                "submitted_calls": len(prediction.get("tool_calls", [])),
            }
            pred_f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            pred_f.flush()
            trace_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_f.flush()
            print(f"[{index}/{len(ids)}] {sid} {prediction['latency_seconds']:.2f}s "
                  f"calls={len(prediction['tool_calls'])} error={error}", flush=True)
            if index in (155, len(ids)):
                manifest["progress"] = {"completed": index, "total": len(ids),
                                         "elapsed_seconds": time.perf_counter() - started_all}
                write_json(OUT / "config_manifest.json", manifest)
    finally:
        pred_f.close()
        trace_f.close()

    manifest["status"] = "predictions_complete_scoring_pending"
    manifest["prediction_run"]["elapsed_seconds"] = time.perf_counter() - started_all
    manifest["prediction_run"]["prediction_sha256"] = sha256(out_predictions)
    manifest["prediction_run"]["trace_sha256"] = sha256(out_traces)
    write_json(OUT / "config_manifest.json", manifest)
    print(f"Prediction run complete: {len(ids)}/{len(ids)}; elapsed {manifest['prediction_run']['elapsed_seconds'] / 3600:.2f} h.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
