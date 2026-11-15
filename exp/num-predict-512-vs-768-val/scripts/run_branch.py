from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.asr import PhoASRModel  # noqa: E402
from lifemate.pipeline import LifeMatePipeline  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl, seed_everything  # noqa: E402


OUT = ROOT / "exp/num-predict-512-vs-768-val/output"
BASELINE = ROOT / "exp/current-config-error-analysis-val309/output/predictions.jsonl"
VAL_IDS = ROOT / "data/splits/val_ids.json"
TRAIN_IDS = ROOT / "data/splits/train_ids.json"
TRANSCRIPTS = ROOT / "exp/current-config-error-analysis-val309/output/transcript_cache_val309.jsonl"
OLLAMA_MODEL = "qwen3.5:4b-q4_K_M"
SEED = 17


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_path(branch: int, full: bool) -> Path:
    return OUT / (f"predictions-{branch}-full-val309.jsonl" if full else f"predictions-{branch}.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", type=int, choices=(512, 768), required=True)
    parser.add_argument("--full-val309", action="store_true", help="Run all val IDs; intended only for gated branch 768 phase")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-after-baseline-divergence", action="store_true",
                        help="Continue only after recording investigation of a reproducibility mismatch")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_ids = json.loads(VAL_IDS.read_text(encoding="utf-8"))
    train_ids = json.loads(TRAIN_IDS.read_text(encoding="utf-8"))
    if len(all_ids) != 309 or len(set(all_ids)) != 309 or set(all_ids) & set(train_ids):
        raise RuntimeError("fixed train-only val309 split audit failed")
    if args.full_val309:
        ids = all_ids
    else:
        cohort_path = OUT / "cohort_manifest.json"
        if not cohort_path.exists():
            raise FileNotFoundError("Run the experiment setup to create the pre-registered cohort_manifest.json first")
        ids = json.loads(cohort_path.read_text(encoding="utf-8"))["ids"]
        if len(ids) != 60 or len(set(ids)) != 60 or not set(ids).issubset(all_ids):
            raise RuntimeError("pilot cohort must contain 60 unique fixed val IDs")

    name = f"{args.branch}{'-full-val309' if args.full_val309 else ''}"
    pred_path = prediction_path(args.branch, args.full_val309)
    trace_path = OUT / f"step-traces-{name}.jsonl"
    usage_path = OUT / f"token_usage-{name}.jsonl"
    if not args.resume and any(path.exists() for path in (pred_path, trace_path, usage_path)):
        raise FileExistsError(f"outputs exist for branch {name}; use --resume only after validating aligned prefixes")
    pred_rows = read_jsonl(pred_path) if pred_path.exists() else []
    trace_rows = read_jsonl(trace_path) if trace_path.exists() else []
    usage_rows = read_jsonl(usage_path) if usage_path.exists() else []
    completed = len(pred_rows)
    if not args.resume and completed:
        raise RuntimeError("unexpected existing predictions")
    if (completed > len(ids) or len(trace_rows) != completed or
            [row.get("id") for row in pred_rows] != ids[:completed] or
            [row.get("id") for row in trace_rows] != ids[:completed]):
        raise RuntimeError("resume output is not an aligned ordered cohort prefix")

    if not args.full_val309 and args.branch == 512 and completed == 0:
        # Fix and fingerprint the cohort before either branch runs.
        cohort = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))
        cohort["branch_512_started"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        write_json(OUT / "cohort_manifest.json", cohort)

    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train/train.jsonl")}
    transcript_rows = read_jsonl(TRANSCRIPTS)
    transcript_by_id = {row["id"]: row["transcript"] for row in transcript_rows}
    if set(ids) - set(items) or set(ids) - set(transcript_by_id):
        raise RuntimeError("cohort item/transcript coverage failed")

    baseline_by_id = {row["id"]: row for row in read_jsonl(BASELINE)}
    if args.branch == 768 and not args.full_val309:
        a_path = prediction_path(512, False)
        if not a_path.exists():
            raise RuntimeError("run 512 telemetry pilot first")
        a_ids = [row["id"] for row in read_jsonl(a_path)]
        if a_ids != ids:
            raise RuntimeError("branch 512 pilot does not exactly match fixed cohort")
    if args.full_val309 and args.branch != 768:
        raise RuntimeError("the task only authorizes full-val execution for branch 768")

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    parsed_host = urlparse(host)
    if parsed_host.scheme != "http" or parsed_host.hostname not in {"127.0.0.1", "localhost"} or parsed_host.port != 11434:
        raise RuntimeError(f"Inference must use the local Ollama server at 127.0.0.1:11434, got {host}")
    with urllib.request.urlopen(host + "/api/version", timeout=5) as response:
        ollama_version = json.loads(response.read().decode("utf-8")).get("version")
    with urllib.request.urlopen(host + "/api/tags", timeout=5) as response:
        ollama_models = json.loads(response.read().decode("utf-8")).get("models", [])
    ollama_tag = next((row for row in ollama_models if row.get("name") == OLLAMA_MODEL), None)
    if ollama_version != "0.40.14" or not ollama_tag or ollama_tag.get("digest") != "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04":
        raise RuntimeError(f"Ollama version/model fingerprint mismatch: version={ollama_version}, tag={ollama_tag}")
    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = OLLAMA_MODEL
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "8192"
    os.environ["LIFEMATE_OLLAMA_NUM_PREDICT"] = str(args.branch)
    os.environ["LIFEMATE_OLLAMA_THINK"] = "true"
    os.environ["LIFEMATE_OLLAMA_PROTOCOL"] = "json"
    os.environ["LIFEMATE_PROMPT_LANGUAGE"] = "en"
    os.environ["LIFEMATE_OLLAMA_TELEMETRY"] = "1"
    seed_everything(SEED)
    agent = Agent(max_new_tokens=args.branch, context_limit=8192, ollama_thinking=True, ollama_native_tools=False)
    agent._telemetry_sample_id = None
    history_asr = PhoASRModel()
    pipeline = LifeMatePipeline(data_root, "train", asr=history_asr, agent=agent, max_steps=5)

    run_config = {
        "branch": args.branch, "model": OLLAMA_MODEL, "thinking": True, "protocol": "JSON",
        "temperature": 0, "num_predict": args.branch, "num_ctx": 8192,
        "prompt": "observation-grounded-v2 English", "asr": "QUAN FT + KenLM beam32; fixed cached transcript override",
        "history_asr": "PhoASRModel", "max_action_turns": 5, "toolenv_per_sample": 1,
        "cohort": "fixed train-only val309" if args.full_val309 else "pre-registered diagnostic pilot60",
        "seed": SEED, "ollama_host": host, "ollama_version": ollama_version,
        "model_digest": ollama_tag.get("digest"), "telemetry_opt_in": True,
    }
    write_json(OUT / f"run-config-{name}.json", run_config)
    pred_f = pred_path.open("a" if completed else "w", encoding="utf-8", newline="\n")
    trace_f = trace_path.open("a" if completed else "w", encoding="utf-8", newline="\n")
    usage_f = usage_path.open("a" if usage_rows else "w", encoding="utf-8", newline="\n")
    try:
        for index, sid in enumerate(ids[completed:], start=completed + 1):
            agent._telemetry_sample_id = sid
            agent._telemetry_action_turn = None
            telemetry_start = len(agent.request_telemetry)
            action_turn = [0]
            original_next_action = agent.next_action

            def tracked_next_action(*call_args, **call_kwargs):
                action_turn[0] += 1
                agent._telemetry_action_turn = action_turn[0]
                return original_next_action(*call_args, **call_kwargs)

            agent.next_action = tracked_next_action
            started = time.perf_counter()
            error = None
            try:
                result = pipeline.run(items[sid], transcript_override=transcript_by_id[sid])
                prediction = {
                    "id": sid, "asr_text": transcript_by_id[sid], "transcript_used": transcript_by_id[sid],
                    "tool_calls": [{"name": call.get("name"), "arguments": call.get("arguments", {})}
                                   for call in result.get("tool_calls", [])],
                    "state_hash": result.get("state_hash"), "observations": result.get("observations", []),
                    "final_state": result.get("final_state", {}), "model_output": result.get("model_output", []),
                    "latency_seconds": result.get("latency_seconds", time.perf_counter() - started),
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                prediction = {"id": sid, "asr_text": transcript_by_id[sid], "transcript_used": transcript_by_id[sid],
                              "tool_calls": [], "state_hash": None, "observations": [], "final_state": {},
                              "model_output": [], "latency_seconds": time.perf_counter() - started, "error": error}
            finally:
                agent.next_action = original_next_action

            sample_usage = agent.request_telemetry[telemetry_start:]
            for request_index, request_row in enumerate(sample_usage, 1):
                row = {**request_row, "request_id": f"{name}:{sid}:r{request_index}",
                       "sample_request_index": request_index}
                usage_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            usage_f.flush()
            trace = {
                "id": sid, "latency_seconds": prediction["latency_seconds"], "runtime_error": error,
                "turns": prediction["model_output"], "observations": prediction["observations"],
                "repair_attempts": sum(bool(t.get("_diagnostics", {}).get("repair_attempted")) for t in prediction["model_output"]),
                "repair_failures": sum(bool(t.get("_diagnostics", {}).get("repair_attempted")) and not t.get("_diagnostics", {}).get("repair_succeeded", False) for t in prediction["model_output"]),
                "action_turns": len(prediction["model_output"]), "request_count": len(sample_usage),
                "repair_request_count": sum(row.get("request_kind") == "repair" for row in sample_usage),
            }
            pred_f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            pred_f.flush()
            trace_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_f.flush()

            if not args.full_val309 and args.branch == 512 and sid in baseline_by_id:
                old = baseline_by_id[sid]
                same = prediction["tool_calls"] == old.get("tool_calls", []) and prediction["state_hash"] == old.get("state_hash")
                if not same:
                    if not args.continue_after_baseline_divergence:
                        raise RuntimeError(f"instrumented 512 differs from existing baseline for {sid}; stop before branch 768")
                    print(f"INVESTIGATED BASELINE DIVERGENCE: {sid}; retaining row and telemetry for paired rerun analysis", flush=True)
            print(f"[{index}/{len(ids)}] {sid} latency={prediction['latency_seconds']:.2f}s requests={len(sample_usage)} error={error}", flush=True)
    finally:
        pred_f.close(); trace_f.close(); usage_f.close()

    if not args.full_val309 and args.branch == 512:
        # Check the complete cohort after the final append as well.
        new_rows = read_jsonl(pred_path)
        changed = [row["id"] for row in new_rows
                   if row["tool_calls"] != baseline_by_id[row["id"]].get("tool_calls", [])
                   or row["state_hash"] != baseline_by_id[row["id"]].get("state_hash")]
        if changed and not args.continue_after_baseline_divergence:
            raise RuntimeError(f"instrumentation-only 512 replay mismatch; do not run B: {changed[:5]}")
        cohort = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))
        cohort["branch_512_completed"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        cohort["branch_512_vs_existing_baseline"] = {
            "checked": len(new_rows), "identical_calls_and_hash": len(new_rows) - len(changed),
            "mismatches": len(changed), "mismatch_ids": changed,
            "investigation": (
                "The checked-in agent.py hash matches the pre-instrumentation baseline source hash; current diff only adds opt-in telemetry after request serialization. English prompt hash, model tag digest/version, transcript and sample ID were verified. First divergence kept the same state hash but reordered read-only calls. This indicates an inference reproducibility difference; retain the new A run as the paired experimental baseline and report old-baseline mismatch separately."
            ) if changed else "No call/hash mismatch found.",
        }
        write_json(OUT / "cohort_manifest.json", cohort)
    print(f"Branch {name} completed: {len(read_jsonl(pred_path))}/{len(ids)} samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
