from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "exp/qwen35-thinking-budget-val60/output"
SOURCE = ROOT / "exp/num-predict-512-vs-768-val/output"
TRANSCRIPTS = ROOT / "exp/current-config-error-analysis-val309/output/transcript_cache_val309.jsonl"
MODEL = "qwen3.5:4b-q4_K_M"
DIGEST = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
HOST = "http://127.0.0.1:11434"
SEED = 17
CAPS = (512, 768, 1024, 1536)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api_json(path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(HOST + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))


def warmup(cap: int, batch: int) -> dict[str, Any]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Return {} as JSON."}],
            "stream": False, "think": True, "format": "json",
            "options": {"temperature": 0, "num_predict": 8, "num_ctx": 8192, "num_batch": batch}}
    start = time.perf_counter()
    response = api_json("/api/chat", body)
    return {"branch_cap": cap, "excluded_from_measurements": True,
            "wall_seconds": time.perf_counter() - start,
            "prompt_eval_count": response.get("prompt_eval_count"), "eval_count": response.get("eval_count"),
            "load_duration_ns": response.get("load_duration"), "total_duration_ns": response.get("total_duration"),
            "timestamp_epoch": time.time()}


def run_block(cap: int, round_index: int, ids: list[str], transcripts: dict[str, str],
              items: dict[str, dict], batch: int, label: str | None = None) -> None:
    from lifemate.agent import Agent
    from lifemate.asr import PhoASRModel
    from lifemate.pipeline import LifeMatePipeline
    from lifemate.utils import seed_everything

    branch = label or str(cap)
    pred_path, trace_path, req_path = (OUT / f"predictions-{branch}.jsonl",
                                       OUT / f"step-traces-{branch}.jsonl",
                                       OUT / f"requests-{branch}.jsonl")
    done_ids = [r["id"] for r in read_jsonl(pred_path)] if pred_path.exists() else []
    if set(ids) & set(done_ids):
        raise RuntimeError(f"Refusing to overwrite/duplicate block IDs for cap {cap}: {set(ids) & set(done_ids)}")

    os.environ.update({"LIFEMATE_LLM_BACKEND": "ollama", "LIFEMATE_OLLAMA_MODEL": MODEL,
        "LIFEMATE_OLLAMA_CONTEXT": "8192", "LIFEMATE_OLLAMA_NUM_PREDICT": str(cap),
        "LIFEMATE_OLLAMA_NUM_BATCH": str(batch), "LIFEMATE_OLLAMA_THINK": "true",
        "LIFEMATE_OLLAMA_PROTOCOL": "json", "LIFEMATE_PROMPT_LANGUAGE": "en",
        "LIFEMATE_OLLAMA_TELEMETRY": "1", "OLLAMA_HOST": HOST})
    seed_everything(SEED)
    agent = Agent(max_new_tokens=cap, context_limit=8192, ollama_thinking=True, ollama_native_tools=False)
    agent._telemetry_sample_id = None
    history_asr = PhoASRModel()
    from lifemate.utils import find_data_root
    pipeline = LifeMatePipeline(find_data_root("train+public_test/train+public_test"), "train",
                                asr=history_asr, agent=agent, max_steps=5)
    preds, traces, requests = [], [], []
    start_block = time.time()
    for offset, sid in enumerate(ids, 1):
        agent._telemetry_sample_id = sid
        start_telemetry = len(agent.request_telemetry)
        action_turn = [0]
        original = agent.next_action

        def tracked(*args, **kwargs):
            action_turn[0] += 1
            agent._telemetry_action_turn = action_turn[0]
            return original(*args, **kwargs)

        agent.next_action = tracked
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
            agent.next_action = original
        usage = agent.request_telemetry[start_telemetry:]
        sample_requests = []
        for i, row in enumerate(usage, 1):
            sample_requests.append({**row, "request_id": f"{branch}:{sid}:r{i}", "sample_request_index": i,
                                    "model_digest": DIGEST, "round": round_index})
        requests.extend(sample_requests)
        traces.append({"id": sid, "cap": cap, "round": round_index, "latency_seconds": pred["latency_seconds"],
                       "runtime_error": error, "turns": pred["model_output"],
                       "request_count": len(usage),
                       "action_turns": len(pred["model_output"]),
                       "repair_request_count": sum(x.get("request_kind") == "repair" for x in usage)})
        preds.append(pred)
        # Flush every completed sample; an interrupted block remains auditable and is never silently resumed.
        write_jsonl(pred_path, (read_jsonl(pred_path) if pred_path.exists() else []) + [preds[-1]])
        write_jsonl(trace_path, (read_jsonl(trace_path) if trace_path.exists() else []) + [traces[-1]])
        write_jsonl(req_path, (read_jsonl(req_path) if req_path.exists() else []) + sample_requests)
        print(f"cap={cap} round={round_index} [{offset}/{len(ids)}] {sid} "
              f"{pred['latency_seconds']:.2f}s requests={len(usage)} error={error}", flush=True)
        if error:
            write_json(OUT / f"run-{branch}.json", {"cap": cap, "branch": branch, "stopped_on_runtime_error": error,
                "samples_completed": len(read_jsonl(pred_path)), "round": round_index,
                "started_epoch": start_block, "ended_epoch": time.time()})
            raise RuntimeError(f"Stopping on first runtime exception for cap={cap}, sample={sid}: {error}")
    meta_path = OUT / f"run-{branch}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"cap": cap, "blocks": []}
    meta.setdefault("blocks", []).append({"round": round_index, "ids": ids,
        "started_epoch": start_block, "ended_epoch": time.time(), "warmup_file": f"warmup-{cap}-round{round_index}.json"})
    meta.update({"num_predict": cap, "num_ctx": 8192, "num_batch": batch,
        "effective_num_batch": batch, "model": MODEL, "model_digest": DIGEST,
        "ollama_version": "0.40.14", "seed": SEED,
        "samples_completed": len(read_jsonl(pred_path)), "execution_order": "balanced 15-ID rounds"})
    write_json(meta_path, meta)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-preflight", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids_a = json.loads((SOURCE / "cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    ids_b = json.loads((ROOT / "exp/inference-latency-optimization-val60/output/cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    if len(ids_a) != 60 or ids_a != ids_b or len(set(ids_a)) != 60:
        raise RuntimeError("The two registered pilot cohort ID sequences do not match exactly")
    version = api_json("/api/version").get("version")
    models = api_json("/api/tags").get("models", [])
    model = next((m for m in models if m.get("name") == MODEL), None)
    if version != "0.40.14" or not model or model.get("digest") != DIGEST:
        raise RuntimeError(f"Ollama fingerprint mismatch: version={version}, model={model}")
    transcript_rows = read_jsonl(TRANSCRIPTS)
    transcripts = {r["id"]: r["transcript"] for r in transcript_rows}
    from lifemate.utils import find_data_root, load_jsonl
    data_root = find_data_root("train+public_test/train+public_test")
    items = {r["id"]: r for r in load_jsonl(data_root / "train/train.jsonl")}
    if set(ids_a) - transcripts.keys() or set(ids_a) - items.keys():
        raise RuntimeError("Cohort is missing a cached transcript or train input")
    if args.only_preflight:
        source_manifest = json.loads((SOURCE / "config_manifest.json").read_text(encoding="utf-8"))
        cohort_manifest = json.loads((SOURCE / "cohort_manifest.json").read_text(encoding="utf-8"))
        write_json(OUT / "cohort_manifest.json", {"experiment": "qwen35-thinking-budget-val60",
            "pilot_type": "fixed enriched train-only diagnostic pilot60", "ids": ids_a,
            "ids_sha256_newline_joined": hashlib.sha256("\n".join(ids_a).encode()).hexdigest(),
            "cohort_source": "exp/num-predict-512-vs-768-val/output/cohort_manifest.json",
            "source_cohort_manifest_sha256": sha256(SOURCE / "cohort_manifest.json"),
            "verified_identical_to": "exp/inference-latency-optimization-val60/output/cohort_manifest.json",
            "pre_run_strata": cohort_manifest.get("pre_run_strata"),
            "not_unbiased_val309_estimate": True, "test_data_used": False})
        input_paths = [ROOT / "COMPETITION.md", ROOT / "TASK.md", ROOT / "AGENTS.md",
            ROOT / "src/lifemate/agent.py", ROOT / "src/lifemate/pipeline.py", ROOT / "src/lifemate/asr.py",
            ROOT / "src/lifemate/tool_executor.py", ROOT / "src/lifemate/schemas.py",
            TRANSCRIPTS, SOURCE / "config_manifest.json", SOURCE / "cohort_manifest.json",
            Path(__file__)]
        write_json(OUT / "config_manifest.json", {"experiment": "qwen35-thinking-budget-val60",
            "locked_common_config": {"model": MODEL, "model_digest": DIGEST, "ollama_version": version,
                "thinking": True, "temperature": 0, "protocol": "JSON", "num_ctx": 8192,
                "num_batch": 512, "prompt": "observation-grounded-v2 English",
                "asr": "QUAN FT + KenLM beam32; fixed cached transcript override",
                "history_asr": "PhoASRModel", "max_action_turns": 5, "toolenv_per_sample": 1,
                "seed": SEED, "host": HOST},
            "branches": {str(cap): {"num_predict": cap, "effective_num_batch": 512} for cap in CAPS},
            "cohort_count": len(ids_a), "cohort_ids_sha256_newline_joined": hashlib.sha256("\n".join(ids_a).encode()).hexdigest(),
            "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in input_paths if p.is_file()},
            "reused_old_predictions": False, "source_config_manifest_sha256": sha256(SOURCE / "config_manifest.json"),
            "source_prompt_sha256_utf8": source_manifest.get("prompt_sha256_utf8"),
            "raw_thinking_saved": False, "payload_hash": "SHA-256 exact serialized Ollama request bytes",
            "effective_batch_verified_in_payload": True, "execution_environment": "native Python with Ollama localhost access; not .venv"})
        print(json.dumps({"cohort_ids": len(ids_a), "cohorts_identical": True, "version": version,
            "model": MODEL, "model_digest": model.get("digest"), "transcripts": len(transcripts),
            "train_samples": len(items)}, indent=2))
        return 0

    if any((OUT / f"predictions-{c}.jsonl").exists() for c in CAPS):
        raise FileExistsError("Prediction artifact exists; this runner never resumes or overwrites a branch")
    # Balanced cap order across four fixed 15-ID blocks. Each sample is run once per cap.
    schedule = [[512, 768, 1024, 1536], [768, 1024, 1536, 512],
                [1024, 1536, 512, 768], [1536, 512, 768, 1024]]
    write_json(OUT / "execution_order.json", {"cap_order_by_round": schedule,
        "rounds": [{"round": i + 1, "ids": ids_a[i * 15:(i + 1) * 15]} for i in range(4)],
        "num_batch": 512, "balanced": True})
    for round_index, order in enumerate(schedule, 1):
        block = ids_a[(round_index - 1) * 15:round_index * 15]
        for cap in order:
            warm = warmup(cap, 512)
            write_json(OUT / f"warmup-{cap}-round{round_index}.json", warm)
            run_block(cap, round_index, block, transcripts, items, 512)
    for cap in CAPS:
        pred_ids = [r["id"] for r in read_jsonl(OUT / f"predictions-{cap}.jsonl")]
        if set(pred_ids) != set(ids_a) or len(pred_ids) != 60 or len(set(pred_ids)) != 60:
            raise RuntimeError(f"Incomplete or duplicate cohort at cap={cap}: {len(pred_ids)} predictions")
        write_json(OUT / f"run-{cap}.json", {**json.loads((OUT / f"run-{cap}.json").read_text()),
                   "status": "complete", "started_at_utc": None, "ended_epoch": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
