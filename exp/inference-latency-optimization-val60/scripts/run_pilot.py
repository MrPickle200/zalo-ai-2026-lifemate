from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lifemate.agent import Agent  # noqa: E402
from lifemate.asr import PhoASRModel  # noqa: E402
from lifemate.pipeline import LifeMatePipeline  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl, seed_everything  # noqa: E402

OUT = ROOT / "exp/inference-latency-optimization-val60/output"
SOURCE = ROOT / "exp/num-predict-512-vs-768-val/output"
MODEL = "qwen3.5:4b-q4_K_M"
HOST = "http://127.0.0.1:11434"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def ollama_get(path: str) -> dict:
    with urllib.request.urlopen(HOST + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def warmup(batch: int) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Return {} as JSON."}],
            "stream": False, "think": True, "format": "json",
            "options": {"temperature": 0, "num_predict": 8, "num_ctx": 8192, "num_batch": batch}}
    request = urllib.request.Request(HOST + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as r:
        result = json.loads(r.read().decode("utf-8"))
    return {"excluded_from_measurements": True, "wall_seconds": time.perf_counter() - started,
            "load_duration_ns": result.get("load_duration"), "total_duration_ns": result.get("total_duration"),
            "prompt_eval_count": result.get("prompt_eval_count"), "eval_count": result.get("eval_count")}


def run_branch(name: str, ids: list[str], transcripts: dict[str, str], items: dict[str, dict], batch: int | None,
               order: list[int] | None = None) -> None:
    files = {key: OUT / f"{key}-{name}.jsonl" for key in ("predictions", "traces", "requests")}
    if any(p.exists() for p in files.values()):
        raise FileExistsError(f"Output exists for {name}; refusing to append or overwrite")
    os.environ.update({"LIFEMATE_LLM_BACKEND": "ollama", "LIFEMATE_OLLAMA_MODEL": MODEL,
                       "LIFEMATE_OLLAMA_CONTEXT": "8192", "LIFEMATE_OLLAMA_NUM_PREDICT": "768",
                       "LIFEMATE_OLLAMA_THINK": "true", "LIFEMATE_OLLAMA_PROTOCOL": "json",
                       "LIFEMATE_PROMPT_LANGUAGE": "en", "LIFEMATE_OLLAMA_TELEMETRY": "1",
                       "OLLAMA_HOST": HOST})
    if batch is None:
        batch = 1024  # Preserve the historical current-default baseline explicitly.
    os.environ["LIFEMATE_OLLAMA_NUM_BATCH"] = str(batch)
    seed_everything(17)
    agent = Agent(max_new_tokens=768, context_limit=8192, ollama_thinking=True, ollama_native_tools=False)
    agent._telemetry_sample_id = None
    history_asr = PhoASRModel()
    pipeline = LifeMatePipeline(find_data_root("train+public_test/train+public_test"), "train", asr=history_asr,
                                agent=agent, max_steps=5)
    preds, traces, requests = [], [], []
    scheduled = list(range(len(ids))) if order is None else order
    started_branch = time.time()
    for position, idx in enumerate(scheduled, 1):
        sid = ids[idx]
        agent._telemetry_sample_id = sid
        start_tlm = len(agent.request_telemetry)
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
            pred = {"id": sid, "asr_text": transcripts[sid], "transcript_used": transcripts[sid], "tool_calls": [],
                    "state_hash": None, "observations": [], "final_state": {}, "model_output": [],
                    "latency_seconds": time.perf_counter() - started}
        finally:
            agent.next_action = original
        usage = agent.request_telemetry[start_tlm:]
        for i, row in enumerate(usage, 1):
            requests.append({**row, "model_digest": "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04",
                             "request_id": f"{name}:{sid}:r{i}", "sample_request_index": i})
        traces.append({"id": sid, "latency_seconds": pred["latency_seconds"], "runtime_error": error,
                       "turns": pred["model_output"], "request_count": len(usage),
                       "repair_request_count": sum(x.get("request_kind") == "repair" for x in usage)})
        preds.append(pred)
        # Flush complete prefixes so an interruption is inspectable, never silently resumed.
        write_jsonl(files["predictions"], preds)
        write_jsonl(files["traces"], traces)
        write_jsonl(files["requests"], requests)
        print(f"{name} [{position}/{len(ids)}] {sid} {pred['latency_seconds']:.2f}s requests={len(usage)} error={error}", flush=True)
        if error and batch is not None:
            write_json(OUT / f"run-{name}.json", {"branch": name, "num_batch": batch,
                       "samples_completed": position, "stopped_on_runtime_error": error,
                       "started_epoch": started_branch, "ended_epoch": time.time()})
            raise RuntimeError(f"stopping batch={batch} branch after first runtime failure on {sid}: {error}")
    write_json(OUT / f"run-{name}.json", {"branch": name, "num_batch_request_override": batch,
               "effective_num_batch": batch, "num_predict": 768,
               "num_ctx": 8192, "samples": len(preds), "started_epoch": started_branch,
               "ended_epoch": time.time(), "errors": sum(bool(x.get("runtime_error")) for x in traces),
               "model_digest": ollama_get("/api/tags").get("models", [{}])[0].get("digest")})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("baseline", "batch-pair"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((SOURCE / "cohort_manifest.json").read_text(encoding="utf-8"))
    ids = cohort["ids"]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise RuntimeError("cohort must contain the exact 60 unique prior IDs")
    transcript_rows = read_jsonl(ROOT / "exp/current-config-error-analysis-val309/output/transcript_cache_val309.jsonl")
    transcripts = {r["id"]: r["transcript"] for r in transcript_rows}
    data_root = find_data_root("train+public_test/train+public_test")
    items = {r["id"]: r for r in load_jsonl(data_root / "train/train.jsonl")}
    if set(ids) - set(transcripts) or set(ids) - set(items):
        raise RuntimeError("cohort lacks a cached transcript or train sample")
    version = ollama_get("/api/version").get("version")
    tags = ollama_get("/api/tags").get("models", [])
    model = next((x for x in tags if x.get("name") == MODEL), None)
    if version != "0.40.14" or not model or model.get("digest") != "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04":
        raise RuntimeError(f"Ollama version/model mismatch: {version=} {model=}")
    if args.branch == "baseline":
        warm = warmup(1024)
        write_json(OUT / "warmup.json", warm)
        run_branch("baseline", ids, transcripts, items, 1024)
    else:
        baseline_path = OUT / "predictions-baseline.jsonl"
        if not baseline_path.exists() or len(read_jsonl(baseline_path)) != 60:
            raise RuntimeError("complete 60-sample baseline is required before the num_batch comparison")
        write_json(OUT / "warmup-batch-pair.json", warmup(512))
        # Balanced four-block crossover: each option runs first and last, with
        # 30 sample positions in each block. Every branch still covers all 60 IDs.
        blocks = [("batch512_a", ids[:30], 512), ("batch1024_a", ids[:30], 1024),
                  ("batch1024_b", ids[30:], 1024), ("batch512_b", ids[30:], 512)]
        for block_name, block_ids, batch in blocks:
            run_branch(block_name, block_ids, transcripts, items, batch)
        for branch, pieces in (("batch512", ("batch512_a", "batch512_b")),
                               ("batch1024", ("batch1024_a", "batch1024_b"))):
            rows_by_kind = {}
            for kind in ("predictions", "traces", "requests"):
                rows = []
                for piece in pieces:
                    rows.extend(read_jsonl(OUT / f"{kind}-{piece}.jsonl"))
                rows_by_kind[kind] = rows
            for kind, rows in rows_by_kind.items():
                if kind == "requests":
                    rows.sort(key=lambda r: (ids.index(r["sample_id"]), r["sample_request_index"]))
                else:
                    rows.sort(key=lambda r: ids.index(r["id"]))
                write_jsonl(OUT / f"{kind}-{branch}.jsonl", rows)
            write_json(OUT / f"run-{branch}.json", {"branch": branch, "num_batch": int(branch[5:]),
                       "num_predict": 768, "num_ctx": 8192, "samples": len(rows_by_kind["predictions"]),
                       "balanced_block_order": ["512:first30", "1024:first30", "1024:last30", "512:last30"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
