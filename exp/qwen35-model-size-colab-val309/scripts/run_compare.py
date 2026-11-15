"""Paired Qwen size screen on cached PhoWhisper val309 transcripts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "train+public_test" / "train+public_test"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "exp" / "btc-scoring" / "scripts"),
                str(ROOT / "exp" / "quan-continuation-cot-v1" / "scripts")]

from lifemate.agent import Agent
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import seed_everything
from run_downstream import inject_cot, canonical_prompt_hash
from score_btc import score_sample

HOST = "http://127.0.0.1:11434"
DEFAULT_MODELS = ["lifemate-sft4:latest", "qwen3.5:9b-q4_K_M"]
EXPECTED_PROMPT_SHA256 = "cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950"
COHORT60 = ROOT / "assets" / "cohort60.json"
VAL_IDS = ROOT / "data" / "splits" / "val_ids.json"
MAIN = ROOT / "assets" / "pho_val309_main.jsonl"
HISTORY = ROOT / "assets" / "pho_val309_history.jsonl"
OUT = ROOT / "output"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def api(path: str) -> dict:
    with urllib.request.urlopen(HOST + path, timeout=30) as response:
        return json.load(response)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_inputs(cohort: str):
    val_ids = json.loads(VAL_IDS.read_text(encoding="utf-8"))
    assert len(val_ids) == 309 and len(set(val_ids)) == 309
    ids = json.loads(COHORT60.read_text(encoding="utf-8"))["ids"] if cohort == "60" else val_ids
    assert len(ids) == int(cohort) and len(set(ids)) == len(ids) and set(ids) <= set(val_ids)
    items = {row["id"]: row for row in read_jsonl(DATA / "train" / "train.jsonl")}
    main = {row["id"]: row["prediction"] for row in read_jsonl(MAIN)}
    history = {row["key"]: row for row in read_jsonl(HISTORY)}
    assert len(main) == 309 and len(history) == 692
    prepared = {}
    for sid in ids:
        assert sid in items and sid in main and main[sid].strip()
        item = dict(items[sid])
        turns = []
        for index, turn in enumerate(item.get("history", [])):
            copy = dict(turn)
            if copy.get("audio"):
                key = f"{sid}:{index}"
                assert key in history and history[key]["audio"] == copy["audio"]
                copy["text"] = history[key]["prediction"]
                assert copy["text"].strip()
                del copy["audio"]
            turns.append(copy)
        item["history"] = turns
        prepared[sid] = item
    return ids, items, prepared, main


def model_digests(models: list[str]) -> dict[str, str]:
    tags = {m["name"]: m["digest"] for m in api("/api/tags")["models"]}
    missing = [name for name in models if name not in tags]
    if missing:
        raise RuntimeError(f"Models missing from Ollama: {missing}")
    return {name: tags[name] for name in models}


def warm_model(model: str) -> None:
    body = json.dumps({"model": model, "prompt": "", "stream": False,
                       "keep_alive": "30m"}).encode("utf-8")
    request = urllib.request.Request(HOST + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError(f"Ollama preload failed for {model}: {result['error']}")


def configure(model: str) -> None:
    os.environ.update({
        "OLLAMA_HOST": HOST,
        "LIFEMATE_LLM_BACKEND": "ollama",
        "LIFEMATE_OLLAMA_MODEL": model,
        "LIFEMATE_OLLAMA_CONTEXT": "8192",
        "LIFEMATE_OLLAMA_NUM_PREDICT": "768",
        "LIFEMATE_OLLAMA_NUM_BATCH": "512",
        "LIFEMATE_OLLAMA_THINK": "true",
        "LIFEMATE_OLLAMA_PROTOCOL": "json",
        "LIFEMATE_PROMPT_LANGUAGE": "en",
    })


def branch_path(cohort: str, model: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in model)
    return OUT / f"predictions_{cohort}_{safe}.jsonl"


def run_branch(cohort: str, model: str, ids: list[str], prepared: dict, main: dict) -> Path:
    configure(model)
    path = branch_path(cohort, model)
    prior = read_jsonl(path) if path.exists() else []
    assert [row["id"] for row in prior] == ids[:len(prior)], "Resume prefix mismatch"
    done = len(prior)
    if done == len(ids):
        return path
    agent = Agent(max_new_tokens=768, context_limit=8192,
                  ollama_thinking=True, ollama_native_tools=False)
    generate = agent._generate
    agent._generate = lambda messages: generate(inject_cot(messages))
    pipe = LifeMatePipeline(DATA, "train", agent=agent, max_steps=5)
    warm_model(model)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for number, sid in enumerate(ids[done:], done + 1):
            start = time.perf_counter()
            try:
                result = pipe.run(prepared[sid], transcript_override=main[sid])
                row = {"id": sid, "tool_calls": result["tool_calls"],
                       "state_hash": result["state_hash"],
                       "latency_seconds": result["latency_seconds"], "error": None}
            except Exception as exc:
                row = {"id": sid, "tool_calls": [], "state_hash": None,
                       "latency_seconds": time.perf_counter() - start,
                       "error": f"{type(exc).__name__}: {exc}"}
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            if number % 10 == 0 or number == len(ids):
                print(f"{model}: {number}/{len(ids)}", flush=True)
    return path


def analyze(cohort: str, models: list[str], ids: list[str], items: dict, metadata: dict) -> None:
    scores = {}
    for model in models:
        predictions = read_jsonl(branch_path(cohort, model))
        assert [row["id"] for row in predictions] == ids
        details = [score_sample(sid, items[sid], row, DATA)
                   for sid, row in zip(ids, predictions)]
        scores[model] = [d["btc_formula_score_before_runtime_gate_diagnostic"] for d in details]
        metadata["results"][model] = {
            "btc_ungated_0_100": 100 * statistics.mean(scores[model]),
            "llm_toolenv_30s_gate_diagnostic_0_100": 100 * statistics.mean(d["official_sample_score"] for d in details),
            "over_30s_llm_toolenv_only": sum(d["runtime_over_30s"] for d in details),
            "runtime_errors": sum(d["runtime_failure"] for d in details),
            "hash_matches": sum(d["hash_match"] for d in details),
            "median_llm_toolenv_seconds": statistics.median(p["latency_seconds"] for p in predictions),
        }
    if len(models) == 2:
        delta = [b - a for a, b in zip(scores[models[0]], scores[models[1]])]
        rng = random.Random(17)
        boot = sorted(100 * statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(5000))
        metadata["paired_second_minus_first"] = {
            "delta_btc_points": 100 * statistics.mean(delta),
            "bootstrap_95pct_CI": [boot[124], boot[4874]],
            "second_wins": sum(x > 0 for x in delta),
            "first_wins": sum(x < 0 for x in delta),
            "ties": sum(x == 0 for x in delta),
        }
    path = OUT / f"summary_{cohort}.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["60", "309"], default="60")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    assert len(args.models) == len(set(args.models))
    OUT.mkdir(exist_ok=True)
    configure(args.models[0])
    prompt_hash = canonical_prompt_hash(True)
    assert prompt_hash == EXPECTED_PROMPT_SHA256, f"Prompt drift: {prompt_hash}"
    ids, items, prepared, main = prepare_inputs(args.cohort)
    digests = model_digests(args.models)
    metadata = {"cohort": args.cohort, "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "models": digests, "ollama_version": api("/api/version")["version"],
                "prompt_sha256": prompt_hash, "settings": {"think": True, "temperature": 0,
                "protocol": "json", "num_ctx": 8192, "num_predict": 768, "num_batch": 512,
                "max_steps": 5, "asr": "cached PhoWhisper large v2"},
                "note": "Model-only diagnostic: runtime excludes PhoWhisper ASR; larger models exceed competition parameter limits.",
                "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (VAL_IDS, MAIN, HISTORY,
                    DATA / "train" / "train.jsonl", DATA / "train" / "train_tools_catalog.json")},
                "results": {}}
    lock = OUT / f"lock_{args.cohort}.json"
    if lock.exists():
        old = json.loads(lock.read_text(encoding="utf-8"))
        assert {k: old[k] for k in old if k != "results"} == {k: metadata[k] for k in metadata if k != "results"}, "Experiment lock changed"
    else:
        lock.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seed_everything(17)
    if not args.analyze_only:
        for model in args.models:
            run_branch(args.cohort, model, ids, prepared, main)
    analyze(args.cohort, args.models, ids, items, metadata)


if __name__ == "__main__":
    main()
