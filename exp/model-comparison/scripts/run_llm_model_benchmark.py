from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.pipeline import MAX_STEPS, LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


def calls_set(calls):
    return {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls}


def score(item, result):
    gold = item.get("label", {}).get("tool_calls", [])
    pred = result["tool_calls"]
    p, g = calls_set(pred), calls_set(gold)
    tp = len(p & g)
    precision = tp / len(p) if p else (1.0 if not g else 0.0)
    recall = tp / len(g) if g else (1.0 if not p else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected = item.get("label", {}).get("final_state", {})
    state = all(result["final_state"].get(k) == v for k, v in expected.items())
    return {"tool_call_f1": f1, "state_match": state, "proxy": (f1 + int(state)) / 2,
            "trajectory_exact": pred == gold,
            "first_full_call_accuracy": bool(pred and gold and pred[0] == gold[0]),
            "first_tool_accuracy": bool(pred and gold and pred[0].get("name") == gold[0].get("name"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-root")
    args = ap.parse_args()
    seed_everything()
    root = find_data_root(args.data_root)
    cache = list(load_jsonl(Path(args.transcripts)))
    items = {r["id"]: r for r in load_jsonl(root / "train" / "train.jsonl")}
    if len(cache) != 30 or len({r["id"] for r in cache}) != 30:
        raise SystemExit("Expected exactly 30 unique cached transcripts")
    missing = set(r["id"] for r in cache) - items.keys()
    if missing:
        raise SystemExit(f"Unknown sample IDs: {sorted(missing)}")
    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = args.model
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "16384"
    agent = Agent(max_new_tokens=128, context_limit=16384)
    pipeline = LifeMatePipeline(root, "train", agent=agent, max_steps=MAX_STEPS)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, cached in enumerate(cache, 1):
            item = items[cached["id"]]
            try:
                result = pipeline.run(item, transcript_override=cached["transcript"])
                row = {**result, "gold_tool_calls": item.get("label", {}).get("tool_calls", []),
                       "metrics": score(item, result)}
            except Exception as exc:
                row = {"id": cached["id"], "tool_calls": [], "metrics": None,
                       "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{i}/30] {cached['id']} {row.get('metrics') or row.get('error')}", flush=True)
    good = [r for r in rows if r["metrics"]]
    agg = {k: sum(r["metrics"][k] for r in good) / len(good) if good else None
           for k in ("tool_call_f1", "state_match", "proxy", "trajectory_exact",
                     "first_full_call_accuracy", "first_tool_accuracy")}
    agg.update(model=args.model, samples=30, runtime_failures=30-len(good),
               malformed_outputs=sum(1 for r in good for a in r["model_output"]
                                     if a.get("action") not in ("stop", "tool_call")),
               average_trajectory_length=sum(len(r["tool_calls"]) for r in good)/len(good) if good else None,
               predictions_file="predictions.jsonl")
    (outdir / "summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
