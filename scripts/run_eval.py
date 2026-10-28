from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


def call_set(calls):
    return {json.dumps({"name": c.get("name"), "arguments": c.get("arguments", {})}, sort_keys=True, ensure_ascii=False)
            for c in calls}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic holdout from labeled train data")
    parser.add_argument("--split", choices=["train"], default="train", help="local CV is restricted to train data")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="outputs/eval/latest.json")
    args = parser.parse_args()
    if args.limit < 1 or args.offset < 0:
        parser.error("--limit must be positive and --offset non-negative")
    seed_everything()
    root = find_data_root(args.data_root)
    items = list(load_jsonl(root / "train" / "train.jsonl"))[args.offset:args.offset + args.limit]
    if not items:
        raise SystemExit("No training samples selected")
    pipeline = LifeMatePipeline(root, "train")
    completed = failures = exact_matches = state_matches = 0
    tp = predicted = gold = 0
    total_latency = 0.0
    errors = []
    for item in items:
        try:
            result = pipeline.run(item)
            prediction = result["tool_calls"]
            expected = item.get("label", {}).get("tool_calls", [])
            pred_set, gold_set = call_set(prediction), call_set(expected)
            tp += len(pred_set & gold_set)
            predicted += len(pred_set)
            gold += len(gold_set)
            exact_matches += int(prediction == expected)
            expected_state = item.get("label", {}).get("final_state", {})
            state_matches += int(all(result["final_state"].get(app) == state for app, state in expected_state.items()))
            total_latency += result["latency_seconds"]
            completed += 1
        except Exception as exc:
            failures += 1
            errors.append({"id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            print(f"{item.get('id')}: {type(exc).__name__}: {exc}", file=sys.stderr)
            break  # model/runtime availability failures are common to remaining samples
    precision = tp / predicted if predicted else (1.0 if not gold else 0.0)
    recall = tp / gold if gold else (1.0 if not predicted else 0.0)
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if completed else None
    report = {"split": "train", "requested_samples": len(items), "samples_run": completed,
              "runtime_failures": failures, "tool_call_f1": f1,
              "tool_call_exact_match_rate": exact_matches / completed if completed else None,
              "state_hash_match_rate": state_matches / completed if completed else None,
              "average_latency_seconds": total_latency / completed if completed else None,
              "errors": errors}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if completed == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())

