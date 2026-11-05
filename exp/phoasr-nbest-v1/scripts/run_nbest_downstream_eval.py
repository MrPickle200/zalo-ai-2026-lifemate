from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.pipeline import MAX_STEPS, LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    a, b = ref.split(), hyp.split()
    return edit_distance(a, b) / max(1, len(a))


def cer(ref: str, hyp: str) -> float:
    a, b = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(a, b) / max(1, len(a))


def call_set(calls):
    return {json.dumps({"name": c.get("name"), "arguments": c.get("arguments", {})},
                       sort_keys=True, ensure_ascii=False) for c in calls}


def consensus_choice(candidates: list[dict]) -> dict:
    # Deterministic medoid: minimize mean token edit distance to all alternatives.
    return min(candidates, key=lambda c: (sum(wer(c["text"], other["text"]) for other in candidates),
                                          c["decode_config"], c["rank"]))


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate cached PhoASR transcripts through the unchanged LifeMate ToolEnv pipeline")
    p.add_argument("--transcripts", default="exp/phoasr-nbest-v1/output/transcripts.jsonl")
    p.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter")
    p.add_argument("--mode", choices=("current", "wer_oracle", "auto_consensus"), required=True)
    p.add_argument("--data-root")
    p.add_argument("--output")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = p.parse_args()
    seed_everything()
    cache = [json.loads(line) for line in Path(args.transcripts).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in cache]
    if len(ids) != len(set(ids)):
        raise SystemExit("Transcript cache contains duplicate sample IDs")
    root = find_data_root(args.data_root)
    items = {r["id"]: r for r in load_jsonl(root / "train" / "train.jsonl")}
    if any(sid not in items for sid in ids):
        raise SystemExit("Transcript cache has IDs missing from the train split")
    agent = Agent(model_path=args.model, adapter_path=args.adapter)
    pipeline = LifeMatePipeline(root, "train", agent=agent, max_steps=args.max_steps)
    output = Path(args.output or (Path("exp/phoasr-nbest-v1/output/hard30") / f"downstream_{args.mode}"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(cache, 1):
            item = items[record["id"]]
            candidates = record["candidates"]
            if args.mode == "current":
                selected = next(c for c in candidates if c["decode_config"] == "production_greedy")
            elif args.mode == "wer_oracle":
                selected = min(candidates, key=lambda c: (wer(record["gold_text"], c["text"]),
                                                          cer(record["gold_text"], c["text"]), c["rank"]))
            else:
                selected = consensus_choice(candidates)
            try:
                result = pipeline.run(item, transcript_override=selected["text"])
                gold_calls = item.get("label", {}).get("tool_calls", [])
                predicted = result["tool_calls"]
                ps, gs = call_set(predicted), call_set(gold_calls)
                tp = len(ps & gs)
                precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
                recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                expected_state = item.get("label", {}).get("final_state", {})
                state_match = all(result["final_state"].get(app) == state for app, state in expected_state.items())
                actions = result["model_output"] or []
                first_gold = gold_calls[0] if gold_calls else None
                first_pred = predicted[0] if predicted else None
                first_tool_accuracy = (first_pred is not None and first_gold is not None and
                                       first_pred.get("name") == first_gold.get("name"))
                first_full_call = (first_pred is not None and first_gold is not None and
                                   first_pred.get("name") == first_gold.get("name") and
                                   first_pred.get("arguments", {}) == first_gold.get("arguments", {}))
                malformed = sum(a.get("action") == "invalid" or a.get("parse_error") is not None for a in actions)
                row = {"id": record["id"], "selected_candidate": selected, "tool_calls": predicted,
                       "model_output": actions, "observations": result["observations"],
                       "final_state": result["final_state"], "state_hash": result["state_hash"],
                       "tool_call_f1": f1, "state_match": state_match,
                       "score": (f1 + int(state_match)) / 2, "exact_trajectory": predicted == gold_calls,
                       "first_tool_accuracy": first_tool_accuracy, "first_full_call_accuracy": first_full_call,
                       "malformed_count": malformed, "trajectory_length": len(predicted),
                       "wer": wer(record["gold_text"], selected["text"]),
                       "cer": cer(record["gold_text"], selected["text"]),
                       "latency_seconds": result["latency_seconds"]}
            except Exception as exc:
                row = {"id": record["id"], "selected_candidate": selected, "runtime_error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            print(f"[{index}/{len(cache)}] {record['id']} score={row.get('score')} error={row.get('runtime_error')}", flush=True)
    valid = [r for r in rows if "score" in r]
    tp = sum(len(call_set(r["tool_calls"]) & call_set(items[r["id"]].get("label", {}).get("tool_calls", []))) for r in valid)
    pred_count = sum(len(call_set(r["tool_calls"])) for r in valid)
    gold_count = sum(len(call_set(items[r["id"]].get("label", {}).get("tool_calls", []))) for r in valid)
    precision = tp / pred_count if pred_count else (1.0 if not gold_count else 0.0)
    recall = tp / gold_count if gold_count else (1.0 if not pred_count else 0.0)
    summary = {"mode": args.mode, "samples": len(rows), "runtime_failures": len(rows) - len(valid),
               "wer": sum(r["wer"] for r in valid) / max(1, len(valid)),
               "cer": sum(r["cer"] for r in valid) / max(1, len(valid)),
               "tool_call_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
               "state_match": sum(r["state_match"] for r in valid) / max(1, len(valid)),
               "proxy": sum(r["score"] for r in valid) / max(1, len(valid)),
               "first_tool_accuracy": sum(bool(r["first_tool_accuracy"]) for r in valid) / max(1, len(valid)),
               "first_full_call_accuracy": sum(bool(r["first_full_call_accuracy"]) for r in valid) / max(1, len(valid)),
               "exact_trajectory": sum(r["exact_trajectory"] for r in valid) / max(1, len(valid)),
               "malformed_count": sum(r["malformed_count"] for r in valid),
               "average_trajectory_length": sum(r["trajectory_length"] for r in valid) / max(1, len(valid)),
               "average_latency_seconds": sum(r["latency_seconds"] for r in valid) / max(1, len(valid)),
               "selection": {"current": "production_greedy", "wer_oracle": "minimum gold WER, then CER, then rank",
                             "auto_consensus": "candidate medoid minimizing average pairwise token WER"}[args.mode],
               "predictions": "predictions.jsonl"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["runtime_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
