from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def call_key(call):
    return json.dumps({"name": call.get("name"), "arguments": call.get("arguments", {})},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def first_divergence(gold, pred):
    for index, expected in enumerate(gold):
        if index >= len(pred):
            return index, "premature_stop"
        actual = pred[index]
        if actual.get("name") != expected.get("name"):
            return index, "wrong_tool" if index == 0 else "wrong_later_action"
        if call_key(actual) != call_key(expected):
            return index, "wrong_arguments"
    if len(pred) > len(gold):
        return len(gold), "extra_tool_call"
    return None, "correct_trajectory"


def summarize(rows, items):
    if not rows:
        raise ValueError("No prediction rows")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    scores = [r.get("score") for r in rows if r.get("score") is not None]
    gold_count = predicted_count = tp = state_match_count = exact_count = 0
    first_exact = second_tool_correct = second_action_correct = second_condition_count = 0
    first_action_count = 0
    consecutive = []
    errors: Counter[str] = Counter()
    cap_hits = malformed = 0
    per_id = {}
    for row in rows:
        item = items[row["id"]]
        gold = item.get("label", {}).get("tool_calls", [])
        pred = row.get("tool_calls", [])
        gs = {call_key(c) for c in gold}
        ps = {call_key(c) for c in pred}
        tp += len(gs & ps)
        gold_count += len(gs)
        predicted_count += len(ps)
        state_match_count += row.get("state_match") is True
        exact = gold == pred
        exact_count += exact
        first_action_count += bool(gold)
        first_ok = bool(gold and pred and call_key(gold[0]) == call_key(pred[0]))
        first_exact += first_ok
        if first_ok and len(gold) >= 2:
            second_condition_count += 1
            second_tool_correct += len(pred) > 1 and pred[1].get("name") == gold[1].get("name")
            second_action_correct += len(pred) > 1 and call_key(pred[1]) == call_key(gold[1])
        n = 0
        for expected, actual in zip(gold, pred):
            if call_key(expected) != call_key(actual):
                break
            n += 1
        consecutive.append(n)
        index, kind = first_divergence(gold, pred)
        errors[kind] += 1
        if any(c.get("name") == "sys.ask_missing_info" for c in gold + pred) and kind != "correct_trajectory":
            errors["missing_info_error"] += 1
        actions = row.get("model_output") or []
        if len(actions) >= 5 and not (actions and actions[-1].get("action") == "stop"):
            cap_hits += 1
        if any(a.get("action") not in {"tool_call", "stop"} for a in actions):
            malformed += 1
        per_id[row["id"]] = {"gold_calls": gold, "predicted_calls": pred, "first_divergence_index": index,
                              "failure": kind, "state_match": row.get("state_match"),
                              "baseline_score": None, "lora_score": row.get("score")}
    precision = tp / predicted_count if predicted_count else (1.0 if not gold_count else 0.0)
    recall = tp / gold_count if gold_count else (1.0 if not predicted_count else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"samples": len(rows), "runtime_failures": sum(r.get("score") is None for r in rows),
            "proxy": sum(scores) / len(scores) if scores else None, "tool_call_f1": f1,
            "final_state_match_rate": state_match_count / len(rows),
            "exact_trajectory_count": exact_count, "exact_trajectory_rate": exact_count / len(rows),
            "first_full_action_accuracy": first_exact / first_action_count if first_action_count else None,
            "first_action_gold_count": first_action_count,
            "second_tool_accuracy_given_correct_first": second_tool_correct / second_condition_count if second_condition_count else None,
            "second_full_action_accuracy_given_correct_first": second_action_correct / second_condition_count if second_condition_count else None,
            "correct_first_condition_count_with_second_action": second_condition_count,
            "average_correct_consecutive_calls_before_divergence": sum(consecutive) / len(consecutive),
            "turn_cap_hits": cap_hits, "malformed_actions": malformed,
            "average_latency_seconds": sum(r.get("latency_seconds", 0.0) for r in rows) / len(rows),
            "error_counts": dict(errors), "per_sample": per_id}


def main():
    p = argparse.ArgumentParser(description="Paired analysis for next-action LoRA validation and hard-30 diagnostic")
    p.add_argument("--predictions", default="exp/qwen-next-action-lora-v1-kaggle-val309/output/kaggle-output-targeted/outputs/eval/qwen-next-action-lora-v1-kaggle-val309/predictions.jsonl")
    p.add_argument("--baseline", default="exp/phoasr-small-baseline-val309/output/predictions.jsonl")
    p.add_argument("--hard-audit", default="exp/agent-architecture-audit-v1-val309/output/architecture-audit.json")
    p.add_argument("--train-manifest", default="train+public_test/train+public_test/train/train.jsonl")
    p.add_argument("--output", default="exp/paired-analysis/qwen-next-action-lora-v1/output/summary.json")
    args = p.parse_args()
    items = {r["id"]: r for r in read_jsonl(Path(args.train_manifest))}
    lora_rows = read_jsonl(Path(args.predictions))
    baseline_rows = read_jsonl(Path(args.baseline))
    if {r["id"] for r in lora_rows} != {r["id"] for r in baseline_rows}:
        raise SystemExit("LoRA validation IDs do not exactly match baseline IDs")
    all_train_ids = set(items)
    val_ids = set(r["id"] for r in lora_rows)
    if not val_ids <= all_train_ids:
        raise SystemExit("Validation prediction contains ID outside labeled train manifest")
    lora = summarize(lora_rows, items)
    baseline = summarize(baseline_rows, items)
    bmap = {r["id"]: r for r in baseline_rows}
    outcomes = Counter()
    paired = []
    for row in lora_rows:
        old = bmap[row["id"]]
        a, b = old.get("score") or 0.0, row.get("score") or 0.0
        category = "improved" if b > a + 1e-12 else "worsened" if b < a - 1e-12 else "unchanged"
        outcomes[category] += 1
        lora["per_sample"][row["id"]]["baseline_score"] = old.get("score")
        paired.append({"id": row["id"], "baseline_score": old.get("score"), "lora_score": row.get("score"),
                       "delta": b - a, "outcome": category,
                       "baseline_state_match": old.get("state_match"), "lora_state_match": row.get("state_match")})
    hard_audit = json.loads(Path(args.hard_audit).read_text(encoding="utf-8"))
    hard_ids = {r["id"] for r in hard_audit["samples"]}
    hard_rows = [r for r in lora_rows if r["id"] in hard_ids]
    if len(hard_rows) != len(hard_ids):
        raise SystemExit("LoRA validation is missing one or more fixed hard-30 IDs")
    hard_lora = summarize(hard_rows, items)
    hard_lora["baseline_final_state_match_rate"] = hard_audit["baseline_state_match_rate"]
    hard_lora["baseline_tool_call_f1"] = hard_audit["baseline_tool_call_f1"]
    result = {"experiment": "qwen-next-action-lora-v1", "baseline": baseline,
              "lora": lora, "paired_outcomes": dict(outcomes),
              "paired_improvement_worsening_unchanged": dict(outcomes),
              "hard30": {"samples": len(hard_rows), "baseline_final_state_match_count": hard_audit["baseline_state_match_count"],
                         "baseline_final_state_match_rate": hard_audit["baseline_state_match_rate"],
                         "baseline_tool_call_f1": hard_audit["baseline_tool_call_f1"],
                         "baseline_proxy": hard_audit["baseline_composite_proxy_mean"],
                         "lora": hard_lora}, "per_sample_paired": paired}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample_paired"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
