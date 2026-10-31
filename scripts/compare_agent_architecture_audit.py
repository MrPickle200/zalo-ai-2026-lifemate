from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def call_key(call):
    return json.dumps({"name": call.get("name"), "arguments": call.get("arguments", {})},
                      sort_keys=True, ensure_ascii=False)


def main():
    root = Path("outputs/eval")
    audit = json.loads((root / "agent-architecture-audit-v1-val309" / "architecture-audit.json").read_text(encoding="utf-8"))
    runs = {
        cap: [json.loads(line) for line in
              (root / f"agent-architecture-audit-v1-iterative-cap{cap}-val30" / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for cap in (6, 7)
    }
    old_rows = {r["id"]: r for r in audit["samples"]}
    details, deltas_by_cap = {}, {}
    for cap, new_rows in runs.items():
        if set(old_rows) != {r["id"] for r in new_rows}:
            raise SystemExit(f"Cap-{cap} run does not match the 30-sample architecture audit ID set")
        paired = Counter()
        new_tp = new_pred = new_gold = 0
        new_state = new_first = new_exact = 0
        pred_lengths = Counter()
        consumed_cap_turns = 0
        deltas = []
        for row in new_rows:
            old = old_rows[row["id"]]
            old_score = old.get("baseline_score") or 0.0
            new_score = row.get("score") or 0.0
            delta = new_score - old_score
            paired["improved" if delta > 1e-12 else "worsened" if delta < -1e-12 else "unchanged"] += 1
            deltas.append({"id": row["id"], "old_score": old_score, "new_score": new_score,
                           "old_state_match": old.get("baseline_state_match"),
                           "new_state_match": row.get("state_match"), "delta": delta})
            gold = old["gold_tool_calls"]
            predicted = row.get("tool_calls", [])
            gs, ps = {call_key(c) for c in gold}, {call_key(c) for c in predicted}
            new_tp += len(gs & ps)
            new_gold += len(gs)
            new_pred += len(ps)
            new_state += row.get("state_match") is True
            new_first += bool(gold and predicted and gold[0] == predicted[0])
            new_exact += gold == predicted
            pred_lengths[len(predicted)] += 1
            consumed_cap_turns += len(row.get("model_output") or []) >= cap
        new_tool_f1 = 2 * new_tp / (new_pred + new_gold) if new_pred + new_gold else 0.0
        details[f"cap{cap}"] = {
            "tool_call_f1": new_tool_f1,
            "state_match_rate": new_state / len(new_rows),
            "composite_proxy_mean": sum((r.get("score") or 0.0) for r in new_rows) / len(new_rows),
            "first_action_exact_count": new_first,
            "full_trajectory_exact_count": new_exact,
            "paired_proxy_outcomes_vs_cap5": dict(paired),
            "predicted_call_count_distribution": dict(sorted(pred_lengths.items())),
            "runs_using_all_cap_turns": consumed_cap_turns,
            "average_latency_seconds": sum(r.get("latency_seconds", 0.0) for r in new_rows) / len(new_rows),
        }
        deltas_by_cap[f"cap{cap}"] = deltas

    old_tool_f1 = audit["baseline_tool_call_f1"]
    result = {
        "experiment": "agent-architecture-audit-v1-iterative-cap7-val30",
        "samples": len(next(iter(runs.values()))),
        "baseline_cap": 5,
        "baseline_tool_call_f1": old_tool_f1,
        "baseline_state_match_rate": audit["baseline_state_match_rate"],
        "baseline_composite_proxy_mean": audit["baseline_composite_proxy_mean"],
        "baseline_average_latency_seconds": audit["baseline_average_latency_seconds"],
        "baseline_first_action_exact_count": audit["baseline_exact_first_action_count"],
        "baseline_full_trajectory_exact_count": audit["baseline_exact_trajectory_count"],
        "iterative_cap_runs": details,
        "per_sample_deltas": deltas_by_cap,
    }
    out = root / "paired-analysis" / "agent-architecture-cap5-vs-cap7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample_deltas"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
