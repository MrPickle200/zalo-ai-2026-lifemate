from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl


def strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield str(value)


def dynamic_argument_keys(item: dict, label: dict) -> dict[int, set[str]]:
    calls = label.get("tool_calls", [])
    results = label.get("tool_results", [])
    present = json.dumps({"transcript": item.get("audio_text", ""), "context": item.get("context", {}),
                          "history": item.get("history", [])}, ensure_ascii=False)
    dynamic = {}
    for step in range(1, min(len(calls), len(results) + 1)):
        prior_values = set(strings(results[:step]))
        keys = {key for key, val in calls[step].get("arguments", {}).items()
                if str(val) in prior_values and str(val) not in present}
        if keys:
            dynamic[step] = keys
    return dynamic


def continuation_category(item: dict, row: dict) -> tuple[str, dict]:
    gold = item.get("label", {}).get("tool_calls", [])
    pred = row.get("tool_calls", [])
    actions = row.get("model_output", [])
    dynamic = dynamic_argument_keys(item, item.get("label", {}))
    detail = {"dynamic_gold_argument_keys_by_step": {str(k + 1): sorted(v) for k, v in dynamic.items()},
              "predicted_actions_after_injected_first_call": actions[1:]}
    # The injected first call is excluded: only the model's continuation is classified.
    for index in range(1, len(gold)):
        if index >= len(pred):
            if any(a.get("action") == "stop" for a in actions[1:]):
                return "Premature STOP", detail
            if len(actions) >= 5:
                return "Turn cap", detail
            return "Other", detail
        expected, actual = gold[index], pred[index]
        if actual.get("name") != expected.get("name"):
            return ("Wrong second tool" if index == 1 else "Other"), detail
        if actual.get("arguments", {}) != expected.get("arguments", {}):
            if index in dynamic and any(actual.get("arguments", {}).get(k) != expected.get("arguments", {}).get(k)
                                        for k in dynamic[index]):
                return "Observation ignored", detail
            return "Wrong later argument", detail
    if len(pred) > len(gold):
        return "Extra call", detail
    if len(pred) < len(gold):
        if any(a.get("action") == "stop" for a in actions[1:]):
            return "Premature STOP", detail
        if len(actions) >= 5:
            return "Turn cap", detail
        return "Other", detail
    return "Other", detail


def main():
    root = find_data_root()
    items = {r["id"]: r for r in load_jsonl(root / "train" / "train.jsonl")}
    path = Path("outputs/eval/oracle-first-full-call-30/predictions.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    counts = Counter()
    sample_details = []
    for row in rows:
        category, detail = continuation_category(items[row["id"]], row)
        counts[category] += 1
        sample_details.append({"id": row["id"], "category": category,
                               "gold_calls": row["gold_tool_calls"], "predicted_calls": row["tool_calls"],
                               "model_output": row["model_output"], **detail})
    ordered = ("Wrong second tool", "Observation ignored", "Wrong later argument", "Premature STOP",
               "Extra call", "Turn cap", "Other")
    result = {"samples": len(rows), "failure_counts": {key: counts[key] for key in ordered},
              "all_final_states_mismatched": sum(not bool(r.get("metrics", {}).get("state_match")) for r in rows),
              "sample_details": sample_details}
    out = Path("outputs/eval/oracle-first-full-call-30")
    (out / "continuation-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paired_path = Path("outputs/eval/paired-analysis/oracle-first-action/summary.json")
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    paired["oracle_full_first_call"]["continuation_failure_counts"] = result["failure_counts"]
    paired["oracle_full_first_call"]["all_final_states_mismatched"] = result["all_final_states_mismatched"]
    paired_path.write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
