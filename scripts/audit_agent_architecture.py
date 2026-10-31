from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl


READ_WORDS = ("get_", "list_", "search", "read_", "guide", "forecast", "history", "menu",
              "transactions", "beneficiaries", "status", "balance", "price")


def flatten_strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from flatten_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from flatten_strings(v)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield str(value)


def is_read(name: str, catalog: dict) -> bool:
    specs = {t["name"]: t for app in catalog.get("apps", []) for t in app.get("tools", [])}
    return specs.get(name, {}).get("kind") == "read" or name.split(".", 1)[-1].startswith(READ_WORDS)


def inspect_item(item: dict, pred: dict, catalog: dict) -> dict:
    label = item.get("label", {})
    gold = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in label.get("tool_calls", [])]
    gold_results = label.get("tool_results", [])
    reads = [is_read(c["name"], catalog) for c in gold]
    writes = [not r and c["name"] != "sys.ask_missing_info" for c, r in zip(gold, reads)]
    transcript = item.get("audio_text", "")
    available_text = json.dumps({"transcript": transcript, "context": item.get("context", {}),
                                 "history": item.get("history", [])}, ensure_ascii=False)
    dependency = None
    dependency_reason = None
    for step in range(1, len(gold)):
        prior_result_text = json.dumps(gold_results[:step], ensure_ascii=False) if len(gold_results) >= step else ""
        prior_values = set(flatten_strings(gold_results[:step]))
        for key, value in gold[step]["arguments"].items():
            val = str(value)
            if (val in prior_values and len(val) >= 3 and val not in available_text
                    and reads[:step] and prior_result_text):
                dependency = step
                dependency_reason = (f"argument {key}={value!r} of step {step + 1} appears in earlier gold tool result "
                                     "but not in transcript/context/history")
                break
        if dependency is not None:
            break
    has_missing = any(c["name"] == "sys.ask_missing_info" for c in gold)
    state_dep = any(reads[i] and any(writes[i + 1:]) for i in range(len(gold)))
    obs_dep = dependency is not None
    if has_missing:
        main_type = "missing_info"
    elif obs_dep:
        main_type = "observation_dependent"
    elif state_dep:
        main_type = "state_dependent"
    elif len(gold) <= 1:
        main_type = "one_shot"
    else:
        main_type = "multi_tool_preplannable"

    predicted = pred.get("tool_calls", [])
    exact_first = bool(gold and predicted and gold[0] == predicted[0])
    exact_trajectory = gold == predicted
    if exact_trajectory:
        main_failure = "correct_trajectory"
    elif not predicted:
        main_failure = "stopped_before_first_call"
    elif not exact_first:
        main_failure = "wrong_first_action"
    elif len(predicted) < len(gold) and predicted == gold[:len(predicted)]:
        main_failure = "stopped_early"
    elif len(predicted) > len(gold) and predicted[:len(gold)] == gold:
        main_failure = "extra_calls"
    elif [c.get("name") for c in predicted] == [c["name"] for c in gold]:
        main_failure = "right_tools_wrong_arguments_or_order"
    else:
        main_failure = "later_tool_or_order_error"

    first_read_missing = bool(gold and reads and reads[0] and (not predicted or predicted[0].get("name") != gold[0]["name"]))
    obs_arg_mismatch = bool(obs_dep and len(predicted) > dependency and
                            predicted[dependency].get("arguments", {}) != gold[dependency]["arguments"])
    return {
        "id": item["id"], "transcript": transcript, "gold_tool_calls": gold,
        "gold_tool_results": gold_results, "number_of_steps": len(gold),
        "requires_previous_observation": "yes" if obs_dep else "unclear" if state_dep else "no",
        "observation_dependency_reason": dependency_reason,
        "read_then_write_state_dependent": state_dep, "has_missing_info": has_missing,
        "main_trajectory_type": main_type,
        "baseline_tool_calls": predicted, "baseline_model_actions": pred.get("model_output", []),
        "baseline_observations": pred.get("observations", []),
        "baseline_final_state": pred.get("final_state", {}),
        "baseline_state_match": pred.get("state_match"),
        "baseline_exact_trajectory": exact_trajectory, "baseline_exact_first_action": exact_first,
        "baseline_main_failure": main_failure, "baseline_missed_first_read": first_read_missing,
        "baseline_observation_argument_mismatch": obs_arg_mismatch,
        "baseline_score": pred.get("score"), "latency_seconds": pred.get("latency_seconds"),
    }


def main():
    root = find_data_root()
    all_items = {r["id"]: r for r in load_jsonl(root / "train" / "train.jsonl")}
    val_ids = json.loads(Path("data/splits/val_ids.json").read_text(encoding="utf-8"))
    predictions_path = Path("outputs/eval/phoasr-small-baseline-val309/predictions.jsonl")
    preds = {r["id"]: r for r in load_jsonl(predictions_path)}
    catalog = json.loads((root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))

    # Diagnostic uses complex held-out examples from the labeled train split, allowing a paired
    # comparison with existing PhoASR baseline outputs without training/tuning on validation labels.
    candidates = [all_items[sid] for sid in val_ids
                  if sid in all_items and sid in preds and len(all_items[sid].get("label", {}).get("tool_calls", [])) >= 2]
    inspected = [inspect_item(item, preds[item["id"]], catalog) for item in candidates]
    inspected.sort(key=lambda r: (r["read_then_write_state_dependent"],
                                  r["requires_previous_observation"] == "yes",
                                  r["number_of_steps"], r["baseline_state_match"] is False), reverse=True)
    inspected = inspected[:30]
    if len(inspected) != 30:
        raise SystemExit(f"Expected 30 paired multi-tool samples, found {len(inspected)}")

    by_type = Counter(r["main_trajectory_type"] for r in inspected)
    failures = Counter(r["baseline_main_failure"] for r in inspected)
    state_matches = sum(r["baseline_state_match"] is True for r in inspected)
    baseline_tp = baseline_pred = baseline_gold = 0
    for row in inspected:
        predicted = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in row["baseline_tool_calls"]}
        expected = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in row["gold_tool_calls"]}
        baseline_tp += len(predicted & expected)
        baseline_pred += len(predicted)
        baseline_gold += len(expected)
    baseline_f1 = (2 * baseline_tp / (baseline_pred + baseline_gold)
                   if baseline_pred + baseline_gold else 0.0)
    result = {
        "experiment": "agent-architecture-audit-v1-val309",
        "selection": "30 highest-complexity multi-tool examples from fixed validation IDs in official train split; paired to existing PhoASR public baseline predictions",
        "no_new_model_calls": True,
        "samples_inspected": len(inspected),
        "gold_step_counts": dict(sorted(Counter(str(r["number_of_steps"]) for r in inspected).items())),
        "trajectory_type_counts": dict(by_type),
        "trajectory_type_percent": {k: round(v / len(inspected), 4) for k, v in by_type.items()},
        "observation_dependent_count": sum(r["requires_previous_observation"] == "yes" for r in inspected),
        "state_dependent_read_then_write_count": sum(r["read_then_write_state_dependent"] for r in inspected),
        "missing_info_count": sum(r["has_missing_info"] for r in inspected),
        "baseline_state_match_count": state_matches,
        "baseline_state_match_rate": state_matches / len(inspected),
        "baseline_tool_call_f1": baseline_f1,
        "baseline_composite_proxy_mean": sum((r["baseline_score"] or 0.0) for r in inspected) / len(inspected),
        "baseline_average_latency_seconds": sum(r.get("latency_seconds", 0.0) for r in inspected) / len(inspected),
        "baseline_exact_trajectory_count": sum(r["baseline_exact_trajectory"] for r in inspected),
        "baseline_exact_first_action_count": sum(r["baseline_exact_first_action"] for r in inspected),
        "baseline_failure_counts": dict(failures),
        "baseline_missed_first_read_count": sum(r["baseline_missed_first_read"] for r in inspected),
        "baseline_observation_argument_mismatch_count": sum(r["baseline_observation_argument_mismatch"] for r in inspected),
        "samples": inspected,
    }
    out = Path("outputs/eval/agent-architecture-audit-v1-val309")
    out.mkdir(parents=True, exist_ok=True)
    (out / "architecture-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "trajectories.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in inspected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
