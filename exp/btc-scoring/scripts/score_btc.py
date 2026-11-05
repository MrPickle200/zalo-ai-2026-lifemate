from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from lifemate.tool_executor import create_tool_env
from lifemate.utils import find_data_root, load_jsonl


def call_key(call: dict[str, Any]) -> str:
    return json.dumps(
        {"name": call.get("name"), "arguments": call.get("arguments", {})},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id in rows:
                raise ValueError(f"Duplicate prediction for {sample_id} at line {line_number}")
            rows[sample_id] = row
    return rows


def score_sample(
    sample_id: str,
    item: dict[str, Any],
    prediction: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    gold_calls = item.get("label", {}).get("tool_calls", [])
    pred_calls = prediction.get("tool_calls") or []

    env = create_tool_env("train", sample_id, data_root)
    replay_error = None
    for step, call in enumerate(gold_calls):
        name = call.get("name")
        arguments = call.get("arguments", {})
        validation = env.validate(name, arguments)
        if not validation.get("ok"):
            replay_error = f"gold call {step} rejected: {validation.get('error')}"
            break
        result = env.call(name, arguments)
        if not result.get("ok"):
            replay_error = f"gold call {step} failed: {result}"
            break

    gold_hash = env.get_state_hash()
    pred_hash = prediction.get("state_hash")
    gold_keys = Counter(call_key(call) for call in gold_calls)
    pred_keys = Counter(call_key(call) for call in pred_calls)
    matched = sum((gold_keys & pred_keys).values())
    unmatched_submitted = len(pred_calls) - matched
    gold_count = len(gold_calls)
    hash_match = pred_hash == gold_hash
    sample_latency = prediction.get("latency_seconds")
    runtime_over_30s = sample_latency is not None and float(sample_latency) > 30.0
    runtime_failure = bool(prediction.get("error"))

    if replay_error or not hash_match or matched == 0:
        score_without_runtime_gate = 0.0
    else:
        precision = matched / (matched + unmatched_submitted)
        recall = matched / gold_count
        score_without_runtime_gate = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sample_score = 0.0 if runtime_failure or runtime_over_30s else score_without_runtime_gate

    proxy_precision = matched / len(pred_calls) if pred_calls else (1.0 if not gold_calls else 0.0)
    proxy_recall = matched / gold_count if gold_count else (1.0 if not pred_calls else 0.0)
    proxy_f1 = (2 * proxy_precision * proxy_recall / (proxy_precision + proxy_recall)
                if proxy_precision + proxy_recall else 0.0)
    expected_state = item.get("label", {}).get("final_state", {})
    predicted_state = prediction.get("final_state") or {}
    predicted_state_matches_label = all(predicted_state.get(app) == value for app, value in expected_state.items())
    local_proxy = (proxy_f1 + int(predicted_state_matches_label)) / 2

    schema_rejections = sum(
        observation.get("result", {}).get("ok") is False
        for observation in prediction.get("observations", [])
    )
    replay_state = env.get_state()
    final_state_match = all(replay_state.get(app) == value for app, value in expected_state.items())

    return {
        "id": sample_id,
        "official_sample_score": sample_score,
        "btc_formula_score_before_runtime_gate_diagnostic": score_without_runtime_gate,
        "matched_calls": matched,
        "unmatched_submitted_calls": unmatched_submitted,
        "gold_calls": gold_count,
        "hash_match": hash_match,
        "gold_state_hash": gold_hash,
        "prediction_state_hash": pred_hash,
        "prediction_latency_seconds": sample_latency,
        "runtime_over_30s": runtime_over_30s,
        "runtime_failure": runtime_failure,
        "schema_rejections": schema_rejections,
        "gold_replay_error": replay_error,
        "gold_final_state_matches_label": final_state_match,
        "predicted_final_state_matches_label": predicted_state_matches_label,
        "local_proxy_score_diagnostic": local_proxy,
        "local_proxy_call_f1_diagnostic": proxy_f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score predictions with the official LifeMate BTC formula.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--details", required=True, help="Output JSONL with per-sample official score diagnostics")
    parser.add_argument("--summary", required=True, help="Output JSON summary")
    args = parser.parse_args()

    data_root = find_data_root(args.data_root)
    split_dir = Path(args.split_dir)
    val_ids = json.loads((split_dir / "val_ids.json").read_text(encoding="utf-8"))
    items = {item["id"]: item for item in load_jsonl(data_root / "train" / "train.jsonl")}
    predictions = read_predictions(Path(args.predictions))

    missing_predictions = sorted(set(val_ids) - set(predictions))
    unexpected_predictions = sorted(set(predictions) - set(val_ids))
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            f"Prediction cohort mismatch: missing={len(missing_predictions)}, "
            f"unexpected={len(unexpected_predictions)}"
        )

    details = []
    for index, sample_id in enumerate(val_ids, 1):
        if sample_id not in items:
            raise KeyError(f"Validation ID not found in train.jsonl: {sample_id}")
        details.append(score_sample(sample_id, items[sample_id], predictions[sample_id], data_root))
        if index % 25 == 0 or index == len(val_ids):
            print(f"[{index}/{len(val_ids)}] replayed gold with ToolEnv", flush=True)

    n = len(details)
    total_matched = sum(row["matched_calls"] for row in details)
    total_submitted = sum(
        len(predictions[row["id"]].get("tool_calls") or []) for row in details
    )
    total_gold = sum(row["gold_calls"] for row in details)
    precision = total_matched / total_submitted if total_submitted else 0.0
    recall = total_matched / total_gold if total_gold else 0.0
    aggregate_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "metric": "BTC official sample score",
        "split": "fixed train-only val309",
        "matching": "exact tool name and arguments; duplicate calls matched by multiplicity",
        "validation_samples": n,
        "official_score_0_100": 100 * sum(row["official_sample_score"] for row in details) / n if n else None,
        "btc_formula_score_before_runtime_gate_diagnostic_0_100": 100 * sum(row["btc_formula_score_before_runtime_gate_diagnostic"] for row in details) / n if n else None,
        "hash_match_samples": sum(row["hash_match"] for row in details),
        "hash_match_rate": sum(row["hash_match"] for row in details) / n if n else None,
        "gold_replay_failures": sum(row["gold_replay_error"] is not None for row in details),
        "gold_replay_state_matches_label": sum(row["gold_final_state_matches_label"] for row in details),
        "runtime_failures": sum(row["runtime_failure"] for row in details),
        "samples_over_30s": sum(row["runtime_over_30s"] for row in details),
        "schema_rejected_submitted_calls": sum(row["schema_rejections"] for row in details),
        "matched_calls_total": total_matched,
        "submitted_calls_total": total_submitted,
        "gold_calls_total": total_gold,
        "aggregate_exact_call_precision": precision,
        "aggregate_exact_call_recall": recall,
        "aggregate_exact_call_f1_diagnostic": aggregate_f1,
        "proxy_score_mean_diagnostic": sum(row["local_proxy_score_diagnostic"] for row in details) / n if n else None,
        "predicted_final_state_matches_label": sum(row["predicted_final_state_matches_label"] for row in details),
    }

    details_path = Path(args.details)
    summary_path = Path(args.summary)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["gold_replay_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
