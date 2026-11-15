from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "btc-scoring" / "scripts"))
from lifemate.utils import find_data_root, load_jsonl  # noqa: E402
from score_btc import score_sample  # noqa: E402

EXP = ROOT / "exp" / "quan-continuation-cot-v1"
OUT = EXP / "output" / "downstream"
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 20261107


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def descriptive(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.fmean(values) if values else None,
            "p50": quantile(values, 0.5), "p90": quantile(values, 0.9),
            "p95": quantile(values, 0.95), "max": max(values) if values else None}


def paired_bootstrap(values: list[float], rng: random.Random) -> dict[str, Any]:
    n = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(BOOTSTRAPS))
    return {"mean": statistics.fmean(values), "ci95_low": means[int(0.025 * BOOTSTRAPS)],
            "ci95_high": means[int(0.975 * BOOTSTRAPS)], "iterations": BOOTSTRAPS}


def aggregate_branch(branch: str, ids: list[str], predictions: dict[str, dict], traces: list[dict],
                     requests: list[dict], scores: dict[str, dict]) -> dict[str, Any]:
    matched = sum(scores[sid]["matched_calls"] for sid in ids)
    unmatched = sum(scores[sid]["unmatched_submitted_calls"] for sid in ids)
    gold = sum(scores[sid]["gold_calls"] for sid in ids)
    precision = matched / (matched + unmatched) if matched + unmatched else 0.0
    recall = matched / gold if gold else 0.0
    call_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    formula = [scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
    local_runtime_gate = [scores[sid]["official_sample_score"] for sid in ids]
    latencies = [float(predictions[sid].get("latency_seconds") or 0) for sid in ids]
    request_latencies = [float(row.get("latency_seconds") or 0) for row in requests]
    eval_counts = [int(row.get("eval_count") or 0) for row in requests]
    prompt_counts = [int(row.get("prompt_eval_count") or 0) for row in requests]
    cap_hits = [row for row in requests if (row.get("eval_count") or 0) >= 768]
    length_rows = [row for row in requests if row.get("done_reason") == "length"]
    repair_requests = sum(row.get("request_kind") == "repair" for row in requests)
    repair_turns = sum(bool(turn.get("_diagnostics", {}).get("repair_attempted"))
                       for sid in ids for turn in predictions[sid].get("model_output", []))
    repair_failures = sum(bool(turn.get("_diagnostics", {}).get("repair_attempted")) and
                          not turn.get("_diagnostics", {}).get("repair_succeeded", False)
                          for sid in ids for turn in predictions[sid].get("model_output", []))
    trace_by_id = {row["id"]: row for row in traces}
    parse_or_runtime_errors = sum(bool(row.get("runtime_error")) for row in traces)
    return {
        "branch": branch, "samples": len(ids), "ollama_requests": len(requests),
        "btc_formula_score_before_runtime_gate_0_100": 100 * statistics.fmean(formula),
        "local_30s_gated_score_diagnostic_0_100": 100 * statistics.fmean(local_runtime_gate),
        "hash_matches": sum(bool(scores[sid]["hash_match"]) for sid in ids),
        "gold_replay_failures": sum(scores[sid]["gold_replay_error"] is not None for sid in ids),
        "M_matched_calls": matched, "U_unmatched_submitted_calls": unmatched, "G_gold_calls": gold,
        "exact_call_precision_diagnostic": precision, "exact_call_recall_diagnostic": recall,
        "aggregate_exact_call_f1_diagnostic": call_f1,
        "final_state_matches_label_diagnostic": sum(bool(scores[sid]["predicted_final_state_matches_label"]) for sid in ids),
        "schema_rejections": sum(scores[sid]["schema_rejections"] for sid in ids),
        "runtime_errors": parse_or_runtime_errors,
        "sample_latency_seconds": descriptive(latencies),
        "request_latency_seconds": descriptive(request_latencies),
        "samples_over_30s_local": sum(value > 30 for value in latencies),
        "requests_over_30s_local": sum(value > 30 for value in request_latencies),
        "total_prompt_eval_tokens": sum(prompt_counts),
        "total_generated_tokens_including_thinking": sum(eval_counts),
        "prompt_eval_tokens_per_request": descriptive([float(v) for v in prompt_counts]),
        "generated_tokens_per_request_including_thinking": descriptive([float(v) for v in eval_counts]),
        "requests_reaching_num_predict": len(cap_hits), "samples_reaching_num_predict": sum(
            any((r.get("eval_count") or 0) >= 768 for r in requests if r.get("sample_id") == sid) for sid in ids),
        "requests_done_reason_length": len(length_rows),
        "requests_done_reason_length_not_num_predict": sum((r.get("eval_count") or 0) < 768 for r in length_rows),
        "repair_requests": repair_requests, "repair_turns_attempted": repair_turns,
        "repair_turns_failed": repair_failures,
        "samples_with_repair": sum(bool(trace_by_id[sid].get("repair_request_count")) for sid in ids),
        "action_turns": sum(len(predictions[sid].get("model_output", [])) for sid in ids),
        "empty_predictions": sum(not bool(predictions[sid].get("tool_calls")) for sid in ids),
        "tool_calls_submitted": sum(len(predictions[sid].get("tool_calls") or []) for sid in ids),
    }


def compute_branch(branch: str, ids: list[str], items: dict[str, dict], data_root: Path) -> tuple[dict, dict, dict, list]:
    pred_rows = rows(OUT / f"predictions-{branch}.jsonl")
    trace_rows = rows(OUT / f"step-traces-{branch}.jsonl")
    request_rows = rows(OUT / f"requests-{branch}.jsonl")
    if [row["id"] for row in pred_rows] != ids or [row["id"] for row in trace_rows] != ids:
        raise RuntimeError(f"Branch {branch} is not a complete ordered cohort")
    pred_map = {row["id"]: row for row in pred_rows}
    score_rows = []
    for index, sid in enumerate(ids, 1):
        score_rows.append(score_sample(sid, items[sid], pred_map[sid], data_root))
        if index % 15 == 0 or index == len(ids):
            print(f"branch={branch} gold ToolEnv replay {index}/{len(ids)}", flush=True)
    score_map = {row["id"]: row for row in score_rows}
    write_jsonl(OUT / f"btc-per-sample-{branch}.jsonl", score_rows)
    metrics = aggregate_branch(branch, ids, pred_map, trace_rows, request_rows, score_map)
    write_json(OUT / f"metrics-{branch}.json", metrics)
    return pred_map, score_map, metrics, request_rows


def pair_metrics(label: str, ids: list[str], left: dict[str, dict], right: dict[str, dict],
                 left_scores: dict[str, dict], right_scores: dict[str, dict],
                 left_requests: list[dict], right_requests: list[dict], rng: random.Random) -> dict:
    formula_delta = [right_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] -
                     left_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
    gated_delta = [right_scores[sid]["official_sample_score"] - left_scores[sid]["official_sample_score"] for sid in ids]
    gated_bootstrap = paired_bootstrap(gated_delta, random.Random(BOOTSTRAP_SEED + 13))
    latency_delta = [float(right[sid].get("latency_seconds") or 0) - float(left[sid].get("latency_seconds") or 0) for sid in ids]
    outcomes = Counter("improved" if d > 1e-12 else "worsened" if d < -1e-12 else "tied" for d in formula_delta)
    left_first = {row["sample_id"]: row for row in left_requests if row.get("sample_request_index") == 1}
    right_first = {row["sample_id"]: row for row in right_requests if row.get("sample_request_index") == 1}
    both_first = [sid for sid in ids if sid in left_first and sid in right_first]
    first_payload_equal = sum(left_first[sid].get("payload_sha256") == right_first[sid].get("payload_sha256") for sid in both_first)
    paired_rows = []
    for sid, score_d, gate_d, latency_d in zip(ids, formula_delta, gated_delta, latency_delta):
        paired_rows.append({"id": sid,
            "left_btc_formula_before_runtime_gate": left_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "right_btc_formula_before_runtime_gate": right_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "btc_formula_delta_right_minus_left": score_d,
            "left_local_30s_gated_score_diagnostic": left_scores[sid]["official_sample_score"],
            "right_local_30s_gated_score_diagnostic": right_scores[sid]["official_sample_score"],
            "local_30s_gated_delta_right_minus_left": gate_d,
            "left_hash_match": left_scores[sid]["hash_match"], "right_hash_match": right_scores[sid]["hash_match"],
            "left_M_U_G": [left_scores[sid]["matched_calls"], left_scores[sid]["unmatched_submitted_calls"], left_scores[sid]["gold_calls"]],
            "right_M_U_G": [right_scores[sid]["matched_calls"], right_scores[sid]["unmatched_submitted_calls"], right_scores[sid]["gold_calls"]],
            "left_latency_seconds": left[sid].get("latency_seconds"), "right_latency_seconds": right[sid].get("latency_seconds"),
            "latency_delta_seconds_right_minus_left": latency_d,
            "first_request_payload_equal": (left_first[sid].get("payload_sha256") == right_first[sid].get("payload_sha256")) if sid in both_first else None})
    write_jsonl(OUT / f"paired-{label}.jsonl", paired_rows)
    return {
        "left_branch": label.split("-vs-")[0], "right_branch": label.split("-vs-")[-1],
        "matching": "exact tool name and arguments; duplicate calls matched by multiplicity",
        "btc_formula": "gold ToolEnv replay hash gate and M>0; no local runtime mask",
        "paired_formula_delta_points_right_minus_left": {**paired_bootstrap(formula_delta, rng),
            "mean": 100 * statistics.fmean(formula_delta),
            "ci95_low": 100 * paired_bootstrap(formula_delta, random.Random(BOOTSTRAP_SEED + 11))["ci95_low"],
            "ci95_high": 100 * paired_bootstrap(formula_delta, random.Random(BOOTSTRAP_SEED + 12))["ci95_high"]},
        "paired_local_30s_gated_score_delta_points_diagnostic": {
            "mean": 100 * statistics.fmean(gated_delta),
            "ci95_low": 100 * gated_bootstrap["ci95_low"],
            "ci95_high": 100 * gated_bootstrap["ci95_high"],
            "iterations": gated_bootstrap["iterations"]},
        "paired_latency_delta_seconds_right_minus_left": paired_bootstrap(latency_delta, random.Random(BOOTSTRAP_SEED + 14)),
        "paired_outcomes_formula_score": dict(outcomes),
        "first_request_payload_hash_matches": first_payload_equal,
        "first_request_payload_hash_compared": len(both_first),
        "same_first_request_payload_but_different_prediction_count": sum(
            left_first[sid].get("payload_sha256") == right_first[sid].get("payload_sha256") and
            (left[sid].get("tool_calls") != right[sid].get("tool_calls") or left[sid].get("state_hash") != right[sid].get("state_hash"))
            for sid in both_first),
        "hash_matches_gained": sum(not left_scores[sid]["hash_match"] and right_scores[sid]["hash_match"] for sid in ids),
        "hash_matches_lost": sum(left_scores[sid]["hash_match"] and not right_scores[sid]["hash_match"] for sid in ids),
        "samples_with_more_unmatched_submitted_calls": sum(
            right_scores[sid]["unmatched_submitted_calls"] > left_scores[sid]["unmatched_submitted_calls"] for sid in ids),
        "gold_replay_failures_left_or_right": [
            sum(left_scores[sid]["gold_replay_error"] is not None for sid in ids),
            sum(right_scores[sid]["gold_replay_error"] is not None for sid in ids)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2", action="store_true")
    parser.add_argument("--step3", action="store_true")
    args = parser.parse_args()
    if args.step2 == args.step3:
        parser.error("choose exactly one of --step2 or --step3")
    cohort = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))
    ids = cohort["ids"]
    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train" / "train.jsonl")}
    rng = random.Random(BOOTSTRAP_SEED)
    if args.step2:
        branches = {name: compute_branch(name, ids, items, data_root) for name in ("A", "B")}
        a_pred, a_score, a_metrics, a_req = branches["A"]
        b_pred, b_score, b_metrics, b_req = branches["B"]
        paired = pair_metrics("A-vs-B", ids, a_pred, b_pred, a_score, b_score, a_req, b_req, rng)
        write_json(OUT / "step2-analysis.json", {"cohort": "fixed enriched subset of held-out val309; screening only",
            "branches": {"A": a_metrics, "B": b_metrics}, "paired_A_vs_B": paired,
            "interpretation": "A and B use the same selected current QUAN checkpoint after early stopping. This paired repeat checks the current 768 branch control; it is not evidence that a different checkpoint improves BTC.",
            "local_runtime_note": "The 30s-masked means use local host latency as a diagnostic only; the main BTC formula scores are unmasked."})
        return 1 if paired["gold_replay_failures_left_or_right"] != [0, 0] else 0
    a_meta = json.loads((OUT / "run-A.json").read_text(encoding="utf-8"))
    if a_meta.get("status") != "complete":
        raise RuntimeError("Step 3 requires completed A prompt-v2 control")
    a_pred = {row["id"]: row for row in rows(OUT / "predictions-A.jsonl")}
    a_traces = rows(OUT / "step-traces-A.jsonl")
    a_req = rows(OUT / "requests-A.jsonl")
    a_score_rows = rows(OUT / "btc-per-sample-A.jsonl")
    a_score = {row["id"]: row for row in a_score_rows}
    if list(a_pred) != ids or set(a_score) != set(ids):
        raise RuntimeError("Step-2 A artifacts are incomplete or misordered")
    cot_pred, cot_score, cot_metrics, cot_req = compute_branch("COT", ids, items, data_root)
    paired = pair_metrics("A-vs-COT", ids, a_pred, cot_pred, a_score, cot_score, a_req, cot_req, rng)
    write_json(OUT / "step3-analysis.json", {"cohort": "same fixed enriched subset of held-out val309; screening only",
        "branches": {"A_prompt_v2_reused_from_step2": aggregate_branch("A", ids, a_pred,
            a_traces, a_req, a_score), "COT": cot_metrics},
        "paired_A_prompt_v2_vs_COT": paired,
        "cot_prompt_delta": "A structured checklist for outcomes, evidence grounding, schema/argument checks, missing required information, and updating completion after tool results. The existing prompt already says to identify unfinished work, read observations and avoid guessing; CoT adds this ordered checklist and an explicit completed-outcome update loop.",
        "raw_chain_of_thought_saved": False,
        "interpretation": "This 60-ID enriched cohort is only a screen. CoT is considered useful only if paired unmasked BTC improves without material runtime/truncation regression.",
        "local_runtime_note": "The 30s-masked means use local host latency as a diagnostic only; the main BTC formula scores are unmasked."})
    return 1 if paired["gold_replay_failures_left_or_right"] != [0, 0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
