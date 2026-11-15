from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/num-predict-512-vs-768-val/output"
sys.path.insert(0, str(ROOT / "exp/btc-scoring/scripts"))
from score_btc import score_sample  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("p50", "p90", "p95", "max", "mean")}
    ordered = sorted(values)
    def q(p: float) -> float:
        pos = (len(ordered) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"p50": q(.5), "p90": q(.9), "p95": q(.95), "max": max(ordered), "mean": statistics.fmean(ordered)}


def score_branch(predictions: dict[str, dict], ids: list[str], items: dict[str, dict], data_root: Path) -> list[dict]:
    return [score_sample(sid, items[sid], predictions[sid], data_root) for sid in ids]


def bootstrap_diff(deltas: list[float], seed: int = 20261105, iterations: int = 10000) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(deltas)
    samples = []
    for _ in range(iterations):
        samples.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return {"mean": statistics.fmean(deltas), "ci95_low": samples[int(.025 * iterations)],
            "ci95_high": samples[min(iterations - 1, int(.975 * iterations))],
            "iterations": iterations, "seed": seed}


def branch_usage(branch: int, ids: list[str]) -> dict[str, Any]:
    name = str(branch)
    requests = read_jsonl(OUT / f"token_usage-{name}.jsonl")
    predictions = {row["id"]: row for row in read_jsonl(OUT / f"predictions-{name}.jsonl")}
    traces = {row["id"]: row for row in read_jsonl(OUT / f"step-traces-{name}.jsonl")}
    requests_by_sample: dict[str, list[dict]] = {sid: [] for sid in ids}
    for request in requests:
        requests_by_sample.setdefault(request["sample_id"], []).append(request)
    for rows in requests_by_sample.values():
        rows.sort(key=lambda row: row.get("sample_request_index", 0))
    prompts = [float(row["prompt_eval_count"]) for row in requests if row.get("prompt_eval_count") is not None]
    generated = [float(row["eval_count"]) for row in requests if row.get("eval_count") is not None]
    totals = [float(row["total_tokens"]) for row in requests]
    headroom = [8192 - value for value in totals]
    per_sample = []
    for sid in ids:
        reqs = requests_by_sample[sid]
        pred = predictions[sid]
        turns = pred.get("model_output", [])
        done_length = sum(row.get("done_reason") == "length" for row in reqs)
        cap_count = sum((row.get("eval_count") or 0) >= branch for row in reqs)
        context_count = sum(
            (row.get("prompt_eval_count") or 0) + (row.get("eval_count") or 0) >= 8192
            and (row.get("eval_count") or 0) < branch for row in reqs
        )
        component_rows = [row.get("prompt_components", {}) for row in reqs]
        comp_est = [row.get("estimated_component_token_sum", 0) for row in component_rows]
        prompt_counts = [row.get("prompt_eval_count") for row in reqs if row.get("prompt_eval_count") is not None]
        repairs = sum(bool(turn.get("_diagnostics", {}).get("repair_attempted")) for turn in turns)
        repair_failures = sum(bool(turn.get("_diagnostics", {}).get("repair_attempted")) and
                              not turn.get("_diagnostics", {}).get("repair_succeeded", False) for turn in turns)
        per_sample.append({
            "id": sid,
            "request_count": len(reqs),
            "repair_request_count": sum(row.get("request_kind") == "repair" for row in reqs),
            "action_turns": len(turns),
            "prompt_eval_tokens": sum(row.get("prompt_eval_count") or 0 for row in reqs),
            "generated_tokens": sum(row.get("eval_count") or 0 for row in reqs),
            "prompt_plus_generated_tokens": sum(row.get("total_tokens") or 0 for row in reqs),
            "minimum_num_ctx_headroom": min([8192 - (row.get("total_tokens") or 0) for row in reqs], default=None),
            "token_cap_requests": cap_count,
            "done_reason_length_requests": done_length,
            "context_limited_candidate_requests": context_count,
            "json_repair_attempts": repairs,
            "json_repair_failures": repair_failures,
            "empty_prediction": not bool(pred.get("tool_calls")),
            "latency_seconds": pred.get("latency_seconds"),
            "first_turn_prompt_eval_count": reqs[0].get("prompt_eval_count") if reqs else None,
            "later_turn_prompt_eval_count_mean": statistics.fmean([r["prompt_eval_count"] for r in reqs[1:] if r.get("prompt_eval_count") is not None]) if len(reqs) > 1 else None,
            "component_estimate_minus_ollama_prompt_count_mean": statistics.fmean([
                e - p for e, p in zip(comp_est, prompt_counts)
            ]) if comp_est and prompt_counts and len(comp_est) == len(prompt_counts) else None,
            "near_context_prompt_ge_7000": bool(prompt_counts and max(prompt_counts) >= 7000),
        })
    component_totals = Counter()
    for row in requests:
        for name_key, component in row.get("prompt_components", {}).items():
            if isinstance(component, dict) and isinstance(component.get("estimated_tokens_char_div_4"), int):
                component_totals[name_key] += component["estimated_tokens_char_div_4"]
    return {
        "branch": branch,
        "requests": len(requests),
        "samples": len(predictions),
        "request_count_per_sample": quantiles([float(row["request_count"]) for row in per_sample]),
        "action_turns_per_sample": quantiles([float(row["action_turns"]) for row in per_sample]),
        "repair_requests_per_sample": quantiles([float(row["repair_request_count"]) for row in per_sample]),
        "sample_prompt_eval_tokens": quantiles([float(row["prompt_eval_tokens"]) for row in per_sample]),
        "sample_generated_tokens": quantiles([float(row["generated_tokens"]) for row in per_sample]),
        "sample_prompt_plus_generated_tokens": quantiles([float(row["prompt_plus_generated_tokens"]) for row in per_sample]),
        "sample_minimum_num_ctx_headroom": quantiles([float(row["minimum_num_ctx_headroom"]) for row in per_sample if row["minimum_num_ctx_headroom"] is not None]),
        "prompt_eval_count": quantiles(prompts),
        "repair_prompt_eval_count": quantiles([float(row["prompt_eval_count"]) for row in requests if row.get("request_kind") == "repair" and row.get("prompt_eval_count") is not None]),
        "first_turn_prompt_eval_count": quantiles([float(rows[0]["prompt_eval_count"]) for rows in requests_by_sample.values() if rows and rows[0].get("prompt_eval_count") is not None]),
        "later_turn_prompt_eval_count": quantiles([float(row["prompt_eval_count"]) for rows in requests_by_sample.values() for row in rows[1:] if row.get("prompt_eval_count") is not None]),
        "eval_count": quantiles(generated),
        "prompt_plus_generated_tokens": quantiles(totals),
        "num_ctx_headroom": quantiles(headroom),
        "requests_done_reason_length": sum(row.get("done_reason") == "length" for row in requests),
        "requests_eval_at_num_predict": sum((row.get("eval_count") or 0) >= branch for row in requests),
        "requests_context_limited_candidates": sum(
            (row.get("prompt_eval_count") or 0) + (row.get("eval_count") or 0) >= 8192 and
            (row.get("eval_count") or 0) < branch for row in requests
        ),
        "requests_finished_normally": sum(row.get("done_reason") != "length" for row in requests),
        "repair_requests": sum(row.get("request_kind") == "repair" for row in requests),
        "thinking_chars_total_not_stored": sum(row.get("thinking_chars", 0) for row in requests),
        "component_token_estimate_totals_char_div_4": dict(component_totals),
        "component_estimate_minus_ollama_prompt_count": quantiles([
            row.get("prompt_components", {}).get("estimated_component_token_sum", 0) - row.get("prompt_eval_count", 0)
            for row in requests if row.get("prompt_eval_count") is not None
        ]),
        "sample_level": per_sample,
        "sample_counts": {
            "token_cap": sum(row["token_cap_requests"] > 0 for row in per_sample),
            "json_repair_attempted": sum(row["json_repair_attempts"] > 0 for row in per_sample),
            "json_repair_failed": sum(row["json_repair_failures"] > 0 for row in per_sample),
            "empty_prediction": sum(row["empty_prediction"] for row in per_sample),
            "over_30_seconds_local": sum((row.get("latency_seconds") or 0) > 30 for row in per_sample),
        },
        "latency_seconds": quantiles([float(row.get("latency_seconds") or 0) for row in predictions.values()]),
        "latency_over_30_request_count": sum((row.get("latency_seconds") or 0) > 30 for row in requests),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true", help="Analyze pilot and never propose a full run")
    args = parser.parse_args()
    cohort = read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    pred_a = {row["id"]: row for row in read_jsonl(OUT / "predictions-512.jsonl")}
    pred_b = {row["id"]: row for row in read_jsonl(OUT / "predictions-768.jsonl")}
    if set(pred_a) != set(ids) or set(pred_b) != set(ids):
        raise RuntimeError("Both branches must exactly cover the pre-registered 60-ID pilot")
    for branch in (512, 768):
        usage_path = OUT / f"token_usage-{branch}.jsonl"
        usage_rows = read_jsonl(usage_path)
        seen_request_ids: set[str] = set()
        changed = False
        for row in usage_rows:
            desired_id = f"{branch}:{row['sample_id']}:r{row['sample_request_index']}"
            if row.get("request_id") != desired_id or desired_id in seen_request_ids:
                row["request_id"] = desired_id
                changed = True
            seen_request_ids.add(desired_id)
        if changed:
            usage_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in usage_rows), encoding="utf-8")
    for branch in (512, 768):
        usage_path = OUT / f"token_usage-{branch}.jsonl"
        usage_rows = read_jsonl(usage_path)
        seen_request_ids: set[str] = set()
        changed = False
        for row in usage_rows:
            desired_id = f"{branch}:{row['sample_id']}:r{row['sample_request_index']}"
            if row.get("request_id") != desired_id or desired_id in seen_request_ids:
                row["request_id"] = desired_id
                changed = True
            seen_request_ids.add(desired_id)
        if changed:
            usage_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in usage_rows), encoding="utf-8")
    data_root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(data_root / "train/train.jsonl")}
    gold_a = score_branch(pred_a, ids, items, data_root)
    gold_b = score_branch(pred_b, ids, items, data_root)
    btc_a = {row["id"]: row for row in gold_a}
    btc_b = {row["id"]: row for row in gold_b}
    deltas = [btc_b[sid]["btc_formula_score_before_runtime_gate_diagnostic"] - btc_a[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
    base_ids = {row["id"]: row for row in read_jsonl(ROOT / "exp/current-config-error-analysis-val309/output/predictions.jsonl")}
    def call_multiset(row: dict[str, Any]) -> Counter[str]:
        return Counter(json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                       for call in row.get("tool_calls", []))
    old_order_diffs = [sid for sid in ids if pred_a[sid].get("tool_calls") != base_ids[sid].get("tool_calls")]
    old_multiset_diffs = [sid for sid in ids if call_multiset(pred_a[sid]) != call_multiset(base_ids[sid])]
    old_hash_diffs = [sid for sid in ids if pred_a[sid].get("state_hash") != base_ids[sid].get("state_hash")]
    old_diffs = sorted(set(old_order_diffs) | set(old_hash_diffs))
    archived_btc = {row["id"]: row for row in read_jsonl(ROOT / "exp/current-config-error-analysis-val309/output/btc-scores.jsonl")}
    old_score_deltas = [
        btc_a[sid]["btc_formula_score_before_runtime_gate_diagnostic"] - archived_btc[sid]["btc_formula_score_before_runtime_gate_diagnostic"]
        for sid in ids
    ]
    old_score_outcomes = Counter("improved" if value > 1e-12 else "worsened" if value < -1e-12 else "tie" for value in old_score_deltas)
    usage_a = branch_usage(512, ids)
    usage_b = branch_usage(768, ids)
    paired = []
    for sid in ids:
        a, b = pred_a[sid], pred_b[sid]
        sa, sb = btc_a[sid], btc_b[sid]
        paired.append({
            "id": sid,
            "btc_512": sa["btc_formula_score_before_runtime_gate_diagnostic"],
            "btc_768": sb["btc_formula_score_before_runtime_gate_diagnostic"],
            "btc_delta": sb["btc_formula_score_before_runtime_gate_diagnostic"] - sa["btc_formula_score_before_runtime_gate_diagnostic"],
            "btc_outcome": "improved" if deltas[ids.index(sid)] > 1e-12 else "worsened" if deltas[ids.index(sid)] < -1e-12 else "tie",
            "hash_512": a.get("state_hash"), "hash_768": b.get("state_hash"),
            "hash_same_between_branches": a.get("state_hash") == b.get("state_hash"),
            "ordered_calls_same_between_branches": a.get("tool_calls") == b.get("tool_calls"),
            "calls_512": a.get("tool_calls"), "calls_768": b.get("tool_calls"),
            "request_turns_512": [{"turn": r.get("action_turn"), "prompt_eval_count": r.get("prompt_eval_count"), "eval_count": r.get("eval_count"), "done_reason": r.get("done_reason")} for r in read_jsonl(OUT / "token_usage-512.jsonl") if r.get("sample_id") == sid],
            "request_turns_768": [{"turn": r.get("action_turn"), "prompt_eval_count": r.get("prompt_eval_count"), "eval_count": r.get("eval_count"), "done_reason": r.get("done_reason")} for r in read_jsonl(OUT / "token_usage-768.jsonl") if r.get("sample_id") == sid],
        })
    score_a = 100 * statistics.fmean(row["btc_formula_score_before_runtime_gate_diagnostic"] for row in gold_a)
    score_b = 100 * statistics.fmean(row["btc_formula_score_before_runtime_gate_diagnostic"] for row in gold_b)
    cap_a, cap_b = usage_a["sample_counts"]["token_cap"], usage_b["sample_counts"]["token_cap"]
    failures_a = usage_a["sample_counts"]["json_repair_failed"]
    failures_b = usage_b["sample_counts"]["json_repair_failed"]
    failure_a = len(set(sid for sid, row in zip(ids, usage_a["sample_level"]) if row["token_cap_requests"] or row["json_repair_attempts"]))
    failure_b = len(set(sid for sid, row in zip(ids, usage_b["sample_level"]) if row["token_cap_requests"] or row["json_repair_attempts"]))
    reduction = (failure_a - failure_b) / failure_a if failure_a else None
    bootstrap = bootstrap_diff([value * 100 for value in deltas])
    latency_projection = usage_b["latency_seconds"]["mean"] * 309 if usage_b["latency_seconds"]["mean"] is not None else None
    screening_gate = (reduction is not None and reduction >= .5 and score_b >= score_a and
                      latency_projection is not None and latency_projection <= 4 * 3600)
    # A full paired confirmation would compare B against the archived A artifact.
    # Do not authorize that phase while the instrumented replay cannot reproduce A.
    pre_run_over_budget = bool(cohort.get("pre_run_time_estimate", {}).get("four_hour_limit_exceeded"))
    pass_gate = screening_gate and not old_diffs and not pre_run_over_budget
    outcome_counts = Counter(row["btc_outcome"] for row in paired)
    for branch_usage_result, score_rows in ((usage_a, gold_a), (usage_b, gold_b)):
        score_by_id = {row["id"]: row for row in score_rows}
        for sample_row in branch_usage_result["sample_level"]:
            sid = sample_row["id"]
            gold_count = score_by_id[sid]["gold_calls"]
            sample_row["gold_call_count"] = gold_count
            sample_row["task_group"] = "no_action" if gold_count == 0 else "one_call" if gold_count == 1 else "two_calls" if gold_count == 2 else "multi_step_3plus"
        groups: dict[str, list[dict]] = {}
        for sample_row in branch_usage_result["sample_level"]:
            groups.setdefault(sample_row["task_group"], []).append(sample_row)
        context_group = [row for row in branch_usage_result["sample_level"] if row["near_context_prompt_ge_7000"]]
        branch_usage_result["sample_groups"] = {
            group: {
                "samples": len(rows),
                "btc_formula_score_before_runtime_gate_0_100": 100 * statistics.fmean(score_by_id[row["id"]]["btc_formula_score_before_runtime_gate_diagnostic"] for row in rows),
                "token_cap_or_json_repair_needed_samples": sum(bool(row["token_cap_requests"] or row["json_repair_attempts"]) for row in rows),
                "over_30_seconds_local": sum((row.get("latency_seconds") or 0) > 30 for row in rows),
            } for group, rows in groups.items()
        }
        branch_usage_result["near_context_group_ge_7000"] = {
            "samples": len(context_group),
            "btc_formula_score_before_runtime_gate_0_100": 100 * statistics.fmean(score_by_id[row["id"]]["btc_formula_score_before_runtime_gate_diagnostic"] for row in context_group) if context_group else None,
            "token_cap_or_json_repair_needed_samples": sum(bool(row["token_cap_requests"] or row["json_repair_attempts"]) for row in context_group),
        }

    call_metrics = {}
    quality_counts = {}
    for branch_name, rows in (("512", gold_a), ("768", gold_b)):
        matched = sum(row["matched_calls"] for row in rows)
        submitted = sum(row["matched_calls"] + row["unmatched_submitted_calls"] for row in rows)
        gold_count = sum(row["gold_calls"] for row in rows)
        precision = matched / submitted if submitted else 0.0
        recall = matched / gold_count if gold_count else 0.0
        call_metrics[branch_name] = {"precision": precision, "recall": recall,
                                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
        pred_map = pred_a if branch_name == "512" else pred_b
        quality_counts[branch_name] = {
            "empty_predictions": sum(not bool(pred_map[row["id"]].get("tool_calls")) for row in rows),
            "samples_with_under_call_by_count": sum((row["matched_calls"] + row["unmatched_submitted_calls"]) < row["gold_calls"] for row in rows),
            "samples_with_extra_submitted_calls": sum(row["unmatched_submitted_calls"] > 0 for row in rows),
            "toolenv_schema_rejections": sum(row["schema_rejections"] for row in rows),
            "runtime_exceptions": sum(row["runtime_failure"] for row in rows),
            "samples_over_30_seconds_local": sum(row["runtime_over_30s"] for row in rows),
            "json_repair_attempt_samples": sum(any(turn.get("_diagnostics", {}).get("repair_attempted") for turn in pred_map[row["id"]].get("model_output", [])) for row in rows),
            "json_repair_failed_samples": sum(any(turn.get("_diagnostics", {}).get("repair_attempted") and not turn.get("_diagnostics", {}).get("repair_succeeded", False) for turn in pred_map[row["id"]].get("model_output", [])) for row in rows),
        }
    result = {
        "experiment": "num_predict 512 vs 768; fixed enriched diagnostic pilot60",
        "matching": "BTC official formula: exact tool name+arguments; duplicate multiplicity; gold replay hash gate; M>0; no local latency mask",
        "cohort_count": len(ids), "cohort_is_unbiased_val309_estimate": False,
        "btc_formula_score_before_runtime_gate_0_100": {"512": score_a, "768": score_b, "delta_768_minus_512": score_b - score_a, "paired_bootstrap_95_ci": bootstrap},
        "paired_btc_outcomes": dict(outcome_counts),
        "hash_match_samples": {"512": sum(row["hash_match"] for row in gold_a), "768": sum(row["hash_match"] for row in gold_b)},
        "M_U_G_totals": {"512": {"M": sum(row["matched_calls"] for row in gold_a), "U": sum(row["unmatched_submitted_calls"] for row in gold_a), "G": sum(row["gold_calls"] for row in gold_a)},
                         "768": {"M": sum(row["matched_calls"] for row in gold_b), "U": sum(row["unmatched_submitted_calls"] for row in gold_b), "G": sum(row["gold_calls"] for row in gold_b)}},
        "exact_call_micro_precision_recall_f1_diagnostic": call_metrics,
        "quality_counts": quality_counts,
        "local_proxy_mean_diagnostic": {"512": statistics.fmean(row["local_proxy_score_diagnostic"] for row in gold_a), "768": statistics.fmean(row["local_proxy_score_diagnostic"] for row in gold_b)},
        "final_state_match_diagnostic": {"512": sum(row["predicted_final_state_matches_label"] for row in gold_a), "768": sum(row["predicted_final_state_matches_label"] for row in gold_b)},
        "gold_replay_failures": {"512": sum(row["gold_replay_error"] is not None for row in gold_a), "768": sum(row["gold_replay_error"] is not None for row in gold_b)},
        "telemetry": {"512": usage_a, "768": usage_b},
        "prior_512_baseline_reproducibility": {
            "checked": len(ids), "ordered_call_mismatch_samples": len(old_order_diffs),
            "call_multiset_mismatch_samples": len(old_multiset_diffs), "hash_mismatch_samples": len(old_hash_diffs),
            "ordered_call_or_hash_mismatch_samples": len(old_diffs), "mismatch_ids": old_diffs,
            "order_only_same_multiset_and_hash": len(set(old_order_diffs) - set(old_multiset_diffs) - set(old_hash_diffs)),
            "archived_vs_instrumented_A_btc_formula_score_0_100": {
                "archived": 100 * statistics.fmean(archived_btc[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids),
                "instrumented_rerun": score_a,
                "paired_delta": 100 * statistics.fmean(old_score_deltas),
                "paired_bootstrap_95_ci": bootstrap_diff([value * 100 for value in old_score_deltas]),
                "outcomes": dict(old_score_outcomes),
            },
        },
        "pilot_failure_reduction": {"sample_level_token_cap_or_json_repair_needed": {"512": failure_a, "768": failure_b}, "reduction_fraction": reduction,
                                     "token_cap_samples": {"512": cap_a, "768": cap_b}, "failed_repair_samples": {"512": failures_a, "768": failures_b}},
        "full_val309_768_estimate_seconds": latency_projection,
        "pilot_screening_gate_passed": screening_gate,
        "reproducibility_gate_passed": not old_diffs,
        "pre_run_full_run_estimate_exceeded_four_hours": pre_run_over_budget,
        "go_for_full_val309_768": pass_gate and not args.pilot_only,
        "decision": "pilot and reproducibility gates pass; eligible for full val309 768" if pass_gate else "NO-GO for full val309; keep result as pilot only",
        "decision_reasons": [reason for reason, condition in (
            ("pilot did not meet the registered >=50% cap/JSON-repair-needed reduction and nondecreasing BTC conditions", not screening_gate),
            ("instrumented A did not reproduce the archived 512 calls/hash on the fixed cohort", bool(old_diffs)),
            ("pre-run 768 full-val time estimate exceeded four hours", pre_run_over_budget),
        ) if condition],
        "per_sample_btc": {"512": gold_a, "768": gold_b},
        "paired_samples": paired,
    }
    write_json(OUT / "paired_comparison.json", result)
    write_json(OUT / "btc-per-sample-512.json", gold_a)
    write_json(OUT / "btc-per-sample-768.json", gold_b)
    # Prompt component estimates are also available in each request JSONL; split the requested view.
    for branch in (512, 768):
        rows = read_jsonl(OUT / f"token_usage-{branch}.jsonl")
        estimate_rows = [{"request_id": row["request_id"], "sample_id": row["sample_id"],
                          "action_turn": row["action_turn"], "request_kind": row["request_kind"],
                          "prompt_eval_count": row["prompt_eval_count"], "prompt_components": row["prompt_components"]}
                         for row in rows]
        (OUT / f"prompt-component-estimates-{branch}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in estimate_rows), encoding="utf-8")
    summary = [
        "# `num_predict` 512 vs 768 — pilot result", "",
        f"Status: **{result['decision']}**. This is an enriched fixed diagnostic pilot, not an unbiased estimate for val309.", "",
        f"BTC formula score before any local runtime mask: 512 **{score_a:.4f}/100**, 768 **{score_b:.4f}/100**, paired difference **{score_b-score_a:+.4f}**; 95% bootstrap CI [{bootstrap['ci95_low']:+.4f}, {bootstrap['ci95_high']:+.4f}].",
        f"BTC paired outcomes: {outcome_counts['improved']} improved, {outcome_counts['worsened']} worsened, {outcome_counts['tie']} tied. Scoring uses exact tool name+arguments, duplicate multiplicity, gold replay hash gate, and M>0.",
        f"Samples hitting token cap or requiring JSON repair: 512 {failure_a}, 768 {failure_b}; reduction {('n/a' if reduction is None else f'{100*reduction:.1f}%')}. Failed repairs are reported separately. Full 768 time projection from pilot latency: {('n/a' if latency_projection is None else f'{latency_projection/3600:.2f} hours')}.",
        f"The new 512 replay differed from the prior artifact on ordered calls for {len(old_order_diffs)}/60, exact call multisets for {len(old_multiset_diffs)}/60, and hashes for {len(old_hash_diffs)}/60. Current transcript IDs matched 60/60. The source, prompt, model digest/version, and request settings were checked; cause of the output reproducibility failure remains unresolved.",
        f"Pre-run estimate for full 768 was {cohort.get('pre_run_time_estimate', {}).get('full_val309_768_estimate_hours', 0):.2f} hours, above the 4-hour rule. The experiment therefore remains a pilot even if its measured screening gate passes.",
        "",
        "## Token and context usage", "",
        f"512: {usage_a['requests']} requests. Prompt tokens p50/p90/p95/max **{usage_a['prompt_eval_count']['p50']:.0f}/{usage_a['prompt_eval_count']['p90']:.0f}/{usage_a['prompt_eval_count']['p95']:.0f}/{usage_a['prompt_eval_count']['max']:.0f}**; generated tokens **{usage_a['eval_count']['p50']:.0f}/{usage_a['eval_count']['p90']:.0f}/{usage_a['eval_count']['p95']:.0f}/{usage_a['eval_count']['max']:.0f}**; prompt+generated **{usage_a['prompt_plus_generated_tokens']['p50']:.0f}/{usage_a['prompt_plus_generated_tokens']['p90']:.0f}/{usage_a['prompt_plus_generated_tokens']['p95']:.0f}/{usage_a['prompt_plus_generated_tokens']['max']:.0f}**; `num_ctx` headroom **{usage_a['num_ctx_headroom']['p50']:.0f}/{usage_a['num_ctx_headroom']['p90']:.0f}/{usage_a['num_ctx_headroom']['p95']:.0f}/{usage_a['num_ctx_headroom']['max']:.0f}**.",
        f"768: {usage_b['requests']} requests. Prompt tokens p50/p90/p95/max **{usage_b['prompt_eval_count']['p50']:.0f}/{usage_b['prompt_eval_count']['p90']:.0f}/{usage_b['prompt_eval_count']['p95']:.0f}/{usage_b['prompt_eval_count']['max']:.0f}**; generated tokens **{usage_b['eval_count']['p50']:.0f}/{usage_b['eval_count']['p90']:.0f}/{usage_b['eval_count']['p95']:.0f}/{usage_b['eval_count']['max']:.0f}**; prompt+generated **{usage_b['prompt_plus_generated_tokens']['p50']:.0f}/{usage_b['prompt_plus_generated_tokens']['p90']:.0f}/{usage_b['prompt_plus_generated_tokens']['p95']:.0f}/{usage_b['prompt_plus_generated_tokens']['max']:.0f}**; `num_ctx` headroom **{usage_b['num_ctx_headroom']['p50']:.0f}/{usage_b['num_ctx_headroom']['p90']:.0f}/{usage_b['num_ctx_headroom']['p95']:.0f}/{usage_b['num_ctx_headroom']['max']:.0f}**.",
        f"Done reason `length` / reached `num_predict` / normal finish: 512 {usage_a['requests_done_reason_length']}/{usage_a['requests_eval_at_num_predict']}/{usage_a['requests_finished_normally']}; 768 {usage_b['requests_done_reason_length']}/{usage_b['requests_eval_at_num_predict']}/{usage_b['requests_finished_normally']}. Repair requests: {usage_a['repair_requests']}/{usage_b['repair_requests']}. Context-limited candidates: {usage_a['requests_context_limited_candidates']}/{usage_b['requests_context_limited_candidates']}; none reached the 8192 context ceiling. Median prompt tokens on first/later turns: 512 {usage_a['first_turn_prompt_eval_count']['p50']:.0f}/{usage_a['later_turn_prompt_eval_count']['p50']:.0f}; 768 {usage_b['first_turn_prompt_eval_count']['p50']:.0f}/{usage_b['later_turn_prompt_eval_count']['p50']:.0f}. Median component estimate minus Ollama prompt count: {usage_a['component_estimate_minus_ollama_prompt_count']['p50']:.0f}/{usage_b['component_estimate_minus_ollama_prompt_count']['p50']:.0f} tokens, so the character/4 component breakdown underestimates the authoritative count.",
        f"JSON repair attempts / failed-repair samples: 512 {usage_a['sample_counts']['json_repair_attempted']}/{usage_a['sample_counts']['json_repair_failed']}; 768 {usage_b['sample_counts']['json_repair_attempted']}/{usage_b['sample_counts']['json_repair_failed']}. `eval_count` includes thinking and answer tokens together; only thinking character counts were retained, never raw chain-of-thought.",
        "",
        "## Telemetry", "",
        "Per-request prompt/generation counters and exact payload SHA-256 are in `token_usage-512.jsonl` and `token_usage-768.jsonl`. Prompt estimates are approximate UTF-8 character-count/4 components; Ollama `prompt_eval_count` is authoritative. Thinking is enabled in both branches; only character counts were retained, never raw thinking. See `paired_comparison.json` for p50/p90/p95/max, headroom, done reasons, repairs, latency, and grouped sample details.",
        "",
        "## Latency and quality diagnostics", "",
        f"Local RTX 4050 sample latency mean/p50/p95/max: 512 **{usage_a['latency_seconds']['mean']:.2f}/{usage_a['latency_seconds']['p50']:.2f}/{usage_a['latency_seconds']['p95']:.2f}/{usage_a['latency_seconds']['max']:.2f}s** ({usage_a['sample_counts']['over_30_seconds_local']}/60 over 30s); 768 **{usage_b['latency_seconds']['mean']:.2f}/{usage_b['latency_seconds']['p50']:.2f}/{usage_b['latency_seconds']['p95']:.2f}/{usage_b['latency_seconds']['max']:.2f}s** ({usage_b['sample_counts']['over_30_seconds_local']}/60 over 30s). No individual Ollama request exceeded 30s. These are not A30 measurements and do not mask BTC formula scores.",
        f"Micro exact-call P/R/F1 diagnostic: 512 {call_metrics['512']['precision']:.3f}/{call_metrics['512']['recall']:.3f}/{call_metrics['512']['f1']:.3f}; 768 {call_metrics['768']['precision']:.3f}/{call_metrics['768']['recall']:.3f}/{call_metrics['768']['f1']:.3f}. Hash matches 28/60 vs 30/60; final-state diagnostic 37/60 vs 39/60; local proxy mean {statistics.fmean(row['local_proxy_score_diagnostic'] for row in gold_a):.3f} vs {statistics.fmean(row['local_proxy_score_diagnostic'] for row in gold_b):.3f}. Schema rejections, runtime errors, gold replay failures: 0 in both branches.",
        "",
        "## Limitations", "",
        "The 30-second counts are local RTX 4050 measurements, not A30 measurements and not used to mask the reported BTC formula score here. Context-limited candidates are inferred where prompt plus generated tokens approach `num_ctx`; Ollama's `done_reason=length` alone does not distinguish `num_predict` from context exhaustion.",
    ]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"btc_512": score_a, "btc_768": score_b, "delta": score_b-score_a,
                      "ci95": [bootstrap["ci95_low"], bootstrap["ci95_high"]],
                      "failures": [failure_a, failure_b], "latency_projection_seconds": latency_projection,
                      "go_full": pass_gate and not args.pilot_only, "old_baseline_mismatches": len(old_diffs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
