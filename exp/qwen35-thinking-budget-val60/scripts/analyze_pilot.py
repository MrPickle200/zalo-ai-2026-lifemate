from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/qwen35-thinking-budget-val60/output"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp/btc-scoring/scripts"))
from lifemate.utils import find_data_root, load_jsonl  # noqa: E402
from score_btc import call_key, score_sample  # noqa: E402

CAPS = (512, 768, 1024, 1536)
SEED = 20261106
BOOTSTRAPS = 20000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    at = (len(v) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return v[lo] + (v[hi] - v[lo]) * (at - lo)


def stats(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.fmean(values) if values else None,
            "p50": quantile(values, .5), "p90": quantile(values, .9),
            "p95": quantile(values, .95), "max": max(values) if values else None}


def bootstrap(deltas: list[float], rng: random.Random) -> dict[str, float]:
    n = len(deltas)
    means = sorted(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(BOOTSTRAPS))
    return {"mean": statistics.fmean(deltas), "ci95_low": means[int(.025 * BOOTSTRAPS)],
            "ci95_high": means[int(.975 * BOOTSTRAPS)], "iterations": BOOTSTRAPS}


def call_metrics(predictions: dict[str, dict], gold_items: dict[str, dict], ids: list[str]) -> dict[str, Any]:
    matched = submitted = gold = 0
    for sid in ids:
        pred = predictions[sid].get("tool_calls") or []
        ref = gold_items[sid].get("label", {}).get("tool_calls", [])
        a, b = Counter(call_key(c) for c in pred), Counter(call_key(c) for c in ref)
        matched += sum((a & b).values())
        submitted += len(pred)
        gold += len(ref)
    precision = matched / submitted if submitted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1_diagnostic": f1,
            "matched_calls": matched, "submitted_calls": submitted, "gold_calls": gold}


def main() -> int:
    cohort = read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    data_root = find_data_root("train+public_test/train+public_test")
    items = {r["id"]: r for r in load_jsonl(data_root / "train/train.jsonl")}
    preds_by_cap: dict[int, dict[str, dict]] = {}
    scores_by_cap: dict[int, dict[str, dict]] = {}
    metrics: dict[str, Any] = {}
    for cap in CAPS:
        preds = rows(OUT / f"predictions-{cap}.jsonl")
        if len(preds) != 60 or len({r["id"] for r in preds}) != 60 or {r["id"] for r in preds} != set(ids):
            raise RuntimeError(f"cap {cap}: prediction IDs are not exactly the fixed 60-ID cohort")
        pred_map = {r["id"]: r for r in preds}
        requests = rows(OUT / f"requests-{cap}.jsonl")
        traces = rows(OUT / f"step-traces-{cap}.jsonl")
        if {r["id"] for r in traces} != set(ids) or any(r.get("num_predict") != cap or r.get("num_ctx") != 8192 or r.get("num_batch") != 512 for r in requests):
            raise RuntimeError(f"cap {cap}: telemetry/sample alignment or effective request options failed")
        sample_scores = [score_sample(sid, items[sid], pred_map[sid], data_root) for sid in ids]
        if any(x["gold_replay_error"] for x in sample_scores):
            raise RuntimeError(f"cap {cap}: a gold ToolEnv replay failed")
        score_map = {r["id"]: r for r in sample_scores}
        preds_by_cap[cap], scores_by_cap[cap] = pred_map, score_map
        write_jsonl(OUT / f"btc-per-sample-{cap}.jsonl", sample_scores)

        by_sample_req = {sid: [] for sid in ids}
        for request in requests:
            by_sample_req[request["sample_id"]].append(request)
        sample_count = {sid: by_sample_req[sid] for sid in ids}
        prompt = [float(x["prompt_eval_count"]) for x in requests if x.get("prompt_eval_count") is not None]
        generated = [float(x["eval_count"]) for x in requests if x.get("eval_count") is not None]
        prompt_generation = [float((x.get("prompt_eval_count") or 0) + (x.get("eval_count") or 0)) for x in requests]
        lengths = [x for x in requests if x.get("done_reason") == "length"]
        cap_hits = [x for x in requests if (x.get("eval_count") or 0) >= cap]
        context_candidates = [x for x in lengths if (x.get("eval_count") or 0) < cap and
                              (x.get("prompt_eval_count") or 0) + (x.get("eval_count") or 0) >= 8192]
        unresolved_lengths = [x for x in lengths if x not in cap_hits and x not in context_candidates]
        lat_s = [float(pred_map[sid].get("latency_seconds") or 0) for sid in ids]
        req_lat = [float(x.get("latency_seconds") or 0) for x in requests]
        repairs = sum(x.get("request_kind") == "repair" for x in requests)
        repair_samples = sum(any(x.get("request_kind") == "repair" for x in sample_count[sid]) for sid in ids)
        repaired_turns = sum(bool(t.get("_diagnostics", {}).get("repair_attempted"))
                             for sid in ids for t in pred_map[sid].get("model_output", []))
        failed_repair_turns = sum(bool(t.get("_diagnostics", {}).get("repair_attempted")) and
                                  not t.get("_diagnostics", {}).get("repair_succeeded", False)
                                  for sid in ids for t in pred_map[sid].get("model_output", []))
        token_cap_samples = sum(any((x.get("eval_count") or 0) >= cap for x in sample_count[sid]) for sid in ids)
        samples_with_cap_reached = sum(any((x.get("eval_count") or 0) >= cap for x in sample_count[sid]) for sid in ids)
        scores = [score_map[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
        hashes = sum(score_map[sid]["hash_match"] for sid in ids)
        matched = sum(score_map[sid]["matched_calls"] for sid in ids)
        unmatched = sum(score_map[sid]["unmatched_submitted_calls"] for sid in ids)
        gold = sum(score_map[sid]["gold_calls"] for sid in ids)
        metrics[str(cap)] = {
            "samples": 60, "requests": len(requests), "btc_formula_score_0_100": 100 * statistics.fmean(scores),
            "hash_matches": hashes, "matched_calls_M": matched, "unmatched_calls_U": unmatched, "gold_calls_G": gold,
            "aggregate_exact_call_precision_diagnostic": matched / (matched + unmatched) if matched + unmatched else 0,
            "aggregate_exact_call_recall_diagnostic": matched / gold if gold else 0,
            "aggregate_exact_call_f1_diagnostic": call_metrics(pred_map, items, ids)["f1_diagnostic"],
            "final_state_matches": sum(score_map[sid]["predicted_final_state_matches_label"] for sid in ids),
            "local_proxy_mean_diagnostic": statistics.fmean(score_map[sid]["local_proxy_score_diagnostic"] for sid in ids),
            "schema_rejections": sum(score_map[sid]["schema_rejections"] for sid in ids),
            "runtime_errors": sum(bool(t.get("runtime_error")) for t in traces),
            "sample_latency_seconds": stats(lat_s), "request_latency_seconds": stats(req_lat),
            "samples_over_30s_local": sum(x > 30 for x in lat_s),
            "requests_over_30s_local": sum(x > 30 for x in req_lat),
            "prompt_tokens_request": stats(prompt), "generated_tokens_request_including_thinking": stats(generated),
            "prompt_plus_generated_tokens_request": stats(prompt_generation),
            "sample_generated_tokens_including_thinking": stats([
                float(sum(x.get("eval_count") or 0 for x in sample_count[sid])) for sid in ids]),
            "sample_prompt_plus_generated_tokens": stats([
                float(sum((x.get("prompt_eval_count") or 0) + (x.get("eval_count") or 0) for x in sample_count[sid])) for sid in ids]),
            "sample_minimum_context_headroom": stats([
                float(min((8192 - ((x.get("prompt_eval_count") or 0) + (x.get("eval_count") or 0))
                           for x in sample_count[sid]), default=8192)) for sid in ids]),
            "total_generated_tokens_including_thinking": sum(x.get("eval_count") or 0 for x in requests),
            "total_prompt_eval_tokens": sum(x.get("prompt_eval_count") or 0 for x in requests),
            "prompt_eval_duration_seconds": sum(x.get("prompt_eval_duration_ns") or 0 for x in requests) / 1e9,
            "decode_duration_seconds": sum(x.get("eval_duration_ns") or 0 for x in requests) / 1e9,
            "load_duration_seconds": sum(x.get("load_duration_ns") or 0 for x in requests) / 1e9,
            "missing_prompt_eval_duration": sum(x.get("prompt_eval_duration_ns") is None for x in requests),
            "missing_decode_duration": sum(x.get("eval_duration_ns") is None for x in requests),
            "missing_load_duration": sum(x.get("load_duration_ns") is None for x in requests),
            "done_reason_length": len(lengths), "requests_reaching_num_predict": len(cap_hits),
            "context_limited_candidates_inferred": len(context_candidates),
            "length_reason_unresolved": len(unresolved_lengths),
            "requests_done_reason_other_than_length": len(requests) - len(lengths),
            "repair_requests": repairs, "samples_with_repair_request": repair_samples,
            "samples_with_num_predict_reached": samples_with_cap_reached,
            "requests_reaching_num_predict_rate": len(cap_hits) / len(requests) if requests else 0.0,
            "repair_turns_attempted": repaired_turns, "repair_turns_failed": failed_repair_turns,
            "action_turns": sum(len(pred_map[sid].get("model_output", [])) for sid in ids),
            "request_count_per_sample": stats([float(len(sample_count[sid])) for sid in ids]),
            "action_turns_per_sample": stats([float(len(pred_map[sid].get("model_output", []))) for sid in ids]),
            "empty_predictions": sum(not bool(pred_map[sid].get("tool_calls")) for sid in ids),
            "token_cap_samples": token_cap_samples,
            "early_stops": sum(any(t.get("action") == "stop" for t in pred_map[sid].get("model_output", [])) for sid in ids),
            "extra_calls_and_duplicates": unmatched,
            "under_calls": sum(max(0, score_map[sid]["gold_calls"] - score_map[sid]["matched_calls"]) for sid in ids),
        }

    rng = random.Random(SEED)
    paired: dict[str, Any] = {}
    base = 512
    for cap in CAPS[1:]:
        score_delta = [scores_by_cap[cap][sid]["btc_formula_score_before_runtime_gate_diagnostic"] -
                       scores_by_cap[base][sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
        latency_delta = [float(preds_by_cap[cap][sid]["latency_seconds"]) -
                         float(preds_by_cap[base][sid]["latency_seconds"]) for sid in ids]
        score_ci, latency_ci = bootstrap(score_delta, rng), bootstrap(latency_delta, rng)
        outcomes = Counter("improved" if d > 1e-12 else "worsened" if d < -1e-12 else "tied" for d in score_delta)
        details = []
        for sid, sd, ld in zip(ids, score_delta, latency_delta):
            details.append({"id": sid, "score_512": scores_by_cap[base][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
                f"score_{cap}": scores_by_cap[cap][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
                "score_delta": sd, "latency_512_seconds": preds_by_cap[base][sid]["latency_seconds"],
                f"latency_{cap}_seconds": preds_by_cap[cap][sid]["latency_seconds"], "latency_delta_seconds": ld,
                "hash_512": scores_by_cap[base][sid]["hash_match"], f"hash_{cap}": scores_by_cap[cap][sid]["hash_match"],
                "matched_calls_512": scores_by_cap[base][sid]["matched_calls"],
                f"matched_calls_{cap}": scores_by_cap[cap][sid]["matched_calls"],
                "unmatched_calls_512": scores_by_cap[base][sid]["unmatched_submitted_calls"],
                f"unmatched_calls_{cap}": scores_by_cap[cap][sid]["unmatched_submitted_calls"]})
        write_jsonl(OUT / f"paired-512-vs-{cap}.jsonl", details)
        paired[str(cap)] = {"score_delta_points_vs_512": {**score_ci, "mean": 100 * score_ci["mean"],
                "ci95_low": 100 * score_ci["ci95_low"], "ci95_high": 100 * score_ci["ci95_high"]},
            "sample_latency_delta_seconds_vs_512": latency_ci,
            "paired_score_outcomes_vs_512": dict(outcomes),
            "hash_matches_gained": sum(not scores_by_cap[base][sid]["hash_match"] and scores_by_cap[cap][sid]["hash_match"] for sid in ids),
            "hash_matches_lost": sum(scores_by_cap[base][sid]["hash_match"] and not scores_by_cap[cap][sid]["hash_match"] for sid in ids),
            "exact_calls_gained": sum(max(0, scores_by_cap[cap][sid]["matched_calls"] - scores_by_cap[base][sid]["matched_calls"]) for sid in ids),
            "exact_calls_lost": sum(max(0, scores_by_cap[base][sid]["matched_calls"] - scores_by_cap[cap][sid]["matched_calls"]) for sid in ids),
            "samples_with_added_unmatched_calls": sum(scores_by_cap[cap][sid]["unmatched_submitted_calls"] > scores_by_cap[base][sid]["unmatched_submitted_calls"] for sid in ids)}

    adjacent: dict[str, Any] = {}
    for lower, upper in zip(CAPS, CAPS[1:]):
        ds = [scores_by_cap[upper][sid]["btc_formula_score_before_runtime_gate_diagnostic"] -
              scores_by_cap[lower][sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
        dl = [float(preds_by_cap[upper][sid]["latency_seconds"]) -
              float(preds_by_cap[lower][sid]["latency_seconds"]) for sid in ids]
        score_ci, latency_ci = bootstrap(ds, rng), bootstrap(dl, rng)
        outcome = Counter("improved" if d > 1e-12 else "worsened" if d < -1e-12 else "tied" for d in ds)
        adjacent[f"{lower}_vs_{upper}"] = {
            "btc_delta_points_upper_minus_lower": {"mean": 100 * score_ci["mean"],
                "ci95_low": 100 * score_ci["ci95_low"], "ci95_high": 100 * score_ci["ci95_high"]},
            "latency_delta_seconds_upper_minus_lower": latency_ci,
            "score_outcomes_upper_vs_lower": dict(outcome)}
        write_jsonl(OUT / f"paired-{lower}-vs-{upper}.jsonl", [{"id": sid,
            "lower_score": scores_by_cap[lower][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "upper_score": scores_by_cap[upper][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "score_delta": scores_by_cap[upper][sid]["btc_formula_score_before_runtime_gate_diagnostic"] - scores_by_cap[lower][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "lower_latency_seconds": preds_by_cap[lower][sid]["latency_seconds"],
            "upper_latency_seconds": preds_by_cap[upper][sid]["latency_seconds"]} for sid in ids])

    trend = adjacent["1024_vs_1536"]["score_outcomes_upper_vs_lower"]
    cap2048_gate = {
        "pre_registered_operational_threshold": "Only consider 2048 if at least 10/60 samples have a request with eval_count >= 1536, and paired 1536-vs-1024 BTC mean delta is positive with more improved than worsened samples.",
        "requests_reaching_num_predict": sum((x.get("eval_count") or 0) >= 1536 for x in rows(OUT / "requests-1536.jsonl")),
        "requests_total": metrics["1536"]["requests"],
        "samples_with_num_predict_reached": metrics["1536"]["samples_with_num_predict_reached"],
        "context_limited_candidates": metrics["1536"]["context_limited_candidates_inferred"],
        "context_limited_requests_are_not_counted_as_cap_hits": True,
        "paired_quality_trend_1536_vs_1024": adjacent["1024_vs_1536"],
        "eligible": metrics["1536"]["samples_with_num_predict_reached"] >= 10 and
                    adjacent["1024_vs_1536"]["btc_delta_points_upper_minus_lower"]["mean"] > 0 and
                    trend.get("improved", 0) > trend.get("worsened", 0)}
    cap2048_gate["decision"] = "Run 2048" if cap2048_gate["eligible"] else "Do not run 2048"

    write_json(OUT / "paired_analysis.json", {"cohort": "fixed enriched train-only pilot60",
        "matching": "exact tool name and arguments; duplicate calls matched by multiplicity",
        "score": "BTC formula with gold ToolEnv replay hash gate and M>0; no local 30-second mask",
        "bootstrap": {"iterations": BOOTSTRAPS, "seed": SEED, "method": "paired sample bootstrap"},
        "branches": metrics, "paired_vs_512": paired, "paired_adjacent_caps": adjacent,
        "cap2048_gate": cap2048_gate,
        "quality_interpretation": "Screening only: 512 replay reproducibility against the archived historical baseline must be reviewed separately."})
    md = ["# Qwen3.5 thinking budget — pilot60 results", "",
        "Cohort: fixed enriched train-only validation pilot60; not an unbiased estimate for val309. All BTC scores use exact tool name and arguments, duplicate multiplicity, gold trajectory replay through ToolEnv, equal prediction/gold state hash, and M > 0. No local runtime mask is applied to this formula score.", "",
        "| `num_predict` | BTC formula /100 | Hash match /60 | M / U / G | Sample latency mean / p50 / p90 / p95 / max (s) | Requests | Generated tokens total | Token cap samples | Repair requests | Context-limited candidates | >30s samples |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cap in CAPS:
        m = metrics[str(cap)]
        l = m["sample_latency_seconds"]
        md.append(f"| {cap} | {m['btc_formula_score_0_100']:.3f} | {m['hash_matches']}/60 | {m['matched_calls_M']} / {m['unmatched_calls_U']} / {m['gold_calls_G']} | {l['mean']:.2f} / {l['p50']:.2f} / {l['p90']:.2f} / {l['p95']:.2f} / {l['max']:.2f} | {m['requests']} | {m['total_generated_tokens_including_thinking']} | {m['token_cap_samples']} | {m['repair_requests']} | {m['context_limited_candidates_inferred']} | {m['samples_over_30s_local']} |")
    md.extend(["", "## Paired comparison against 512", "",
        "| Cap | BTC delta (95% paired bootstrap CI), points | Improved / worse / tied | Sample latency delta (95% CI), seconds | Hash gain / loss | Exact calls gained / lost | Added unmatched-call samples |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for cap in CAPS[1:]:
        p = paired[str(cap)]
        s, l = p["score_delta_points_vs_512"], p["sample_latency_delta_seconds_vs_512"]
        o = p["paired_score_outcomes_vs_512"]
        md.append(f"| {cap} | {s['mean']:+.3f} [{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}] | {o.get('improved',0)} / {o.get('worsened',0)} / {o.get('tied',0)} | {l['mean']:+.2f} [{l['ci95_low']:+.2f}, {l['ci95_high']:+.2f}] | {p['hash_matches_gained']} / {p['hash_matches_lost']} | {p['exact_calls_gained']} / {p['exact_calls_lost']} | {p['samples_with_added_unmatched_calls']} |")
    md.extend(["", "## Limitations and interpretation", "",
        "The cohort is enriched for prior token-cap and repair failures. These results screen the mechanism and do not estimate full val309 performance. Re-run results at cap 512 must be compared with the archived baseline; any divergence limits quality conclusions. Local RTX 4050 times and the 30-second counts are diagnostic only and do not establish A30 compliance.",
        "`eval_count` is total generated tokens, including thinking and answer; raw thinking was not saved. `done_reason=length` is split into cap reached, inferred context-limited candidate, and unresolved. The context classification is conservative/inferred from prompt+generation counts; `num_ctx` remains 8192 and high prompt lengths can reduce usable generation room.",
        "Per-request telemetry, payload hashes, sample scores, and paired outcomes are in the adjacent JSONL/JSON artifacts. GPU/process metrics should be interpreted from the recorded host observations if available.", ""])
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"btc_scores": {cap: metrics[str(cap)]["btc_formula_score_0_100"] for cap in CAPS},
        "paired_vs_512": paired, "status": "analysis complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
