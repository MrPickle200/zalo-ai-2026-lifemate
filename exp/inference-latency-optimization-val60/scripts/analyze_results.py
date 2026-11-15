from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/inference-latency-optimization-val60/output"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp/btc-scoring/scripts"))
from score_btc import score_sample  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    a = sorted(xs)
    pos = (len(a) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def summarize_latency(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs) if xs else None, "p50": quantile(xs, .5), "p90": quantile(xs, .9),
            "p95": quantile(xs, .95), "max": max(xs) if xs else None,
            "over_30s": sum(x > 30 for x in xs)}


def main() -> int:
    cohort = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))
    ids = cohort["ids"]
    data_root = find_data_root("train+public_test/train+public_test")
    items = {r["id"]: r for r in load_jsonl(data_root / "train/train.jsonl")}
    branches = [x for x in ("baseline", "batch512", "batch1024") if (OUT / f"predictions-{x}.jsonl").exists()]
    predictions, request_rows, metrics = {}, {}, {}
    for branch in branches:
        run_meta_path = OUT / f"run-{branch}.json"
        if run_meta_path.exists():
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            run_meta.setdefault("effective_num_batch", 1024 if branch in {"baseline", "batch1024"} else 512)
            run_meta.setdefault("num_batch_request_override", run_meta.get("num_batch", run_meta["effective_num_batch"]))
            write(run_meta_path, run_meta)
        pr = rows(OUT / f"predictions-{branch}.jsonl")
        rq = rows(OUT / f"requests-{branch}.jsonl")
        for request in rq:
            request.setdefault("model_digest", "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04")
        (OUT / f"requests-{branch}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rq), encoding="utf-8")
        if [x["id"] for x in pr] != ids or len({x["id"] for x in pr}) != 60:
            raise RuntimeError(f"{branch} predictions do not exactly match the locked cohort order")
        predictions[branch] = {x["id"]: x for x in pr}
        request_rows[branch] = rq
        latencies = [float(x["latency_seconds"]) for x in pr]
        req_lat = [float(x["latency_seconds"]) for x in rq]
        metric = {"sample_latency_seconds": summarize_latency(latencies),
                  "request_latency_seconds": summarize_latency(req_lat),
                  "request_count": len(rq), "errors": sum(bool(x.get("runtime_error")) for x in rows(OUT / f"traces-{branch}.jsonl")),
                  "requests_over_30s": sum(x > 30 for x in req_lat),
                  "prompt_eval_tokens": sum(x.get("prompt_eval_count") or 0 for x in rq),
                  "generated_tokens": sum(x.get("eval_count") or 0 for x in rq),
                  "prompt_eval_seconds": sum((x.get("prompt_eval_duration_ns") or 0) / 1e9 for x in rq),
                  "decode_seconds": sum((x.get("eval_duration_ns") or 0) / 1e9 for x in rq),
                  "load_seconds": sum((x.get("load_duration_ns") or 0) / 1e9 for x in rq),
                  "prompt_tokens_per_second_mean_request": statistics.fmean([x["prompt_tokens_per_second"] for x in rq if x.get("prompt_tokens_per_second")]) if any(x.get("prompt_tokens_per_second") for x in rq) else None,
                  "generated_tokens_per_second_mean_request": statistics.fmean([x["generated_tokens_per_second"] for x in rq if x.get("generated_tokens_per_second")]) if any(x.get("generated_tokens_per_second") for x in rq) else None,
                  "prompt_eval_cached_count_sum": sum(x.get("prompt_eval_cached_count") or 0 for x in rq),
                  "missing_prompt_eval_duration": sum(x.get("prompt_eval_duration_ns") is None for x in rq),
                  "missing_eval_duration": sum(x.get("eval_duration_ns") is None for x in rq),
                  "missing_load_duration": sum(x.get("load_duration_ns") is None for x in rq)}
        scored = [score_sample(sid, items[sid], predictions[branch][sid], data_root) for sid in ids]
        metric["btc_formula_score_before_runtime_gate_diagnostic_0_100"] = 100 * statistics.fmean(x["btc_formula_score_before_runtime_gate_diagnostic"] for x in scored)
        metric["hash_matches"] = sum(x["hash_match"] for x in scored)
        metric["predicted_final_state_matches_label"] = sum(x["predicted_final_state_matches_label"] for x in scored)
        metric["exact_matched_calls"] = sum(x["matched_calls"] for x in scored)
        metric["unmatched_submitted_calls"] = sum(x["unmatched_submitted_calls"] for x in scored)
        metric["gold_calls"] = sum(x["gold_calls"] for x in scored)
        metric["gold_replay_failures"] = sum(x["gold_replay_error"] is not None for x in scored)
        metric["schema_rejected_submitted_calls"] = sum(x["schema_rejections"] for x in scored)
        metric["runtime_failures"] = sum(x["runtime_failure"] for x in scored)
        write(OUT / f"btc-summary-{branch}.json", metric)
        (OUT / f"btc-details-{branch}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scored), encoding="utf-8")
        metrics[branch] = metric
    if "batch512" in branches and "batch1024" in branches:
        a, b = predictions["batch512"], predictions["batch1024"]
        deltas = [b[sid]["latency_seconds"] - a[sid]["latency_seconds"] for sid in ids]
        score_a = {x["id"]: x for x in rows(OUT / "btc-details-batch512.jsonl")}
        score_b = {x["id"]: x for x in rows(OUT / "btc-details-batch1024.jsonl")}
        score_deltas = [score_b[sid]["btc_formula_score_before_runtime_gate_diagnostic"] -
                        score_a[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in ids]
        rng = random.Random(20261105)
        boot, median_boot, score_boot = [], [], []
        for _ in range(10000):
            sample_idx = [rng.randrange(60) for _ in range(60)]
            boot.append(statistics.fmean(deltas[i] for i in sample_idx))
            median_boot.append(statistics.median(deltas[i] for i in sample_idx))
            score_boot.append(100 * statistics.fmean(score_deltas[i] for i in sample_idx))
        boot.sort()
        median_boot.sort()
        score_boot.sort()
        call_multiset_mismatch = 0
        ordered_mismatch = 0
        hash_mismatch = 0
        for sid in ids:
            ca = Counter(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for x in a[sid]["tool_calls"])
            cb = Counter(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for x in b[sid]["tool_calls"])
            ordered_mismatch += a[sid]["tool_calls"] != b[sid]["tool_calls"]
            call_multiset_mismatch += ca != cb
            hash_mismatch += a[sid]["state_hash"] != b[sid]["state_hash"]
        paired_rows = []
        for sid, latency_delta, score_delta in zip(ids, deltas, score_deltas):
            paired_rows.append({"id": sid, "latency_batch512_seconds": a[sid]["latency_seconds"],
                                "latency_batch1024_seconds": b[sid]["latency_seconds"],
                                "latency_delta_1024_minus_512_seconds": latency_delta,
                                "btc_formula_score_batch512": score_a[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
                                "btc_formula_score_batch1024": score_b[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
                                "btc_formula_score_delta_1024_minus_512": score_delta,
                                "hash_match_batch512": score_a[sid]["hash_match"],
                                "hash_match_batch1024": score_b[sid]["hash_match"]})
        (OUT / "paired-batch-deltas.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in paired_rows), encoding="utf-8")
        paired = {"sample_latency_delta_seconds_1024_minus_512": summarize_latency(deltas),
                  "mean_sample_latency_delta_bootstrap_95_ci_seconds": [boot[249], boot[9749]],
                  "median_sample_latency_delta_bootstrap_95_ci_seconds": [median_boot[249], median_boot[9749]],
                  "btc_formula_score_delta_1024_minus_512_points": 100 * statistics.fmean(score_deltas),
                  "btc_formula_score_delta_bootstrap_95_ci_points": [score_boot[249], score_boot[9749]],
                  "paired_median_latency_reduction_fraction": ((metrics["batch512"]["sample_latency_seconds"]["p50"] - metrics["batch1024"]["sample_latency_seconds"]["p50"]) /
                                                                metrics["batch512"]["sample_latency_seconds"]["p50"]),
                  "p95_not_worse": metrics["batch1024"]["sample_latency_seconds"]["p95"] <= metrics["batch512"]["sample_latency_seconds"]["p95"],
                  "samples_over_30s_not_worse": metrics["batch1024"]["sample_latency_seconds"]["over_30s"] <= metrics["batch512"]["sample_latency_seconds"]["over_30s"],
                  "runtime_errors_in_either_branch": metrics["batch512"]["errors"] + metrics["batch1024"]["errors"],
                  "ordered_call_mismatch_samples": ordered_mismatch, "call_multiset_mismatch_samples": call_multiset_mismatch,
                  "state_hash_mismatch_samples": hash_mismatch,
                  "quality_reproducibility_caveat": "Baseline reproduction was previously unstable; these paired scores are only diagnostic."}
        write(OUT / "paired-batch-comparison.json", paired)
    write(OUT / "analysis.json", {"cohort": "fixed enriched train-only pilot60", "branches": metrics})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
