from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/current-config-error-analysis-val309/output"
PILOT = ROOT / "exp/qwen35-track-c-quan-vs-phoasr-val309/output/pilot60/quan-ft-beam32"


def read_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> int:
    pilot_ids = json.loads((PILOT.parent / "pilot_ids.json").read_text(encoding="utf-8"))
    current_preds = read_jsonl(OUT / "predictions.jsonl")
    old_preds = read_jsonl(PILOT / "predictions.jsonl")
    current_scores = read_jsonl(OUT / "btc-scores.jsonl")
    old_scores = read_jsonl(PILOT / "btc-scores.jsonl")
    if len(pilot_ids) != 60 or any(sid not in current_preds or sid not in old_preds for sid in pilot_ids):
        raise RuntimeError("Pilot60 comparison IDs are incomplete")

    comparisons = []
    for sid in pilot_ids:
        new, old = current_preds[sid], old_preds[sid]
        same_calls = new.get("tool_calls", []) == old.get("tool_calls", [])
        same_hash = new.get("state_hash") == old.get("state_hash")
        comparisons.append({
            "id": sid,
            "calls_identical": same_calls,
            "hash_identical": same_hash,
            "prediction_identical_calls_and_hash": same_calls and same_hash,
            "old_calls": old.get("tool_calls", []),
            "new_calls": new.get("tool_calls", []),
            "old_hash": old.get("state_hash"),
            "new_hash": new.get("state_hash"),
            "old_runtime_seconds": old.get("latency_seconds"),
            "new_runtime_seconds": new.get("latency_seconds"),
            "old_formula_score": old_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "new_formula_score": current_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            # Earlier pilot scorer stored the pre-mask BTC formula in official_sample_score.
            # Reapply the same 30s competition runtime rule used by this run.
            "old_runtime_gated_score": (0.0 if old_scores[sid]["runtime_failure"] or old_scores[sid]["runtime_over_30s"]
                                         else old_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"]),
            "new_runtime_gated_score": current_scores[sid]["official_sample_score"],
        })
    deltas = [100 * (row["new_runtime_gated_score"] - row["old_runtime_gated_score"]) for row in comparisons]
    formula_deltas = [100 * (row["new_formula_score"] - row["old_formula_score"]) for row in comparisons]
    n = len(deltas)
    rng = random.Random(20261104)
    formula_means = sorted(sum(formula_deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(10_000))
    rng = random.Random(20261104)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(10_000))
    report = {
        "cohort": "fixed stratified train-only val309 pilot60 IDs only",
        "matching": "exact tool name and arguments with duplicate multiplicity; ToolEnv hash gate",
        "comparison": "previous Track C QUAN pilot60 run vs this full-val run on same 60 IDs",
        "same_calls": sum(row["calls_identical"] for row in comparisons),
        "same_hash": sum(row["hash_identical"] for row in comparisons),
        "same_calls_and_hash": sum(row["prediction_identical_calls_and_hash"] for row in comparisons),
        "old_samples_over_30s": sum(row["old_runtime_seconds"] is not None and row["old_runtime_seconds"] > 30 for row in comparisons),
        "new_samples_over_30s": sum(row["new_runtime_seconds"] is not None and row["new_runtime_seconds"] > 30 for row in comparisons),
        "paired_formula_score_difference_new_minus_old_0_100_mean": sum(formula_deltas) / n,
        "paired_formula_score_difference_new_minus_old_0_100": {
            "mean": sum(formula_deltas) / n,
            "ci95_percentile": [formula_means[int(.025 * 10_000)], formula_means[min(9_999, int(.975 * 10_000))]],
            "resamples": 10_000, "seed": 20261104,
        },
        "runtime_gated_difference_is_timing_only_when_predictions_identical": True,
        "different_sample_evidence": [row for row in comparisons if not row["prediction_identical_calls_and_hash"]],
        "paired_runtime_gated_score_difference_new_minus_old_0_100": {
            "mean": sum(deltas) / n,
            "ci95_percentile": [means[int(.025 * 10_000)], means[min(9_999, int(.975 * 10_000))]],
            "resamples": 10_000,
            "seed": 20261104,
            "improved": sum(delta > 1e-12 for delta in deltas),
            "worsened": sum(delta < -1e-12 for delta in deltas),
            "tied": sum(abs(delta) <= 1e-12 for delta in deltas),
        },
        "per_sample": comparisons,
    }
    (OUT / "paired_pilot60_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_sample", "different_sample_evidence"}}, ensure_ascii=False, indent=2))
    print(f"different_samples={len(report['different_sample_evidence'])}; details={OUT / 'paired_pilot60_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
