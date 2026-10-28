from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from run_eval import ERROR_TYPES, classify_error
from lifemate.utils import find_data_root, load_jsonl


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def categories(rows: list[dict], items: dict, oracle_by_id: dict) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        if row.get("score") is None or row["score"] >= 1:
            continue
        label = items[row["id"]].get("label", {})
        oracle = oracle_by_id[row["id"]]
        kind = classify_error(label.get("tool_calls", []), row.get("tool_calls", []),
                              row.get("asr_text", ""), items[row["id"]].get("audio_text", ""),
                              bool(oracle.get("exact_match") and oracle.get("state_match")))
        row["error_type"] = kind
        counts[kind] += 1
    return counts


def category_summary(counts: Counter) -> tuple[dict, dict]:
    total = sum(counts.values())
    return ({kind: counts[kind] for kind in ERROR_TYPES},
            {kind: counts[kind] / total if total else 0.0 for kind in ERROR_TYPES})


def paired_summary(before: list[dict], after: list[dict], before_errors: Counter,
                   after_errors: Counter, before_name: str, after_name: str) -> dict:
    before_by_id = {row["id"]: row for row in before}
    after_by_id = {row["id"]: row for row in after}
    ids = sorted(set(before_by_id) & set(after_by_id))
    deltas = [after_by_id[sample_id]["score"] - before_by_id[sample_id]["score"] for sample_id in ids]
    before_score = sum(before_by_id[sample_id]["score"] for sample_id in ids) / len(ids)
    after_score = sum(after_by_id[sample_id]["score"] for sample_id in ids) / len(ids)
    before_f1 = sum(before_by_id[sample_id].get("tool_call_f1", 0.0) for sample_id in ids) / len(ids)
    after_f1 = sum(after_by_id[sample_id].get("tool_call_f1", 0.0) for sample_id in ids) / len(ids)
    def micro_f1(rows_by_id: dict[str, dict]) -> float:
        tp = sum(rows_by_id[sample_id].get("tp", 0) for sample_id in ids)
        predicted = sum(rows_by_id[sample_id].get("predicted_calls", 0) for sample_id in ids)
        gold = sum(rows_by_id[sample_id].get("gold_calls", 0) for sample_id in ids)
        precision = tp / predicted if predicted else (1.0 if not gold else 0.0)
        recall = tp / gold if gold else (1.0 if not predicted else 0.0)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    before_micro_f1 = micro_f1(before_by_id)
    after_micro_f1 = micro_f1(after_by_id)
    before_state = sum(bool(before_by_id[sample_id].get("state_match")) for sample_id in ids) / len(ids)
    after_state = sum(bool(after_by_id[sample_id].get("state_match")) for sample_id in ids) / len(ids)
    before_counts, before_pct = category_summary(before_errors)
    after_counts, after_pct = category_summary(after_errors)
    return {"before": before_name, "after": after_name, "samples": len(ids),
            "before_proxy_mean": before_score, "after_proxy_mean": after_score,
            "proxy_gain": after_score - before_score,
            "before_tool_call_f1": before_micro_f1, "after_tool_call_f1": after_micro_f1,
            "before_mean_sample_tool_call_f1": before_f1, "after_mean_sample_tool_call_f1": after_f1,
            "before_final_state_match": before_state, "after_final_state_match": after_state,
            "paired_improved": sum(delta > 0 for delta in deltas),
            "paired_worsened": sum(delta < 0 for delta in deltas),
            "paired_tied": sum(delta == 0 for delta in deltas),
            "before_error_counts": before_counts, "after_error_counts": after_counts,
            "before_error_percent_of_incorrect": before_pct,
            "after_error_percent_of_incorrect": after_pct,
            "metric_note": "Local proxy: per-sample mean of tool-call F1 and labeled final-state exact match; official combined formula is unavailable."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a retrieval experiment against baseline and oracle")
    parser.add_argument("--retrieval-experiment", default="retrieval-v1-val309")
    args = parser.parse_args()
    root = find_data_root()
    items = {x["id"]: x for x in load_jsonl(root / "train" / "train.jsonl")}
    out = Path("outputs/eval")
    baseline_dir = out / "baseline-v1-val309"
    oracle_dir = out / "oracle-asr-v1-val309"
    retrieval_dir = out / args.retrieval_experiment
    baseline = read_rows(baseline_dir / "predictions.jsonl")
    oracle = read_rows(oracle_dir / "predictions.jsonl")
    retrieval = read_rows(retrieval_dir / "predictions.jsonl")
    baseline_by_id = {r["id"]: r for r in baseline}
    oracle_by_id = {r["id"]: r for r in oracle}
    retrieval_by_id = {r["id"]: r for r in retrieval}
    if set(baseline_by_id) != set(oracle_by_id) or set(baseline_by_id) != set(retrieval_by_id):
        raise SystemExit("Baseline, oracle, and retrieval validation IDs differ")

    baseline_errors = categories(baseline, items, oracle_by_id)
    retrieval_errors = categories(retrieval, items, oracle_by_id)
    for rows, directory, mode in ((baseline, baseline_dir, "baseline"), (retrieval, retrieval_dir, "retrieval")):
        (directory / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        summary_path = directory / "summary.json"
        report = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        counts = baseline_errors if mode == "baseline" else retrieval_errors
        report["error_counts"], report["error_percent_of_incorrect"] = category_summary(counts)
        report["error_analysis_note"] = "Heuristic attribution; ASR_SLOT_ERROR requires oracle exact tool calls and labeled final-state match."
        report["average_latency_seconds"] = sum(r.get("latency_seconds", 0.0) for r in rows) / len(rows)
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    oracle_summary_path = oracle_dir / "summary.json"
    oracle_report = json.loads(oracle_summary_path.read_text(encoding="utf-8-sig"))
    oracle_report["average_latency_seconds"] = sum(r.get("latency_seconds", 0.0) for r in oracle) / len(oracle)
    oracle_summary_path.write_text(json.dumps(oracle_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    paired_dir = out / "paired-analysis"
    comparisons = paired_dir / "baseline-vs-oracle"
    comparisons.mkdir(parents=True, exist_ok=True)
    oracle_pair = paired_summary(baseline, oracle, baseline_errors, Counter(), "baseline-v1", "oracle-asr-v1")
    (comparisons / "summary.json").write_text(json.dumps(oracle_pair, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    experiment_slug = args.retrieval_experiment.removesuffix("-val309")
    comparisons = paired_dir / f"baseline-vs-{experiment_slug}"
    comparisons.mkdir(parents=True, exist_ok=True)
    retrieval_pair = paired_summary(baseline, retrieval, baseline_errors, retrieval_errors, "baseline-v1", experiment_slug)
    regressions = []
    for sample_id in sorted(baseline_by_id):
        before, after = baseline_by_id[sample_id], retrieval_by_id[sample_id]
        if after["score"] < before["score"]:
            regressions.append({"id": sample_id, "baseline_score": before["score"],
                                "retrieval_score": after["score"], "transcript": after.get("asr_text", ""),
                                "candidate_tools": after.get("candidate_tools"),
                                "retrieved_examples": after.get("retrieved_examples"),
                                "gold_tool_calls": items[sample_id].get("label", {}).get("tool_calls", []),
                                "predicted_tool_calls": after.get("tool_calls", [])})
    (comparisons / "summary.json").write_text(json.dumps(retrieval_pair, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (comparisons / "regressions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regressions), encoding="utf-8")
    (paired_dir / "summary.json").write_text(json.dumps(retrieval_pair, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(retrieval_pair, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
