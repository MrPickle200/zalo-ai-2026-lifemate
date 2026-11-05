from __future__ import annotations

import json
from pathlib import Path
import sys
import _bootstrap
sys.path.insert(0, str(_bootstrap.ROOT / "src"))
from lifemate.utils import find_data_root, load_jsonl


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired(before: list[dict], after: list[dict], labels: dict[str, dict] | None = None) -> dict:
    b, a = {r["id"]: r for r in before}, {r["id"]: r for r in after}
    ids = [sid for sid in b if sid in a]
    paired_counts = {"improved": 0, "worsened": 0, "unchanged": 0}
    regressions, gains = [], []
    for sid in ids:
        delta = a[sid]["score"] - b[sid]["score"]
        key = "improved" if delta > 1e-12 else "worsened" if delta < -1e-12 else "unchanged"
        paired_counts[key] += 1
        if key == "worsened":
            gold_calls = (labels or {}).get(sid, {}).get("label", {}).get("tool_calls", [])
            regressions.append({"id": sid, "public_score": b[sid]["score"], "hybrid_score": a[sid]["score"],
                                "public_state_match": b[sid].get("state_match"), "hybrid_state_match": a[sid].get("state_match"),
                                "transcript": b[sid].get("asr_text"), "dual_transcript": a[sid].get("transcript_used"),
                                "gold_tool_calls": gold_calls,
                                "public_tool_calls": b[sid].get("tool_calls"), "hybrid_tool_calls": a[sid].get("tool_calls")})
        elif key == "improved":
            gains.append(sid)
    n = len(ids)
    def mean(rows, field):
        return sum(float(r.get(field, 0) or 0) for r in rows) / len(rows) if rows else None
    return {"samples": n, **paired_counts,
            "paired_public_proxy": mean([b[s] for s in ids], "score"),
            "paired_hybrid_proxy": mean([a[s] for s in ids], "score"),
            "public_micro_tool_f1": _micro([b[s] for s in ids]),
            "hybrid_micro_tool_f1": _micro([a[s] for s in ids]),
            "public_final_state_match": mean([b[s] for s in ids], "state_match"),
            "hybrid_final_state_match": mean([a[s] for s in ids], "state_match"),
            "regressions": regressions, "improved_ids": gains}


def _micro(rows):
    tp = sum(r.get("tp", 0) for r in rows)
    p = sum(r.get("predicted_calls", 0) for r in rows)
    g = sum(r.get("gold_calls", 0) for r in rows)
    precision = tp / p if p else (1.0 if not g else 0.0)
    recall = tp / g if g else (1.0 if not p else 0.0)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    root = Path("exp")
    public = read(root / "phoasr-small-baseline-val309" / "output" / "predictions.jsonl")
    ft = read(root / "phoasr-lifemate-ft-v1-val309" / "output" / "predictions.jsonl")
    dual = read(root / "phoasr-dual-transcript-v1-val309" / "output" / "predictions.jsonl")
    labels = {r["id"]: r for r in load_jsonl(find_data_root() / "train" / "train.jsonl")}
    report = {"dual_vs_public": paired(public, dual, labels), "ft_vs_public": paired(public, ft, labels),
              "dual_vs_ft": paired(ft, dual, labels),
              "comparison_note": "Paired per-sample score is the existing evaluator's mean of sample tool-call F1 and labeled final-state match; aggregate tool F1 is micro-averaged.",
              "selection": "PhoASR public baseline; dual transcript regressed on all aggregate downstream metrics and is not selected."}
    asr_pairs = {r["sample_id"]: r for r in read(root / "paired-analysis" / "phoasr-public-vs-ft-v1" / "output" / "paired_transcripts.jsonl")}
    regression_ids = {r["id"] for r in report["dual_vs_public"]["regressions"]}
    improved_ids = set(report["dual_vs_public"]["improved_ids"])
    report["transcript_pair_outcomes"] = {
        "regressions_where_transcripts_were_identical": sum(asr_pairs[sid]["phoasr_public_transcript"].strip().casefold() == asr_pairs[sid]["phoasr_ft_transcript"].strip().casefold() for sid in regression_ids),
        "regressions_where_transcripts_differed": sum(asr_pairs[sid]["phoasr_public_transcript"].strip().casefold() != asr_pairs[sid]["phoasr_ft_transcript"].strip().casefold() for sid in regression_ids),
        "improvements_where_transcripts_were_identical": sum(asr_pairs[sid]["phoasr_public_transcript"].strip().casefold() == asr_pairs[sid]["phoasr_ft_transcript"].strip().casefold() for sid in improved_ids),
        "improvements_where_transcripts_differed": sum(asr_pairs[sid]["phoasr_public_transcript"].strip().casefold() != asr_pairs[sid]["phoasr_ft_transcript"].strip().casefold() for sid in improved_ids)}
    target = root / "paired-analysis" / "phoasr-hybrid-v1" / "output"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "dual-regressions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report["dual_vs_public"]["regressions"]), encoding="utf-8")
    slim = {k: {x: v for x, v in value.items() if x not in {"regressions", "improved_ids"}} for k, value in report.items() if isinstance(value, dict)}
    print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
