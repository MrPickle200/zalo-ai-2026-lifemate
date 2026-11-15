from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from inference_common import edit_distance, normalize, read_jsonl, score_rows

NUMBER_WORDS = {
    "không", "một", "hai", "ba", "bốn", "tư", "năm", "sáu", "bảy", "tám", "chín",
    "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "rưỡi", "lẻ", "linh", "mốt", "lăm",
}
NEGATIONS = {"không", "chẳng", "chưa", "đừng", "chớ", "khỏi"}
TIME_WORDS = {
    "ngày", "tháng", "năm", "giờ", "phút", "sáng", "trưa", "chiều", "tối", "đêm",
    "hôm", "mai", "mốt", "tuần", "thứ", "lịch", "ngàymai", "ngàymốt",
}


def word_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def char_rate(reference: str, hypothesis: str) -> float:
    ref = list("".join(normalize(reference)))
    hyp = list("".join(normalize(hypothesis)))
    return edit_distance(ref, hyp) / max(1, len(ref))


def exactness(reference: str, hypothesis: str) -> dict:
    ref_tokens, hyp_tokens = normalize(reference), normalize(hypothesis)
    ref_numbers = [t for t in ref_tokens if t in NUMBER_WORDS or re.fullmatch(r"\d+(?:[.,]\d+)?", t)]
    hyp_numbers = [t for t in hyp_tokens if t in NUMBER_WORDS or re.fullmatch(r"\d+(?:[.,]\d+)?", t)]
    ref_times = [t for t in ref_tokens if t in TIME_WORDS or re.fullmatch(r"\d{1,2}:\d{2}", t)]
    hyp_times = [t for t in hyp_tokens if t in TIME_WORDS or re.fullmatch(r"\d{1,2}:\d{2}", t)]
    ref_neg = bool(set(ref_tokens) & NEGATIONS)
    hyp_neg = bool(set(hyp_tokens) & NEGATIONS)
    return {
        "numeric_present": bool(ref_numbers), "numeric_exact": ref_numbers == hyp_numbers if ref_numbers else None,
        "negation_exact": ref_neg == hyp_neg,
        "time_present": bool(ref_times), "time_exact": ref_times == hyp_times if ref_times else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired WER/CER and critical-slot analysis for a QUAN epoch")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    baseline, candidate = read_jsonl(args.baseline), read_jsonl(args.candidate)
    if [r["id"] for r in baseline] != [r["id"] for r in candidate] or len(baseline) != 309:
        raise ValueError("Expected matching, ordered baseline/candidate predictions for all 309 validation IDs")

    delta_wer, delta_cer = [], []
    wins = {"candidate": 0, "baseline": 0, "tie": 0}
    slots = {name: {"present": 0, "baseline_exact": 0, "candidate_exact": 0}
             for name in ("numeric", "time", "negation")}
    per_sample = []
    for before, after in zip(baseline, candidate):
        ref = before.get("reference") or ""
        b_hyp, c_hyp = before.get("prediction") or "", after.get("prediction") or ""
        b_wer, c_wer = word_rate(ref, b_hyp), word_rate(ref, c_hyp)
        b_cer, c_cer = char_rate(ref, b_hyp), char_rate(ref, c_hyp)
        delta_wer.append(c_wer - b_wer)
        delta_cer.append(c_cer - b_cer)
        wins["candidate" if c_wer < b_wer else "baseline" if b_wer < c_wer else "tie"] += 1
        b_slot, c_slot = exactness(ref, b_hyp), exactness(ref, c_hyp)
        for name in slots:
            present_key = "numeric_present" if name == "numeric" else "time_present" if name == "time" else None
            b_key = "numeric_exact" if name == "numeric" else "time_exact" if name == "time" else "negation_exact"
            present = b_slot[present_key] if present_key else True
            if present:
                slots[name]["present"] += 1
                slots[name]["baseline_exact"] += bool(b_slot[b_key])
                slots[name]["candidate_exact"] += bool(c_slot[b_key])
        per_sample.append({"id": before["id"], "baseline_wer": b_wer, "candidate_wer": c_wer,
                           "delta_wer_candidate_minus_baseline": c_wer - b_wer,
                           "baseline_cer": b_cer, "candidate_cer": c_cer,
                           "delta_cer_candidate_minus_baseline": c_cer - b_cer,
                           "baseline_exactness": b_slot, "candidate_exactness": c_slot})

    rng = random.Random(args.seed)
    wer_boot = []
    cer_boot = []
    for _ in range(args.bootstrap_samples):
        indices = [rng.randrange(len(delta_wer)) for _ in delta_wer]
        wer_boot.append(sum(delta_wer[i] for i in indices) / len(indices))
        cer_boot.append(sum(delta_cer[i] for i in indices) / len(indices))

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    summary = {
        "samples": len(candidate), "baseline_metrics": score_rows(baseline),
        "candidate_metrics": score_rows(candidate),
        "paired": {"wer_delta_mean_candidate_minus_baseline": sum(delta_wer) / len(delta_wer),
                   "wer_delta_95pct_bootstrap_ci": interval(wer_boot),
                   "cer_delta_mean_candidate_minus_baseline": sum(delta_cer) / len(delta_cer),
                   "cer_delta_95pct_bootstrap_ci": interval(cer_boot), "utterance_wer_wins": wins},
        "critical_slot_exactness": {
            name: {**counts,
                   "baseline_rate": counts["baseline_exact"] / max(1, counts["present"]),
                   "candidate_rate": counts["candidate_exact"] / max(1, counts["present"])}
            for name, counts in slots.items()},
        "normalization": "lowercase and remove punctuation; WER uses Vietnamese whitespace tokens; CER removes whitespace",
        "bootstrap": {"resamples": args.bootstrap_samples, "seed": args.seed,
                      "unit": "paired validation utterance"},
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
