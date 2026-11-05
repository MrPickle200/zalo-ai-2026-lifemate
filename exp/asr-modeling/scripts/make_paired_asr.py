from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from analyze_phoasr_ft import corpus_rates, read_jsonl
from lifemate.utils import find_data_root, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paired PhoASR transcript artifact for fixed validation IDs")
    parser.add_argument("--output-root", default="exp")
    parser.add_argument("--public", default="phoasr-small-baseline-val309")
    parser.add_argument("--ft", default="phoasr-lifemate-ft-v1-val309")
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    args = parser.parse_args()
    root = Path(args.output_root)
    data = {row["id"]: row for row in load_jsonl(find_data_root(args.data_root) / "train" / "train.jsonl")}
    pub = {r["id"]: r for r in read_jsonl(root / args.public / "output" / "predictions.jsonl")}
    ft = {r["id"]: r for r in read_jsonl(root / args.ft / "output" / "predictions.jsonl")}
    ids = [sid for sid in pub if sid in ft and sid in data]
    out = root / "paired-analysis" / "phoasr-public-vs-ft-v1" / "output"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    same = public_better = ft_better = both_wrong = 0
    refs, public_hyps, ft_hyps = [], [], []
    differences = []
    for sid in ids:
        ref = data[sid].get("audio_text", "")
        a, b = pub[sid].get("asr_text") or "", ft[sid].get("asr_text") or ""
        wer_a = corpus_rates([ref], [a])["wer"]
        wer_b = corpus_rates([ref], [b])["wer"]
        same += a.strip().casefold() == b.strip().casefold()
        public_better += wer_a < wer_b
        ft_better += wer_b < wer_a
        both_wrong += wer_a > 0 and wer_b > 0
        row = {"sample_id": sid, "ground_truth_transcript": ref, "phoasr_public_transcript": a,
               "phoasr_ft_transcript": b, "wer_public": wer_a, "wer_ft": wer_b,
               "avg_logprob_public": None, "avg_logprob_ft": None,
               "no_speech_prob_public": None, "no_speech_prob_ft": None,
               "compression_ratio_public": None, "compression_ratio_ft": None}
        rows.append(row)
        refs.append(ref); public_hyps.append(a); ft_hyps.append(b)
        if a.strip().casefold() != b.strip().casefold():
            # Preserve transcript differences containing likely slot-bearing terms for quick manual review.
            if re.search(r"\d|\b(?:triệu|nghìn|đồng|giờ|ngày|tháng|thứ|anh|chị|minh|lan|ngân|hà nội|sài gòn)\b", ref + " " + a + " " + b, re.I):
                differences.append(row)
    (out / "paired_transcripts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"samples": len(ids), "transcripts_identical": same, "transcripts_different": len(ids)-same,
               "wer_public_better": public_better, "wer_ft_better": ft_better, "wer_tied": len(ids)-public_better-ft_better,
               "both_transcripts_wrong": both_wrong,
               "corpus_wer_public": corpus_rates(refs, public_hyps)["wer"],
               "corpus_wer_ft": corpus_rates(refs, ft_hyps)["wer"],
               "confidence_metadata_available": False,
               "confidence_note": "Existing PhoASR inference artifacts contain decoded text only; comparable token confidence metadata was not recorded.",
               "slot_relevant_differences": differences[:60]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "slot_relevant_differences"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
