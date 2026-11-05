from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import edit_distance, normalize, read_jsonl, score_rows


def main() -> int:
    project = _bootstrap.ROOT
    parser = argparse.ArgumentParser(description="Compare paired PhoASR and QUAN validation transcripts")
    parser.add_argument("--val-ids", default=str(project / "data" / "splits" / "val_ids.json"))
    parser.add_argument("--phoasr", default=str(project / "exp" / "phoasr-inference-val309" / "output" / "predictions.jsonl"))
    parser.add_argument("--quan", default=str(project / "exp" / "quan-transcribe-lm-val309" / "output" / "predictions.jsonl"))
    parser.add_argument("--output", default=str(project / "exp" / "paired-analysis" / "phoasr-vs-quan-transcribe-val309" / "output" / "summary.json"))
    args = parser.parse_args()

    ids = json.loads(Path(args.val_ids).read_text(encoding="utf-8-sig"))
    pho_rows = {row["id"]: row for row in read_jsonl(Path(args.phoasr))}
    quan_rows = {row["id"]: row for row in read_jsonl(Path(args.quan))}
    expected = set(ids)
    if set(pho_rows) != expected or set(quan_rows) != expected:
        raise ValueError(
            f"Need exact validation coverage: expected={len(ids)}, PhoASR={len(pho_rows)}, QUAN={len(quan_rows)}"
        )
    pho = [pho_rows[sid] for sid in ids]
    quan = [quan_rows[sid] for sid in ids]
    wins = {"quan": 0, "phoasr": 0, "tie": 0}
    paired_deltas = []
    disagreements = []
    for p, q in zip(pho, quan):
        ref = normalize(p.get("reference") or "")
        p_hyp, q_hyp = normalize(p.get("prediction") or ""), normalize(q.get("prediction") or "")
        p_edits, q_edits = edit_distance(ref, p_hyp), edit_distance(ref, q_hyp)
        p_wer, q_wer = p_edits / max(1, len(ref)), q_edits / max(1, len(ref))
        paired_deltas.append(q_wer - p_wer)
        wins["quan" if q_wer < p_wer else "phoasr" if p_wer < q_wer else "tie"] += 1
        if p_hyp != q_hyp:
            disagreements.append({
                "id": p["id"], "reference": p.get("reference", ""),
                "phoasr": p.get("prediction", ""), "quan": q.get("prediction", ""),
                "phoasr_wer": p_wer, "quan_wer": q_wer,
            })
    summary = {
        "dataset": "all train examples" if Path(args.val_ids).name == "train_ids.json" else "validation IDs",
        "sample_ids_file": str(Path(args.val_ids).resolve()), "samples": len(ids),
        "phoasr": score_rows(pho), "quan_transcribe_kenlm_beam": score_rows(quan),
        "per_utterance_wer_wins": wins,
        "mean_paired_utterance_wer_delta_quan_minus_phoasr":
            sum(paired_deltas) / len(paired_deltas) if paired_deltas else None,
        "normalization":
            "lowercase and punctuation removed; WER uses Vietnamese whitespace tokens; CER removes whitespace",
        "decoder": {
            "phoasr": "greedy, num_beams=1",
            "quan_transcribe": "KenLM CTC beam search, width=32, alpha=0.3, beta=1.0",
        },
        "latency_note": "Both were measured sequentially on one RTX 4050 Laptop GPU; per-sample timing includes audio decoding/resampling and excludes model load.",
        "normalization_caveat": "No digit-to-Vietnamese-number-word conversion is applied. QUAN emits numbers as words, so formatting differences can count as WER/CER errors.",
        "transcript_disagreements": disagreements,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "transcript_disagreements"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
