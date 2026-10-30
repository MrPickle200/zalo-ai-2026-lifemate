from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        cur = [i]
        for j, b in enumerate(right, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]


def corpus_rates(refs: list[str], hyps: list[str]) -> dict:
    word_edits = word_count = char_edits = char_count = 0
    for ref, hyp in zip(refs, hyps):
        rw, hw = tokens(ref), tokens(hyp)
        rc, hc = list("".join(rw)), list("".join(hw))
        word_edits += edit_distance(rw, hw)
        word_count += len(rw)
        char_edits += edit_distance(rc, hc)
        char_count += len(rc)
    return {"wer": word_edits / max(1, word_count), "cer": char_edits / max(1, char_count)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--output-root", default="outputs/eval")
    parser.add_argument("--public", default="phoasr-small-baseline-val309")
    parser.add_argument("--ft", default="phoasr-lifemate-ft-v1-val309")
    args = parser.parse_args()
    data = {row["id"]: row for row in load_jsonl(find_data_root(args.data_root) / "train" / "train.jsonl")}
    public_rows = {row["id"]: row for row in read_jsonl(Path(args.output_root) / args.public / "predictions.jsonl")}
    ft_rows = {row["id"]: row for row in read_jsonl(Path(args.output_root) / args.ft / "predictions.jsonl")}
    ids = [sample_id for sample_id in public_rows if sample_id in ft_rows]
    refs = [data[sample_id].get("audio_text", "") for sample_id in ids]
    pub_hyps = [public_rows[sample_id].get("asr_text") or "" for sample_id in ids]
    ft_hyps = [ft_rows[sample_id].get("asr_text") or "" for sample_id in ids]
    pub_summary = json.loads((Path(args.output_root) / args.public / "summary.json").read_text(encoding="utf-8-sig"))
    ft_summary = json.loads((Path(args.output_root) / args.ft / "summary.json").read_text(encoding="utf-8-sig"))
    counts = {"improved": 0, "worsened": 0, "unchanged": 0}
    examples = []
    for sample_id in ids:
        before, after = public_rows[sample_id], ft_rows[sample_id]
        if before.get("score") is not None and after.get("score") is not None:
            delta = after["score"] - before["score"]
            counts["improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"] += 1
        ref = data[sample_id].get("audio_text", "")
        pw, fw = corpus_rates([ref], [before.get("asr_text") or ""])["wer"], corpus_rates([ref], [after.get("asr_text") or ""])["wer"]
        if fw < pw:
            examples.append({"id": sample_id, "reference": ref, "phoasr_public": before.get("asr_text", ""),
                             "phoasr_ft": after.get("asr_text", ""), "public_utterance_wer": pw,
                             "ft_utterance_wer": fw, "public_score": before.get("score"), "ft_score": after.get("score")})
    pub, ft = pub_summary, ft_summary
    oracle = {"proxy": .2920, "tool_f1": .2429, "state_match": .3301}
    metrics = {"proxy": "composite_proxy_mean", "tool_f1": "tool_call_f1", "state_match": "state_match_rate"}
    outcome = {}
    for key, field in metrics.items():
        before, after = pub[field], ft[field]
        outcome[key] = {"phoasr_public": before, "phoasr_ft": after, "ft_gain": after-before,
                        "remaining_oracle_gap": oracle[key]-after,
                        "oracle_gap_closed_fraction_of_public_gap": (after-before)/(oracle[key]-before) if oracle[key] != before else None,
                        "direction": "improved" if after > before else "worsened" if after < before else "unchanged"}
    report = {"samples": len(ids), "runtime_failures": ft_summary["runtime_failures"],
              "asr": {"phoasr_public": corpus_rates(refs, pub_hyps), "phoasr_lifemate_ft": corpus_rates(refs, ft_hyps)},
              "downstream": outcome, "paired_sample_scores": counts,
              "public_runtime": {"mean_seconds": pub_summary.get("average_latency_seconds"),
                                 "ft_mean_seconds": ft_summary.get("average_latency_seconds")},
              "wer_improved_examples": sorted(examples, key=lambda x: x["public_utterance_wer"]-x["ft_utterance_wer"], reverse=True)[:20],
              "metric_note": "WER uses Vietnamese whitespace tokenization after lowercase/punctuation removal; CER removes whitespace. Downstream comparison is paired on common validation IDs."}
    target = Path(args.output_root) / "paired-analysis" / "phoasr-public-vs-ft-v1"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
