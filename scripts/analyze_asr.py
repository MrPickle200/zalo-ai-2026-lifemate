from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl


def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def edits(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, chars: bool = False) -> float:
    ref, hyp = norm(reference), norm(hypothesis)
    if chars:
        ref, hyp = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    else:
        ref, hyp = ref.split(), hyp.split()
    return edits(ref, hyp) / max(1, len(ref))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ASR and paired downstream results on fixed validation IDs")
    parser.add_argument("--experiment", default="phoasr-small-baseline-val309")
    parser.add_argument("--baseline", default="baseline-v1-val309")
    parser.add_argument("--output-root", default="outputs/eval")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    root, out = find_data_root(args.data_root), Path(args.output_root)
    items = {x["id"]: x for x in load_jsonl(root / "train" / "train.jsonl")}
    base = read(out / args.baseline / "predictions.jsonl")
    pho = read(out / args.experiment / "predictions.jsonl")
    base_by, pho_by = {x["id"]: x for x in base}, {x["id"]: x for x in pho}
    ids = sorted(set(base_by) & set(pho_by))
    def asr_scores(rows, key):
        refs = [items[i].get("audio_text", "") for i in ids]
        hyps = [rows[i].get(key) or "" for i in ids]
        word_refs = [norm(r).split() for r in refs]
        word_hyps = [norm(h).split() for h in hyps]
        char_refs = [list(norm(r).replace(" ", "")) for r in refs]
        char_hyps = [list(norm(h).replace(" ", "")) for h in hyps]
        total_words = sum(map(len, word_refs))
        total_chars = sum(map(len, char_refs))
        return {"wer": sum(edits(r, h) for r, h in zip(word_refs, word_hyps)) / max(1, total_words),
                "cer": sum(edits(r, h) for r, h in zip(char_refs, char_hyps)) / max(1, total_chars),
                "macro_utterance_wer": sum(error_rate(r, h) for r, h in zip(refs, hyps)) / len(ids),
                "macro_utterance_cer": sum(error_rate(r, h, True) for r, h in zip(refs, hyps)) / len(ids)}
    improved = worsened = unchanged = 0
    examples = []
    for sample_id in ids:
        before, after = base_by[sample_id], pho_by[sample_id]
        if before.get("score") is None or after.get("score") is None:
            continue
        delta = after["score"] - before["score"]
        improved += delta > 0
        worsened += delta < 0
        unchanged += delta == 0
        if error_rate(items[sample_id].get("audio_text", ""), after.get("asr_text", "")) < error_rate(
                items[sample_id].get("audio_text", ""), before.get("asr_text", "")):
            examples.append({"id": sample_id, "reference": items[sample_id].get("audio_text", ""),
                             "whisper": before.get("asr_text", ""), "phoasr": after.get("asr_text", ""),
                             "wer_whisper": error_rate(items[sample_id].get("audio_text", ""), before.get("asr_text", "")),
                             "wer_phoasr": error_rate(items[sample_id].get("audio_text", ""), after.get("asr_text", "")),
                             "baseline_score": before["score"], "phoasr_score": after["score"]})
    base_sum = json.loads((out / args.baseline / "summary.json").read_text(encoding="utf-8-sig"))
    pho_sum = json.loads((out / args.experiment / "summary.json").read_text(encoding="utf-8-sig"))
    proxy_gain = pho_sum["composite_proxy_mean"] - base_sum["composite_proxy_mean"]
    baseline_proxy, pho_proxy, oracle_proxy = base_sum["composite_proxy_mean"], pho_sum["composite_proxy_mean"], .2920
    def gap(base_value, new_value, oracle_value):
        return (new_value - base_value) / (oracle_value - base_value) if oracle_value != base_value else None
    report = {"samples": len(ids), "wer_cer": {"whisper_small": asr_scores(base_by, "asr_text"),
              "phoasr_whisper_small": asr_scores(pho_by, "asr_text")},
              "baseline": {"proxy": baseline_proxy, "tool_f1": base_sum["tool_call_f1"], "state_match": base_sum["state_match_rate"]},
              "phoasr": {"proxy": pho_proxy, "tool_f1": pho_sum["tool_call_f1"], "state_match": pho_sum["state_match_rate"]},
              "paired": {"improved": improved, "worsened": worsened, "unchanged": unchanged,
                         "proxy_gain": proxy_gain, "oracle_gap_closed_proxy": gap(baseline_proxy, pho_proxy, oracle_proxy),
                         "oracle_gap_closed_tool_f1": gap(base_sum["tool_call_f1"], pho_sum["tool_call_f1"], .2429),
                         "oracle_gap_closed_state": gap(base_sum["state_match_rate"], pho_sum["state_match_rate"], .3301)},
              "improved_asr_examples": sorted(examples, key=lambda x: x["wer_whisper"] - x["wer_phoasr"], reverse=True)[:20],
              "metric_note": "Corpus WER/CER are micro-averaged edit counts over the validation corpus; macro utterance rates are also reported. Punctuation/case excluded, Vietnamese whitespace tokenization for WER, whitespace removed for CER."}
    target = out / "paired-analysis" / f"baseline-vs-{args.experiment.removesuffix('-val309')}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "asr-analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
