from __future__ import annotations

import json
import random
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import edit_distance, normalize
from lifemate.utils import find_data_root, load_jsonl


ROOT = Path("exp/qwen35-track-c-quan-vs-phoasr-val309/output/pilot60")
BRANCHES = ("phoasr-ft", "quan-ft-beam32")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap(values: list[float]) -> dict:
    rng = random.Random(20261104)
    n, iterations = len(values), 10_000
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return {"mean": sum(values) / n, "ci95_percentile": [samples[int(.025 * iterations)], samples[int(.975 * iterations)]],
            "resamples": iterations, "seed": 20261104}


def corpus_wer_cer(rows: list[tuple[str, str]]) -> dict:
    word_edits = word_count = char_edits = char_count = 0
    for reference, prediction in rows:
        ref, hyp = normalize(reference), normalize(prediction)
        ref_chars, hyp_chars = list("".join(ref)), list("".join(hyp))
        word_edits += edit_distance(ref, hyp); word_count += len(ref)
        char_edits += edit_distance(ref_chars, hyp_chars); char_count += len(ref_chars)
    return {"corpus_wer": word_edits / max(1, word_count), "corpus_cer": char_edits / max(1, char_count)}


def themes(reference: str, pho: str, quan: str) -> list[str]:
    flags = []
    if re.search(r"\d|\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|trăm|nghìn|triệu)\b", reference.lower()) and pho != quan:
        flags.append("number_or_amount_transcript_diff")
    if re.search(r"\b(độ|°c|km|kilô|kg|gram|lít|ml|đồng|usd|eur|yên|phút|giây)\b", reference.lower()) and pho != quan:
        flags.append("unit_or_currency_transcript_diff")
    if re.search(r"\b(hôm|ngày|tháng|năm|giờ|sáng|chiều|tối|mai|nay)\b", reference.lower()) and pho != quan:
        flags.append("date_or_time_transcript_diff")
    if re.search(r"\b(không|đừng|chưa|chẳng|hủy|huỷ|hoãn|nếu|trừ)\b", reference.lower()) and pho != quan:
        flags.append("negation_or_conditional_transcript_diff")
    if re.search(r"\b(tôi|anh|chị|bạn|cô|chú|bác|ông|bà)\b", reference.lower()) and pho != quan:
        flags.append("person_reference_transcript_diff")
    if re.search(r"\b(cho|gọi|nhắn|đến|tại|với)\s+[A-ZÀ-Ỹ][\wÀ-ỹ]+", reference) and pho != quan:
        flags.append("possible_named_entity_transcript_diff")
    if not flags:
        flags.append("other_or_no_transcript_diff")
    return flags


def main() -> int:
    project = _bootstrap.ROOT
    ids = json.loads((ROOT / "pilot_ids.json").read_text(encoding="utf-8"))
    root = find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in load_jsonl(root / "train/train.jsonl")}
    prediction_rows = {branch: read_jsonl(ROOT / branch / "predictions.jsonl") for branch in BRANCHES}
    score_rows = {branch: read_jsonl(ROOT / branch / "btc-scores.jsonl") for branch in BRANCHES}
    runtime = {branch: read_json(ROOT / branch / "runtime-summary.json") for branch in BRANCHES}
    transcript_cache = {branch: read_jsonl(ROOT / branch / "transcripts-val309.jsonl") for branch in BRANCHES}

    scores = {}
    corpus = {}
    for branch in BRANCHES:
        scores[branch] = [float(score_rows[branch][sid]["btc_formula_score_before_runtime_gate_diagnostic"]) for sid in ids]
        corpus[branch] = corpus_wer_cer([(items[sid].get("audio_text", ""), transcript_cache[branch][sid].get("transcript", "")) for sid in ids])
        detail_rows = []
        for sid in ids:
            row = score_rows[branch][sid]
            row["runtime_masked_sample_score_diagnostic"] = row.get("runtime_masked_sample_score_diagnostic", row["official_sample_score"])
            row["official_sample_score"] = row["btc_formula_score_before_runtime_gate_diagnostic"]
            detail_rows.append(row)
        (ROOT / branch / "btc-scores.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in detail_rows), encoding="utf-8")
        score_summary = read_json(ROOT / branch / "btc-summary.json")
        masked = score_summary.get("runtime_masked_score_diagnostic_0_100", score_summary["official_score_0_100"])
        score_summary["split"] = "fixed train-only val309, stratified pilot60 cohort"
        score_summary["official_score_0_100"] = 100 * sum(scores[branch]) / len(ids)
        score_summary["runtime_masked_score_diagnostic_0_100"] = masked
        score_summary["score_note"] = "official formula score is reported before separate runtime mask; runtime_masked_score_diagnostic is not the official formula"
        write_json(ROOT / branch / "btc-summary.json", score_summary)

    deltas = [q - p for p, q in zip(scores["phoasr-ft"], scores["quan-ft-beam32"])]
    paired = {sid: deltas[i] for i, sid in enumerate(ids)}
    improved = [sid for sid in ids if paired[sid] > 1e-12]
    worsened = [sid for sid in ids if paired[sid] < -1e-12]
    tied = [sid for sid in ids if abs(paired[sid]) <= 1e-12]
    analysis = []
    for sid in ids:
        item = items[sid]
        pho_text = transcript_cache["phoasr-ft"][sid].get("transcript", "")
        quan_text = transcript_cache["quan-ft-beam32"][sid].get("transcript", "")
        pho_calls = prediction_rows["phoasr-ft"][sid].get("tool_calls", [])
        quan_calls = prediction_rows["quan-ft-beam32"][sid].get("tool_calls", [])
        gold_calls = item.get("label", {}).get("tool_calls", [])
        analysis.append({"id": sid, "btc_delta_quan_minus_phoasr_0_100": 100 * paired[sid],
            "phoasr_btc": 100 * score_rows["phoasr-ft"][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "quan_btc": 100 * score_rows["quan-ft-beam32"][sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "hash_match": {b: score_rows[b][sid]["hash_match"] for b in BRANCHES},
            "wer_cer": {b: corpus_wer_cer([(item.get("audio_text", ""), transcript_cache[b][sid].get("transcript", "")) for _ in [0]]) for b in BRANCHES},
            "themes": themes(item.get("audio_text", ""), pho_text, quan_text),
            "reference": item.get("audio_text", ""), "phoasr_transcript": pho_text, "quan_transcript": quan_text,
            "gold_calls": gold_calls, "phoasr_calls": pho_calls, "quan_calls": quan_calls,
            "gold_tool_names": [call.get("name") for call in gold_calls],
            "predicted_tool_names": {"phoasr": [call.get("name") for call in pho_calls], "quan": [call.get("name") for call in quan_calls]},
            "gold_exact_call_matches": {b: score_rows[b][sid]["matched_calls"] for b in BRANCHES},
            "gold_argument_mismatches": {b: len(gold_calls) - score_rows[b][sid]["matched_calls"] for b in BRANCHES},
            "premature_stop_or_under_call": {"phoasr": len(pho_calls) < len(gold_calls), "quan": len(quan_calls) < len(gold_calls)},
            "over_call_vs_gold": {"phoasr": len(pho_calls) > len(gold_calls), "quan": len(quan_calls) > len(gold_calls)}})

    paired_report = {"metric": "BTC official formula score before runtime mask", "split": "fixed train-only val309 stratified pilot60",
        "matching": "exact tool name and arguments; duplicate multiplicity; hash gate; gold replay via ToolEnv",
        "phoasr_btc_0_100": 100 * sum(scores["phoasr-ft"]) / len(ids),
        "quan_btc_0_100": 100 * sum(scores["quan-ft-beam32"]) / len(ids),
        "paired_difference_quan_minus_phoasr_0_100": bootstrap([100 * value for value in deltas]),
        "improved_samples": len(improved), "worsened_samples": len(worsened), "tied_samples": len(tied),
        "improved_ids": improved, "worsened_ids": worsened, "tied_ids": tied,
        "corpus_wer_cer": corpus,
        "branches": {branch: {"mean_latency_seconds": runtime[branch]["mean_latency_seconds"],
            "p95_latency_seconds": runtime[branch]["p95_latency_seconds"], "max_latency_seconds": runtime[branch]["max_latency_seconds"],
            "samples_over_30s": runtime[branch]["samples_over_30s"], "runtime_failures": runtime[branch]["runtime_failures"],
            "schema_rejections": runtime[branch]["schema_rejections"], "parse_schema_failures": runtime[branch]["parse_schema_failures"],
            "exact_matched_calls": read_json(ROOT / branch / "btc-summary.json")["matched_calls_total"],
            "aggregate_precision": read_json(ROOT / branch / "btc-summary.json")["aggregate_exact_call_precision"],
            "aggregate_recall": read_json(ROOT / branch / "btc-summary.json")["aggregate_exact_call_recall"],
            "hash_match_rate": read_json(ROOT / branch / "btc-summary.json")["hash_match_rate"]} for branch in BRANCHES},
        "analysis_samples": analysis,
        "interpretation_limit": "Pilot60 only. The fixed validation set had 309 IDs; no full paired run was started because the pre-run estimate exceeded four hours."}
    write_json(ROOT / "paired-analysis.json", paired_report)
    write_json(ROOT.parent / "summary.json", {"status": "pilot_complete", "cohort": "fixed stratified pilot60",
        "paired_report": "pilot60/paired-analysis.json", "full_val309_run": "not started; estimated paired model runtime 19,694 seconds (>4 hours)"})
    print(json.dumps({"phoasr_btc": paired_report["phoasr_btc_0_100"], "quan_btc": paired_report["quan_btc_0_100"],
        "paired_difference": paired_report["paired_difference_quan_minus_phoasr_0_100"],
        "improved": len(improved), "worsened": len(worsened), "tied": len(tied), "corpus_wer_cer": corpus}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
