from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.utils import find_data_root, load_jsonl  # noqa: E402

OUT = ROOT / "exp/current-config-error-analysis-val309/output"
IDS_PATH = ROOT / "data/splits/val_ids.json"
PRED_PATH = OUT / "predictions.jsonl"
SCORE_PATH = OUT / "btc-scores.jsonl"
TRACE_PATH = OUT / "step-traces.jsonl"
TRANSCRIPT_PATH = OUT / "transcript_cache_val309.jsonl"

PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
NUM_WORDS = {"không", "một", "hai", "ba", "bốn", "tư", "năm", "sáu", "bảy", "tám", "chín",
             "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "rưỡi", "lẻ", "linh", "mốt", "lăm"}
NEGATIONS = {"không", "chẳng", "chưa", "đừng", "chớ", "khỏi"}
TIME_MARKERS = {"ngày", "tháng", "năm", "giờ", "phút", "sáng", "trưa", "chiều", "tối", "đêm",
                "hôm", "mai", "mốt", "tuần", "thứ", "lịch", "ngàymai", "ngàymốt"}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join(PUNCT.sub(" ", str(text).lower()).split())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def call_key(call: dict[str, Any]) -> str:
    return json.dumps({"name": call.get("name"), "arguments": call.get("arguments", {})},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def exact_call_lists(gold: list[dict], pred: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    remaining = Counter(call_key(c) for c in pred)
    matched, missing = [], []
    for call in gold:
        key = call_key(call)
        if remaining[key]:
            remaining[key] -= 1
            matched.append(call)
        else:
            missing.append(call)
    extras = []
    for call in pred:
        key = call_key(call)
        if remaining[key]:
            extras.append(call)
            remaining[key] -= 1
    return matched, missing, extras


def arg_and_name_error_counts(gold: list[dict], pred: list[dict]) -> tuple[int, int]:
    g, p = defaultdict(Counter), defaultdict(Counter)
    for c in gold:
        g[c.get("name")][call_key(c)] += 1
    for c in pred:
        p[c.get("name")][call_key(c)] += 1
    arg_errors = 0
    for name in set(g) & set(p):
        g_exact, p_exact = sum(g[name].values()), sum(p[name].values())
        matched = sum((g[name] & p[name]).values())
        arg_errors += min(g_exact, p_exact) - matched
    gold_left = sum((sum(counter.values()) for name, counter in g.items()))
    pred_left = sum((sum(counter.values()) for name, counter in p.items()))
    exact = sum((g[name] & p[name]).total() for name in set(g) | set(p))
    # Different tool names among the unmatched calls are candidate name errors;
    # exact U/M accounting remains in the scorer fields.
    gold_names, pred_names = Counter(c.get("name") for c in gold), Counter(c.get("name") for c in pred)
    same_name_unmatched = sum(max(0, gold_names[n] - pred_names[n]) for n in set(gold_names) | set(pred_names))
    same_name_unmatched -= arg_errors
    name_errors = max(0, min(len(gold) - exact - arg_errors, len(pred) - exact - arg_errors))
    return arg_errors, name_errors


def number_tokens(text: str) -> list[str]:
    tokens = norm(text).split()
    return [token for token in tokens if token in NUM_WORDS or re.fullmatch(r"\d+(?:[.,]\d+)?", token)]


def critical_slots(ref: str, hyp: str) -> dict[str, Any]:
    rt, ht = norm(ref).split(), norm(hyp).split()
    rn, hn = number_tokens(ref), number_tokens(hyp)
    ref_neg = bool(set(rt) & NEGATIONS)
    hyp_neg = bool(set(ht) & NEGATIONS)
    ref_time = [t for t in rt if t in TIME_MARKERS or re.fullmatch(r"\d{1,2}:\d{2}", t)]
    hyp_time = [t for t in ht if t in TIME_MARKERS or re.fullmatch(r"\d{1,2}:\d{2}", t)]
    return {
        "numeric_tokens_reference": rn,
        "numeric_tokens_asr": hn,
        "numeric_slot_present": bool(rn),
        "numeric_slot_exact": rn == hn if rn else None,
        "negation_reference": ref_neg,
        "negation_asr": hyp_neg,
        "negation_polarity_exact": ref_neg == hyp_neg,
        "time_markers_reference": ref_time,
        "time_markers_asr": hyp_time,
        "time_slot_present": bool(ref_time),
        "time_slot_exact_markers": ref_time == hyp_time if ref_time else None,
    }


def diff_tokens(ref: str, hyp: str) -> list[dict[str, str]]:
    a, b = norm(ref).split(), norm(hyp).split()
    # Produce a compact, local evidence span without pretending it explains causality.
    import difflib
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [{"reference": " ".join(a[i1:i2]), "asr": " ".join(b[j1:j2])}
            for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"][:12]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))]


def group_summary(records: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(label, "unknown"))].append(record)
    output = []
    for group, group_rows in sorted(groups.items()):
        matched = sum(row["M"] for row in group_rows)
        submitted = sum(row["predicted_call_count"] for row in group_rows)
        gold = sum(row["gold_call_count"] for row in group_rows)
        output.append({
            "group": group, "samples": len(group_rows),
            "btc_formula_before_runtime_gate_mean_0_1": sum(row["btc_formula_before_runtime_gate"] for row in group_rows) / len(group_rows),
            "runtime_gated_score_mean_0_1": sum(row["btc_score_runtime_gated"] for row in group_rows) / len(group_rows),
            "hash_matches": sum(row["hash_match"] for row in group_rows),
            "M": matched, "U": sum(row["U"] for row in group_rows), "gold_calls": gold,
            "predicted_calls": submitted,
            "aggregate_exact_precision": matched / submitted if submitted else 0.0,
            "aggregate_exact_recall": matched / gold if gold else 0.0,
        })
    return output


def main() -> int:
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    data_root = find_data_root("train+public_test/train+public_test")
    items = {item["id"]: item for item in load_jsonl(data_root / "train/train.jsonl")}
    pred_rows = {row["id"]: row for row in rows(PRED_PATH)}
    score_rows = {row["id"]: row for row in rows(SCORE_PATH)}
    trace_rows = {row["id"]: row for row in rows(TRACE_PATH)}
    transcript_rows = {row["id"]: row for row in rows(TRANSCRIPT_PATH)}
    if any(set(mapping) != set(ids) for mapping in (pred_rows, score_rows, trace_rows, transcript_rows)):
        raise RuntimeError("Analysis inputs do not exactly cover fixed val309")

    records = []
    word_ref = word_err = char_ref = char_err = 0
    all_slots = Counter()
    app_call_rows: dict[str, Counter] = defaultdict(Counter)
    tool_call_rows: dict[str, Counter] = defaultdict(Counter)
    app_sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tool_sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sid in ids:
        item, pred, score, trace = items[sid], pred_rows[sid], score_rows[sid], trace_rows[sid]
        gold = item.get("label", {}).get("tool_calls", []) or []
        calls = pred.get("tool_calls", []) or []
        matched, missing, extras = exact_call_lists(gold, calls)
        M, U, G = score["matched_calls"], score["unmatched_submitted_calls"], score["gold_calls"]
        if (M, U, G) != (len(matched), len(extras), len(gold)):
            raise RuntimeError(f"Scorer/call multiplicity mismatch for {sid}")
        arg_errors, name_errors = arg_and_name_error_counts(gold, calls)
        transcript = transcript_rows[sid]["transcript"]
        reference = transcript_rows[sid].get("reference", item.get("audio_text", ""))
        rw, hw = norm(reference).split(), norm(transcript).split()
        rc, hc = list("".join(rw)), list("".join(hw))
        wdist, cdist = edit_distance(rw, hw), edit_distance(rc, hc)
        word_ref += len(rw); word_err += wdist
        char_ref += len(rc); char_err += cdist
        slots = critical_slots(reference, transcript)
        if slots["numeric_slot_present"]:
            all_slots["numeric_tested"] += 1
            all_slots["numeric_exact"] += slots["numeric_slot_exact"]
        all_slots["negation_tested"] += 1
        all_slots["negation_exact"] += slots["negation_polarity_exact"]
        if slots["time_slot_present"]:
            all_slots["time_tested"] += 1
            all_slots["time_exact"] += slots["time_slot_exact_markers"]

        latency = float(score["prediction_latency_seconds"] or 0.0)
        runtime_failure = bool(score["runtime_failure"])
        schema_rejections = int(score["schema_rejections"])
        turns = trace.get("turns", []) or []
        repair_attempts = sum(bool(t.get("_diagnostics", {}).get("repair_attempted")) for t in turns)
        repair_failures = sum(bool(t.get("_diagnostics", {}).get("repair_attempted")) and
                              not t.get("_diagnostics", {}).get("repair_succeeded", False) for t in turns)
        length_stops = sum(t.get("_diagnostics", {}).get("done_reason") == "length" for t in turns)
        failed_turns = [t.get("_diagnostics", {}) for t in turns
                        if t.get("_diagnostics", {}).get("repair_attempted") and
                        not t.get("_diagnostics", {}).get("repair_succeeded", False)]
        failed_errors = [str(d.get("initial_schema_error") or "") for d in failed_turns]
        token_cap_failure = any(d.get("done_reason") == "length" for d in failed_turns)
        parse_failure = any("json" in error.lower() or "delimiter" in error.lower() or "JSON object" in error
                            for error in failed_errors)
        action_protocol_failure = any("Expected action=" in error for error in failed_errors)
        gold_names, pred_names = [c.get("name") for c in gold], [c.get("name") for c in calls]
        gold_ask = any(name == "sys.ask_missing_info" for name in gold_names)
        pred_ask = any(name == "sys.ask_missing_info" for name in pred_names)
        if not gold:
            task_type = "no_action_or_out_of_scope"
        elif gold_ask:
            task_type = "ask_missing_information"
        else:
            task_type = "execute_request"
        history = item.get("history") or item.get("conversation_history") or item.get("messages") or []
        has_history = bool(history)
        critical_asr_error = ((slots["numeric_slot_present"] and not slots["numeric_slot_exact"]) or
                              (slots["negation_reference"] != slots["negation_asr"]) or
                              (slots["time_slot_present"] and not slots["time_slot_exact_markers"]))
        if runtime_failure:
            primary, confidence = "protocol_runtime_failure", "certain"
        elif schema_rejections:
            primary, confidence = "toolenv_schema_rejection", "certain"
        elif token_cap_failure:
            primary, confidence = "token_cap_truncation", "high"
        elif parse_failure:
            primary, confidence = "json_parse_or_extraction_failure", "high"
        elif action_protocol_failure:
            primary, confidence = "json_action_protocol_failure", "high"
        elif repair_failures:
            primary, confidence = "schema_or_validator_repair_failure", "high"
        elif length_stops:
            primary, confidence = "generation_token_cap_hit", "medium"
        elif score["hash_match"] and Counter(map(call_key, gold)) == Counter(map(call_key, calls)):
            primary, confidence = "no_observed_action_error", "high"
        elif not gold and not calls:
            primary, confidence = ("no_action_hash_mismatch", "uncertain")
        elif gold_ask and not pred_ask:
            primary, confidence = "missing_information_policy", "high"
        elif not calls and gold:
            primary, confidence = "early_stop_or_under_call", "high"
        elif arg_errors:
            primary, confidence = "argument_grounding", "medium"
        elif name_errors:
            primary, confidence = "tool_selection_or_order", "medium"
        elif len(missing) > 0 and len(extras) > 0:
            primary, confidence = "tool_selection_sequence_or_overcall", "medium"
        elif len(missing) > 0:
            primary, confidence = "under_call_or_early_stop", "medium"
        elif len(extras) > 0:
            primary, confidence = "over_call_or_duplicate", "medium"
        elif Counter(map(call_key, gold)) == Counter(map(call_key, calls)):
            primary, confidence = "tool_order_or_toolenv_hash", "uncertain"
        else:
            primary, confidence = "uncertain", "low"
        secondary = []
        if critical_asr_error:
            secondary.append("critical ASR slot mismatch; causal link to action not proven")
        if length_stops:
            secondary.append(f"generation ended at num_predict cap in {length_stops} turn(s)")
        if has_history:
            secondary.append("conversation history present")
        if any(obs.get("result", {}).get("ok") is False for obs in pred.get("observations", [])):
            secondary.append("ToolEnv observation reported failure")
        if task_type == "no_action_or_out_of_scope" and calls:
            secondary.append("gold has no tool call; inspect whether request is out of scope or no-op")
        if len(turns) >= 5 and len(missing) > 0:
            secondary.append("five action-turn cap reached with gold calls missing")

        name_freq_gold, name_freq_pred = Counter(gold_names), Counter(pred_names)
        exact_precision = M / len(calls) if calls else 0.0
        exact_recall = M / len(gold) if gold else 0.0
        if score["hash_match"] and Counter(map(call_key, gold)) == Counter(map(call_key, calls)):
            outcome = "complete_exact_hash_match"
        elif task_type == "ask_missing_information":
            outcome = "ask_missing_info_gold"
        elif not calls:
            outcome = "empty_prediction"
        elif score["hash_match"] and M == G and U == 0:
            outcome = "complete_exact_hash_match"
        elif score["hash_match"]:
            outcome = "hash_match_with_call_difference"
        else:
            outcome = "executed_hash_mismatch"
        wer = wdist / max(1, len(rw))
        cer = cdist / max(1, len(rc))
        wer_bucket = "0" if wer == 0 else "(0,.10]" if wer <= .10 else "(.10,.30]" if wer <= .30 else ">.30"
        record = {
            "id": sid,
            "btc_score_runtime_gated": score["official_sample_score"],
            "btc_formula_before_runtime_gate": score["btc_formula_score_before_runtime_gate_diagnostic"],
            "gold_hash": score["gold_state_hash"], "predicted_hash": score["prediction_state_hash"],
            "hash_match": score["hash_match"], "M": M, "U": U, "G": G,
            "exact_matched_calls": matched, "missing_exact_calls": missing, "unmatched_submitted_calls": extras,
            "tool_name_mismatch_pairs_diagnostic": name_errors, "same_tool_wrong_arguments_diagnostic": arg_errors,
            "tool_call_precision": exact_precision, "tool_call_recall": exact_recall,
            "gold_call_count": len(gold), "predicted_call_count": len(calls),
            "gold_tools": gold_names, "predicted_tools": pred_names,
            "latency_seconds": latency,
            "runtime_status": "exception" if runtime_failure else "over_30s" if latency > 30 else "within_30s",
            "runtime_failure": runtime_failure, "schema_rejections": schema_rejections,
            "json_repair_attempts": repair_attempts, "json_repair_failures": repair_failures,
            "failed_repair_errors": failed_errors, "token_cap_failure": token_cap_failure,
            "json_parse_failure": parse_failure, "action_protocol_failure": action_protocol_failure,
            "duplicate_excess_calls": sum(max(0, Counter(map(call_key, calls))[key] - Counter(map(call_key, gold))[key])
                                           for key in set(map(call_key, calls)) if Counter(map(call_key, gold))[key] > 0),
            "generation_length_stops": length_stops,
            "task_type": task_type, "outcome": outcome, "has_history": has_history,
            "primary_error_cause": primary, "cause_confidence": confidence,
            "secondary_evidence_flags": secondary,
            "asr_reference": reference, "asr_transcript": transcript,
            "raw_wer": wer, "raw_cer": cer, "wer_bucket": wer_bucket,
            "asr_token_differences": diff_tokens(reference, transcript),
            "critical_asr_slots": slots,
            "numeric_asr_group": ("not_tested" if not slots["numeric_slot_present"] else
                                  "exact" if slots["numeric_slot_exact"] else "mismatch"),
            "negation_asr_group": "exact" if slots["negation_polarity_exact"] else "mismatch",
            "time_asr_group": ("not_tested" if not slots["time_slot_present"] else
                               "exact" if slots["time_slot_exact_markers"] else "mismatch"),
            "first_action": turns[0] if turns else None,
            "observations": pred.get("observations", []),
        }
        records.append(record)
        for call in gold:
            name = call.get("name", "unknown")
            app = name.split(".", 1)[0]
            tool_call_rows[name]["gold"] += 1; app_call_rows[app]["gold"] += 1
        for call in calls:
            name = call.get("name", "unknown")
            app = name.split(".", 1)[0]
            tool_call_rows[name]["predicted"] += 1; app_call_rows[app]["predicted"] += 1
        for call in matched:
            name = call.get("name", "unknown"); app = name.split(".", 1)[0]
            tool_call_rows[name]["matched"] += 1; app_call_rows[app]["matched"] += 1
        for call in missing:
            name = call.get("name", "unknown"); app = name.split(".", 1)[0]
            tool_call_rows[name]["missing_exact"] += 1; app_call_rows[app]["missing_exact"] += 1
        for call in extras:
            name = call.get("name", "unknown"); app = name.split(".", 1)[0]
            tool_call_rows[name]["extra"] += 1; app_call_rows[app]["extra"] += 1
        apps_in_gold = set()
        for name in set(gold_names):
            tool_call_rows[name]["gold_sample_support"] += 1
            apps_in_gold.add(name.split(".", 1)[0])
            tool_sample_rows[name].append(record)
        for app in apps_in_gold:
            app_call_rows[app]["gold_sample_support"] += 1
            app_sample_rows[app].append(record)
        for name in set(pred_names):
            tool_call_rows[name]["predicted_sample_support"] += 1
            app_call_rows[name.split(".", 1)[0]]["predicted_sample_support"] += 1

    if len(records) != 309:
        raise RuntimeError(f"Expected 309 analysis rows, got {len(records)}")
    with (OUT / "per_sample_error_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in record.items()})
    (OUT / "per_sample_error_table.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    latency = [r["latency_seconds"] for r in records]
    root_counts = Counter(r["primary_error_cause"] for r in records)
    if sum(root_counts.values()) != 309:
        raise RuntimeError("Primary cause counts do not sum to 309")
    total_m, total_u, total_g = sum(r["M"] for r in records), sum(r["U"] for r in records), sum(r["G"] for r in records)
    exact_p = total_m / (total_m + total_u) if total_m + total_u else 0.0
    exact_r = total_m / total_g if total_g else 0.0
    exact_f1 = 2 * exact_p * exact_r / (exact_p + exact_r) if exact_p + exact_r else 0.0
    summary = {
        "cohort": "fixed train-only val309, 309 predictions, one prediction ToolEnv per sample",
        "matching": "exact tool name and arguments; duplicate calls by multiplicity; hash gate; gold trajectory replayed through ToolEnv",
        "score_interpretation": "BTC formula before runtime gate and BTC score with the competition 30-second zero-score gate are reported separately",
        "btc_formula_score_before_runtime_gate_0_100": 100 * sum(r["btc_formula_before_runtime_gate"] for r in records) / 309,
        "runtime_gated_score_0_100": 100 * sum(r["btc_score_runtime_gated"] for r in records) / 309,
        "hash_match_samples": sum(r["hash_match"] for r in records),
        "hash_mismatch_samples": sum(not r["hash_match"] for r in records),
        "zero_runtime_gated_score_samples": sum(r["btc_score_runtime_gated"] == 0 for r in records),
        "samples_without_exact_matched_call": sum(r["M"] == 0 for r in records),
        "empty_prediction_samples": sum(r["predicted_call_count"] == 0 for r in records),
        "gold_ask_missing_info_samples": sum(r["task_type"] == "ask_missing_information" for r in records),
        "predicted_ask_missing_info_samples": sum("sys.ask_missing_info" in r["predicted_tools"] for r in records),
        "gold_no_action_or_out_of_scope_samples": sum(r["task_type"] == "no_action_or_out_of_scope" for r in records),
        "correct_no_action_samples": sum(r["task_type"] == "no_action_or_out_of_scope" and
                                          r["outcome"] == "complete_exact_hash_match" for r in records),
        "complete_exact_hash_match_samples": sum(r["outcome"] == "complete_exact_hash_match" for r in records),
        "M": total_m, "U": total_u, "G": total_g,
        "missing_exact_calls_total": total_g - total_m,
        "extra_submitted_calls_total": total_u,
        "same_tool_wrong_argument_calls_diagnostic": sum(r["same_tool_wrong_arguments_diagnostic"] for r in records),
        "tool_name_mismatch_pairs_diagnostic": sum(r["tool_name_mismatch_pairs_diagnostic"] for r in records),
        "duplicate_excess_calls": sum(r["duplicate_excess_calls"] for r in records),
        "exact_matched_call_precision": exact_p, "exact_matched_call_recall": exact_r,
        "exact_matched_call_f1_diagnostic": exact_f1,
        "runtime_failures": sum(r["runtime_failure"] for r in records),
        "schema_rejected_submitted_calls": sum(r["schema_rejections"] for r in records),
        "json_repair_attempts": sum(r["json_repair_attempts"] for r in records),
        "json_repair_failures": sum(r["json_repair_failures"] for r in records),
        "samples_with_failed_json_repairs": sum(r["json_repair_failures"] > 0 for r in records),
        "samples_with_token_cap_failure": sum(r["token_cap_failure"] for r in records),
        "samples_with_json_parse_failure": sum(r["json_parse_failure"] for r in records),
        "samples_with_action_protocol_failure": sum(r["action_protocol_failure"] for r in records),
        "samples_with_generation_length_stop": sum(r["generation_length_stops"] > 0 for r in records),
        "runtime_seconds_measurement": "LifeMatePipeline.run wall time on local RTX 4050 laptop GPU; includes local Ollama calls and ToolEnv, excludes gold replay; not an A30 benchmark",
        "latency_seconds": {"mean": sum(latency) / len(latency), "p50": percentile(latency, .50),
                            "p95": percentile(latency, .95), "max": max(latency),
                            "over_30s": sum(x > 30 for x in latency)},
        "ASR_raw_metrics": {"normalization": "lowercase, punctuation removed; whitespace Vietnamese tokens; CER removes spaces",
                            "corpus_WER": word_err / max(1, word_ref), "corpus_CER": char_err / max(1, char_ref),
                            "reference_words": word_ref, "word_edits": word_err,
                            "reference_characters_no_spaces": char_ref, "char_edits": char_err},
        "critical_slot_metrics": {
            "numeric_token_exact": {"correct": all_slots["numeric_exact"], "tested": all_slots["numeric_tested"],
                                     "accuracy": all_slots["numeric_exact"] / all_slots["numeric_tested"] if all_slots["numeric_tested"] else None},
            "negation_polarity_exact": {"correct": all_slots["negation_exact"], "tested": all_slots["negation_tested"],
                                        "accuracy": all_slots["negation_exact"] / all_slots["negation_tested"] if all_slots["negation_tested"] else None},
            "time_marker_sequence_exact": {"correct": all_slots["time_exact"], "tested": all_slots["time_tested"],
                                           "accuracy": all_slots["time_exact"] / all_slots["time_tested"] if all_slots["time_tested"] else None},
            "limitation": "These deterministic text-slot checks identify mismatches, not causality or entity correctness; names/places require manual trace review.",
        },
        "primary_cause_counts": dict(sorted(root_counts.items())),
        "cause_confidence_counts": dict(Counter(r["cause_confidence"] for r in records)),
        "groups": {
            "by_app": [{"app": k, **dict(v),
                        "gold_sample_support": len(app_sample_rows[k]),
                        "mean_btc_formula_score_for_gold_app_samples": sum(r["btc_formula_before_runtime_gate"] for r in app_sample_rows[k]) / len(app_sample_rows[k]) if app_sample_rows[k] else None,
                        "mean_runtime_gated_score_for_gold_app_samples": sum(r["btc_score_runtime_gated"] for r in app_sample_rows[k]) / len(app_sample_rows[k]) if app_sample_rows[k] else None}
                       for k, v in sorted(app_call_rows.items())],
            "by_tool": [{"tool": k, **dict(v),
                         "gold_sample_support": len(tool_sample_rows[k]),
                         "mean_btc_formula_score_for_gold_tool_samples": sum(r["btc_formula_before_runtime_gate"] for r in tool_sample_rows[k]) / len(tool_sample_rows[k]) if tool_sample_rows[k] else None,
                         "mean_runtime_gated_score_for_gold_tool_samples": sum(r["btc_score_runtime_gated"] for r in tool_sample_rows[k]) / len(tool_sample_rows[k]) if tool_sample_rows[k] else None}
                        for k, v in sorted(tool_call_rows.items())],
            "by_gold_call_count": group_summary(records, "gold_call_count"),
            "by_task_type": group_summary(records, "task_type"),
            "by_history": group_summary(records, "has_history"),
            "by_wer_bucket": group_summary(records, "wer_bucket"),
            "by_numeric_slot_group": group_summary(records, "numeric_asr_group"),
            "by_negation_slot_group": group_summary(records, "negation_asr_group"),
            "by_time_slot_group": group_summary(records, "time_asr_group"),
        },
        "pilot60_paired_comparison": json.loads((OUT / "paired_pilot60_comparison.json").read_text(encoding="utf-8")),
        "artifacts": {
            "per_sample_csv": "per_sample_error_table.csv",
            "per_sample_jsonl": "per_sample_error_table.jsonl",
            "official_scores": "btc-scores.jsonl",
            "official_summary": "official_summary.json",
            "traces": "step-traces.jsonl",
            "pilot60_pair": "paired_pilot60_comparison.json",
        },
    }
    (OUT / "error_taxonomy.json").write_text(json.dumps({
        "primary_cause_counts": summary["primary_cause_counts"],
        "cause_confidence_counts": summary["cause_confidence_counts"],
        "categories": {
            "protocol_runtime_failure": "exception, Ollama/ToolEnv failure or other runtime exception",
            "json_parse_or_extraction_failure": "model output did not contain a parseable JSON object",
            "json_action_protocol_failure": "JSON was present but did not contain an allowed action shape",
            "token_cap_truncation": "failed repair turn ended at the configured num_predict cap",
            "toolenv_schema_rejection": "ToolEnv authoritative schema rejected a submitted call",
            "schema_or_validator_repair_failure": "local action validation or repair failed for another schema reason",
            "missing_information_policy": "gold asks for missing information but prediction does not",
            "early_stop_or_under_call": "gold action trajectory exists but prediction emitted no call",
            "argument_grounding": "same tool selected but exact arguments differ; transcript effects remain a secondary hypothesis",
            "tool_selection_or_order": "tool identity differs or call sequence/state hash does not reconcile",
            "over_call_or_duplicate": "submitted calls remain unmatched after exact multiset matching",
            "no_observed_action_error": "exact multiset and state hash match, including valid no-action requests",
            "uncertain": "available trace does not support a more specific causal claim",
        },
        "causal_policy": "Primary categories describe directly observed failure layer. ASR mismatch, history, and failed observations are secondary evidence unless a causal link is directly established. Uncertain is retained instead of assigning recovered score.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "error_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"groups", "pilot60_paired_comparison"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
