from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.pipeline import LifeMatePipeline
from lifemate.asr import ASRModel, PhoASRModel
from lifemate.retrieval import DEFAULT_TOP_K, MAX_CANDIDATE_TOOLS, MIN_CONFIDENCE
from lifemate.utils import find_data_root, load_jsonl, seed_everything

ERROR_TYPES = ("ASR_SLOT_ERROR", "WRONG_TOOL", "WRONG_ARGUMENT", "PLANNING_ERROR",
               "MISSING_INFO_ERROR", "OUT_OF_SCOPE_ERROR", "OTHER")


def call_set(calls):
    return {json.dumps({"name": c.get("name"), "arguments": c.get("arguments", {})}, sort_keys=True, ensure_ascii=False)
            for c in calls}


def classify_error(gold, pred, asr_text: str, transcript: str, oracle_correct: bool = False) -> str:
    if oracle_correct:
        return "ASR_SLOT_ERROR"
    gold_names = [x.get("name") for x in gold]
    pred_names = [x.get("name") for x in pred]
    if not gold and pred:
        return "OUT_OF_SCOPE_ERROR"
    if "sys.ask_missing_info" in gold_names or "sys.ask_missing_info" in pred_names:
        return "MISSING_INFO_ERROR"
    if Counter(gold_names) != Counter(pred_names):
        return "WRONG_TOOL"
    if pred_names != gold_names:
        return "PLANNING_ERROR"
    if call_set(gold) != call_set(pred):
        return "WRONG_ARGUMENT"
    # Remaining mismatches are state effects caused by an omitted, extra, or semantically bad call.
    return "OTHER"


def metrics(rows):
    completed = [r for r in rows if r.get("score") is not None]
    tp = sum(r["tp"] for r in completed)
    predicted = sum(r["predicted_calls"] for r in completed)
    gold = sum(r["gold_calls"] for r in completed)
    precision = tp / predicted if predicted else (1.0 if not gold else 0.0)
    recall = tp / gold if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"samples_run": len(completed), "runtime_failures": len(rows) - len(completed),
            "tool_call_f1": f1,
            "tool_call_exact_match_rate": sum(r["exact_match"] for r in completed) / len(completed) if completed else None,
            "state_match_rate": sum(r["state_match"] for r in completed) / len(completed) if completed else None,
            "composite_proxy_mean": sum(r["score"] for r in completed) / len(completed) if completed else None,
            "average_latency_seconds": sum(r.get("latency_seconds", 0.0) for r in completed) / len(completed) if completed else None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a fixed train-only holdout through the local ToolEnv pipeline")
    parser.add_argument("--mode", choices=("baseline", "oracle", "retrieval", "soft-retrieval", "curated-fewshot"), default="baseline")
    parser.add_argument("--limit", type=int, help="optional prefix of the fixed validation IDs")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--asr", choices=("whisper-small", "phoasr-whisper-small"), default="whisper-small")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output-root", default="outputs/eval")
    parser.add_argument("--experiment-name", help="override the default output experiment directory")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="number of retrieved examples used to vote for candidate tools")
    parser.add_argument("--max-candidate-tools", type=int, default=MAX_CANDIDATE_TOOLS)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.offset < 0 or args.top_k < 1 or args.max_candidate_tools < 1:
        parser.error("limit, top-k, candidate count, and offset must be positive (offset may be zero)")
    seed_everything()
    root = find_data_root(args.data_root)
    items = list(load_jsonl(root / "train" / "train.jsonl"))
    by_id = {item["id"]: item for item in items}
    split_dir = Path(args.split_dir)
    val_ids = json.loads((split_dir / "val_ids.json").read_text(encoding="utf-8"))
    ids = val_ids[args.offset: args.offset + args.limit if args.limit is not None else None]
    if not ids:
        raise SystemExit("No validation samples selected")
    retrieval_on = args.mode in {"retrieval", "soft-retrieval", "curated-fewshot"}
    asr = PhoASRModel() if args.asr == "phoasr-whisper-small" else ASRModel()
    pipeline = LifeMatePipeline(root, "train", asr=asr, retrieval_assisted=retrieval_on,
                                retrieval_top_k=args.top_k,
                                retrieval_max_candidate_tools=args.max_candidate_tools,
                                soft_retrieval=args.mode == "soft-retrieval",
                                curated_fewshot=args.mode == "curated-fewshot")
    rows = []
    output_root = Path(args.output_root)
    experiment = args.experiment_name or ({"phoasr-whisper-small": "phoasr-small-baseline-val309"}.get(args.asr) if args.asr != "whisper-small" and args.mode == "baseline" else None) or {"baseline": "baseline-v1-val309", "oracle": "oracle-asr-v1-val309",
                                          "retrieval": "retrieval-v1-val309", "soft-retrieval": "soft-retrieval-v1-val309",
                                          "curated-fewshot": "curated-fewshot-v1-val309"}[args.mode]
    outdir = output_root / experiment
    outdir.mkdir(parents=True, exist_ok=True)
    prediction_path = outdir / "predictions.jsonl"
    prediction_stream = prediction_path.open("w", encoding="utf-8")
    for i, sample_id in enumerate(ids, 1):
        item = by_id[sample_id]
        expected = item.get("label", {}).get("tool_calls", [])
        transcript = item.get("audio_text", "") if args.mode == "oracle" else None
        try:
            result = pipeline.run(item, transcript_override=transcript)
            pred = result["tool_calls"]
            ps, gs = call_set(pred), call_set(expected)
            tp = len(ps & gs)
            expected_state = item.get("label", {}).get("final_state", {})
            state_match = all(result["final_state"].get(app) == state for app, state in expected_state.items())
            precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
            recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            exact = pred == expected
            row = {"id": sample_id, "asr_text": result["asr_text"], "model_output": result["model_output"],
                   "transcript_used": result["input_transcript"], "tool_calls": pred,
                   "score": (f1 + int(state_match)) / 2, "tool_call_f1": f1,
                   "exact_match": exact, "state_match": state_match,
                   "predicted_calls": len(ps), "gold_calls": len(gs), "tp": tp,
                   "latency_seconds": result["latency_seconds"], "state_hash": result["state_hash"],
                   "final_state": result["final_state"], "observations": result["observations"],
                   "retrieved_examples": result["retrieved_examples"],
                   "candidate_tools": result["candidate_tools"],
                   "retrieval_confidence": result["retrieval_confidence"]}
        except Exception as exc:
            row = {"id": sample_id, "asr_text": "", "model_output": None, "transcript_used": transcript,
                   "tool_calls": [], "score": None, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{i}/{len(ids)}] {sample_id}: {row['error']}", file=sys.stderr)
        rows.append(row)
        prediction_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        prediction_stream.flush()
        print(f"[{i}/{len(ids)}] {sample_id} score={row.get('score')} exact={row.get('exact_match')} state={row.get('state_match')}", flush=True)
    prediction_stream.close()
    # Error attribution uses paired baseline/oracle outputs when available; the oracle comparison is descriptive only.
    oracle_rows_path = output_root / "oracle-asr-v1-val309" / "predictions.jsonl"
    oracle_correct = {}
    if args.mode in {"baseline", "retrieval", "soft-retrieval", "curated-fewshot"} and oracle_rows_path.is_file():
        for line in oracle_rows_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            oracle_correct[r["id"]] = bool(r.get("exact_match") and r.get("state_match"))
    error_counts = Counter()
    for row in rows:
        if row.get("score") is not None and row["score"] < 1:
            item = by_id[row["id"]]
            typ = classify_error(item.get("label", {}).get("tool_calls", []), row.get("tool_calls", []),
                                 row.get("asr_text", ""), item.get("audio_text", ""), oracle_correct.get(row["id"], False))
            row["error_type"] = typ
            error_counts[typ] += 1
    report = {"mode": args.mode, "asr": args.asr, "split": "train", "validation_samples": len(ids), **metrics(rows),
              "error_counts": {t: error_counts[t] for t in ERROR_TYPES},
              "error_percent_of_incorrect": {t: error_counts[t] / sum(error_counts.values()) if sum(error_counts.values()) else 0 for t in ERROR_TYPES},
              "metric_note": "composite_proxy_mean = mean((per-sample tool-call F1 + exact labeled final-state match)/2); official scorer formula not available in the local kit",
              "predictions_file": "predictions.jsonl"}
    if retrieval_on:
        report["retrieval_config"] = {"method": "word and adjacent word-bigram TF-IDF cosine",
                                      "corpus": "train_ids.json only", "top_k": args.top_k,
                                      "few_shot_mode": args.mode,
                                      "full_tool_set": args.mode in {"soft-retrieval", "curated-fewshot"},
                                      "max_candidate_tools": (None if args.mode in {"soft-retrieval", "curated-fewshot"}
                                                              else args.max_candidate_tools),
                                      "min_similarity": MIN_CONFIDENCE,
                                      "fallback": "all tools available for sample",
                                      "fallback_samples": sum(not row.get("candidate_tools") for row in rows),
                                      "average_shortlist_size": sum(len(row.get("candidate_tools") or []) for row in rows) / len(rows),
                                      "few_shot_count": min(DEFAULT_TOP_K, args.top_k),
                                      "missing_info_demos": [d["category"] for d in pipeline.retriever.missing_info_demos]}
    (outdir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["runtime_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
