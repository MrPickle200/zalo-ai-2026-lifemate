from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("exp/qwen35-thinking-tool-protocol-val309/output")
FOLDERS = {"A": "A-nothink-json", "B": "B-nothink-native", "C": "C-think-json", "D": "D-think-native"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=FOLDERS, required=True)
    args = parser.parse_args()
    ids = json.loads((ROOT / "val309_ids.json").read_text(encoding="utf-8"))
    out = ROOT / FOLDERS[args.variant] / "val309"
    pred = rows(out / "predictions.jsonl")
    diag = rows(out / "diagnostics.jsonl")
    completed = [row["id"] for row in pred]
    if completed != ids[:len(completed)] or len(diag) != len(pred):
        raise ValueError("Partial artifacts are not a prefix-aligned cohort")
    latency = sorted(float(row.get("latency_seconds", 0)) for row in pred)
    turns = [turn for row in diag for turn in row.get("turns", [])]
    summary = {
        "status": "partial; not a full val309 evaluation",
        "variant": args.variant,
        "cohort": "prefix of fixed train-only val309",
        "completed_samples": len(pred),
        "total_cohort_samples": len(ids),
        "remaining_samples": len(ids) - len(pred),
        "last_completed_id": completed[-1] if completed else None,
        "next_sample_id": ids[len(completed)] if len(completed) < len(ids) else None,
        "resume_note": "The next ID has no committed prediction row. The stop may have arrived after its ToolEnv/request started; --resume reruns it from scratch with a fresh ToolEnv, so preserve this partial attempt and do not treat it as a complete scored trajectory.",
        "official_btc_score": None,
        "reason": "User requested an early stop; no full-cohort BTC score or paired inference is reported.",
        "runtime_failures": sum("error" in row for row in pred),
        "mean_latency_seconds": sum(latency) / len(latency) if latency else None,
        "p95_latency_seconds": latency[min(len(latency)-1, int(.95 * len(latency)))] if latency else None,
        "max_latency_seconds": max(latency) if latency else None,
        "samples_over_30s": sum(value > 30 for value in latency),
        "turns": len(turns),
        "ollama_requests": sum(turn.get("request_count", 1) for turn in turns),
        "prompt_eval_tokens_recorded": sum((turn.get("prompt_eval_count") or 0) + (turn.get("initial_prompt_eval_count") or 0) for turn in turns),
        "eval_tokens_recorded": sum((turn.get("eval_count") or 0) + (turn.get("initial_eval_count") or 0) for turn in turns),
        "context_failures": sum(row.get("context_failures", 0) for row in diag),
        "schema_rejections": sum(row.get("schema_rejections", 0) for row in diag),
        "schema_repair_turns": sum(row.get("schema_repair_turns", 0) for row in diag),
        "parse_schema_failures": sum(row.get("parse_schema_failures", 0) for row in diag),
        "native_overcalls": sum(row.get("native_overcalls", 0) for row in diag),
        "stop_early_samples": sum(row.get("stop_early", 0) for row in diag),
        "mean_action_calls_per_sample": sum(row.get("action_count", 0) for row in diag) / len(diag) if diag else None,
        "diagnostic_files": ["predictions.jsonl", "diagnostics.jsonl", "../../C-run.log"],
    }
    (out / "partial_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
