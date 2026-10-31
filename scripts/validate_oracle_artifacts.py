from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/eval")
BASELINE = ROOT / "phoasr-small-baseline-val309" / "predictions.jsonl"
MODES = {
    "tool_name": ("oracle-first-tool-name-30", 0),
    "full_call": ("oracle-first-full-call-30", 1),
    "two_calls": ("oracle-first-two-calls-30", 2),
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    baseline = {r["id"]: r for r in read_rows(BASELINE)}
    report = {}
    for mode, (dirname, inject_count) in MODES.items():
        outdir = ROOT / dirname
        rows = read_rows(outdir / "predictions.jsonl")
        summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
        transcript_match = all(r.get("asr_text") == baseline.get(r["id"], {}).get("asr_text") for r in rows)
        if mode == "tool_name":
            injection_correct = sum(bool(r["tool_calls"] and
                                         r["tool_calls"][0].get("name") == r["gold_first_call"].get("name"))
                                    for r in rows)
            actual_observation_prefix = True
        else:
            injection_correct = sum(r.get("tool_calls", [])[:inject_count] ==
                                     r.get("gold_tool_calls", [])[:inject_count] for r in rows)
            actual_observation_prefix = all(
                [o.get("tool") for o in r.get("observations", [])[:inject_count]] ==
                [c.get("name") for c in r.get("gold_tool_calls", [])[:inject_count]]
                for r in rows
            )
        report[mode] = {"rows": len(rows), "runtime_failures": summary["runtime_failures"],
                        "transcripts_match_baseline": transcript_match,
                        "first_intervention_correct": f"{injection_correct}/{len(rows)}",
                        "injected_calls_executed_and_observed": actual_observation_prefix,
                        "state_matches": int(summary["final_state_match_rate"] * len(rows))}
        if len(rows) != 30 or summary["runtime_failures"] or not transcript_match or not actual_observation_prefix:
            raise SystemExit(f"Artifact validation failed for {mode}: {report[mode]}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
