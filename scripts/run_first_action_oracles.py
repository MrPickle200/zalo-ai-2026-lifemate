from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.pipeline import MAX_STEPS, LifeMatePipeline
from lifemate.tool_executor import available_tool_names
from lifemate.utils import find_data_root, load_jsonl, seed_everything


AUDIT_DIR = Path("outputs/eval/agent-architecture-audit-v1-val309")
BASELINE_DIR = Path("outputs/eval/phoasr-small-baseline-val309")
EXPERIMENTS = {
    "tool_name": Path("outputs/eval/oracle-first-tool-name-30"),
    "full_call": Path("outputs/eval/oracle-first-full-call-30"),
    "two_calls": Path("outputs/eval/oracle-first-two-calls-30"),
}


def call_set(calls):
    return {json.dumps({"name": c.get("name"), "arguments": c.get("arguments", {})},
                       sort_keys=True, ensure_ascii=False) for c in calls}


def score(item: dict, result: dict) -> dict:
    gold = item.get("label", {}).get("tool_calls", [])
    pred = result["tool_calls"]
    gs, ps = call_set(gold), call_set(pred)
    tp = len(gs & ps)
    precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    final_expected = item.get("label", {}).get("final_state", {})
    state_match = all(result["final_state"].get(app) == state for app, state in final_expected.items())
    return {"tool_call_f1": f1, "state_match": state_match,
            "proxy": (f1 + int(state_match)) / 2,
            "trajectory_exact": pred == gold,
            "first_tool_name_correct": bool(pred and gold and pred[0].get("name") == gold[0].get("name")),
            "first_full_call_correct": bool(pred and gold and pred[0] == gold[0])}


def aggregate(rows: list[dict], kind: str) -> dict:
    completed = [r for r in rows if r.get("metrics") is not None]
    tp = predicted = gold_calls = 0
    for row in completed:
        p, g = call_set(row["tool_calls"]), call_set(row["gold_tool_calls"])
        tp += len(p & g)
        predicted += len(p)
        gold_calls += len(g)
    f1 = 2 * tp / (predicted + gold_calls) if predicted + gold_calls else 0.0
    return {
        "experiment": {"tool-name": "oracle-first-tool-name-30", "full-call": "oracle-first-full-call-30",
                       "first-two-calls": "oracle-first-two-calls-30"}[kind],
        "intervention": ("Force the gold first tool name; Qwen generates its arguments, then all later turns use baseline policy."
                        if kind == "tool-name" else
                        "Execute the exact gold first two calls; all later turns use baseline policy."
                        if kind == "first-two-calls" else
                        "Execute the exact gold first call; all later turns use baseline policy."),
        "samples": len(rows), "runtime_failures": len(rows) - len(completed), "max_steps": MAX_STEPS,
        "tool_call_f1": f1,
        "final_state_match_rate": sum(r["metrics"]["state_match"] for r in completed) / len(completed) if completed else 0.0,
        "proxy_mean": sum(r["metrics"]["proxy"] for r in completed) / len(completed) if completed else 0.0,
        "first_tool_name_correct_count": sum(r["metrics"]["first_tool_name_correct"] for r in completed),
        "first_full_call_correct_count": sum(r["metrics"]["first_full_call_correct"] for r in completed),
        "trajectory_exact_count": sum(r["metrics"]["trajectory_exact"] for r in completed),
        "average_latency_seconds": sum(r.get("latency_seconds", 0.0) for r in completed) / len(completed) if completed else None,
        "predictions_file": "predictions.jsonl",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run first-action oracle diagnostics on the fixed 30-sample subset")
    parser.add_argument("--mode", choices=("all", "tool_name", "full_call", "two_calls"), default="all")
    parser.add_argument("--summaries-only", action="store_true",
                        help="rebuild paired summary from completed experiment summaries without model calls")
    args = parser.parse_args()
    seed_everything()
    root = find_data_root()
    ids = json.loads((AUDIT_DIR / "sample_ids.json").read_text(encoding="utf-8"))
    items = {r["id"]: r for r in load_jsonl(root / "train" / "train.jsonl")}
    baseline_asr = {r["id"]: r for r in load_jsonl(BASELINE_DIR / "predictions.jsonl")}
    audit = json.loads((AUDIT_DIR / "architecture-audit.json").read_text(encoding="utf-8"))
    expected_ids = {r["id"] for r in audit["samples"]}
    if set(ids) != expected_ids or any(sid not in items or sid not in baseline_asr for sid in ids):
        raise SystemExit("Diagnostic IDs do not match the audited set or baseline PhoASR predictions")

    catalog = json.loads((root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    tool_catalog = {tool["name"]: tool for app in catalog.get("apps", []) for tool in app.get("tools", [])}
    outputs = {}
    selected_modes = [] if args.summaries_only else list(EXPERIMENTS) if args.mode == "all" else [args.mode]
    for mode in selected_modes:
        outdir = EXPERIMENTS[mode]
        outdir.mkdir(parents=True, exist_ok=True)
        agent = Agent(max_new_tokens=128, context_limit=6144)
        pipeline = LifeMatePipeline(root, "train", agent=agent, max_steps=MAX_STEPS)
        rows = []
        outpath = outdir / "predictions.jsonl"
        with outpath.open("w", encoding="utf-8", newline="\n") as stream:
            for i, sample_id in enumerate(ids, 1):
                item = items[sample_id]
                gold = item.get("label", {}).get("tool_calls", [])
                if not gold:
                    raise ValueError(f"{sample_id}: selected diagnostic item has no gold first call")
                allowed = set(available_tool_names(item))
                if gold[0]["name"] not in allowed or gold[0]["name"] not in tool_catalog:
                    raise ValueError(f"{sample_id}: first gold tool is not installed/in catalog")
                transcript = baseline_asr[sample_id].get("asr_text", "")
                if not transcript:
                    raise ValueError(f"{sample_id}: baseline PhoASR transcript is empty")
                try:
                    kwargs = ({"forced_first_tool_name": gold[0]["name"]} if mode == "tool_name"
                              else {"injected_first_calls": [{"name": c["name"], "arguments": c.get("arguments", {})}
                                                             for c in gold[:2]]} if mode == "two_calls"
                              else {"injected_first_call": {"name": gold[0]["name"],
                                                             "arguments": gold[0].get("arguments", {})}})
                    result = pipeline.run(item, transcript_override=transcript, **kwargs)
                    metrics = score(item, result)
                    row = {"id": sample_id, "mode": mode, "asr_text": transcript,
                           "gold_first_call": gold[0], "gold_tool_calls": gold,
                           "tool_calls": result["tool_calls"], "model_output": result["model_output"],
                           "observations": result["observations"], "final_state": result["final_state"],
                           "state_hash": result["state_hash"], "metrics": metrics,
                           "latency_seconds": result["latency_seconds"],
                           "intervention": {"tool_name": "tool_name_only", "full_call": "full_first_call",
                                            "two_calls": "full_first_two_calls"}[mode]}
                except Exception as exc:
                    row = {"id": sample_id, "mode": mode, "asr_text": transcript,
                           "gold_first_call": gold[0], "gold_tool_calls": gold,
                           "tool_calls": [], "metrics": None,
                           "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[{mode} {i}/{len(ids)}] {sample_id}: {row['error']}", flush=True)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                print(f"[{mode} {i}/{len(ids)}] {sample_id} proxy="
                      f"{row.get('metrics', {}).get('proxy')} state={row.get('metrics', {}).get('state_match')}", flush=True)
        summary = aggregate(rows, {"tool_name": "tool-name", "full_call": "full-call",
                                   "two_calls": "first-two-calls"}[mode])
        (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs[mode] = rows

    comparison = {}
    for mode, rows in outputs.items():
        comparisons = Counter()
        for row in rows:
            if row.get("metrics") is None:
                continue
            old = next(r for r in audit["samples"] if r["id"] == row["id"])
            delta = row["metrics"]["proxy"] - (old.get("baseline_score") or 0.0)
            comparisons["improved" if delta > 1e-12 else "worsened" if delta < -1e-12 else "unchanged"] += 1
        comparison[mode] = dict(comparisons)
    paired_dir = Path("outputs/eval/paired-analysis/oracle-first-action")
    paired_dir.mkdir(parents=True, exist_ok=True)
    paired_path = paired_dir / "summary.json"
    if paired_path.is_file():
        old_paired = json.loads(paired_path.read_text(encoding="utf-8"))
        old_outcomes = old_paired.get("paired_proxy_outcomes_vs_baseline", {})
        old_outcomes.update(comparison)
        comparison = old_outcomes
    oracle_a = json.loads((EXPERIMENTS["tool_name"] / "summary.json").read_text(encoding="utf-8"))
    oracle_b = json.loads((EXPERIMENTS["full_call"] / "summary.json").read_text(encoding="utf-8"))
    oracle_c = json.loads((EXPERIMENTS["two_calls"] / "summary.json").read_text(encoding="utf-8"))
    paired = {"baseline": {"first_action_accuracy": audit["baseline_exact_first_action_count"] / len(ids),
                            "tool_call_f1": audit["baseline_tool_call_f1"],
                            "final_state_match_rate": audit["baseline_state_match_rate"],
                            "trajectory_exact_rate": audit["baseline_exact_trajectory_count"] / len(ids),
                            "proxy_mean": audit["baseline_composite_proxy_mean"]},
              "oracle_tool_name": oracle_a,
              "oracle_full_first_call": oracle_b,
              "oracle_first_two_calls": oracle_c,
              "metric_deltas": {
                  "oracle_full_first_call_minus_tool_name": {
                      key: oracle_b[bkey] - oracle_a[akey] for key, bkey, akey in (
                          ("tool_call_f1", "tool_call_f1", "tool_call_f1"),
                          ("final_state_match_rate", "final_state_match_rate", "final_state_match_rate"),
                          ("proxy_mean", "proxy_mean", "proxy_mean"))},
                  "oracle_first_two_calls_minus_full_first_call": {
                      key: oracle_c[bkey] - oracle_b[bkey] for key, bkey in (
                          ("tool_call_f1", "tool_call_f1"),
                          ("final_state_match_rate", "final_state_match_rate"),
                          ("proxy_mean", "proxy_mean"))}},
              "paired_proxy_outcomes_vs_baseline": comparison}
    paired_path.write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(paired, ensure_ascii=False, indent=2))
    return 0 if all(json.loads((EXPERIMENTS[m] / "summary.json").read_text(encoding="utf-8"))["runtime_failures"] == 0
                    for m in selected_modes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
