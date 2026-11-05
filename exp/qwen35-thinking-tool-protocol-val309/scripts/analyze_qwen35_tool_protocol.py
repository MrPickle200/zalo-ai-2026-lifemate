from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("exp/qwen35-thinking-tool-protocol-val309/output")
OUT = Path("exp/paired-analysis/qwen35-thinking-tool-protocol-val309/output")
BRANCHES = {"A": "A-nothink-json", "B": "B-nothink-native", "C": "C-think-json", "D": "D-think-native"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores, preds = {}, {}
    available = {}
    for branch, folder in BRANCHES.items():
        base = ROOT / folder / "val309"
        if (base / "official_scores.jsonl").is_file() and (base / "official_summary.json").is_file():
            available[branch] = folder
            scores[branch] = {r["id"]: r for r in read_jsonl(base / "official_scores.jsonl")}
            preds[branch] = {r["id"]: r for r in read_jsonl(base / "predictions.jsonl")}
    ids = json.loads((ROOT / "val309_ids.json").read_text(encoding="utf-8"))
    if not {"A", "B"}.issubset(available) or any(set(scores[b]) != set(ids) for b in available):
        raise ValueError("Each available full branch must have exactly the fixed val309 score rows")
    arrays = {b: np.array([scores[b][sid]["official_sample_score"] for sid in ids], dtype=float) for b in available}
    contrasts = {"B_minus_A_native_no_think": arrays["B"] - arrays["A"]}
    if "C" in available:
        contrasts["C_minus_A_thinking_json"] = arrays["C"] - arrays["A"]
    if "D" in available and "C" in available:
        contrasts["D_minus_C_native_think"] = arrays["D"] - arrays["C"]
    if "D" in available:
        contrasts["D_minus_B_thinking_native"] = arrays["D"] - arrays["B"]
    if {"A", "B", "C", "D"}.issubset(available):
        contrasts["interaction_(D-C)-(B-A)"] = (arrays["D"] - arrays["C"]) - (arrays["B"] - arrays["A"])
    rng = np.random.default_rng(20261102)
    boot_idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    report = {"cohort": "fixed train-only val309", "matching": "exact tool name and arguments; duplicate calls by multiplicity; BTC hash gate",
              "bootstrap": {"method": "paired sample bootstrap with replacement", "replicates": 10000, "seed": 20261102},
              "branches": {}, "paired_differences": {},
              "historical_references": {"Qwen3 same prompt": 3.0868, "PhoASR/Qwen3 prior baseline": 4.9145},
              "smoke_12_diagnostics": {b: json.loads((ROOT / folder / "smoke" / "summary.json").read_text(encoding="utf-8"))
                                       for b, folder in BRANCHES.items()
                                       if (ROOT / folder / "smoke" / "summary.json").is_file()},
              "full_val309_variants_missing": [b for b in BRANCHES if b not in available]}
    for b in available:
        summary = json.loads((ROOT / available[b] / "val309" / "official_summary.json").read_text(encoding="utf-8"))
        report["branches"][b] = summary
    for name, values in contrasts.items():
        means = values[boot_idx].mean(axis=1)
        report["paired_differences"][name] = {"mean_points": float(values.mean() * 100),
            "ci95_points": [float(np.quantile(means, .025) * 100), float(np.quantile(means, .975) * 100)],
            "samples_positive": int((values > 0).sum()), "samples_negative": int((values < 0).sum()),
            "samples_tied": int((values == 0).sum())}

    # Fixed gold cohort metadata for subgroup diagnostics.
    import sys
    sys.path.insert(0, "src")
    from lifemate.utils import find_data_root, load_jsonl
    data_root = find_data_root(None)
    items = {x["id"]: x for x in load_jsonl(data_root / "train" / "train.jsonl")}
    strata = {}
    length_groups = defaultdict(list)
    missing_groups = defaultdict(list)
    groups = Counter()
    for sid in ids:
        gold = items[sid].get("label", {}).get("tool_calls", [])
        length = "0" if len(gold) == 0 else "1" if len(gold) == 1 else "2" if len(gold) == 2 else "3+"
        missing = any(c.get("name") == "sys.ask_missing_info" for c in gold)
        key = f"gold_calls={length};ask_missing_info={str(missing).lower()}"
        groups[key] += 1
        strata.setdefault(key, []).append(sid)
        length_groups[length].append(sid)
        missing_groups[str(missing).lower()].append(sid)
    report["subgroups"] = {}
    for key, group_ids in strata.items():
        indices = np.array([ids.index(sid) for sid in group_ids])
        report["subgroups"][key] = {"n": len(group_ids), "mean_official_score_points": {
            b: float(arrays[b][indices].mean() * 100) for b in available}}
    report["subgroups"]["marginal_gold_trajectory_length"] = {}
    for key, group_ids in length_groups.items():
        indices = np.array([ids.index(sid) for sid in group_ids])
        report["subgroups"]["marginal_gold_trajectory_length"][key] = {"n": len(group_ids),
            "mean_official_score_points": {b: float(arrays[b][indices].mean() * 100) for b in available}}
    report["subgroups"]["marginal_ask_missing_info"] = {}
    for key, group_ids in missing_groups.items():
        indices = np.array([ids.index(sid) for sid in group_ids])
        report["subgroups"]["marginal_ask_missing_info"][key] = {"n": len(group_ids),
            "mean_official_score_points": {b: float(arrays[b][indices].mean() * 100) for b in available}}

    disagreements = []
    for i, sid in enumerate(ids):
        vals = {b: float(arrays[b][i] * 100) for b in available}
        if len(set(vals.values())) < 2:
            continue
        disagreements.append({"id": sid, "score_range_points": max(vals.values()) - min(vals.values()),
            "scores": vals, "transcript": preds["A"][sid].get("transcript_used", ""),
            "gold_calls": items[sid].get("label", {}).get("tool_calls", []),
            "branches": {b: {"tool_calls": preds[b][sid].get("tool_calls", []),
                             "observations": preds[b][sid].get("observations", [])} for b in available}})
    report["paired_disagreement_examples"] = sorted(disagreements, key=lambda r: (-r["score_range_points"], r["id"]))[:12]
    report["conclusion_threshold"] = {"go_minimum_gain_points": 0.5, "requires_95pct_ci_excluding_zero": True,
                                       "must_not_increase_schema_context_runtime_errors": True,
                                       "production_default_changed": False,
                                       "full_factorial_interaction_estimable": "D" in available}
    (OUT / "paired-bootstrap-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"branches": {b: x["official_score_0_100"] for b, x in report["branches"].items()},
                      "paired_differences": report["paired_differences"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
