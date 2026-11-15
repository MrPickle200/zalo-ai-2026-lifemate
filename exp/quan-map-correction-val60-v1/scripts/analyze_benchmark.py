from __future__ import annotations
import json, random
from pathlib import Path
import prepare_benchmark as prep

ROOT, OUT = prep.ROOT, prep.OUT
ANALYZER = ROOT / "exp/quan-continuation-cot-v1/scripts/analyze_downstream.py"

def main():
    lock = prep.read_json(OUT / "benchmark_lock.json")
    if lock.get("status") != "inference_complete": raise RuntimeError("Three inference branches are not complete")
    runner_module = prep.module("val60_analysis_runner", prep.NORMALIZATION_RUNNER)
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    analyzer = prep.module("quan_map_btc_analyzer", ANALYZER)
    analyzer.OUT = OUT
    cohort = prep.read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    data_root = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    outputs = {branch: analyzer.compute_branch(branch, ids, items, data_root)
               for branch in ("A", "B", "C")}
    predictions, scores, metrics, requests = {}, {}, {}, {}
    for branch in ("A", "B", "C"):
        predictions[branch], scores[branch], metrics[branch], requests[branch] = outputs[branch]
    rng = random.Random(20261109)
    paired = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        paired[f"{left}_vs_{right}"] = analyzer.pair_metrics(
            f"{left}-vs-{right}", ids, predictions[left], predictions[right], scores[left], scores[right],
            requests[left], requests[right], rng)
    transcript = prep.read_json(OUT / "transcript_analysis.json")
    report = {
        "cohort": "fixed enriched held-out val309 subset of 60 IDs; screening only",
        "branches": {
            "A": {"input": "raw QUAN epoch-06 transcript",
                  "WER_project": transcript["WER_project_normalization"]["A_QUAN_raw"],
                  "WER_number_value": transcript["WER_number_value_normalization"]["A_QUAN_raw"],
                  "BTC": metrics["A"]},
            "B": {"input": "QUAN epoch-06 transcript; Vietnamese number normalization then train-only map",
                  "WER_project": transcript["WER_project_normalization"]["B_number_normalization_plus_map"],
                  "WER_number_value": transcript["WER_number_value_normalization"]["B_number_normalization_plus_map"],
                  "BTC": metrics["B"]},
            "C": {"input": "original BTC reference script (oracle transcript)",
                  "WER_project": transcript["WER_project_normalization"]["C_BTC_script"],
                  "WER_number_value": transcript["WER_number_value_normalization"]["C_BTC_script"],
                  "BTC": metrics["C"]}},
        "map_accuracy": transcript["map_accuracy"],
        "paired_diagnostics": paired,
        "BTC_formula": "exact tool name and arguments; duplicate multiplicity; gold ToolEnv state-hash gate; M>0; unmasked formula",
        "runtime_gate_note": "Local 30s-gated score uses this host's latency and is a diagnostic, not an A30 timeout determination.",
        "limits": ["The 60-ID cohort was enriched for previous failures and is a screening cohort.",
                   "There was one Qwen run per branch; outputs can vary across repeats even at temperature 0.",
                   "Validation references were used only for WER, map accuracy, the oracle branch and BTC scoring; the map was built from train IDs only."]}
    prep.write_json(OUT / "benchmark_results.json", report)
    lines = ["# QUAN transcript map benchmark — val60", "",
        "Same fixed 60 held-out validation tasks, same Qwen SFT epoch-2 + CoT prompt, PhoASR history path, tool settings, and one prediction ToolEnv per task. Branch order was balanced across three 20-task blocks.", "",
        "## Results", "",
        "| Branch | Input transcript | Project WER | Number-value WER | BTC formula /100 | Local 30s-gated diagnostic | Hash matches | Runtime errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for branch in ("A", "B", "C"):
        row = report["branches"][branch]; btc = row["BTC"]
        lines.append(f"| {branch} | {row['input']} | {row['WER_project']['corpus_wer']:.4f} | {row['WER_number_value']['corpus_wer']:.4f} | {btc['btc_formula_score_before_runtime_gate_0_100']:.2f} | {btc['local_30s_gated_score_diagnostic_0_100']:.2f} | {btc['hash_matches']}/60 | {btc['runtime_errors']} |")
    m = report["map_accuracy"]
    precision = "n/a" if m["precision"] is None else f"{m['precision']:.1%}"
    lines += ["", "## Map correction accuracy", "",
        f"{m['exactly_correct']}/{m['applications']} exact corrections ({precision}) across {m['changed_utterances']} tasks. The Vietnamese number parser normalized {m['number_spans_normalized_before_map']} QUAN spans before map matching.",
        "", "## Paired BTC score deltas (right minus left)", "",
        "| Comparison | Mean points | 95% paired bootstrap CI | Improved / worsened / tied |",
        "|---|---:|---:|---:|"]
    for key, row in paired.items():
        d = row["paired_formula_delta_points_right_minus_left"]
        o = row["paired_outcomes_formula_score"]
        lines.append(f"| {key} | {d['mean']:+.2f} | [{d['ci95_low']:+.2f}, {d['ci95_high']:+.2f}] | {o.get('improved',0)} / {o.get('worsened',0)} / {o.get('tied',0)} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limits"]]
    lines += ["- Project WER retains number words; number-value WER parses both sides so equivalent word and digit forms are not counted as ASR errors.",
              "- BTC formula scores are unmasked. The local 30-second gate is reported separately as a host-specific diagnostic.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"branches": {b: {"WER_project": report["branches"][b]["WER_project"]["corpus_wer"],
        "WER_number_value": report["branches"][b]["WER_number_value"]["corpus_wer"],
        "BTC_formula": report["branches"][b]["BTC"]["btc_formula_score_before_runtime_gate_0_100"],
        "local_30s_gate": report["branches"][b]["BTC"]["local_30s_gated_score_diagnostic_0_100"]} for b in ("A","B","C")},
        "map_accuracy": m, "paired_deltas": {k:v["paired_formula_delta_points_right_minus_left"] for k,v in paired.items()}},
        ensure_ascii=True, indent=2))
    return 1 if any(metrics[b]["gold_replay_failures"] for b in ("A","B","C")) else 0

if __name__ == "__main__":
    raise SystemExit(main())

