from __future__ import annotations
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp/quan-map-correction-val60-v1/scripts"))
import prepare_benchmark as prep

EXP = ROOT / "exp/quan-map-correction-val309-v1"
OUT = EXP / "output" / "benchmark"
ANALYZER = ROOT / "exp/quan-continuation-cot-v1/scripts/analyze_downstream.py"
BRANCHES = ("A", "B", "C", "D")
SEED = 20261110

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    lock_path = OUT / "benchmark_lock.json"
    lock = read_json(lock_path)
    run_meta = read_json(OUT / "inference_run.json")
    if lock.get("status") not in {"inference_complete", "complete"} or run_meta.get("status") != "complete":
        raise RuntimeError("All four val309 inference branches must be complete before analysis")
    cohort = read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    if len(ids) != 309 or cohort.get("is_entire_split") is not True:
        raise RuntimeError("Refusing to analyze anything other than all 309 val IDs")

    analyzer = prep.module("quan_map_val309_analyzer", ANALYZER)
    analyzer.OUT = OUT
    analyzer.BOOTSTRAPS = 20000
    analyzer.BOOTSTRAP_SEED = SEED
    runner_module = prep.module("quan_map_val309_analysis_runner",
        ROOT / "exp/quan-transcript-normalization-val60/scripts/run_benchmark.py")
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    runner.OUT = OUT
    data_root = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}

    results = {}
    for branch in BRANCHES:
        results[branch] = analyzer.compute_branch(branch, ids, items, data_root)
    predictions, scores, metrics, requests = {}, {}, {}, {}
    for branch in BRANCHES:
        predictions[branch], scores[branch], metrics[branch], requests[branch] = results[branch]

    transcript = read_json(OUT / "transcript_analysis.json")
    runtime_failures_by_branch = {}
    for branch in BRANCHES:
        trace_rows = [json.loads(line) for line in (OUT / f"step-traces-{branch}.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        runtime_failures_by_branch[branch] = [row["id"] for row in trace_rows if row.get("runtime_error")]
    run_meta["ollama_version_segments"] = lock.get("ollama_version_segments", [])
    run_meta["qwen_model_digest"] = lock["qwen"]["digest"]
    run_meta["runtime_failure_ids_by_branch"] = runtime_failures_by_branch
    write_json(OUT / "inference_run.json", run_meta)
    report = {
        "split": "entire held-out val split (309/309 IDs)",
        "cohort_ids_sha256_newline_joined": cohort["ids_sha256_newline_joined"],
        "branches": {},
        "ollama_version_segments": lock.get("ollama_version_segments", []),
        "qwen_model_digest": lock["qwen"]["digest"],
        "runtime_failure_ids_by_branch": runtime_failures_by_branch,
        "map_accuracy": transcript["map_accuracy"],
        "paired_diagnostics": {},
        "paired_bootstrap": {"iterations": analyzer.BOOTSTRAPS, "seed": SEED},
        "BTC_formula": "exact tool name and arguments; duplicate multiplicity; gold ToolEnv state-hash gate; M>0; unmasked formula",
        "runtime_gate_note": "Local 30s-gated score uses local host latency and is a diagnostic, not an A30 timeout determination.",
        "limits": [
            "This is the complete 309-ID held-out validation set; the enriched val60 benchmark was a screening subset.",
            "One Qwen run per branch; deterministic temperature 0 does not eliminate all run variance.",
            "The map rules were frozen from train-only candidates before this comparison; val labels are used only for WER, map audit, the oracle branch and BTC evaluation.",
        ],
    }
    input_descriptions = {
        "A": "raw QUAN epoch-06 transcript",
        "B": "QUAN epoch-06 transcript; Vietnamese number normalization then frozen train-only map",
        "C": "original BTC reference script (oracle transcript)",
        "D": "raw QUAN-transcribe continuation checkpoint, epoch 02 (two epochs)",
    }
    for branch in BRANCHES:
        report["branches"][branch] = {
            "input": input_descriptions[branch],
            "WER_project": transcript["WER_project_normalization"][branch],
            "WER_number_value": transcript["WER_number_value_normalization"][branch],
            "BTC": metrics[branch],
        }

    rng = random.Random(SEED)
    comparisons = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
    for left, right in comparisons:
        label = f"{left}-vs-{right}"
        report["paired_diagnostics"][f"{left}_vs_{right}"] = analyzer.pair_metrics(
            label, ids, predictions[left], predictions[right], scores[left], scores[right],
            requests[left], requests[right], rng)

    write_json(OUT / "benchmark_results.json", report)
    version_note = "; ".join(f"rounds {segment['rounds']}: Ollama {segment['version']}" for segment in lock.get("ollama_version_segments", []))
    lines = [
        "# QUAN transcript map comparison — full val309", "",
        "All 309 held-out validation IDs. Same Qwen SFT epoch-2 + CoT model, prompt, PhoASR history path, ToolEnv setup, and action limit for all branches. A/B/C use the epoch-06 source (C is the original BTC script oracle); D uses the QUAN-transcribe epoch-02 continuation. The four branch orders rotate across 20-ID blocks.",
        f"Ollama versions: {version_note}. The Qwen model digest is unchanged, and all four branches for each ID block used the same version.",
        "Runtime failure: train_1725 exceeded the configured 8,192-token context in all four branches (8,236–8,239 tokens); no tool calls were generated, and its official BTC sample score is 0 in each branch.", "",
        "## Results", "",
        "| Branch | Transcript input | Project WER | Number-value WER | BTC formula /100 | Local 30s diagnostic | Hash matches | Runtime errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        row = report["branches"][branch]
        btc = row["BTC"]
        lines.append(f"| {branch} | {row['input']} | {row['WER_project']['corpus_wer']:.4f} | {row['WER_number_value']['corpus_wer']:.4f} | {btc['btc_formula_score_before_runtime_gate_0_100']:.2f} | {btc['local_30s_gated_score_diagnostic_0_100']:.2f} | {btc['hash_matches']}/309 | {btc['runtime_errors']} |")

    map_result = report["map_accuracy"]
    precision = "n/a" if map_result["precision"] is None else f"{map_result['precision']:.1%}"
    lines += [
        "", "## Train-map correction audit", "",
        f"{map_result['exactly_correct']}/{map_result['applications']} exact corrections ({precision}) across {map_result['changed_utterances']} tasks. The Vietnamese number parser normalized {map_result['number_spans_normalized_before_map']} epoch-06 transcript spans before map matching.",
        "", "## Paired BTC score differences (right minus left)", "",
        "| Comparison | Mean points | 95% paired bootstrap CI | Improved / worsened / tied |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["paired_diagnostics"].items():
        delta = row["paired_formula_delta_points_right_minus_left"]
        outcomes = row["paired_outcomes_formula_score"]
        lines.append(f"| {key} | {delta['mean']:+.2f} | [{delta['ci95_low']:+.2f}, {delta['ci95_high']:+.2f}] | {outcomes.get('improved',0)} / {outcomes.get('worsened',0)} / {outcomes.get('tied',0)} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limits"]]
    lines += [
        "- Project WER retains number words; number-value WER parses both references and transcripts, so equivalent word and digit forms are not counted as ASR errors.",
        "- BTC formula scores are unmasked. The local 30-second gate is reported separately as a host-specific diagnostic.",
        "- BTC matching uses exact tool name and arguments with duplicate multiplicity, gold ToolEnv state-hash gate, and M>0.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    readme = f"""# QUAN transcript map comparison — full val309

## Scope

This is a paired comparison on all 309 held-out validation IDs, with one Qwen SFT epoch-2 + CoT run per branch:

- **A:** raw QUAN epoch-06 transcript.
- **B:** epoch-06 transcript, Vietnamese number normalization, then the frozen train-only exact phrase map.
- **C:** original BTC reference script as an oracle transcript.
- **D:** raw QUAN-transcribe continuation checkpoint at epoch 02.

All branches share the same Qwen model/prompt, PhoASR history, ToolEnv configuration, and 5-action limit. Branch order rotates across 20-ID blocks. The map rules were frozen using train-only candidates; validation labels were not used to select rules.

Ollama versions: {version_note}. The Qwen model digest is unchanged, and all four branches for each ID block used the same version. The same task, train_1725, exceeded the configured 8,192-token context in all four branches (8,236–8,239 tokens); no tool calls were generated, and its official BTC sample score is 0 in each branch.

## Results

| Branch | Project WER | Number-value WER | BTC formula /100 | Local 30s diagnostic |
|---|---:|---:|---:|---:|
| A — epoch-06 QUAN raw | {transcript['WER_project_normalization']['A']['corpus_wer']:.4f} | {transcript['WER_number_value_normalization']['A']['corpus_wer']:.4f} | {metrics['A']['btc_formula_score_before_runtime_gate_0_100']:.2f} | {metrics['A']['local_30s_gated_score_diagnostic_0_100']:.2f} |
| B — epoch-06 + number parser/map | {transcript['WER_project_normalization']['B']['corpus_wer']:.4f} | {transcript['WER_number_value_normalization']['B']['corpus_wer']:.4f} | {metrics['B']['btc_formula_score_before_runtime_gate_0_100']:.2f} | {metrics['B']['local_30s_gated_score_diagnostic_0_100']:.2f} |
| C — BTC oracle script | {transcript['WER_project_normalization']['C']['corpus_wer']:.4f} | {transcript['WER_number_value_normalization']['C']['corpus_wer']:.4f} | {metrics['C']['btc_formula_score_before_runtime_gate_0_100']:.2f} | {metrics['C']['local_30s_gated_score_diagnostic_0_100']:.2f} |
| D — QUAN epoch 02 raw | {transcript['WER_project_normalization']['D']['corpus_wer']:.4f} | {transcript['WER_number_value_normalization']['D']['corpus_wer']:.4f} | {metrics['D']['btc_formula_score_before_runtime_gate_0_100']:.2f} | {metrics['D']['local_30s_gated_score_diagnostic_0_100']:.2f} |

Train-map audit: {map_result['exactly_correct']}/{map_result['applications']} exact corrections ({precision}) across {map_result['changed_utterances']} IDs.

## Reproduction

Run from the project root with native system Python outside the virtual environment:

1. python exp/quan-map-correction-val309-v1/scripts/prepare_val309.py
2. python exp/quan-map-correction-val309-v1/scripts/run_val309.py
3. python exp/quan-map-correction-val309-v1/scripts/analyze_val309.py

The run script resumes completed blocks and staged per-ID work. Inference artifacts, per-sample BTC scores, paired bootstrap outputs, source hashes and run progress are under output/benchmark/.
"""
    (EXP / "README.md").write_text(readme, encoding="utf-8")

    lock_path = OUT / "benchmark_lock.json"
    lock = read_json(lock_path)
    lock["status"] = "complete"
    lock["analysis_completed_epoch"] = __import__("time").time()
    lock["analysis_seed"] = SEED
    lock["paired_bootstrap_iterations"] = analyzer.BOOTSTRAPS
    tracked = [
        "cohort_manifest.json", "execution_order.json", "transcript_analysis.json",
        "branch_transcripts.jsonl", "map_applications.jsonl", "gold_references.jsonl",
        "inference_run.json", "run-progress.json", "benchmark_results.json", "summary.md",
    ]
    for branch in BRANCHES:
        tracked += [
            f"predictions-{branch}.jsonl", f"step-traces-{branch}.jsonl",
            f"requests-{branch}.jsonl", f"run-{branch}.json", f"btc-per-sample-{branch}.jsonl",
            f"metrics-{branch}.json",
        ]
    for left, right in comparisons:
        tracked.append(f"paired-{left}-vs-{right}.jsonl")
    for rel in tracked:
        full = OUT / rel
        if full.exists():
            lock.setdefault("completed_output_hashes", {})[rel] = prep.sha(full)
    for script in ("prepare_val309.py", "run_val309.py"):
        key = f"exp/quan-map-correction-val309-v1/scripts/{script}"
        lock["source_hashes"][key] = prep.sha(ROOT / key)
    key = "exp/quan-map-correction-val309-v1/scripts/analyze_val309.py"
    lock["source_hashes"][key] = prep.sha(Path(__file__))
    write_json(lock_path, lock)

    insight_header = "## QUAN transcript map comparison on full val309 (2026-11-10)"
    a_d = report["paired_diagnostics"]["A_vs_D"]["paired_formula_delta_points_right_minus_left"]
    insight = f"""
{insight_header}

- Ran all four transcript inputs on all 309 held-out validation IDs using the same Qwen SFT epoch-2 + CoT setup. A/B use epoch-06 QUAN transcripts (B normalizes Vietnamese number phrases before the frozen train-only map), C uses the BTC-provided reference script as an oracle, and D uses the raw QUAN-transcribe epoch-02 continuation.
- Full metrics, map audit and paired bootstrap comparisons are recorded at exp/quan-map-correction-val309-v1/output/benchmark/summary.md. On this full split, map precision was {map_result['exactly_correct']}/{map_result['applications']} ({precision}); paired D−A BTC difference was {a_d['mean']:+.2f} points (95% CI [{a_d['ci95_low']:+.2f},{a_d['ci95_high']:+.2f}]).
- Validation labels were used for evaluation/reporting only; train-only map rules were frozen before the run. No submission was produced.
- Ollama changed from 0.40.15 to 0.40.16 after round 10; model digest and prompt stayed fixed, and each ID block remained version-matched across all four branches. train_1725 exceeded the 8,192-token context in all branches and was scored as a recorded runtime failure.
"""
    insights_path = ROOT / "INSIGHTS.md"
    existing = insights_path.read_text(encoding="utf-8")
    if insight_header not in existing:
        insights_path.write_text(existing.rstrip() + "\n\n" + insight, encoding="utf-8")

    work_header = "## 2026-11-10 — QUAN four-branch comparison on full val309"
    work = f"""
{work_header}

- Completed A/B/C/D downstream inference for all 309 validation IDs with the locked Qwen SFT epoch-2 + CoT model and shared ToolEnv configuration. A/B used epoch-06 QUAN transcripts, C used the original BTC script, and D used the raw epoch-02 QUAN continuation. The branches ran in rotating 20-ID blocks.
- Official BTC per-sample replay, full-split WER/map analysis, and six paired 20,000-bootstrap comparisons are saved under exp/quan-map-correction-val309-v1/output/benchmark/. Runtime counts: {run_meta['runtime_errors']}.
- The complete split was used for evaluation only; map rules remained frozen from train data. No public/private test or submission was used.
- Ollama versions were 0.40.15 for rounds 1–10 and 0.40.16 for rounds 11–16 with unchanged model digest/prompt. The same train_1725 context overflow was recorded in all branches.
"""
    worklog_path = ROOT / "WORKLOG.md"
    worklog = worklog_path.read_text(encoding="utf-8")
    if work_header not in worklog:
        worklog_path.write_text(worklog.rstrip() + "\n\n" + work, encoding="utf-8")

    progress_path = ROOT / "PROGRESS.md"
    progress = progress_path.read_text(encoding="utf-8")
    new_row = (
        "| quan-map-correction-val309-v1 | Full 309-ID val; A/B epoch-06 QUAN raw/map, C BTC oracle, D QUAN epoch-02 raw; same Qwen SFT2+CoT | "
        f"WER project A/B/C/D {transcript['WER_project_normalization']['A']['corpus_wer']:.4f}/{transcript['WER_project_normalization']['B']['corpus_wer']:.4f}/{transcript['WER_project_normalization']['C']['corpus_wer']:.4f}/{transcript['WER_project_normalization']['D']['corpus_wer']:.4f}; "
        f"BTC A/B/C/D {metrics['A']['btc_formula_score_before_runtime_gate_0_100']:.2f}/{metrics['B']['btc_formula_score_before_runtime_gate_0_100']:.2f}/{metrics['C']['btc_formula_score_before_runtime_gate_0_100']:.2f}/{metrics['D']['btc_formula_score_before_runtime_gate_0_100']:.2f}; "
        f"map exact {map_result['exactly_correct']}/{map_result['applications']} | complete 1,236/1,236; runtime errors {run_meta['runtime_errors']}; no submission |")
    rows = progress.splitlines()
    if not any(line.startswith("| quan-map-correction-val309-v1 |") for line in rows):
        rows.append(new_row)
        progress_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(json.dumps({"branches": {b: {
        "WER_project": report["branches"][b]["WER_project"]["corpus_wer"],
        "WER_number_value": report["branches"][b]["WER_number_value"]["corpus_wer"],
        "BTC": metrics[b]["btc_formula_score_before_runtime_gate_0_100"],
        "local_30s": metrics[b]["local_30s_gated_score_diagnostic_0_100"],
        "runtime_errors": metrics[b]["runtime_errors"],
        "gold_replay_failures": metrics[b]["gold_replay_failures"],
    } for b in BRANCHES},
        "map_accuracy": map_result,
        "paired": {k: v["paired_formula_delta_points_right_minus_left"]
                   for k, v in report["paired_diagnostics"].items()}},
        ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

