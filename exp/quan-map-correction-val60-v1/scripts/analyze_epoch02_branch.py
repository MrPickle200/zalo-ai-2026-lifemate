from __future__ import annotations
import hashlib, json, random, statistics, sys
from pathlib import Path
import prepare_benchmark as prep

ROOT, OUT = prep.ROOT, prep.OUT
ANALYZER = ROOT / "exp/quan-continuation-cot-v1/scripts/analyze_downstream.py"

def jread(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def jwrite(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    lock = jread(OUT / "benchmark_lock.json")
    run_d = jread(OUT / "inference-D.json")
    if lock.get("status") != "complete" or run_d.get("status") != "complete" or run_d.get("samples") != 60:
        raise RuntimeError("Locked A/B/C and 60-sample D inference must be complete")
    if run_d["qwen_digest"] != lock["qwen"]["digest"] or run_d["prompt_sha256_utf8"] != lock["qwen"]["prompt_sha256_utf8"]:
        raise RuntimeError("D did not use the locked Qwen model/prompt")
    runner_module = prep.module("val60_epoch02_analysis_runner", prep.NORMALIZATION_RUNNER)
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    analyzer = prep.module("val60_epoch02_btc_analyzer", ANALYZER)
    analyzer.OUT = OUT
    cohort = jread(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    d_transcripts = prep.read_jsonl(OUT / "transcripts-D-epoch02.jsonl")
    predictions_d = prep.read_jsonl(OUT / "predictions-D.jsonl")
    if [row["id"] for row in d_transcripts] != ids or [row["id"] for row in predictions_d] != ids:
        raise RuntimeError("D transcript/prediction artifacts do not cover the locked cohort in order")
    if any(pred.get("transcript_used") != src["prediction"] for pred, src in zip(predictions_d, d_transcripts)):
        raise RuntimeError("D downstream predictions do not match the locked epoch-02 transcript file")

    data_root = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    pred_d, score_d, metrics_d, req_d = analyzer.compute_branch("D", ids, items, data_root)
    if metrics_d["runtime_errors"] or metrics_d["gold_replay_failures"]:
        raise RuntimeError("D has inference runtime or gold ToolEnv replay failures")
    report = jread(OUT / "benchmark_results.json")
    transcript_d = jread(OUT / "transcript-analysis-D.json")
    report["branches"]["D"] = {
        "input": "raw QUAN-transcribe continuation checkpoint, epoch 02 (two epochs)",
        "WER_project": transcript_d["WER_project_normalization"],
        "WER_number_value": transcript_d["WER_number_value_normalization"],
        "BTC": metrics_d
    }
    paired = report["paired_diagnostics"]
    rng = random.Random(20261109)
    for left in ("A", "B", "C"):
        if left not in report["branches"]:
            raise RuntimeError(f"Missing existing branch {left}")
        left_predictions = {row["id"]: row for row in analyzer.rows(OUT / f"predictions-{left}.jsonl")}
        left_scores = {row["id"]: row for row in analyzer.rows(OUT / f"btc-per-sample-{left}.jsonl")}
        left_requests = analyzer.rows(OUT / f"requests-{left}.jsonl")
        if list(left_predictions) != ids or set(left_scores) != set(ids):
            raise RuntimeError(f"Existing branch {left} artifacts are incomplete")
        paired[f"{left}_vs_D"] = analyzer.pair_metrics(
            f"{left}-vs-D", ids, left_predictions, pred_d, left_scores, score_d,
            left_requests, req_d, rng)
    jwrite(OUT / "benchmark_results.json", report)

    lines = [
        "# QUAN transcript map benchmark — val60", "",
        "Same fixed 60 held-out validation tasks and Qwen SFT epoch-2 + CoT setup, prompt, PhoASR history path, and ToolEnv settings. A/B/C ran in balanced 20-task blocks; D was added afterward in three 20-task blocks using the QUAN-transcribe epoch-02 continuation transcript.", "",
        "## Results", "",
        "| Branch | Input transcript | Project WER | Number-value WER | BTC formula /100 | Local 30s-gated diagnostic | Hash matches | Runtime errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|"
    ]
    for branch in ("A", "B", "C", "D"):
        row, btc = report["branches"][branch], report["branches"][branch]["BTC"]
        lines.append(f"| {branch} | {row['input']} | {row['WER_project']['corpus_wer']:.4f} | {row['WER_number_value']['corpus_wer']:.4f} | {btc['btc_formula_score_before_runtime_gate_0_100']:.2f} | {btc['local_30s_gated_score_diagnostic_0_100']:.2f} | {btc['hash_matches']}/60 | {btc['runtime_errors']} |")
    m = report["map_accuracy"]
    precision = "n/a" if m["precision"] is None else f"{m['precision']:.1%}"
    lines += ["", "## Map correction accuracy", "",
        f"{m['exactly_correct']}/{m['applications']} exact corrections ({precision}) across {m['changed_utterances']} tasks. The Vietnamese number parser normalized {m['number_spans_normalized_before_map']} QUAN spans before map matching.",
        "", "## Paired BTC score deltas (right minus left)", "",
        "| Comparison | Mean points | 95% paired bootstrap CI | Improved / worsened / tied |",
        "|---|---:|---:|---:|"]
    for key, row in paired.items():
        delta = row["paired_formula_delta_points_right_minus_left"]
        outcomes = row["paired_outcomes_formula_score"]
        lines.append(f"| {key} | {delta['mean']:+.2f} | [{delta['ci95_low']:+.2f}, {delta['ci95_high']:+.2f}] | {outcomes.get('improved',0)} / {outcomes.get('worsened',0)} / {outcomes.get('tied',0)} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limits"]]
    lines += ["- Project WER retains number words; number-value WER parses both sides so equivalent word and digit forms are not counted as ASR errors.",
              "- BTC formula scores are unmasked. The local 30-second gate is reported separately as a host-specific diagnostic.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


    # Add the separately requested continuation branch to the run record.
    elapsed = float(run_d["elapsed_seconds"])
    run_meta = jread(OUT / "inference_run.json")
    run_meta.setdefault("samples_per_branch", {})["D"] = 60
    run_meta["additional_branch_D"] = {
        "checkpoint": "exp/quan-continuation-cot-v1/output/checkpoints/epoch-02",
        "checkpoint_sha256": run_d["checkpoint_sha256"],
        "samples": 60, "elapsed_seconds": elapsed,
        "completed_epoch": run_d["ended_epoch"],
        "order": "three sequential 20-ID blocks after A/B/C",
        "qwen_digest": run_d["qwen_digest"]
    }
    run_meta["total_samples_across_branches"] = sum(run_meta["samples_per_branch"].values())
    run_meta["total_elapsed_seconds_A_B_C_D"] = float(run_meta.get("elapsed_seconds", 0)) + elapsed
    jwrite(OUT / "inference_run.json", run_meta)

    # Extend the experiment lock with D inputs and all new derived outputs.
    lock["branch_D"] = {
        "status": "complete", "checkpoint": run_d["checkpoint_sha256"],
        "validation_transcripts": run_d["transcript_file_sha256"],
        "cohort_ids_sha256_newline_joined": cohort["ids_sha256_newline_joined"],
        "samples": 60, "elapsed_seconds": elapsed,
        "qwen_model": run_d["qwen_model"], "qwen_digest": run_d["qwen_digest"],
        "prompt_sha256_utf8": run_d["prompt_sha256_utf8"],
        "input": "raw epoch-02 QUAN transcript; no map",
        "WER_project": transcript_d["WER_project_normalization"]["corpus_wer"],
        "WER_number_value": transcript_d["WER_number_value_normalization"]["corpus_wer"],
        "btc_formula_score": metrics_d["btc_formula_score_before_runtime_gate_0_100"],
        "gold_replay_failures": metrics_d["gold_replay_failures"],
        "runtime_errors": metrics_d["runtime_errors"]
    }
    lock.setdefault("source_hashes", {})["exp\\quan-map-correction-val60-v1\\scripts\\analyze_epoch02_branch.py"] = sha(Path(__file__))
    new_outputs = [
        "transcripts-D-epoch02.jsonl", "transcript-analysis-D.json", "inference-D.json",
        "run-progress-D.json", "predictions-D.jsonl", "step-traces-D.jsonl",
        "requests-D.jsonl", "btc-per-sample-D.jsonl", "metrics-D.json",
        "paired-A-vs-D.jsonl", "paired-B-vs-D.jsonl", "paired-C-vs-D.jsonl",
        "benchmark_results.json", "summary.md", "inference_run.json"
    ]
    output_hashes = lock.setdefault("completed_output_hashes", {})
    for name in new_outputs:
        output_hashes[name] = sha(OUT / name)
    lock["analysis_completed"] = True
    import time
    lock["epoch02_branch_analysis_completed_epoch"] = time.time()
    jwrite(OUT / "benchmark_lock.json", lock)

    # Update the experiment README without changing the original A/B/C record.
    readme_path = ROOT / "exp/quan-map-correction-val60-v1/README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace("This experiment runs three transcript inputs", "This experiment runs four transcript inputs")
    readme = readme.replace("- **C — BTC script:** original BTC audio_text reference, as an oracle transcript.",
        "- **C — BTC script:** original BTC audio_text reference, as an oracle transcript.\n- **D — QUAN epoch 02:** raw transcript from the two-epoch QUAN continuation checkpoint, without the map.")
    readme = readme.replace(
        "The 180 downstream tasks completed in 3,208.9 seconds, with zero inference runtime failures and zero gold replay failures.",
        f"The A/B/C run completed 180 downstream tasks in 3,208.9 seconds. The added D run completed 60 tasks in {elapsed:.1f} seconds. All four branches had zero inference runtime failures and zero gold replay failures.")
    readme = readme.replace(
        "| C — BTC oracle script | 0.0000 | 0.0000 | 32.33 | 32.33 |",
        "| C — BTC oracle script | 0.0000 | 0.0000 | 32.33 | 32.33 |\n| D — QUAN continuation epoch 02 | " +
        f"{transcript_d['WER_project_normalization']['corpus_wer']:.4f} | {transcript_d['WER_number_value_normalization']['corpus_wer']:.4f} | " +
        f"{metrics_d['btc_formula_score_before_runtime_gate_0_100']:.2f} | {metrics_d['local_30s_gated_score_diagnostic_0_100']:.2f} |")
    readme = readme.replace(
        "The map branch was worse by point estimate for both WER and BTC; the paired BTC interval includes zero. The zero exact-correction precision is a strong warning against applying these candidate rules globally. The oracle branch's BTC score is an upper-bound diagnostic, not a deployable ASR result.",
        f"The map branch was worse by point estimate for both WER and BTC; the paired BTC interval includes zero. The D epoch-02 continuation measured project WER {transcript_d['WER_project_normalization']['corpus_wer']:.4f}, number-value WER {transcript_d['WER_number_value_normalization']['corpus_wer']:.4f}, and downstream BTC {metrics_d['btc_formula_score_before_runtime_gate_0_100']:.2f}/100 on this screen. The zero exact-correction precision is a strong warning against applying these candidate rules globally. The oracle branch's BTC score is an upper-bound diagnostic, not a deployable ASR result.")
    readme = readme.replace(
        "3. python exp/quan-map-correction-val60-v1/scripts/analyze_benchmark.py",
        "3. python exp/quan-map-correction-val60-v1/scripts/analyze_benchmark.py\n4. python exp/quan-map-correction-val60-v1/scripts/run_epoch02_branch.py\n5. python exp/quan-map-correction-val60-v1/scripts/analyze_epoch02_branch.py")
    readme_path.write_text(readme, encoding="utf-8")

    d_vs_a = paired["A_vs_D"]["paired_formula_delta_points_right_minus_left"]
    insight_header = "## QUAN-transcribe epoch-02 continuation on val60 (2026-11-10)"
    insight = f"""
{insight_header}

- The raw two-epoch continuation transcript scored project WER {transcript_d['WER_project_normalization']['corpus_wer']:.4f} and number-value WER {transcript_d['WER_number_value_normalization']['corpus_wer']:.4f} on the fixed enriched 60-ID screen; downstream BTC formula score was {metrics_d['btc_formula_score_before_runtime_gate_0_100']:.2f}/100, with zero inference runtime errors and zero gold replay failures.
- Compared with raw QUAN epoch-06 branch A, D changed downstream BTC by {d_vs_a['mean']:+.2f} points (paired 95% CI [{d_vs_a['ci95_low']:+.2f},{d_vs_a['ci95_high']:+.2f}]). This single-run enriched screen is diagnostic; do not infer a general epoch effect from it.
- D used the locked Qwen SFT epoch-2 + CoT model/prompt and raw epoch-02 QUAN transcript, with no map. Details: exp/quan-map-correction-val60-v1/output/benchmark/summary.md.
"""
    insights_path = ROOT / "INSIGHTS.md"
    insights = insights_path.read_text(encoding="utf-8")
    if insight_header not in insights:
        insights_path.write_text(insights.rstrip() + "\n\n" + insight, encoding="utf-8")

    work_header = "## 2026-11-10 — QUAN continuation epoch-02 downstream branch"
    work = f"""
{work_header}

- Added branch D on the fixed enriched held-out val60 cohort using the raw QUAN-transcribe two-epoch continuation checkpoint and the same Qwen SFT epoch-2 + CoT setup. System Python outside the project .venv reached the local Ollama model. All 60 predictions completed; inference runtime errors and official gold ToolEnv replay failures were both zero.
- Transcript WER was {transcript_d['WER_project_normalization']['corpus_wer']:.4f} project-normalized / {transcript_d['WER_number_value_normalization']['corpus_wer']:.4f} number-value-normalized. BTC formula score was {metrics_d['btc_formula_score_before_runtime_gate_0_100']:.2f}/100; local 30s-gated diagnostic was {metrics_d['local_30s_gated_score_diagnostic_0_100']:.2f}.
- Added paired D-vs-A/B/C analysis and updated the benchmark summary, lock, README, progress and insights records. This remains a one-run enriched-cohort screen; no full-val309 or submission was run.
"""
    worklog_path = ROOT / "WORKLOG.md"
    worklog = worklog_path.read_text(encoding="utf-8")
    if work_header not in worklog:
        worklog_path.write_text(worklog.rstrip() + "\n\n" + work, encoding="utf-8")

    progress_path = ROOT / "PROGRESS.md"
    progress = progress_path.read_text(encoding="utf-8")
    old_prefix = "| quan-map-correction-val60-v1 |"
    new_row = (
        "| quan-map-correction-val60-v1 | Raw QUAN epoch-06/map/BTC oracle plus QUAN continuation epoch-02; same Qwen SFT2+CoT, fixed enriched held-out val60 | "
        f"WER project A/B/C/D 0.1757/0.2046/0.0000/{transcript_d['WER_project_normalization']['corpus_wer']:.4f}; "
        f"number-value D {transcript_d['WER_number_value_normalization']['corpus_wer']:.4f}; map 0/28; "
        f"BTC A/B/C/D 23.29/19.83/32.33/{metrics_d['btc_formula_score_before_runtime_gate_0_100']:.2f}; "
        f"A−D paired {d_vs_a['mean']:+.2f} [{d_vs_a['ci95_low']:+.2f},{d_vs_a['ci95_high']:+.2f}] | "
        "complete diagnostic; 240/240, 0 runtime/replay failures; no submission |")
    rows = progress.splitlines()
    replaced = False
    for index, line in enumerate(rows):
        if line.startswith(old_prefix):
            rows[index] = new_row
            replaced = True
            break
    if replaced:
        progress_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(json.dumps({
        "D": {"WER_project": transcript_d["WER_project_normalization"]["corpus_wer"],
              "WER_number_value": transcript_d["WER_number_value_normalization"]["corpus_wer"],
              "BTC": metrics_d["btc_formula_score_before_runtime_gate_0_100"],
              "local_30s": metrics_d["local_30s_gated_score_diagnostic_0_100"],
              "runtime_errors": metrics_d["runtime_errors"],
              "gold_replay_failures": metrics_d["gold_replay_failures"]},
        "paired_D": {name: paired[f"{name}_vs_D"]["paired_formula_delta_points_right_minus_left"]
                     for name in ("A", "B", "C")},
        "total_samples": run_meta["total_samples_across_branches"],
        "total_elapsed_seconds_A_B_C_D": run_meta["total_elapsed_seconds_A_B_C_D"]
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

