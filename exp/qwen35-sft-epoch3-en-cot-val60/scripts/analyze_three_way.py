from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "btc-scoring" / "scripts"))
EXP = ROOT / "exp" / "qwen35-sft-epoch3-en-cot-val60"
OUT = EXP / "output"
INFER = OUT / "inference"
PROV = OUT / "model-provenance"
E2_OUT = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "inference"
E3_VI_OUT = ROOT / "exp" / "qwen35-sft-epoch3-cot-val60" / "output" / "inference"
ANALYZER_PATH = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "analyze_downstream.py"
EXPECTED_IDS_HASH = "87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analyzer():
    spec = importlib.util.spec_from_file_location("epoch3_en_shared_analyzer", ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = INFER
    return module


def read_branch(folder: Path, branch: str, expected_ids: list[str]):
    predictions_list = rows(folder / f"predictions-{branch}.jsonl")
    scores_list = rows(folder / f"btc-per-sample-{branch}.jsonl")
    requests = rows(folder / f"requests-{branch}.jsonl")
    if [row["id"] for row in predictions_list] != expected_ids:
        raise RuntimeError(f"{folder}: prediction order/cohort mismatch")
    if [row["id"] for row in scores_list] != expected_ids or not requests:
        raise RuntimeError(f"{folder}: score rows are incomplete or misordered")
    return ({row["id"]: row for row in predictions_list},
            {row["id"]: row for row in scores_list}, requests)


def main() -> int:
    lock = read_json(INFER / "run_lock.json")
    run_meta = read_json(INFER / "run-COT.json")
    config = read_json(INFER / "config_manifest.json")
    if lock.get("status") != "locked" or run_meta.get("status") != "complete":
        raise RuntimeError("A complete, locked English epoch-3 + CoT run is required")
    if lock.get("config_manifest_sha256") != sha256(INFER / "config_manifest.json"):
        raise RuntimeError("Inference configuration no longer matches its run lock")
    if run_meta.get("model_digest") != lock.get("epoch3_en_model_digest"):
        raise RuntimeError("Completed inference model digest differs from its locked digest")
    ids = read_json(INFER / "cohort_manifest.json")["ids"]
    ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if len(ids) != 60 or ids_hash != EXPECTED_IDS_HASH:
        raise RuntimeError("New branch is not the exact ordered enriched cohort60")
    if lock.get("cohort_ids_sha256_newline_joined") != ids_hash:
        raise RuntimeError("Cohort manifest differs from the inference lock")
    if run_meta.get("runtime_errors") != 0:
        raise RuntimeError("The new run has runtime errors; inspect before comparing")
    predictions_rows = rows(INFER / "predictions-COT.jsonl")
    if any(row.get("toolenv_count") != 1 for row in predictions_rows):
        raise RuntimeError("A new-branch prediction did not use exactly one ToolEnv")

    e2_ids = read_json(E2_OUT / "cohort_manifest.json")["ids"]
    vi_ids = read_json(E3_VI_OUT / "cohort_manifest.json")["ids"]
    if ids != e2_ids or ids != vi_ids:
        raise RuntimeError("The three runs do not use the exact same ordered 60-ID cohort")
    e2_run, e2_metrics = read_json(E2_OUT / "run-COT.json"), read_json(E2_OUT / "metrics-COT.json")
    vi_run, vi_metrics = read_json(E3_VI_OUT / "run-COT.json"), read_json(E3_VI_OUT / "metrics-COT.json")
    for label, run, metrics in (("epoch2_C", e2_run, e2_metrics), ("epoch3_vi_COT", vi_run, vi_metrics)):
        if run.get("status") != "complete" or run.get("samples_completed") != 60 or metrics.get("samples") != 60:
            raise RuntimeError(f"Reused baseline {label} is incomplete")
        if metrics.get("gold_replay_failures") != 0 or metrics.get("runtime_errors") != 0:
            raise RuntimeError(f"Reused baseline {label} has replay/runtime errors")

    analyzer = load_analyzer()
    data_root = analyzer.find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in analyzer.load_jsonl(data_root / "train" / "train.jsonl")}
    if set(ids) - items.keys():
        raise RuntimeError("A fixed cohort item is missing from the official train split")
    new_pred, new_scores, new_metrics, new_requests = analyzer.compute_branch("COT", ids, items, data_root)
    if new_metrics["gold_replay_failures"] != 0:
        raise RuntimeError("At least one new-branch gold ToolEnv replay failed")
    new_metrics["branch"] = "epoch3_en_COT"
    write_json(INFER / "metrics-COT.json", new_metrics)

    e2_pred, e2_scores, e2_requests = read_branch(E2_OUT, "COT", ids)
    vi_pred, vi_scores, vi_requests = read_branch(E3_VI_OUT, "COT", ids)
    # The baseline result files remain untouched; their file hashes are recorded below.
    paired_e2 = analyzer.pair_metrics("epoch2-C-vs-epoch3-en-COT", ids,
        e2_pred, new_pred, e2_scores, new_scores, e2_requests, new_requests,
        random.Random(analyzer.BOOTSTRAP_SEED))
    paired_vi = analyzer.pair_metrics("epoch3-vi-vs-epoch3-en-COT", ids,
        vi_pred, new_pred, vi_scores, new_scores, vi_requests, new_requests,
        random.Random(analyzer.BOOTSTRAP_SEED + 1))
    if paired_e2["gold_replay_failures_left_or_right"] != [0, 0] or paired_vi["gold_replay_failures_left_or_right"] != [0, 0]:
        raise RuntimeError("Gold replay errors are present in a paired comparison")

    artifacts = {}
    for label, folder in (("epoch2_C", E2_OUT), ("epoch3_vi_COT", E3_VI_OUT)):
        artifacts[label] = {name: sha256(folder / name) for name in (
            "cohort_manifest.json", "config_manifest.json", "run-COT.json", "metrics-COT.json",
            "predictions-COT.jsonl", "btc-per-sample-COT.jsonl", "requests-COT.jsonl")}
    analysis = {
        "experiment": EXP.name,
        "cohort": {"description": "same fixed enriched subset of held-out val309; preliminary screening cohort only",
            "samples": len(ids), "ids_sha256_newline_joined": ids_hash},
        "reused_baselines": {
            "epoch2_C": {"official_btc_formula_score_before_runtime_gate_0_100": e2_metrics["btc_formula_score_before_runtime_gate_0_100"],
                "local_30s_gated_diagnostic_0_100": e2_metrics["local_30s_gated_score_diagnostic_0_100"],
                "hash_matches": e2_metrics["hash_matches"],
                "M_U_G": [e2_metrics["M_matched_calls"], e2_metrics["U_unmatched_submitted_calls"], e2_metrics["G_gold_calls"]],
                "run_artifacts": "exp/qwen35-sft-benchmark-v1/output/inference/", "rerun": False},
            "epoch3_vietnamese_COT": {"official_btc_formula_score_before_runtime_gate_0_100": vi_metrics["btc_formula_score_before_runtime_gate_0_100"],
                "local_30s_gated_diagnostic_0_100": vi_metrics["local_30s_gated_score_diagnostic_0_100"],
                "hash_matches": vi_metrics["hash_matches"],
                "M_U_G": [vi_metrics["M_matched_calls"], vi_metrics["U_unmatched_submitted_calls"], vi_metrics["G_gold_calls"]],
                "run_artifacts": "exp/qwen35-sft-epoch3-cot-val60/output/inference/", "rerun": False},
        },
        "epoch3_english_COT": new_metrics,
        "paired_epoch2_C_vs_epoch3_english_COT": paired_e2,
        "paired_epoch3_vietnamese_COT_vs_epoch3_english_COT": paired_vi,
        "reused_baseline_artifact_sha256": artifacts,
        "provenance": {"adapter_archive_sha256": read_json(PROV / "local-audit.json")["archive"]["sha256"],
            "adapter_sha256": read_json(PROV / "local-audit.json")["adapter"]["final"]["sha256"],
            "epoch2_continuation_source_adapter_sha256": read_json(PROV / "local-audit.json")["training_report"]["continuation_source_adapter_sha256"],
            "continuation_source_matches_epoch2": read_json(PROV / "local-audit.json")["adapter"]["matches_continuation_source_archive"],
            "english_sft_train_sha256": read_json(PROV / "local-audit.json")["training_report"]["sft_train_sha256"],
            "remote_training_report_has_exact_base_weight_hash": False,
            "model_tag": run_meta["model"], "model_digest": run_meta["model_digest"],
            "cot_prompt_sha256_utf8": config["agent"]["cot_prompt_sha256_utf8"],
            "asr_checkpoint_sha256": config["asr"]["selected_checkpoint_sha256"],
            "transcript_file_sha256": config["asr"]["transcript_file_sha256"]},
        "comparison_controls": {"same_ordered_ids": True, "same_base_runtime_prompt_and_CoT": True,
            "same_QUAN_ASR_transcripts_and_history_ASR": True,
            "same_ollama_version_and_decoding_settings": True,
            "only_new_branch_rerun": True, "prediction_toolenv_instances_per_sample": 1},
        "official_scoring": {"match": "exact tool name and arguments with duplicate calls matched by multiplicity",
            "sample_score": "zero when ToolEnv state hash differs from gold or no exact call; otherwise harmonic mean of exact-call precision and recall",
            "main_score": "mean BTC formula score on 0-100 scale before local 30-second runtime mask",
            "local_30_second_gate": "reported as a separate diagnostic, not BTC score"},
        "limitations": ["The 60 examples are a fixed enriched screening subset, not all val309.",
            "Both historical branches were reused without repeating their model inference, so run-to-run variance is not estimated.",
            "Paired bootstrap intervals quantify item variation in this cohort and are not corrected for two comparisons.",
            "The remote training report binds the English SFT data and the epoch-2 adapter source, but lacks an exact base-weight hash."],
        "raw_chain_of_thought_saved": False,
        "public_private_test_used": False,
        "environment_files_read": False,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "inference_config_sha256": lock["config_manifest_sha256"],
    }
    write_json(INFER / "three-way-analysis.json", analysis)

    def metric_line(label: str, metrics: dict[str, Any]) -> str:
        latency = metrics["sample_latency_seconds"]
        return (f"| {label} | {metrics['btc_formula_score_before_runtime_gate_0_100']:.2f} | "
            f"{metrics['local_30s_gated_score_diagnostic_0_100']:.2f} | {metrics['hash_matches']}/60 | "
            f"{metrics['M_matched_calls']} / {metrics['U_unmatched_submitted_calls']} / {metrics['G_gold_calls']} | "
            f"{metrics['samples_over_30s_local']} | {latency['mean']:.2f}s / {latency['p95']:.2f}s | {metrics['runtime_errors']} |")

    lines = ["# Qwen3.5 English-prompt epoch 3 + CoT — fixed cohort60", "",
        "## Scope", "",
        "Compared the newly fine-tuned English-prompt epoch-3 adapter + CoT against two completed runs: epoch-2 branch C and the Vietnamese-prompt epoch-3 adapter + CoT. Both historical runs were read from their saved artifacts and were not rerun.",
        "All three runs use the same ordered 60-ID enriched subset of held-out val309, English runtime system prompt plus the same screened CoT checklist, the same QUAN transcripts and PhoASR history, Ollama version and decoding settings. Each prediction used exactly one official ToolEnv.", "",
        "## Scores", "",
        "| Branch | BTC formula before runtime gate | Local 30s-gated diagnostic | Hash matches | M / U / G | Local >30s | Mean / p95 latency | Runtime errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_line("Epoch 2 SFT + CoT, branch C (reused)", e2_metrics),
        metric_line("Epoch 3 Vietnamese-prompt SFT + CoT (reused)", vi_metrics),
        metric_line("Epoch 3 English-prompt SFT + CoT (new)", new_metrics), "",
        "Calls are matched by exact tool name and arguments, with duplicate calls counted by multiplicity. Per-sample BTC score is zero if the predicted ToolEnv state hash differs from the gold replay or no exact tool call matches; otherwise it is the harmonic mean of exact-call precision and recall. The 30-second local runtime mask is only a separate diagnostic.", "",
        "## Paired deltas", "",
        "Positive values mean the English-prompt epoch-3 branch scored higher than the named reused baseline.", "",
        "| Comparison | BTC delta, points | 95% paired bootstrap CI | Improved / worsened / tied samples | Local gated delta, points | 95% CI |",
        "|---|---:|---:|---:|---:|---:|"]
    for label, paired in (("English epoch 3 − epoch 2 C", paired_e2), ("English epoch 3 − Vietnamese epoch 3", paired_vi)):
        d = paired["paired_formula_delta_points_right_minus_left"]
        g = paired["paired_local_30s_gated_score_delta_points_diagnostic"]
        outcome = paired["paired_outcomes_formula_score"]
        lines.append(f"| {label} | {d['mean']:+.2f} | [{d['ci95_low']:+.2f}, {d['ci95_high']:+.2f}] | "
            f"{outcome.get('improved', 0)} / {outcome.get('worsened', 0)} / {outcome.get('tied', 0)} | "
            f"{g['mean']:+.2f} | [{g['ci95_low']:+.2f}, {g['ci95_high']:+.2f}] |")
    lines += ["", "## Provenance and limits", "",
        f"- English archive SHA-256: `{analysis['provenance']['adapter_archive_sha256']}`; adapter SHA-256: `{analysis['provenance']['adapter_sha256']}`.",
        f"- Continuation source matches the epoch-2 adapter: **{analysis['provenance']['continuation_source_matches_epoch2']}**. The English SFT data SHA-256 matches the training report and local manifest.",
        "- The remote training report does not include an exact base-weight hash, so local Kaggle version-1 base lineage is verified but exact remote base bytes are not.",
        "- The cohort is an enriched 60-example screen, not full val309. The two historical runs were reused and not repeated; paired intervals do not estimate run variance and are not adjusted for the two comparisons.",
        "- The English branch changes both the extra training epoch and the SFT prompt language relative to epoch 2. The comparison to Vietnamese epoch 3 isolates the prompt-language training variant more directly, while both use the same English runtime prompt.",
        "- No public/private test data were used. Raw chain-of-thought was not saved.", "",
        "## Artifacts", "",
        "New predictions, request telemetry, traces, BTC rows, and run metadata are under `output/inference/`. Reused baseline outputs remain at their original paths. The machine-readable results and per-ID paired rows are recorded alongside this report.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "analyzed", "scores": {
        "epoch2_C": e2_metrics["btc_formula_score_before_runtime_gate_0_100"],
        "epoch3_vi_COT": vi_metrics["btc_formula_score_before_runtime_gate_0_100"],
        "epoch3_en_COT": new_metrics["btc_formula_score_before_runtime_gate_0_100"]},
        "paired_deltas": {"vs_epoch2_C": paired_e2["paired_formula_delta_points_right_minus_left"],
            "vs_epoch3_vi_COT": paired_vi["paired_formula_delta_points_right_minus_left"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
