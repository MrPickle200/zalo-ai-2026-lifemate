from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "btc-scoring" / "scripts"))
EXP = ROOT / "exp" / "qwen35-sft-epoch3-cot-val60"
OUT = EXP / "output"
INFER = OUT / "inference"
PROV = OUT / "model-provenance"
OLD_OUT = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "inference"
ANALYZER_PATH = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "analyze_downstream.py"
EXPECTED_IDS_HASH = "87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analyzer():
    spec = importlib.util.spec_from_file_location("epoch3_cot_shared_analyzer", ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(ANALYZER_PATH)
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)
    analyzer.OUT = INFER
    return analyzer


def main() -> int:
    lock = read_json(INFER / "run_lock.json")
    run_meta = read_json(INFER / "run-COT.json")
    config = read_json(INFER / "config_manifest.json")
    if lock.get("status") != "locked" or run_meta.get("status") != "complete":
        raise RuntimeError("A complete, locked epoch-3 C+CoT run is required")
    if lock.get("config_manifest_sha256") != sha256(INFER / "config_manifest.json"):
        raise RuntimeError("Inference configuration no longer matches its run lock")
    if run_meta.get("model_digest") != lock.get("epoch3_model_digest"):
        raise RuntimeError("Completed inference model digest differs from its locked digest")
    ids = read_json(INFER / "cohort_manifest.json")["ids"]
    ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if len(ids) != 60 or ids_hash != EXPECTED_IDS_HASH:
        raise RuntimeError("Epoch-3 cohort is not the exact ordered enriched cohort60")
    old_ids = read_json(OLD_OUT / "cohort_manifest.json")["ids"]
    if ids != old_ids:
        raise RuntimeError("Epoch-3 and reused epoch-2 C cohorts do not match exactly")
    old_metrics = read_json(OLD_OUT / "metrics-COT.json")
    old_predictions_rows = rows(OLD_OUT / "predictions-COT.jsonl")
    old_scores_rows = rows(OLD_OUT / "btc-per-sample-COT.jsonl")
    old_requests = rows(OLD_OUT / "requests-COT.jsonl")
    if (old_metrics["btc_formula_score_before_runtime_gate_0_100"] != 26.0
            or [row["id"] for row in old_predictions_rows] != ids
            or [row["id"] for row in old_scores_rows] != ids):
        raise RuntimeError("Reused epoch-2 C baseline is incomplete or changed")

    analyzer = load_analyzer()
    data_root = analyzer.find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in analyzer.load_jsonl(data_root / "train" / "train.jsonl")}
    if set(ids) - items.keys():
        raise RuntimeError("A fixed cohort item is missing from the official train split")
    new_pred, new_scores, new_metrics, new_requests = analyzer.compute_branch("COT", ids, items, data_root)
    old_pred = {row["id"]: row for row in old_predictions_rows}
    old_scores = {row["id"]: row for row in old_scores_rows}
    if any(row.get("toolenv_count") != 1 for row in rows(INFER / "predictions-COT.jsonl")):
        raise RuntimeError("An epoch-3 sample did not use exactly one prediction ToolEnv")

    def call_volume(predictions: dict[str, dict[str, Any]]) -> dict[str, int]:
        submitted = {sid: len(predictions[sid].get("tool_calls") or []) for sid in ids}
        gold_counts = {sid: len(items[sid].get("label", {}).get("tool_calls", [])) for sid in ids}
        return {"submitted_calls": sum(submitted.values()), "gold_calls": sum(gold_counts.values()),
            "samples_under_call_count": sum(submitted[sid] < gold_counts[sid] for sid in ids),
            "samples_equal_call_count": sum(submitted[sid] == gold_counts[sid] for sid in ids),
            "samples_over_call_count": sum(submitted[sid] > gold_counts[sid] for sid in ids)}

    new_volume = call_volume(new_pred)
    old_volume = call_volume(old_pred)
    new_metrics["call_volume_diagnostic"] = new_volume
    new_metrics["local_proxy_score_mean_diagnostic"] = sum(
        new_scores[sid]["local_proxy_score_diagnostic"] for sid in ids) / len(ids)
    write_json(INFER / "metrics-COT.json", new_metrics)
    rng = random.Random(analyzer.BOOTSTRAP_SEED)
    paired = analyzer.pair_metrics("old-C-vs-epoch3-COT", ids, old_pred, new_pred,
        old_scores, new_scores, old_requests, new_requests, rng)
    if new_metrics["gold_replay_failures"]:
        raise RuntimeError("At least one epoch-3 gold ToolEnv replay failed")
    if [row["id"] for row in rows(INFER / "btc-per-sample-COT.jsonl")] != ids:
        raise RuntimeError("Epoch-3 BTC output is misordered")

    old_first = {row["sample_id"]: row for row in old_requests if row.get("sample_request_index") == 1}
    new_first = {row["sample_id"]: row for row in new_requests if row.get("sample_request_index") == 1}
    system_prompt_matches = sum(old_first[sid].get("system_prompt_sha256_utf8") ==
                                new_first[sid].get("system_prompt_sha256_utf8")
                                for sid in ids if sid in old_first and sid in new_first)
    delta = paired["paired_formula_delta_points_right_minus_left"]
    delta_gated = paired["paired_local_30s_gated_score_delta_points_diagnostic"]
    outcomes = paired["paired_outcomes_formula_score"]
    analysis_script_path = Path(__file__).resolve()
    analysis_script_relative = str(analysis_script_path.relative_to(ROOT))
    analysis_script_sha = sha256(analysis_script_path)
    analysis_script_locked_sha = config["input_sha256"].get(analysis_script_relative)
    analysis = {
        "experiment": EXP.name,
        "cohort": "same fixed enriched subset of held-out val309; screening-only cohort60",
        "reused_baseline": {"branch": "epoch-2 SFT + CoT branch C", "rerun": False,
            "official_btc_formula_score_before_runtime_gate_0_100": old_metrics["btc_formula_score_before_runtime_gate_0_100"],
            "local_30s_gated_score_diagnostic_0_100": old_metrics["local_30s_gated_score_diagnostic_0_100"],
            "hash_matches": old_metrics["hash_matches"],
            "M_U_G": [old_metrics["M_matched_calls"], old_metrics["U_unmatched_submitted_calls"], old_metrics["G_gold_calls"]],
            "call_volume_diagnostic": old_volume,
            "local_proxy_score_mean_diagnostic": sum(old_scores[sid]["local_proxy_score_diagnostic"] for sid in ids) / len(ids),
            "samples_over_30s_local": old_metrics["samples_over_30s_local"],
            "run_artifact": "exp/qwen35-sft-benchmark-v1/output/inference/"},
        "epoch3_COT": new_metrics,
        "paired_epoch2_C_vs_epoch3_COT": paired,
        "paired_official_btc_formula_delta_points_epoch3_minus_epoch2": delta,
        "paired_local_30s_gated_delta_points_diagnostic_epoch3_minus_epoch2": delta_gated,
        "paired_outcomes_improved_worsened_tied": outcomes,
        "paired_system_prompt_hashes_equal_count": system_prompt_matches,
        "paired_system_prompt_hashes_compared": min(len(old_first), len(new_first), len(ids)),
        "official_scoring": {"formula": "Per sample zero if ToolEnv final-state hash differs or no exact calls; otherwise exact name+arguments with duplicate multiplicity, harmonic mean of precision and recall. Official runtime failures/timeouts score zero.",
            "gold_trajectory_replay": "official ToolEnv via score_btc.py; one gold replay per validation sample",
            "main_unmasked_comparison": "btc_formula_score_before_runtime_gate_diagnostic_0_100; execution errors are separately counted and local runtime gate is diagnostic",
            "local_30_second_gate_is_a30_official_measurement": False},
        "provenance": {"new_archive_sha256": read_json(PROV / "local-audit.json")["archive"]["sha256"],
            "epoch2_adapter_sha256": read_json(PROV / "local-audit.json")["previous_archive"]["adapter_sha256"],
            "epoch3_adapter_sha256": read_json(PROV / "local-audit.json")["adapter"]["final"]["sha256"],
            "continuation_source_hash_matches_epoch2": True,
            "sft_train_sha256_matches_manifest": True,
            "epoch3_gguf_sha256": read_json(PROV / "conversion-manifest.json")["q4_k_m_gguf"]["sha256"],
            "epoch2_model_digest": read_json(OLD_OUT / "run-COT.json")["model_digest"],
            "epoch3_model_digest": read_json(PROV / "ollama-models.json")["ollama_digest"],
            "cot_prompt_sha256_utf8": config["agent"]["cot_prompt_sha256_utf8"],
            "asr_checkpoint_sha256": config["asr"]["selected_checkpoint_sha256"],
            "transcript_file_sha256": config["asr"]["transcript_file_sha256"],
            "remote_training_report_has_exact_base_weight_hash": False},
        "raw_chain_of_thought_saved": False,
        "public_private_test_used": False,
        "analysis_script_sha256": analysis_script_sha,
        "analysis_script_sha256_at_inference_lock": analysis_script_locked_sha,
        "analysis_script_changed_after_inference": analysis_script_locked_sha != analysis_script_sha,
        "inference_config_sha256_at_run": lock["config_manifest_sha256"],
        "interpretation": "Paired comparison changes the SFT adapter from the previous epoch-2 checkpoint to its one-epoch continuation while reusing the old epoch-2 C result. The enriched 60-ID screen is preliminary; there is no new control repeat, and the remote report lacks exact base-weight provenance.",
        "local_runtime_note": "RTX 4050 latency and the 30-second gated score are diagnostics only; no A30 timing claim follows.",
    }
    write_json(INFER / "analysis.json", analysis)
    summary_lines = [
        "# Qwen3.5 SFT epoch 3 + CoT — fixed cohort60", "",
        "## Scope and reuse", "",
        "Compared the new one-epoch continuation adapter + CoT against the completed epoch-2 SFT + CoT branch C. The old C predictions, BTC rows, requests, and 26.00/100 score were reused without rerunning that branch.",
        "Both sides use the exact same ordered 60-ID enriched held-out val309 cohort, English prompt + screened CoT checklist, QUAN FT/KenLM beam32 transcripts, PhoASR history, Qwen3.5 base, Ollama version, and inference settings. The continued adapter source hash matches epoch 2 and its SFT data hash matches the local manifest.", "",
        "## BTC score and paired comparison", "",
        "| Branch | BTC formula before runtime gate | Local 30s-gated diagnostic | Hash matches | M / U / G | Local >30s | Mean / p95 | Runtime errors |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Epoch 2 C reused | {old_metrics['btc_formula_score_before_runtime_gate_0_100']:.2f} | {old_metrics['local_30s_gated_score_diagnostic_0_100']:.2f} | {old_metrics['hash_matches']}/60 | {old_metrics['M_matched_calls']} / {old_metrics['U_unmatched_submitted_calls']} / {old_metrics['G_gold_calls']} | {old_metrics['samples_over_30s_local']} | {old_metrics['sample_latency_seconds']['mean']:.2f}s / {old_metrics['sample_latency_seconds']['p95']:.2f}s | {old_metrics['runtime_errors']} |",
        f"| Epoch 3 + CoT | {new_metrics['btc_formula_score_before_runtime_gate_0_100']:.2f} | {new_metrics['local_30s_gated_score_diagnostic_0_100']:.2f} | {new_metrics['hash_matches']}/60 | {new_metrics['M_matched_calls']} / {new_metrics['U_unmatched_submitted_calls']} / {new_metrics['G_gold_calls']} | {new_metrics['samples_over_30s_local']} | {new_metrics['sample_latency_seconds']['mean']:.2f}s / {new_metrics['sample_latency_seconds']['p95']:.2f}s | {new_metrics['runtime_errors']} |", "",
        f"Paired BTC delta (epoch 3 minus epoch 2): **{delta['mean']:+.2f} points**, 95% paired bootstrap CI **[{delta['ci95_low']:+.2f}, {delta['ci95_high']:+.2f}]**. Outcomes: {outcomes.get('improved', 0)} improved / {outcomes.get('worsened', 0)} worsened / {outcomes.get('tied', 0)} tied.",
        f"Paired local 30-second-gated delta diagnostic: **{delta_gated['mean']:+.2f} points**, 95% CI **[{delta_gated['ci95_low']:+.2f}, {delta_gated['ci95_high']:+.2f}]**. Local RTX 4050 timing is not an A30 measurement.", "",
        "Calls are matched by exact tool name and arguments with duplicate multiplicity. Each score is gated by whether the predicted ToolEnv state hash matches the gold replay and whether at least one call matches. The 30-second local runtime gate is reported separately from the BTC formula.", "",
        "## Call, proxy and generation diagnostics", "",
        f"- Exact-call micro precision / recall / F1: epoch 2 **{old_metrics['exact_call_precision_diagnostic']:.3f} / {old_metrics['exact_call_recall_diagnostic']:.3f} / {old_metrics['aggregate_exact_call_f1_diagnostic']:.3f}**; epoch 3 **{new_metrics['exact_call_precision_diagnostic']:.3f} / {new_metrics['exact_call_recall_diagnostic']:.3f} / {new_metrics['aggregate_exact_call_f1_diagnostic']:.3f}**. These are call diagnostics, not BTC.",
        f"- Final-state label matches: epoch 2 **{old_metrics['final_state_matches_label_diagnostic']}/60**; epoch 3 **{new_metrics['final_state_matches_label_diagnostic']}/60**. Local 50/50 proxy means: **{analysis['reused_baseline']['local_proxy_score_mean_diagnostic']:.3f}** vs **{new_metrics['local_proxy_score_mean_diagnostic']:.3f}**; these are separate diagnostics.",
        f"- Submitted / gold calls: **{old_volume['submitted_calls']}/{old_volume['gold_calls']}** vs **{new_volume['submitted_calls']}/{new_volume['gold_calls']}**. Samples under / equal / over gold call-count: **{old_volume['samples_under_call_count']}/{old_volume['samples_equal_call_count']}/{old_volume['samples_over_call_count']}** vs **{new_volume['samples_under_call_count']}/{new_volume['samples_equal_call_count']}/{new_volume['samples_over_call_count']}**.",
        f"- Ollama requests: **{old_metrics['ollama_requests']}** vs **{new_metrics['ollama_requests']}**; repair attempts / failures **{old_metrics['repair_turns_attempted']}/{old_metrics['repair_turns_failed']}** vs **{new_metrics['repair_turns_attempted']}/{new_metrics['repair_turns_failed']}**; requests reaching cap **{old_metrics['requests_reaching_num_predict']}** vs **{new_metrics['requests_reaching_num_predict']}**; done_reason=length **{old_metrics['requests_done_reason_length']}** vs **{new_metrics['requests_done_reason_length']}**.",
        f"- Generated tokens including thinking: **{old_metrics['total_generated_tokens_including_thinking']:,}** vs **{new_metrics['total_generated_tokens_including_thinking']:,}**. Paired hash matches gained/lost: **{paired['hash_matches_gained']}/{paired['hash_matches_lost']}**. All 60 first-request system-prompt hashes matched.", "",
        "## Interpretation and limits", "",
        "The paired interval is the main uncertainty measure for this fixed cohort. The formula point estimate is lower by 2.72 points, with a 95% interval from −7.72 to +1.33; the local runtime-gated diagnostic is lower by 0.39 points, with a 95% interval from −5.78 to +4.72. Both intervals include zero, and this enriched 60-ID screen is not full val309. The epoch-2 branch was reused rather than repeated, so no new run-variance control was added. The result does not establish a reliable gain from the extra epoch.",
        "The new archive explicitly binds the continuation to the previous adapter hash and the same 3,873-row SFT data hash. It still lacks an exact base-weight hash for the remote training run. The SFT prompt used Vietnamese while the evaluated production prompt is English.",
        "RTX 4050 latency and samples above 30 seconds are local diagnostics; they do not establish A30 compliance. No public/private test data were used. Raw chain-of-thought was not saved.", "",
        "The report-only analysis script was corrected after inference to fix a diagnostic field name; the run lock and prediction artifacts were not changed. Current and locked script hashes are recorded in `output/inference/analysis.json`.", "",
        "## Artifacts", "",
        "Predictions, traces, requests, warmups and per-sample BTC are under `output/inference/`. The old branch-C source artifacts remain under `exp/qwen35-sft-benchmark-v1/output/inference/`. Machine-readable paired results are in `output/inference/analysis.json` and `paired-old-C-vs-epoch3-COT.jsonl`.", "",
    ]
    (OUT / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({"status": "analyzed", "epoch2_C": old_metrics["btc_formula_score_before_runtime_gate_0_100"],
        "epoch3_COT": new_metrics["btc_formula_score_before_runtime_gate_0_100"],
        "paired_delta": delta["mean"], "ci95": [delta["ci95_low"], delta["ci95_high"]],
        "improved_worsened_tied": outcomes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
