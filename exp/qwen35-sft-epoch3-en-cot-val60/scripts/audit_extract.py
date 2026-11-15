from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "qwen35-sft-epoch3-en-cot-val60"
OUT = EXP / "output"
PROV = OUT / "model-provenance"
NEW_ZIP = ROOT / "exp" / "qwen35-sft-next-action-v1" / "adapter-3-epochs-en.zip"
OLD_ZIP = ROOT / "exp" / "qwen35-sft-next-action-v1" / "adapter-2-epochs.zip"
OLD_EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OLD_OUT = OLD_EXP / "output" / "inference"
OLD_PROV = OLD_EXP / "output" / "model-provenance"
VI_EXP = ROOT / "exp" / "qwen35-sft-epoch3-cot-val60"
VI_OUT = VI_EXP / "output" / "inference"
PREFIX = "qwen35_lifemate_sft_v1_prompt_en_continuation_epoch3/"
OLD_PREFIX = "qwen35_lifemate_sft_v1/"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_reused_config(config: dict[str, Any], config_path: Path, cohort_path: Path,
                         lock_path: Path, allowed_source_exceptions: set[str]) -> list[dict[str, str]]:
    lock = read_json(lock_path)
    if lock.get("status") != "locked":
        raise RuntimeError(f"Reused run lock is not locked: {lock_path}")
    if lock.get("config_manifest_sha256") != sha256_file(config_path):
        raise RuntimeError(f"Reused config does not match its run lock: {config_path}")
    if lock.get("cohort_manifest_sha256") != sha256_file(cohort_path):
        raise RuntimeError(f"Reused cohort manifest does not match its run lock: {cohort_path}")
    exceptions = []
    for relative, expected in config.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file():
            if relative in allowed_source_exceptions:
                exceptions.append({"path": relative, "expected_sha256": expected, "reason": "source file no longer present"})
                continue
            raise FileNotFoundError(f"Reused config input is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            if relative in allowed_source_exceptions:
                exceptions.append({"path": relative, "expected_sha256": expected,
                    "actual_sha256": actual, "reason": "report-only analysis source was edited after inference"})
                continue
            raise RuntimeError(f"Reused config input hash changed: {relative}")
    return exceptions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    if not NEW_ZIP.is_file() or not OLD_ZIP.is_file():
        raise FileNotFoundError("Both the new continuation archive and epoch-2 source archive are required")
    if (OUT / "model-provenance" / "local-audit.json").exists():
        raise FileExistsError("English epoch-3 audit exists; refusing to overwrite")

    new_zip_hash, old_zip_hash = sha256_file(NEW_ZIP), sha256_file(OLD_ZIP)
    new_prefix = PREFIX
    old_prefix = OLD_PREFIX
    with zipfile.ZipFile(NEW_ZIP) as archive, zipfile.ZipFile(OLD_ZIP) as old_archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
        members = set(archive.namelist())
        old_members = set(old_archive.namelist())
        required = [new_prefix + "adapter_final/" + name for name in ADAPTER_FILES]
        required += [new_prefix + "training_report.json", new_prefix + "checkpoints/checkpoint-243/trainer_state.json"]
        if set(required) - members:
            raise FileNotFoundError(f"Missing expected epoch-3 archive members: {sorted(set(required)-members)}")
        final_bytes = archive.read(new_prefix + "adapter_final/adapter_model.safetensors")
        checkpoint_bytes = archive.read(new_prefix + "checkpoints/checkpoint-243/adapter_model.safetensors")
        final_hash = sha256_bytes(final_bytes)
        if final_bytes != checkpoint_bytes:
            raise RuntimeError("adapter_final differs from final checkpoint-243")
        report = json.loads(archive.read(new_prefix + "training_report.json"))
        config = json.loads(archive.read(new_prefix + "adapter_final/adapter_config.json"))
        old_adapter = old_archive.read(old_prefix + "adapter_final/adapter_model.safetensors")
        old_adapter_hash = sha256_bytes(old_adapter)
        if report.get("continuation_source_adapter_sha256") != old_adapter_hash:
            raise RuntimeError("Continuation source adapter hash does not match the previous epoch-2 archive")
        if report.get("run_type") != "continuation_from_epoch2_adapter_english_prompt" or report.get("epochs_requested") != 1.0:
            raise RuntimeError("Archive report does not identify one continuation epoch from epoch 2")
        if report.get("train_rows") != 3873:
            raise RuntimeError("Unexpected continuation training row count")
        adapter_base = config.get("base_model_name_or_path", "").replace("\\", "/")
        expected_base = "/kaggle/input/models/limamateus/qwen3-5-4b/pytorch/default/1/Qwen3.5-4B"
        if adapter_base != expected_base:
            raise RuntimeError(f"Unexpected adapter base path: {adapter_base}")

        file_hashes = {name: sha256_bytes(archive.read(new_prefix + "adapter_final/" + name))
                       for name in ADAPTER_FILES}
        old_report = json.loads(old_archive.read(old_prefix + "training_report.json"))

    dataset_manifest_path = ROOT / "exp" / "qwen35-sft-next-action-v1" / "kaggle-upload-prompt-en-v1" / "dataset_manifest.json"
    train_jsonl = dataset_manifest_path.with_name("sft_train.jsonl")
    dataset = read_json(dataset_manifest_path)
    train_hash = sha256_file(train_jsonl)
    if report.get("sft_train_sha256") != dataset.get("sft_train_sha256") or train_hash != dataset.get("sft_train_sha256"):
        raise RuntimeError("Continuation report, local SFT file, and dataset manifest hashes do not match")

    base_manifest_source = OLD_PROV / "kaggle-version1-sha256.json"
    base_manifest = read_json(base_manifest_source)
    base_dir = ROOT / base_manifest["base_directory"]
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError(f"Hashed Kaggle version-1 base is missing: {base_dir}")
    PROV.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_manifest_source, PROV / "kaggle-version1-sha256.json")

    # Validate the saved lock and artifact hashes. One source note was later removed from
    # the workspace, so preserve its recorded SHA and mark the exception explicitly.
    runner_path = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "run_downstream.py"
    old_config = read_json(OLD_OUT / "config_manifest.json")
    old_source_exceptions = verify_reused_config(old_config, OLD_OUT / "config_manifest.json",
        OLD_OUT / "cohort_manifest.json", OLD_OUT / "benchmark_lock.json", {"TASK_benchmark.md"})
    old_run = read_json(OLD_OUT / "run-COT.json")
    old_metrics = read_json(OLD_OUT / "metrics-COT.json")
    old_scores = read_jsonl(OLD_OUT / "btc-per-sample-COT.jsonl")
    old_predictions = read_jsonl(OLD_OUT / "predictions-COT.jsonl")
    old_requests = read_jsonl(OLD_OUT / "requests-COT.jsonl")
    ids = read_json(OLD_OUT / "cohort_manifest.json")["ids"]
    expected = {"num_predict": 768, "num_ctx": 8192, "num_batch": 512, "temperature": 0,
                "thinking": True, "protocol": "JSON", "max_action_turns": 5}
    old_run_settings = {key: old_run.get(key) for key in expected}
    if old_run.get("status") != "complete" or old_run.get("samples_completed") != 60 or old_run_settings != expected:
        raise RuntimeError("Previous epoch-2 branch C is incomplete or has a different inference setup")
    if old_config["agent"].get("cot_prompt_sha256_utf8") != "cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950":
        raise RuntimeError("Previous branch-C CoT checklist hash differs from the screened prompt")
    if old_config["asr"].get("selected_checkpoint_sha256") != "e31be8516152587134fb62e0d1690954221a0f02c53b6549cf79db58c8eb18ae":
        raise RuntimeError("Previous branch-C QUAN checkpoint differs from the selected checkpoint")
    if old_run.get("model_digest") != "973ae821f84016daad8db7ebf494fd2ddf98db7d1f14b5502ffb608cbfc8304d":
        raise RuntimeError("Previous epoch-2 SFT tag digest differs from the recorded branch-C digest")
    if not (len(ids) == 60 and [r["id"] for r in old_predictions] == ids
            and [r["id"] for r in old_scores] == ids and len(old_requests) > 0):
        raise RuntimeError("Previous branch-C outputs do not exactly cover the fixed 60-ID cohort")
    if old_metrics.get("btc_formula_score_before_runtime_gate_0_100") != 26.0:
        raise RuntimeError("Previous branch-C result is not the known 26.00/100 baseline")

    # The other comparison branch is the already completed Vietnamese 3-epoch + CoT run.
    # Validate its immutable source artifacts and reuse its recorded metrics; do not rerun it.
    vi_config = read_json(VI_OUT / "config_manifest.json")
    vi_source_exceptions = verify_reused_config(vi_config, VI_OUT / "config_manifest.json",
        VI_OUT / "cohort_manifest.json", VI_OUT / "run_lock.json",
        {"exp\\qwen35-sft-epoch3-cot-val60\\scripts\\analyze_epoch3.py"})
    vi_run = read_json(VI_OUT / "run-COT.json")
    vi_metrics = read_json(VI_OUT / "metrics-COT.json")
    vi_scores = read_jsonl(VI_OUT / "btc-per-sample-COT.jsonl")
    vi_predictions = read_jsonl(VI_OUT / "predictions-COT.jsonl")
    vi_requests = read_jsonl(VI_OUT / "requests-COT.jsonl")
    vi_ids = read_json(VI_OUT / "cohort_manifest.json")["ids"]
    if (vi_run.get("status") != "complete" or vi_run.get("samples_completed") != 60
            or vi_metrics.get("samples") != 60 or vi_metrics.get("gold_replay_failures") != 0
            or vi_metrics.get("runtime_errors") != 0):
        raise RuntimeError("Reusable Vietnamese epoch-3 + CoT branch is incomplete or has errors")
    if (vi_ids != ids or [row["id"] for row in vi_predictions] != ids
            or [row["id"] for row in vi_scores] != ids or len(vi_requests) == 0):
        raise RuntimeError("Vietnamese epoch-3 + CoT artifacts do not cover the same ordered cohort60")
    vi_score = vi_metrics.get("btc_formula_score_before_runtime_gate_0_100")
    if vi_score is None or abs(float(vi_score) - 23.27777777777778) > 1e-9:
        raise RuntimeError("Unexpected reused Vietnamese epoch-3 + CoT BTC baseline")
    if vi_config["agent"].get("cot_prompt_sha256_utf8") != old_config["agent"].get("cot_prompt_sha256_utf8"):
        raise RuntimeError("Vietnamese epoch-3 + CoT prompt differs from the shared screened prompt")
    for key in ("selected_checkpoint_sha256", "transcript_file_sha256", "kenlm_config_sha256"):
        if vi_config["asr"].get(key) != old_config["asr"].get(key):
            raise RuntimeError(f"Vietnamese epoch-3 + CoT ASR input differs from branch C: {key}")
    setting_keys = ("thinking", "temperature", "protocol", "num_predict", "num_ctx", "num_batch", "seed_python")
    if {key: vi_config["ollama"].get(key) for key in setting_keys} != {key: old_config["ollama"].get(key) for key in setting_keys}:
        raise RuntimeError("Vietnamese epoch-3 + CoT inference settings differ from branch C")

    for directory in (PROV / "adapter-final", PROV / "training", OUT / "merged-hf", OUT / "gguf",
                      OUT / "ollama", OUT / "smoke", OUT / "inference"):
        directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(NEW_ZIP) as archive:
        for name in ADAPTER_FILES:
            destination = PROV / "adapter-final" / name
            if destination.exists():
                raise FileExistsError(destination)
            destination.write_bytes(archive.read(new_prefix + "adapter_final/" + name))
        for member, destination_name in ((new_prefix + "training_report.json", "training_report.json"),
                                         (new_prefix + "checkpoints/checkpoint-243/trainer_state.json", "trainer_state-checkpoint-243.json")):
            (PROV / "training" / destination_name).write_bytes(archive.read(member))

    local_audit = {
        "experiment": EXP.name,
        "archive": {"path": str(NEW_ZIP.relative_to(ROOT)), "size_bytes": NEW_ZIP.stat().st_size,
                    "sha256": new_zip_hash, "entries": len(members)},
        "previous_archive": {"path": str(OLD_ZIP.relative_to(ROOT)), "size_bytes": OLD_ZIP.stat().st_size,
                             "sha256": old_zip_hash, "adapter_sha256": old_adapter_hash},
        "adapter": {"final": {"size_bytes": len(final_bytes), "sha256": final_hash},
                    "checkpoint_243": {"size_bytes": len(checkpoint_bytes), "sha256": sha256_bytes(checkpoint_bytes)},
                    "final_matches_checkpoint_243": True,
                    "matches_continuation_source_archive": report["continuation_source_adapter_sha256"] == old_adapter_hash,
                    "target_modules": config["target_modules"], "rank": config["r"], "alpha": config["lora_alpha"],
                    "base_model_name_or_path": config["base_model_name_or_path"], "files_sha256": file_hashes},
        "training_report": {key: report.get(key) for key in ("run_type", "continuation_source_adapter_sha256",
            "sft_train_sha256", "epochs_requested", "learning_rate", "optimizer_state", "scheduler_state",
            "warmup_steps", "seed", "world_size", "effective_batch_size", "max_seq_length", "train_rows",
            "trainable_parameters", "train_runtime_seconds", "train_loss")},
        "old_epoch2_training_report": {key: old_report.get(key) for key in ("epochs_requested", "learning_rate", "train_rows", "train_loss")},
        "dataset": {"manifest_path": str(dataset_manifest_path.relative_to(ROOT)),
                    "manifest_sha256": sha256_file(dataset_manifest_path), "actual_sft_train_sha256": train_hash,
                    "reported_sft_train_sha256": report["sft_train_sha256"], "rows": dataset["sft_examples"],
                    "training_report_binds_source": True},
        "base": {"source": base_manifest["kaggle_model_instance_version"],
                 "directory": base_manifest["base_directory"],
                 "manifest_sha256": sha256_file(base_manifest_source),
                 "input_hash_in_remote_training_report": False,
                 "note": "Continuation lineage and model path are verified; remote training report still does not identify exact base-weight bytes."},
        "old_epoch2_branch_C_reuse": {
            "experiment": "qwen35-sft-benchmark-v1", "status": old_run["status"], "samples": len(ids),
            "cohort_ids_sha256_newline_joined": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "model": old_run["model"], "model_digest": old_run["model_digest"],
            "official_btc_0_100": old_metrics["btc_formula_score_before_runtime_gate_0_100"],
            "hash_matches": old_metrics["hash_matches"], "M_U_G": [old_metrics["M_matched_calls"],
                old_metrics["U_unmatched_submitted_calls"], old_metrics["G_gold_calls"]],
            "predictions_sha256": sha256_file(OLD_OUT / "predictions-COT.jsonl"),
            "scores_sha256": sha256_file(OLD_OUT / "btc-per-sample-COT.jsonl"),
            "requests_sha256": sha256_file(OLD_OUT / "requests-COT.jsonl"),
            "prompt_sha256": old_config["agent"]["cot_prompt_sha256_utf8"],
            "transcript_sha256": old_config["asr"]["transcript_file_sha256"],
            "asr_checkpoint_sha256": old_config["asr"]["selected_checkpoint_sha256"],
            "ollama_version": old_config["ollama"]["version"],
            "settings": expected},
        "old_vietnamese_epoch3_COT_reuse": {
            "experiment": VI_EXP.name, "status": vi_run["status"], "samples": len(vi_ids),
            "cohort_ids_sha256_newline_joined": hashlib.sha256("\n".join(vi_ids).encode()).hexdigest(),
            "model": vi_run["model"], "model_digest": vi_run["model_digest"],
            "official_btc_0_100": vi_score, "local_30s_gated_diagnostic_0_100": vi_metrics["local_30s_gated_score_diagnostic_0_100"],
            "hash_matches": vi_metrics["hash_matches"], "M_U_G": [vi_metrics["M_matched_calls"],
                vi_metrics["U_unmatched_submitted_calls"], vi_metrics["G_gold_calls"]],
            "predictions_sha256": sha256_file(VI_OUT / "predictions-COT.jsonl"),
            "scores_sha256": sha256_file(VI_OUT / "btc-per-sample-COT.jsonl"),
            "requests_sha256": sha256_file(VI_OUT / "requests-COT.jsonl"),
            "prompt_sha256": vi_config["agent"]["cot_prompt_sha256_utf8"],
            "transcript_sha256": vi_config["asr"]["transcript_file_sha256"],
            "asr_checkpoint_sha256": vi_config["asr"]["selected_checkpoint_sha256"],
            "ollama_version": vi_config["ollama"]["version"], "rerun": False},
        "source_files_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__).resolve(), runner_path, dataset_manifest_path, OLD_OUT / "config_manifest.json",
            OLD_OUT / "run-COT.json", OLD_OUT / "metrics-COT.json", OLD_OUT / "predictions-COT.jsonl",
            OLD_OUT / "btc-per-sample-COT.jsonl", OLD_OUT / "requests-COT.jsonl",
            VI_OUT / "config_manifest.json", VI_OUT / "run-COT.json", VI_OUT / "metrics-COT.json",
            VI_OUT / "predictions-COT.jsonl", VI_OUT / "btc-per-sample-COT.jsonl", VI_OUT / "requests-COT.jsonl")},
        "reused_source_lock_exceptions": {"epoch2_C": old_source_exceptions,
            "epoch3_vietnamese_COT": vi_source_exceptions},
        "public_private_test_used": False,
        "environment_files_read": False,
    }
    write_json(PROV / "local-audit.json", local_audit)
    shutil.copy2(OLD_OUT / "cohort_manifest.json", OUT / "inference" / "reused-cohort-manifest.json")
    write_json(OUT / "preflight.json", {"status": "audited_and_extracted", "cohort_samples": 60,
        "cohort_ids_sha256_newline_joined": local_audit["old_epoch2_branch_C_reuse"]["cohort_ids_sha256_newline_joined"],
        "continuation_adapter_matches_previous": True, "train_data_sha256_matches": True,
        "epoch2_C_reused_without_rerun": True, "epoch2_C_official_btc_0_100": 26.0,
        "epoch3_Vietnamese_COT_reused_without_rerun": True, "epoch3_Vietnamese_COT_official_btc_0_100": vi_score,
        "base_model": local_audit["base"], "new_adapter_sha256": final_hash,
        "new_archive_sha256": new_zip_hash, "new_train_loss": report.get("train_loss"),
        "notes": ["No train/eval labels were accessed by this audit.",
                  "The continuation report lacks an exact base-weight hash."]})
    print(json.dumps({"status": "audited_and_extracted", "new_adapter_sha256": final_hash,
        "continuation_source_matches_previous": True, "sft_train_sha256_matches": True,
        "cohort_samples": len(ids), "reused_epoch2_C_btc": 26.0,
        "archive_sha256": new_zip_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
