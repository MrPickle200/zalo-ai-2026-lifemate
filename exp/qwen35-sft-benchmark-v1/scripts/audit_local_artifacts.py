from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "exp" / "qwen35-sft-next-action-v1"
ARCHIVE = SOURCE / "results.zip"
UPLOAD = SOURCE / "kaggle-upload"
NOTEBOOK = SOURCE / "kaggle-notebook" / "qwen35_lifemate_sft.ipynb"
OUT = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "model-provenance"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    manifest_path = UPLOAD / "dataset_manifest.json"
    sft_path = UPLOAD / "sft_train.jsonl"
    if not manifest_path.is_file() or not sft_path.is_file() or not NOTEBOOK.is_file():
        raise FileNotFoundError("SFT package or source notebook is incomplete")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    archive_inventory = []
    selected: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            row = {"path": info.filename, "size_bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
            if info.filename in {
                "qwen35_lifemate_sft_v1/adapter_final/adapter_model.safetensors",
                "qwen35_lifemate_sft_v1/checkpoints/checkpoint-486/adapter_model.safetensors",
                "qwen35_lifemate_sft_v1/adapter_final/adapter_config.json",
                "qwen35_lifemate_sft_v1/adapter_final/tokenizer.json",
                "qwen35_lifemate_sft_v1/adapter_final/tokenizer_config.json",
                "qwen35_lifemate_sft_v1/adapter_final/chat_template.jinja",
                "qwen35_lifemate_sft_v1/training_report.json",
            }:
                data = archive.read(info.filename)
                row["sha256"] = sha256_bytes(data)
                selected[info.filename] = {"size_bytes": info.file_size, "sha256": row["sha256"]}
                if info.filename.endswith("adapter_config.json"):
                    selected[info.filename]["config"] = json.loads(data)
                elif info.filename.endswith("tokenizer_config.json"):
                    selected[info.filename]["config"] = json.loads(data)
                elif info.filename.endswith("training_report.json"):
                    report = json.loads(data)
                    selected[info.filename]["report"] = {
                        key: report.get(key) for key in
                        ("epochs_requested", "learning_rate", "world_size", "effective_batch_size",
                         "max_seq_length", "train_rows", "train_runtime_seconds", "train_loss")
                    }
                    selected[info.filename]["report_has_training_input_hash"] = any(
                        "hash" in key.lower() or "sha256" in key.lower() for key in report
                    )
            archive_inventory.append(row)

    sft_hash = sha256_file(sft_path)
    with sft_path.open(encoding="utf-8") as stream:
        sft_rows = sum(1 for line in stream if line.strip())
    adapter_final = selected["qwen35_lifemate_sft_v1/adapter_final/adapter_model.safetensors"]
    epoch2 = selected["qwen35_lifemate_sft_v1/checkpoints/checkpoint-486/adapter_model.safetensors"]
    adapter_equal = adapter_final["sha256"] == epoch2["sha256"]
    adapter_config = selected["qwen35_lifemate_sft_v1/adapter_final/adapter_config.json"]["config"]
    model_path = adapter_config.get("base_model_name_or_path")
    exact_name_dirs = [path for path in (ROOT / "models").iterdir()
                       if path.is_dir() and path.name.casefold() == "qwen3.5-4b"]
    sft_manifest_match = sft_hash == manifest.get("sft_train_sha256")
    train_rows_match = sft_rows == manifest.get("sft_examples")

    notebook_digest = sha256_file(NOTEBOOK)
    audit = {
        "experiment": "qwen35-sft-benchmark-v1",
        "archive": {"path": str(ARCHIVE.relative_to(ROOT)), "size_bytes": ARCHIVE.stat().st_size,
                    "sha256": sha256_file(ARCHIVE), "entries": len(archive_inventory)},
        "archive_inventory_file": "archive-inventory.json",
        "adapter": {
            "final": adapter_final,
            "epoch_2_checkpoint_486": epoch2,
            "final_matches_epoch_2": adapter_equal,
            "target_modules": adapter_config.get("target_modules"),
            "rank": adapter_config.get("r"), "alpha": adapter_config.get("lora_alpha"),
            "dropout": adapter_config.get("lora_dropout"), "bias": adapter_config.get("bias"),
            "peft_type": adapter_config.get("peft_type"), "peft_version": adapter_config.get("peft_version"),
            "adapter_base_model_path": model_path, "base_revision_in_adapter_config": adapter_config.get("revision"),
        },
        "tokenizer": {
            path.rsplit("/", 1)[-1]: {"size_bytes": value["size_bytes"], "sha256": value["sha256"]}
            for path, value in selected.items()
            if path.endswith(("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"))
        },
        "training_report": selected["qwen35_lifemate_sft_v1/training_report.json"],
        "dataset": {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "sft_rows": sft_rows, "manifest_sft_examples": manifest.get("sft_examples"),
            "actual_sft_train_sha256": sft_hash,
            "manifest_sft_train_sha256": manifest.get("sft_train_sha256"),
            "sft_hash_matches_manifest": sft_manifest_match,
            "row_count_matches_manifest": train_rows_match,
            "training_report_records_input_hash": selected[
                "qwen35_lifemate_sft_v1/training_report.json"].get("report_has_training_input_hash"),
            "note": "Manifest/file consistency is verified locally; training_report.json does not bind the executed training run to this exact input hash.",
        },
        "notebook": {
            "path": str(NOTEBOOK.relative_to(ROOT)), "sha256": notebook_digest,
            "kaggle_kernel_id_or_source_metadata_present": any(
                key in notebook.get("metadata", {}) for key in ("kaggle", "kernel_id", "kernel_sources", "dataset_sources")
            ),
            "model_discovery_uses_input_files": "MODEL_DIR = unique_path(model_configs" in notebook_source,
            "adapter_or_base_weight_hash_in_notebook": any(
                phrase in notebook_source.lower() for phrase in ("model sha256", "weight sha256", "base_hash")
            ),
        },
        "base_provenance": {
            "kaggle_model_ref": "limamateus/qwen3-5-4b",
            "training_input_path": model_path,
            "framework": "PyTorch", "variant": "default", "version": 1,
            "model_page": "https://www.kaggle.com/models/limamateus/qwen3-5-4b",
            "exact_file_inventory_and_weight_hashes_available_locally": False,
            "base_config_model_index_and_weight_shards_in_adapter_archive": False,
            "exact_named_base_directory_under_project_models": [str(path.relative_to(ROOT)) for path in exact_name_dirs],
            "kaggle_api_token_available_at_audit": bool(os.environ.get("KAGGLE_API_TOKEN")),
            "status": "blocked_unverified_exact_base_weights",
            "reason": "The adapter records a Kaggle mount path and no revision/hash; results.zip contains only adapters/tokenizer/training metadata. A base file inventory/hash is still required before conversion.",
        },
        "conversion_or_benchmark_started": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "archive-inventory.json", archive_inventory)
    write_json(OUT / "local-audit.json", audit)
    print(json.dumps({"archive_sha256": audit["archive"]["sha256"],
                      "adapter_sha256": adapter_final["sha256"],
                      "adapter_matches_epoch2": adapter_equal,
                      "sft_rows": sft_rows,
                      "sft_hash_matches_manifest": sft_manifest_match,
                      "base_status": audit["base_provenance"]["status"]}, indent=2))
    return 0 if adapter_equal and sft_manifest_match and train_rows_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
