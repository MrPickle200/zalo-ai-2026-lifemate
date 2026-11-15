from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "exp" / "qwen35-sft-pho-synthetic-v1" / "output" / "adapter_final-4-epochs.zip"
OUT = ROOT / "exp" / "qwen35-sft-pho-synthetic-v1" / "output" / "adapter-final-4epoch-benchmark"
PROVENANCE = OUT / "model-provenance"
ADAPTER_PREFIX = "adapter_final/"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_adapter(target: Path) -> dict[str, str]:
    target.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        for filename in ADAPTER_FILES:
            member = ADAPTER_PREFIX + filename
            if member not in names:
                raise FileNotFoundError(f"Missing {member} in source adapter_final-4-epochs.zip")
            data = archive.read(member)
            destination = target / filename
            expected_hash = hashlib.sha256(data).hexdigest()
            if destination.exists():
                if sha256_file(destination) != expected_hash:
                    raise RuntimeError(f"Existing adapter extraction differs from adapter_final-4-epochs.zip: {filename}")
            else:
                destination.write_bytes(data)
            hashes[filename] = expected_hash
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Qwen3.5-4B directory downloaded from Kaggle model version 1")
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError(f"No config.json under base directory {base_dir}")
    base_manifest_path = ROOT / "exp" / "qwen35-sft-epoch3-en-cot-val60" / "output" / "model-provenance" / "kaggle-version1-sha256.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError("Run fetch_kaggle_base.py and verify its version-1 file manifest first")
    base_manifest = read_json(base_manifest_path)
    if (ROOT / base_manifest["base_directory"]).resolve() != base_dir:
        raise RuntimeError("Base directory does not match the independently hashed Kaggle v1 path")
    manifest_files = {row["path"]: row for row in base_manifest["files"]}
    current_files = {path.relative_to(base_dir).as_posix(): path
                     for path in base_dir.rglob("*") if path.is_file()
                     and ".cache" not in path.relative_to(base_dir).parts}
    if set(current_files) != set(manifest_files):
        raise RuntimeError("Kaggle base file set changed after its version-1 hash manifest was written")
    for relative, path in current_files.items():
        expected = manifest_files[relative]
        if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Kaggle version-1 file hash/size changed: {relative}")

    adapter_dir = PROVENANCE / "adapter-final"
    adapter_hashes = extract_adapter(adapter_dir)

    import accelerate
    import peft
    import psutil
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    config = read_json(base_dir / "config.json")
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError(f"Unexpected base model_type: {config.get('model_type')}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, use_fast=True)
    sft_tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True, use_fast=True)
    same_vocab = base_tokenizer.get_vocab() == sft_tokenizer.get_vocab()
    same_chat_template = base_tokenizer.chat_template == sft_tokenizer.chat_template
    if not same_vocab:
        raise RuntimeError("Adapter tokenizer vocabulary differs from the exact Kaggle base tokenizer")
    if not same_chat_template:
        raise RuntimeError("Adapter tokenizer chat template differs from the exact Kaggle base template")

    old_merge_manifest = read_json(ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" /
                                   "model-provenance" / "merge-manifest.json")
    base_device_map = old_merge_manifest["merge"]["hf_device_map"]
    adapter_device_map = {f"base_model.model.{name}": device for name, device in base_device_map.items()}
    # Reuse the exact successful epoch-2 placement. Automatic placement can vary with current free RAM
    # and put an unused MTP module on disk, which PEFT 0.21.2 cannot rename in its offload index.
    max_memory = old_merge_manifest["merge"]["max_memory"]
    if "disk" in base_device_map.values():
        raise RuntimeError("The recorded epoch-2 device map unexpectedly includes disk offload")
    offload_dir = OUT / "merge-offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForMultimodalLM.from_pretrained(
        base_dir, dtype="auto", low_cpu_mem_usage=True, device_map=base_device_map,
        offload_buffers=True, local_files_only=True,
    )
    model.eval()
    model = PeftModel.from_pretrained(
        model, adapter_dir, is_trainable=False, local_files_only=True,
        device_map=adapter_device_map,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    probe_ids = sft_tokenizer("LifeMate tool selection: return one JSON action.", return_tensors="pt")["input_ids"].to(input_device)
    with torch.inference_mode():
        before_logits = model(input_ids=probe_ids).logits[:, -1, :].float().cpu()
    merged = model.merge_and_unload(safe_merge=True)
    merged.eval()
    merged_input_device = merged.get_input_embeddings().weight.device
    with torch.inference_mode():
        after_logits = merged(input_ids=probe_ids.to(merged_input_device)).logits[:, -1, :].float().cpu()
    max_abs_diff = float((before_logits - after_logits).abs().max())
    same_argmax = int(before_logits.argmax()) == int(after_logits.argmax())
    if not same_argmax or not torch.isfinite(after_logits).all():
        raise RuntimeError(f"Merge fidelity check failed: same_argmax={same_argmax}, max_abs_diff={max_abs_diff}")

    merged_dir = OUT / "merged-hf"
    if merged_dir.exists():
        if any(merged_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty merged model directory {merged_dir}")
        merged_dir.rmdir()
    merged_dir.mkdir(parents=True)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    sft_tokenizer.save_pretrained(merged_dir)
    chat_template = adapter_dir / "chat_template.jinja"
    if chat_template.is_file() and not (merged_dir / "chat_template.jinja").exists():
        shutil.copy2(chat_template, merged_dir / "chat_template.jinja")

    merged_files = [{"path": path.relative_to(merged_dir).as_posix(), "size_bytes": path.stat().st_size,
                     "sha256": sha256_file(path)}
                    for path in sorted(p for p in merged_dir.rglob("*") if p.is_file())]
    report = {
        "status": "merged",
        "base_source": "Kaggle model limamateus/qwen3-5-4b/pytorch/default/1; same verified base as epoch-2 branch C",
        "base_directory": str(base_dir.relative_to(ROOT)),
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "adapter_archive_sha256": sha256_file(ARCHIVE),
        "adapter_files_sha256": adapter_hashes,
        "tokenizer_comparison": {
            "base_tokenizer_json_sha256": sha256_file(base_dir / "tokenizer.json"),
            "adapter_saved_tokenizer_json_sha256": adapter_hashes["tokenizer.json"],
            "base_adapter_vocab_equal": same_vocab,
            "base_adapter_chat_template_equal": same_chat_template,
            "note": "The serialized tokenizer.json bytes differ, but the effective vocabulary and exact chat template match.",
        },
        "merge": {"implementation": "PEFT PeftModel.merge_and_unload(safe_merge=True)",
                  "dtype": str(next(merged.parameters()).dtype), "max_memory": max_memory,
                  "hf_device_map": getattr(merged, "hf_device_map", None),
                  "probe": "raw text next-token logits before and after merge",
                  "probe_same_argmax": same_argmax, "probe_max_abs_logit_diff": max_abs_diff},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "accelerate": accelerate.__version__},
        "merged_files": merged_files,
        "merged_file_count": len(merged_files),
        "merged_tree_sha256": hashlib.sha256(
            "".join(f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n" for row in merged_files).encode("utf-8")
        ).hexdigest(),
    }
    write_json(PROVENANCE / "merge-manifest.json", report)
    print(json.dumps({"status": report["status"], "merged_tree_sha256": report["merged_tree_sha256"],
                      "probe_same_argmax": same_argmax,
                      "probe_max_abs_logit_diff": max_abs_diff, "files": len(merged_files)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


