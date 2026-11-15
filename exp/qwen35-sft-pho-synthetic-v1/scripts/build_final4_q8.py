from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\zalo-ai-challenge-2026\lifemate")
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/adapter-final-4epoch-q8-benchmark"
PROV = OUT / "model-provenance"
GGUF = OUT / "gguf"
LOGS = OUT / "conversion-logs"
BASE = ROOT / "models/kaggle-qwen3-5-4b-v1/Qwen3.5-4B"
SOURCE_MERGE = EXP / "scripts/merge_adapter_final4.py"
SOURCE_CONVERT = EXP / "scripts/convert_adapter_final4.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_merge():
    manifest_path = PROV / "merge-manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for row in manifest["merged_files"]:
            path = OUT / "merged-hf" / row["path"]
            if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha(path) != row["sha256"]:
                raise RuntimeError(f"Existing merged file differs: {path}")
        return manifest
    merge = load_module("final4_q8_reused_merge", SOURCE_MERGE)
    merge.OUT = OUT
    merge.PROVENANCE = PROV
    saved_argv = sys.argv
    try:
        sys.argv = [str(SOURCE_MERGE), "--base-dir", str(BASE)]
        if merge.main() != 0:
            raise RuntimeError("Adapter merge failed.")
    finally:
        sys.argv = saved_argv
    return read_json(manifest_path)


def ensure_q8(merge_manifest):
    convert = load_module("final4_q8_reused_converter", SOURCE_CONVERT)
    commit = subprocess.run(
        ["git", "-C", str(convert.LLAMA), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != convert.EXPECTED_COMMIT:
        raise RuntimeError(f"Pinned llama.cpp commit differs: {commit}")
    if sha(convert.QUANTIZER) != read_json(EXP / "output/adapter-final-4epoch-benchmark/model-provenance/conversion-manifest.json")["quantizer_sha256"]:
        raise RuntimeError("Pinned quantizer changed.")
    GGUF.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    bf16 = GGUF / "adapter-final-4epoch-bf16.gguf"
    q8 = GGUF / "adapter-final-4epoch-Q8_0.gguf"
    manifest_path = PROV / "conversion-manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["q8_0_gguf"]["sha256"] != sha(q8):
            raise RuntimeError("Existing Q8 file differs from conversion manifest.")
        return manifest
    if q8.exists():
        raise FileExistsError(f"Unverified Q8 file exists: {q8}")
    if not bf16.exists():
        convert_command = [
            sys.executable, str(convert.LLAMA / "convert_hf_to_gguf.py"),
            str(OUT / "merged-hf"), "--outfile", str(bf16),
            "--outtype", "bf16", "--no-mtp",
        ]
        convert_seconds = convert.run_logged(
            convert_command, LOGS / "adapter-final-4epoch-convert.log"
        )
    else:
        convert_command = ["already-present", str(bf16)]
        convert_seconds = None
    quantize_command = [str(convert.QUANTIZER), str(bf16), str(q8), "Q8_0"]
    quantize_seconds = convert.run_logged(
        quantize_command, LOGS / "adapter-final-4epoch-quantize-q8.log"
    )
    manifest = {
        "status": "converted_and_quantized",
        "source": "verified Kaggle v1 base plus adapter_final-4-epochs.zip",
        "merge_manifest_sha256": sha(PROV / "merge-manifest.json"),
        "merge_tree_sha256": merge_manifest["merged_tree_sha256"],
        "adapter_archive_sha256": merge_manifest["adapter_archive_sha256"],
        "llama_cpp_commit": commit,
        "quantizer": str(convert.QUANTIZER.relative_to(ROOT)),
        "quantizer_sha256": sha(convert.QUANTIZER),
        "quantization": "Q8_0",
        "convert_command": convert_command,
        "convert_seconds": convert_seconds,
        "bf16_gguf": convert.record(bf16),
        "quantize_command": quantize_command,
        "quantize_seconds": quantize_seconds,
        "q8_0_gguf": convert.record(q8),
        "build_script_sha256": sha(Path(__file__).resolve()),
    }
    write_json(manifest_path, manifest)
    return manifest


def main():
    merge_manifest = ensure_merge()
    convert_manifest = ensure_q8(merge_manifest)
    print(json.dumps({
        "status": convert_manifest["status"],
        "model": convert_manifest["q8_0_gguf"],
        "merge_tree_sha256": merge_manifest["merged_tree_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()


