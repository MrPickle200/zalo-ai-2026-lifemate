from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OUT = EXP / "output"
PROVENANCE = OUT / "model-provenance"
LLAMA = EXP / "third_party" / "llama.cpp"
LLAMA_BUILD = OUT / "llama-build-gcc"
BASE = ROOT / "models" / "kaggle-qwen3-5-4b-v1" / "Qwen3.5-4B"
MERGED = OUT / "merged-hf"
GGUF_DIR = OUT / "gguf"
LOG_DIR = OUT / "conversion-logs"
EXPECTED_COMMIT = "c0ab6e5c0bab3fb21e6cb9d2371f3c41d85ea775"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log_path: Path) -> float:
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite conversion log {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        result = process.wait()
    elapsed = time.time() - started
    if result:
        raise RuntimeError(f"Command failed ({result}); see {log_path.relative_to(ROOT)}")
    return elapsed


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def main() -> int:
    for required in (BASE / "config.json", MERGED / "config.json", LLAMA / "convert_hf_to_gguf.py"):
        if not required.is_file():
            raise FileNotFoundError(required)
    git = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True)
    commit = git.stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"llama.cpp commit changed: expected {EXPECTED_COMMIT}, found {commit}")
    converter_source = (LLAMA / "conversion" / "qwen.py").read_text(encoding="utf-8")
    if "Qwen3_5ForConditionalGeneration" not in converter_source:
        raise RuntimeError("Pinned llama.cpp converter does not register the Qwen3.5 architecture")
    quantizer_candidates = [LLAMA_BUILD / "bin" / "llama-quantize.exe",
                            LLAMA_BUILD / "bin" / "Release" / "llama-quantize.exe"]
    quantizer = next((path for path in quantizer_candidates if path.is_file()), None)
    if quantizer is None:
        raise FileNotFoundError("Build llama-quantize first; expected it under output/llama-build/bin")

    conversion_manifest_path = PROVENANCE / "conversion-manifest.json"
    if conversion_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {conversion_manifest_path}")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = GGUF_DIR / "in-progress"
    if tmp_dir.exists():
        raise FileExistsError(f"A previous conversion left {tmp_dir}; inspect before continuing")
    tmp_dir.mkdir()

    py = sys.executable
    branches = {"base": BASE, "sft": MERGED}
    records: dict[str, dict[str, Any]] = {}
    for branch, model_dir in branches.items():
        bf16_path = tmp_dir / f"{branch}-bf16.gguf"
        q4_path = tmp_dir / f"{branch}-Q4_K_M.gguf"
        convert_command = [py, str(LLAMA / "convert_hf_to_gguf.py"), str(model_dir),
                           "--outfile", str(bf16_path), "--outtype", "bf16", "--no-mtp"]
        quantize_command = [str(quantizer), str(bf16_path), str(q4_path), "Q4_K_M"]
        convert_seconds = run_logged(convert_command, LOG_DIR / f"{branch}-convert.log")
        quantize_seconds = run_logged(quantize_command, LOG_DIR / f"{branch}-quantize.log")
        records[branch] = {"hf_model_dir": str(model_dir.relative_to(ROOT)),
                           "convert_command": convert_command, "convert_seconds": convert_seconds,
                           "bf16_gguf": file_record(bf16_path), "quantize_command": quantize_command,
                           "quantize_seconds": quantize_seconds, "q4_k_m_gguf": file_record(q4_path)}

    final_paths: dict[str, dict[str, Path]] = {}
    for branch in branches:
        final_paths[branch] = {}
        for kind in ("bf16", "Q4_K_M"):
            source = tmp_dir / f"{branch}-{kind}.gguf"
            destination = GGUF_DIR / f"{branch}-{kind}.gguf"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            source.replace(destination)
            final_paths[branch][kind] = destination
    tmp_dir.rmdir()

    for branch in branches:
        records[branch]["bf16_gguf"] = file_record(final_paths[branch]["bf16"])
        records[branch]["q4_k_m_gguf"] = file_record(final_paths[branch]["Q4_K_M"])
    manifest = {
        "status": "converted_and_quantized",
        "llama_cpp_commit": commit,
        "llama_cpp_license": "MIT",
        "converter": "convert_hf_to_gguf.py",
        "converter_mode": "text-only Qwen3.5; --outtype bf16 --no-mtp",
        "quantizer": str(quantizer.relative_to(ROOT)),
        "quantization": "Q4_K_M",
        "same_converter_and_quantizer_for_base_and_sft": True,
        "merged_tree_sha256": json.loads((PROVENANCE / "merge-manifest.json").read_text(encoding="utf-8"))["merged_tree_sha256"],
        "branches": records,
    }
    write_json(conversion_manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "llama_cpp_commit": commit,
                      "base_q4_sha256": records["base"]["q4_k_m_gguf"]["sha256"],
                      "sft_q4_sha256": records["sft"]["q4_k_m_gguf"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
