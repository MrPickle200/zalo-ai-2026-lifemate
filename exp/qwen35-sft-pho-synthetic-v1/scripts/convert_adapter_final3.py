from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output" / "adapter-final-3epoch-benchmark"
PROV = OUT / "model-provenance"
LLAMA = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "third_party" / "llama.cpp"
QUANTIZER = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "llama-build-gcc" / "bin" / "llama-quantize.exe"
MERGED = OUT / "merged-hf"
GGUF = OUT / "gguf"
LOGS = OUT / "conversion-logs"
EXPECTED_COMMIT = "c0ab6e5c0bab3fb21e6cb9d2371f3c41d85ea775"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log_path: Path) -> float:
    if log_path.exists():
        raise FileExistsError(log_path)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log_path.relative_to(ROOT)}")
    return time.time() - started


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def main() -> int:
    for path in (MERGED / "config.json", LLAMA / "convert_hf_to_gguf.py", QUANTIZER):
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Pinned llama.cpp commit mismatch: {commit}")
    if (PROV / "conversion-manifest.json").exists():
        raise FileExistsError("Conversion manifest exists; refusing to overwrite")
    GGUF.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    bf16 = GGUF / "adapter-final-3epoch-bf16.gguf"
    q4 = GGUF / "adapter-final-3epoch-Q4_K_M.gguf"
    if bf16.exists() or q4.exists():
        raise FileExistsError("GGUF output already exists")
    convert_command = [sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
                       "--outfile", str(bf16), "--outtype", "bf16", "--no-mtp"]
    quantize_command = [str(QUANTIZER), str(bf16), str(q4), "Q4_K_M"]
    convert_seconds = run_logged(convert_command, LOGS / "adapter-final-3epoch-convert.log")
    quantize_seconds = run_logged(quantize_command, LOGS / "adapter-final-3epoch-quantize.log")
    manifest = {
        "status": "converted_and_quantized", "llama_cpp_commit": commit,
        "llama_cpp_license": "MIT", "converter": str((LLAMA / "convert_hf_to_gguf.py").relative_to(ROOT)),
        "converter_mode": "Qwen3.5 text model; --outtype bf16 --no-mtp",
        "quantizer": str(QUANTIZER.relative_to(ROOT)), "quantizer_sha256": sha256_file(QUANTIZER),
        "quantization": "Q4_K_M", "convert_command": convert_command,
        "convert_seconds": convert_seconds, "bf16_gguf": record(bf16),
        "quantize_command": quantize_command, "quantize_seconds": quantize_seconds,
        "q4_k_m_gguf": record(q4),
        "epoch2_comparison_gguf_sha256": json.loads((ROOT / "exp" / "qwen35-sft-benchmark-v1" /
            "output" / "model-provenance" / "conversion-manifest.json").read_text(encoding="utf-8"))[
                "branches"]["sft"]["q4_k_m_gguf"]["sha256"],
        "note": "Same pinned llama.cpp revision, converter mode, quantizer binary and Q4_K_M setting as old branch C; only SFT weights differ."}
    manifest["generated_bf16_gguf_cleanup"] = "Temporary BF16 GGUF and merged-HF shards are deleted after Q4_K_M is verified."
    write_json(PROV / "conversion-manifest.json", manifest)
    if sha256_file(q4) != manifest["q4_k_m_gguf"]["sha256"]:
        raise RuntimeError("Final Q4_K_M hash differs from the conversion manifest")
    bf16.unlink()
    shutil.rmtree(MERGED)
    print(json.dumps({"status": manifest["status"], "bf16": manifest["bf16_gguf"],
                      "q4_k_m": manifest["q4_k_m_gguf"], "llama_cpp_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
