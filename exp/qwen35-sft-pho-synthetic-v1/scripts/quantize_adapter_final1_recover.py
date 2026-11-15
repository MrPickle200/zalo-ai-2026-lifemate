from __future__ import annotations
import hashlib, json, shutil, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp" / "qwen35-sft-pho-synthetic-v1" / "output" / "adapter-final-1epoch-benchmark"
PROV = OUT / "model-provenance"
LOGS = OUT / "conversion-logs"
BF16 = OUT / "gguf" / "adapter-final-1epoch-bf16.gguf"
Q4 = OUT / "gguf" / "adapter-final-1epoch-Q4_K_M.gguf"
LLAMA = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "third_party" / "llama.cpp"
QUANTIZER = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "llama-build-gcc" / "bin" / "llama-quantize.exe"
EXPECTED_COMMIT = "c0ab6e5c0bab3fb21e6cb9d2371f3c41d85ea775"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    if (PROV / "conversion-manifest.json").exists():
        raise FileExistsError("Conversion manifest already exists; refusing to overwrite")
    if not BF16.is_file() or BF16.stat().st_size < 7_000_000_000:
        raise RuntimeError(f"Expected complete BF16 GGUF missing or too small: {BF16}")
    commit = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Pinned llama.cpp commit mismatch: {commit}")
    temp_root = Path.home() / "AppData" / "Local" / "Temp" / "lifemate-qwen35-adapter-final1"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_bf16 = temp_root / BF16.name
    temp_q4 = temp_root / Q4.name
    if temp_bf16.exists():
        if sha256(temp_bf16) != sha256(BF16):
            raise RuntimeError(f"Temporary BF16 exists with different hash: {temp_bf16}")
        BF16.unlink()
    else:
        shutil.move(str(BF16), str(temp_bf16))
    prior_q4 = {"existed": Q4.exists(), "size_bytes": Q4.stat().st_size if Q4.exists() else 0,
                "sha256": sha256(Q4) if Q4.exists() else None,
                "note": "Partial output from the first quantization attempt; replaced atomically after successful retry."}
    if temp_q4.exists():
        raise FileExistsError(temp_q4)
    log_path = LOGS / "adapter-final-1epoch-quantize-retry.log"
    if log_path.exists():
        raise FileExistsError(log_path)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen([str(QUANTIZER), str(temp_bf16), str(temp_q4), "Q4_K_M"], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end="", flush=True)
        code = process.wait()
    elapsed = time.time() - started
    if code != 0 or not temp_q4.is_file() or temp_q4.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"Quantization retry failed (exit={code}); see {log_path}")
    stage = Q4.with_suffix(Q4.suffix + ".staging")
    shutil.copyfile(temp_q4, stage)
    if sha256(stage) != sha256(temp_q4):
        raise RuntimeError("Staged Q4 hash differs from quantizer output")
    stage.replace(Q4)
    if sha256(Q4) != sha256(temp_q4):
        raise RuntimeError("Final Q4 hash differs from quantizer output")
    def rec(path: Path, label: str) -> dict:
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path), "label": label}
    manifest = {
        "status": "converted_and_quantized", "llama_cpp_commit": commit, "llama_cpp_license": "MIT",
        "converter": str((LLAMA / "convert_hf_to_gguf.py").relative_to(ROOT)),
        "converter_mode": "Qwen3.5 text model; --outtype bf16 --no-mtp",
        "quantizer": str(QUANTIZER.relative_to(ROOT)), "quantizer_sha256": sha256(QUANTIZER),
        "quantization": "Q4_K_M", "quantize_command": [str(QUANTIZER), str(temp_bf16), str(temp_q4), "Q4_K_M"],
        "quantize_seconds_retry": elapsed, "bf16_gguf": rec(temp_bf16, "Retained on C: temporary storage because D: lacked capacity"),
        "q4_k_m_gguf": rec(Q4, "Benchmark model in experiment output"),
        "first_quantization_attempt": {"status": "failed_due_to_output_disk_capacity", "log": str((LOGS / "adapter-final-1epoch-quantize.log").relative_to(ROOT)), "partial_q4_replaced": prior_q4},
        "retry_log": str(log_path.relative_to(ROOT)),
        "epoch2_comparison_gguf_sha256": json.loads((ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "model-provenance" / "conversion-manifest.json").read_text(encoding="utf-8"))["branches"]["sft"]["q4_k_m_gguf"]["sha256"],
        "note": "Same pinned llama.cpp revision, converter mode, quantizer binary and Q4_K_M setting as comparison branches. BF16 retained at the recorded temporary path; final Q4 is stored in the project."
    }
    PROV.mkdir(parents=True, exist_ok=True)
    (PROV / "conversion-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "bf16": manifest["bf16_gguf"], "q4_k_m": manifest["q4_k_m_gguf"], "failed_partial": prior_q4}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
