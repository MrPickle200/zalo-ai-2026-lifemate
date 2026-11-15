from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OUT = EXP / "output"
LLAMA = EXP / "third_party" / "llama.cpp"
BUILD = OUT / "llama-build-gcc"
LOGS = OUT / "conversion-logs"
MANIFEST = OUT / "model-provenance" / "llama-toolchain-manifest.json"
THREADS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path) -> float:
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite build log {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        result = process.wait()
    elapsed = time.time() - started
    if result:
        raise RuntimeError(f"Build command failed ({result}); see {log_path.relative_to(ROOT)}")
    return elapsed


def version(command: str) -> str:
    result = subprocess.run([command, "--version"], check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).splitlines()[0]


def main() -> int:
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    gcc = shutil.which("gcc")
    gxx = shutil.which("c++")
    if not all((cmake, ninja, gcc, gxx)):
        raise FileNotFoundError("CMake, Ninja, GCC and G++ must be available on PATH")
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite {MANIFEST}")
    commit = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != "c0ab6e5c0bab3fb21e6cb9d2371f3c41d85ea775":
        raise RuntimeError(f"Unexpected llama.cpp commit: {commit}")

    configure = [cmake, "-S", str(LLAMA), "-B", str(BUILD), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_SERVER=OFF",
                 "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                 "-DLLAMA_BUILD_TOOLS=ON", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF",
                 "-DGGML_NATIVE=OFF", "-DGGML_AVX2=OFF", "-DGGML_CPU_REPACK=OFF",
                 "-DGGML_OPENMP=ON", "-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00",
                 "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00"]
    configure_seconds = run_logged(configure, LOGS / "llama-cmake-configure-gcc-stable.log")
    build = [cmake, "--build", str(BUILD), "--target", "llama-quantize", "--parallel", str(THREADS)]
    build_seconds = run_logged(build, LOGS / "llama-cmake-build-quantizer-gcc-stable.log")
    candidates = [BUILD / "bin" / "llama-quantize.exe", BUILD / "llama-quantize.exe"]
    binary = next((path for path in candidates if path.is_file()), None)
    if binary is None:
        raise FileNotFoundError(f"Built llama-quantize.exe not found under {BUILD}")
    result = subprocess.run([str(binary), "--help"], cwd=ROOT, capture_output=True, text=True)
    help_output = (result.stdout or result.stderr).splitlines()
    manifest: dict[str, Any] = {
        "status": "built",
        "llama_cpp_commit": commit,
        "llama_cpp_license": "MIT",
        "generator": "Ninja",
        "backend": "CPU, generic x86 (GGML_CUDA=OFF, GGML_AVX2=OFF, GGML_NATIVE=OFF)",
        "cmake": version(cmake), "ninja": version(ninja),
        "gcc": version(gcc), "gxx": version(gxx),
        "threads": THREADS,
        "configure_command": configure,
        "build_command": build,
        "configure_seconds": configure_seconds,
        "build_seconds": build_seconds,
        "quantizer": {"path": str(binary.relative_to(ROOT)), "size_bytes": binary.stat().st_size,
                      "sha256": sha256_file(binary), "help_output_first_lines": help_output[:3]},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "llama_cpp_commit": commit,
                      "quantizer": manifest["quantizer"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
