from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OUT = EXP / "output"
PROVENANCE = OUT / "model-provenance"
GGUF_DIR = OUT / "gguf"
OLLAMA_DIR = OUT / "ollama"
CONVERSION = PROVENANCE / "conversion-manifest.json"
OLLAMA_EXE = Path(os.environ.get("OLLAMA_EXE", r"C:\Users\PC\AppData\Local\Programs\Ollama\ollama.exe"))
HOST = "http://127.0.0.1:11434"
MODELS = {
    "base": "lifemate-qwen35-kagglev1-base-q4km:latest",
    "sft": "lifemate-qwen35-kagglev1-sft-epoch2-q4km:latest",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(HOST + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run this script with system Python, outside .venv, to reach local Ollama")
    if not OLLAMA_EXE.is_file():
        raise FileNotFoundError(f"Ollama executable not found: {OLLAMA_EXE}")
    manifest = json.loads(CONVERSION.read_text(encoding="utf-8"))
    if manifest.get("status") != "converted_and_quantized":
        raise RuntimeError("GGUF conversion manifest is incomplete")
    version = api("/api/version")
    current = api("/api/tags").get("models", [])
    by_name = {model["name"]: model for model in current}
    for tag in MODELS.values():
        if tag in by_name:
            raise FileExistsError(f"Refusing to overwrite existing Ollama tag {tag}")

    OLLAMA_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for branch, tag in MODELS.items():
        artifact = manifest["branches"][branch]["q4_k_m_gguf"]
        model_path = ROOT / artifact["path"]
        if not model_path.is_file() or model_path.stat().st_size != artifact["size_bytes"]:
            raise RuntimeError(f"GGUF size mismatch for {branch}: {model_path}")
        actual_hash = sha256_file(model_path)
        if actual_hash != artifact["sha256"]:
            raise RuntimeError(f"GGUF SHA-256 mismatch for {branch}: {model_path}")

        model_file = OLLAMA_DIR / f"{branch}.Modelfile"
        log_file = OLLAMA_DIR / f"{branch}-create.log"
        if model_file.exists() or log_file.exists():
            raise FileExistsError(f"Refusing to overwrite existing Ollama artifacts for {branch}")
        model_file.write_text(
            "\n".join((
                f"FROM {model_path.as_posix()}",
                "PARAMETER num_ctx 8192",
                "PARAMETER num_batch 512",
                "PARAMETER num_predict 768",
                "PARAMETER temperature 0",
                "",
            )), encoding="utf-8")
        command = [str(OLLAMA_EXE), "create", tag, "-f", str(model_file)]
        started = time.time()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                errors="replace", timeout=3600)
        log_file.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Ollama create failed for {branch} ({result.returncode}); see {log_file}")

        tags = api("/api/tags").get("models", [])
        created = next((model for model in tags if model.get("name") == tag), None)
        if created is None:
            raise RuntimeError(f"Ollama did not expose the created tag {tag} through /api/tags")
        shown = api("/api/show", {"model": tag})
        records[branch] = {
            "tag": tag,
            "gguf": artifact,
            "modelfile": str(model_file.relative_to(ROOT)),
            "modelfile_sha256": sha256_file(model_file),
            "create_log": str(log_file.relative_to(ROOT)),
            "create_seconds": time.time() - started,
            "ollama_digest": created.get("digest"),
            "ollama_size_bytes": created.get("size"),
            "model_info": shown.get("model_info", {}),
            "template": shown.get("template"),
            "parameters": shown.get("parameters"),
        }

    result = {
        "status": "registered",
        "python_executable": sys.executable,
        "ollama_version": version.get("version"),
        "ollama_host": HOST,
        "settings": {"num_ctx": 8192, "num_batch": 512, "num_predict": 768,
                     "temperature": 0, "format": "json", "think": True},
        "branches": records,
    }
    output = PROVENANCE / "ollama-models.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "ollama_version": result["ollama_version"],
                      "tags": {branch: record["tag"] for branch, record in records.items()},
                      "digests": {branch: record["ollama_digest"] for branch, record in records.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
