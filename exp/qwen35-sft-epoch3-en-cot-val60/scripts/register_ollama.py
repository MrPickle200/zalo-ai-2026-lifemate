from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "qwen35-sft-epoch3-en-cot-val60"
OUT = EXP / "output"
PROV = OUT / "model-provenance"
GGUF = OUT / "gguf" / "sft-epoch3-en-Q4_K_M.gguf"
OLLAMA_DIR = OUT / "ollama"
TAG = "lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
HOST = "http://127.0.0.1:11434"
OLLAMA_EXE = Path(os.environ.get("OLLAMA_EXE", r"C:\Users\PC\AppData\Local\Programs\Ollama\ollama.exe"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(HOST + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run with system Python outside .venv; it must reach local Ollama")
    if not OLLAMA_EXE.is_file():
        raise FileNotFoundError(OLLAMA_EXE)
    conversion = json.loads((PROV / "conversion-manifest.json").read_text(encoding="utf-8"))
    if conversion.get("status") != "converted_and_quantized":
        raise RuntimeError("Conversion is incomplete")
    artifact = conversion["q4_k_m_gguf"]
    if (ROOT / artifact["path"]).resolve() != GGUF.resolve() or not GGUF.is_file():
        raise RuntimeError("Conversion manifest points to an unexpected Q4 GGUF")
    if GGUF.stat().st_size != artifact["size_bytes"] or sha256_file(GGUF) != artifact["sha256"]:
        raise RuntimeError("Q4 GGUF hash/size mismatch")

    version = api("/api/version").get("version")
    expected_version = json.loads((OLD_PROV := ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" /
                                   "model-provenance" / "ollama-models.json").read_text(encoding="utf-8"))["ollama_version"]
    if version != expected_version:
        raise RuntimeError(f"Ollama version changed from branch C: expected {expected_version}, found {version}")
    if any(row.get("name") == TAG for row in api("/api/tags").get("models", [])):
        raise FileExistsError(f"Refusing to overwrite Ollama tag {TAG}")
    if (PROV / "ollama-models.json").exists():
        raise FileExistsError("Ollama manifest exists; refusing to overwrite")
    OLLAMA_DIR.mkdir(parents=True, exist_ok=True)
    modelfile = OLLAMA_DIR / "sft-epoch3-en.Modelfile"
    create_log = OLLAMA_DIR / "sft-epoch3-en-create.log"
    if modelfile.exists() or create_log.exists():
        raise FileExistsError("Ollama registration artifacts already exist")
    modelfile.write_text("\n".join((f"FROM {GGUF.as_posix()}", "PARAMETER num_ctx 8192",
        "PARAMETER num_batch 512", "PARAMETER num_predict 768", "PARAMETER temperature 0", "")), encoding="utf-8")
    started = time.time()
    result = subprocess.run([str(OLLAMA_EXE), "create", TAG, "-f", str(modelfile)],
                            cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=3600)
    create_log.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"ollama create failed ({result.returncode}); see {create_log}")
    tag_rows = api("/api/tags").get("models", [])
    found = next((row for row in tag_rows if row.get("name") == TAG), None)
    if found is None:
        raise RuntimeError("Ollama did not expose the registered epoch-3 tag")
    shown = api("/api/show", {"model": TAG})
    model_info = shown.get("model_info", {})
    if model_info.get("general.architecture") != "qwen35" or model_info.get("general.quantization_version") != 2:
        raise RuntimeError(f"Unexpected Ollama model metadata: {model_info}")
    manifest = {
        "status": "registered", "python_executable": sys.executable, "ollama_version": version,
        "host": HOST, "tag": TAG, "ollama_digest": found.get("digest"),
        "ollama_size_bytes": found.get("size"), "model_info": model_info,
        "template": shown.get("template"), "parameters": shown.get("parameters"),
        "gguf": artifact, "modelfile": str(modelfile.relative_to(ROOT)),
        "modelfile_sha256": sha256_file(modelfile), "create_log": str(create_log.relative_to(ROOT)),
        "create_seconds": time.time() - started,
        "settings": {"thinking": True, "temperature": 0, "protocol": "JSON",
                     "num_predict": 768, "num_ctx": 8192, "num_batch": 512},
        "reference_epoch2": {"tag": "lifemate-qwen35-kagglev1-sft-epoch2-q4km:latest",
            "digest": json.loads(OLD_PROV.read_text(encoding="utf-8"))["branches"]["sft"]["ollama_digest"],
            "same_ollama_version": True},
    }
    write_json(PROV / "ollama-models.json", manifest)
    print(json.dumps({"status": "registered", "tag": TAG, "digest": found.get("digest"),
                      "ollama_version": version, "size_bytes": found.get("size")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
