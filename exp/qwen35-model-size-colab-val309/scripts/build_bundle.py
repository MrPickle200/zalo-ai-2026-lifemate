"""Build the portable, model-only Colab validation bundle from the local project."""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
DEST = SOURCE / "exp/qwen35-model-size-colab-val309/output/colab-resources/lifemate-colab-qwen-scale"
ARCHIVE = DEST / "lifemate-val309-resources.zip"
MODEL_NAME = "sft-epoch3-en-Q4_K_M.gguf"
MODEL_SOURCE = SOURCE / "exp/qwen35-sft-epoch3-en-cot-val60/output/gguf" / MODEL_NAME
MODEL_DEST = DEST / MODEL_NAME


def add_file(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive.write(path, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    assert SOURCE.is_dir() and MODEL_SOURCE.is_file()
    files = {
        "run_compare.py": SOURCE / "exp/qwen35-model-size-colab-val309/scripts/run_compare.py",
        "README.md": DEST / "README.md",
        "Modelfile": SOURCE / "exp/qwen35-model-size-colab-val309/scripts/Modelfile",
        "data/splits/val_ids.json": SOURCE / "data/splits/val_ids.json",
        "assets/cohort60.json": SOURCE / "exp/inference-latency-optimization-val60/output/cohort_manifest.json",
        "assets/pho_val309_main.jsonl": SOURCE / "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho/predictions.jsonl",
        "assets/pho_val309_history.jsonl": SOURCE / "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho/history_predictions.jsonl",
        "train+public_test/train+public_test/train/train.jsonl": SOURCE / "train+public_test/train+public_test/train/train.jsonl",
        "train+public_test/train+public_test/train/train_tools_catalog.json": SOURCE / "train+public_test/train+public_test/train/train_tools_catalog.json",
        "train+public_test/train+public_test/README.md": SOURCE / "train+public_test/train+public_test/README.md",
        "train+public_test/train+public_test/VERSION": SOURCE / "train+public_test/train+public_test/VERSION",
        "exp/btc-scoring/scripts/score_btc.py": SOURCE / "exp/btc-scoring/scripts/score_btc.py",
        "exp/btc-scoring/scripts/_bootstrap.py": SOURCE / "exp/btc-scoring/scripts/_bootstrap.py",
        "exp/quan-continuation-cot-v1/scripts/run_downstream.py": SOURCE / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
    }
    for path in (SOURCE / "src/lifemate").glob("*.py"):
        files[f"src/lifemate/{path.name}"] = path
    train = SOURCE / "train+public_test/train+public_test/train"
    for directory in (train / "zaic_tools", train / "train_environment"):
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(SOURCE)).replace("\\", "/")] = path
    if ARCHIVE.exists() or MODEL_DEST.exists():
        raise FileExistsError("Destination already contains bundle output")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            add_file(archive, path, name)
    with MODEL_SOURCE.open("rb") as source, MODEL_DEST.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive integrity failure: {bad}")
        assert len(archive.namelist()) == len(files)
    checksums = [(ARCHIVE, sha256(ARCHIVE)), (MODEL_DEST, sha256(MODEL_DEST))]
    (DEST / "SHA256SUMS.txt").write_text(
        "".join(f"{checksum}  {path.name}  {path.stat().st_size} bytes\n"
                for path, checksum in checksums), encoding="utf-8")
    print(f"Files in ZIP: {len(files)}")
    for path, checksum in checksums:
        print(f"{path.name}: {path.stat().st_size:,} bytes; SHA-256 {checksum}")


if __name__ == "__main__":
    main()
