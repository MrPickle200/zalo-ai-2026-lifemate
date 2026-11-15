from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
DEST = EXP / "output/colab-resources"
ARCHIVE = DEST / "phowhisper-large-train-inference-colab.zip"
DATA = ROOT / "train+public_test/train+public_test"
TRAIN_IDS = ROOT / "data/splits/train_ids.json"
MODEL = ROOT / "models/pho-whisper-large-zaloai"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    ids = json.loads(TRAIN_IDS.read_text(encoding="utf-8-sig"))
    train_rows = read_jsonl(DATA / "train/train.jsonl")
    by_id = {row["id"]: row for row in train_rows}
    assert len(ids) == 1236 and len(ids) == len(set(ids)) and set(ids) <= set(by_id)

    manifest = []
    audio_files: dict[str, Path] = {}
    for sid in ids:
        item = by_id[sid]
        main_audio = item["audio"]
        manifest.append({"key": f"{sid}:main", "id": sid, "kind": "main",
                         "history_index": None, "audio": main_audio})
        references = [main_audio]
        for index, turn in enumerate(item.get("history", [])):
            ref = turn.get("audio")
            if not ref:
                continue
            manifest.append({"key": f"{sid}:{index}", "id": sid, "kind": "history",
                             "history_index": index, "audio": ref})
            references.append(ref)
        for ref in references:
            path = (DATA / "train" / ref).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Train audio not found for {sid}: {ref}")
            audio_files[path.relative_to((DATA / "train").resolve()).as_posix()] = path

    assert len(manifest) == 4035
    assert len(audio_files) == 4035
    assert sum(row["kind"] == "main" for row in manifest) == 1236
    assert sum(row["kind"] == "history" for row in manifest) == 2799
    assert all((MODEL / name).is_file() for name in
               ("model.safetensors", "config.json", "generation_config.json", "processor_config.json",
                "tokenizer.json", "tokenizer_config.json", "LICENSE"))

    manifest_text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                            for row in manifest)
    audio_bytes = sum(path.stat().st_size for path in audio_files.values())
    model_bytes = sum(path.stat().st_size for path in MODEL.iterdir() if path.is_file())
    package_manifest = {
        "purpose": "PhoWhisper Large v2 inference on the fixed BTC train audio cohort",
        "train_ids": len(ids),
        "main_audio": 1236,
        "history_audio": 2799,
        "audio_clips": len(audio_files),
        "audio_bytes": audio_bytes,
        "model_files": sum(path.is_file() for path in MODEL.iterdir()),
        "model_bytes": model_bytes,
        "train_ids_sha256": sha256(TRAIN_IDS),
        "label_data_included": False,
        "validation_data_included": False,
        "test_data_included": False,
    }

    static_files = {
        "README.md": EXP / "asr_bundle/README.md",
        "scripts/run_train_asr.py": EXP / "asr_bundle/scripts/run_train_asr.py",
        "assets/train_ids.json": TRAIN_IDS,
        "src/lifemate/__init__.py": ROOT / "src/lifemate/__init__.py",
        "src/lifemate/asr.py": ROOT / "src/lifemate/asr.py",
    }
    for file_path in static_files.values():
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name, file_path in sorted(static_files.items()):
            archive.write(file_path, name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        archive.writestr("assets/train_audio_manifest.jsonl", manifest_text,
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        archive.writestr("assets/package_manifest.json",
                         json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        for relative, file_path in sorted(audio_files.items()):
            archive.write(file_path, f"assets/audio/{relative}",
                          compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        for file_path in sorted(path for path in MODEL.iterdir() if path.is_file()):
            method = zipfile.ZIP_STORED if file_path.name == "model.safetensors" else zipfile.ZIP_DEFLATED
            archive.write(file_path, f"model/pho-whisper-large-zaloai/{file_path.name}",
                          compress_type=method, compresslevel=1 if method == zipfile.ZIP_DEFLATED else None)
    temporary.replace(ARCHIVE)

    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate archive entries"
        assert sum(name.startswith("assets/audio/") and name.lower().endswith(".wav") for name in names) == 4035
        assert "model/pho-whisper-large-zaloai/model.safetensors" in names
        assert not any("train.jsonl" in name or "validation" in name.lower() or "test" in name.lower()
                       for name in names)
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity error in {bad}")
    print(json.dumps({"archive": str(ARCHIVE), "zip_bytes": ARCHIVE.stat().st_size,
                      **package_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
