from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
DEST = EXP / "output/colab-resources"
ARCHIVE = DEST / "lifemate-colab-qwen35-pho-synthetic-direct-v1.zip"
DATA = ROOT / "train+public_test/train+public_test"
TRAIN_IDS = ROOT / "data/splits/train_ids.json"
SFT = ROOT / "exp/qwen35-sft-next-action-v1/kaggle-upload-prompt-en-v1/sft_train.jsonl"
VAL_PHO = ROOT / "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def add_source(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    archive.write(source, archive_name)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    train_ids = json.loads(TRAIN_IDS.read_text(encoding="utf-8-sig"))
    assert len(train_ids) == 1236 and len(train_ids) == len(set(train_ids))
    train_rows = jsonl_rows(DATA / "train/train.jsonl")
    by_id = {row["id"]: row for row in train_rows}
    assert set(train_ids) <= set(by_id)
    selected = [by_id[sid] for sid in train_ids]
    assert len(selected) == 1236

    audio_paths: set[Path] = set()
    for row in selected:
        refs = [row.get("audio")]
        refs.extend(turn.get("audio") for turn in row.get("history", []) if turn.get("audio"))
        for ref in refs:
            if not ref:
                continue
            candidates = (DATA / "train" / ref, DATA / ref)
            audio = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if audio is None:
                raise FileNotFoundError(f"Missing train audio for {row['id']}: {ref}")
            audio_paths.add(audio)
    assert len(audio_paths) == 4035, f"Expected 4,035 train-only audio clips, got {len(audio_paths)}"

    source_files = {
        "README.md": EXP / "README.md",
        "exp/qwen35-sft-pho-synthetic-v1/scripts/transcribe_train.py": EXP / "scripts/transcribe_train.py",
        "exp/qwen35-sft-pho-synthetic-v1/scripts/build_dataset.py": EXP / "scripts/build_dataset.py",
        "exp/qwen35-sft-pho-synthetic-v1/scripts/build_direct_dataset.py": EXP / "scripts/build_direct_dataset.py",
        "exp/qwen35-sft-pho-synthetic-v1/scripts/run_colab.py": EXP / "scripts/run_colab.py",
        "exp/qwen35-sft-pho-synthetic-v1/output/data/synthetic_utterances_1236.jsonl": EXP / "output/data/synthetic_utterances_1236.jsonl",
        "data/splits/train_ids.json": TRAIN_IDS,
        "data/splits/val_ids.json": ROOT / "data/splits/val_ids.json",
        "exp/qwen35-sft-next-action-v1/kaggle-upload-prompt-en-v1/sft_train.jsonl": SFT,
        "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho/predictions.jsonl": VAL_PHO / "predictions.jsonl",
        "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho/history_predictions.jsonl": VAL_PHO / "history_predictions.jsonl",
    }
    for path in (ROOT / "src/lifemate").glob("*.py"):
        source_files[f"src/lifemate/{path.name}"] = path

    audio_bytes = sum(path.stat().st_size for path in audio_paths)
    manifest = {
        "experiment": "qwen35-sft-pho-synthetic-v1",
        "train_trajectories": len(selected),
        "train_audio_clips": len(audio_paths),
        "train_audio_bytes": audio_bytes,
        "source_sft_rows": len(jsonl_rows(SFT)),
        "validation_only_phowhisper_main": 309,
        "validation_only_phowhisper_history": 692,
        "test_data_included": False,
        "model_weights_included": False,
        "note": "Train IDs only. PhoWhisper model weights are loaded separately from Drive. Includes the 1,236 directly authored synthetic utterances; the Colab runner uses build_direct_dataset.py. Validation transcripts remain evaluation-only.",
        "train_ids_sha256": sha256_file(TRAIN_IDS),
        "official_train_jsonl_sha256": sha256_file(DATA / "train/train.jsonl"),
        "source_sft_sha256": sha256_file(SFT),
    }

    temp_archive = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(source_files.items()):
            add_source(archive, path, name)
        rows_text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
        archive.writestr("train+public_test/train+public_test/train/train.jsonl", rows_text)
        for audio in sorted(audio_paths):
            arcname = "train+public_test/train+public_test/train/" + audio.relative_to(DATA / "train").as_posix()
            archive.write(audio, arcname)
        archive.writestr("assets/package_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temp_archive.replace(ARCHIVE)

    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        assert bad is None, f"Archive integrity failure: {bad}"
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate archive members"
        assert sum(name.lower().endswith(".wav") for name in names) == 4035
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size, **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
