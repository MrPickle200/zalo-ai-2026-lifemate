from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
DATA = Path(os.environ.get("LIFEMATE_DATA_ROOT", ROOT / "train+public_test/train+public_test")).expanduser().resolve()
TRAIN_IDS = ROOT / "data/splits/train_ids.json"
VAL_IDS = ROOT / "data/splits/val_ids.json"
SOURCE_SFT = ROOT / "exp/qwen35-sft-next-action-v1/kaggle-upload-prompt-en-v1/sft_train.jsonl"
ASR = Path(os.environ.get("LIFEMATE_ASR_TRANSCRIPTS", EXP / "output/asr/train_audio_transcripts.jsonl")).expanduser().resolve()
VAL_PHO = ROOT / "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho"
OUT = Path(os.environ.get("LIFEMATE_DATASET_OUTPUT_DIR", EXP / "output/data")).expanduser().resolve()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transcript_maps() -> tuple[dict, dict]:
    rows = read_jsonl(ASR)
    main = {row["id"]: row["transcript"] for row in rows if row["kind"] == "main"}
    history = {row["key"]: row for row in rows if row["kind"] == "history"}
    assert len(main) == 1236 and len(history) == len(rows) - 1236
    assert all(text.strip() for text in main.values())
    assert all(row["transcript"].strip() for row in history.values())
    return main, history


def rewrite_phowhisper(base: dict, item: dict, main: dict, history: dict) -> dict:
    import copy

    row = copy.deepcopy(base)
    payload = json.loads(row["messages"][1]["content"])
    sample_id = row["sample_id"]
    if sample_id not in main:
        raise KeyError(f"Missing PhoWhisper main transcript: {sample_id}")
    payload["transcript"] = main[sample_id]
    rewritten_history = []
    for index, turn in enumerate(item.get("history", [])):
        converted = copy.deepcopy(turn)
        if converted.get("audio"):
            key = f"{sample_id}:{index}"
            saved = history.get(key)
            if not saved or saved["audio"] != converted["audio"]:
                raise KeyError(f"Missing/mismatched PhoWhisper history transcript: {key}")
            converted["text"] = saved["transcript"]
            converted.pop("audio", None)
        rewritten_history.append(converted)
    payload["history"] = rewritten_history
    row["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    row["source"] = "btc_gold_phowhisper_large"
    return row


def main() -> None:
    from build_direct_dataset import main as build_direct
    build_direct()


if __name__ == "__main__":
    main()
