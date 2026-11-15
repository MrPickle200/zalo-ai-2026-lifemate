from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.asr import PhoASRModel
from lifemate.utils import find_data_root, load_jsonl, resolve_audio

EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = Path(os.environ.get("LIFEMATE_ASR_OUTPUT_DIR", EXP / "output/asr")).expanduser().resolve()
MODEL = Path(os.environ.get("LIFEMATE_PHOWHISPER_MODEL", ROOT / "models/pho-whisper-large-zaloai")).expanduser().resolve()
TRAIN_IDS = ROOT / "data/splits/train_ids.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    data = find_data_root(os.environ.get("LIFEMATE_DATA_ROOT", "train+public_test/train+public_test"))
    ids = json.loads(TRAIN_IDS.read_text(encoding="utf-8-sig"))
    items = {row["id"]: row for row in load_jsonl(data / "train/train.jsonl")}
    assert len(ids) == 1236 and len(set(ids)) == len(ids) and set(ids) <= set(items)
    clips = []
    for sid in ids:
        item = items[sid]
        clips.append({"key": f"{sid}:main", "id": sid, "kind": "main", "history_index": None,
                      "audio": item["audio"], "reference": item.get("audio_text", "")})
        for index, turn in enumerate(item.get("history", [])):
            if turn.get("audio"):
                clips.append({"key": f"{sid}:{index}", "id": sid, "kind": "history",
                              "history_index": index, "audio": turn["audio"],
                              "reference": turn.get("audio_text", turn.get("text", ""))})
    assert len([row for row in clips if row["kind"] == "main"]) == 1236
    assert len({row["audio"] for row in clips}) == len(clips)

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "train_audio_transcripts.jsonl"
    prior = read_jsonl(target)
    saved = {row["key"]: row for row in prior}
    expected = {row["key"] for row in clips}
    assert len(prior) == len(saved) and set(saved) <= expected
    model_hash = sha256(MODEL / "model.safetensors")
    lock = {"split": "train_only", "train_ids_sha256": sha256(TRAIN_IDS),
            "official_train_sha256": sha256(data / "train/train.jsonl"),
            "model_sha256": model_hash,
            "release_sha256": sha256(MODEL / "release.json"),
            "clips": len(clips), "main": 1236,
            "history": len(clips) - 1236,
            "decode": {"language": "vi", "task": "transcribe", "num_beams": 1,
                       "do_sample": False, "max_new_tokens": 256, "sample_rate": 16000}}
    lock_path = OUT / "asr_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text(encoding="utf-8")) == lock, "ASR source/config lock changed"
    else:
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cache = tempfile.TemporaryDirectory(prefix="lifemate-phowhisper-train-")
    asr = PhoASRModel(MODEL, cache.name, device="cuda")
    start = time.perf_counter()
    asr._load()
    model_load_seconds = time.perf_counter() - start
    started = time.perf_counter()
    for number, clip in enumerate(clips, 1):
        if clip["key"] in saved:
            continue
        item = items[clip["id"]]
        audio = resolve_audio(data, data / "train", item, clip["audio"])
        tick = time.perf_counter()
        transcript = asr.transcribe(str(audio))
        if not transcript.strip():
            raise RuntimeError(f"Empty PhoWhisper transcript for {clip['key']}")
        saved[clip["key"]] = {**clip, "transcript": transcript,
                              "inference_seconds": time.perf_counter() - tick}
        if number % 25 == 0 or number == len(clips):
            write_jsonl(target, [saved[row["key"]] for row in clips if row["key"] in saved])
            elapsed = time.perf_counter() - started
            print(f"PhoWhisper train {len(saved)}/{len(clips)} elapsed={elapsed/3600:.2f}h", flush=True)
    rows = [saved[row["key"]] for row in clips]
    assert len(rows) == len(clips) and all(row["transcript"].strip() for row in rows)
    write_jsonl(target, rows)
    summary = {"status": "complete", "clips": len(rows), "main": 1236,
               "history": len(rows) - 1236, "model_load_seconds": model_load_seconds,
               "inference_seconds": round(sum(row["inference_seconds"] for row in rows), 3),
               "model_sha256": model_hash, "transcripts_sha256": sha256(target),
               "split": "fixed train IDs only; validation labels excluded"}
    (OUT / "train_audio_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    cache.cleanup()


if __name__ == "__main__":
    main()
