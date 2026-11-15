from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "asr-modeling" / "scripts"))

from lifemate.utils import resolve_audio  # noqa: E402
from inference_common import read_audio_16k  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="QUAN-transcribe history audio for the fixed LifeMate train/val split")
    parser.add_argument("--data-root", type=Path, default=ROOT / "train+public_test" / "train+public_test")
    parser.add_argument("--train-ids", type=Path, default=ROOT / "data" / "splits" / "train_ids.json")
    parser.add_argument("--val-ids", type=Path, default=ROOT / "data" / "splits" / "val_ids.json")
    parser.add_argument("--output", type=Path, default=ROOT / "exp" / "qwen35-sft-next-action-v1" / "output" / "data" / "quan_history_transcripts.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "quan-transcribe")
    parser.add_argument("--lm", type=Path, default=ROOT / "models" / "quan-transcribe" / "lm")
    parser.add_argument("--beam-width", type=int, default=32)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    train_dir = data_root / "train"
    rows = read_jsonl(train_dir / "train.jsonl")
    by_id = {row["id"]: row for row in rows}
    train_ids = json.loads(args.train_ids.read_text(encoding="utf-8-sig"))
    val_ids = json.loads(args.val_ids.read_text(encoding="utf-8-sig"))
    assert len(train_ids) == 1236 and len(val_ids) == 309
    assert not (set(train_ids) & set(val_ids))
    assert set(train_ids) | set(val_ids) == set(by_id)

    # Transcribe history audio only. Current utterance QUAN transcripts already exist
    # in the train1545 ASR comparison artifact and are joined by sample ID later.
    requests: dict[str, dict] = {}
    for sample_id in [*train_ids, *val_ids]:
        sample = by_id[sample_id]
        for turn_index, turn in enumerate(sample.get("history", [])):
            audio_ref = turn.get("audio")
            if audio_ref:
                requests.setdefault(audio_ref, {"sample_id": sample_id, "turn_index": turn_index, "audio_ref": audio_ref})

    prior = {row["audio_ref"]: row for row in read_jsonl(args.output)}
    unknown = set(prior) - set(requests)
    if unknown:
        raise ValueError(f"Cached history transcripts contain unknown audio refs: {sorted(unknown)[:3]}")

    inference_dir = args.model / "inference"
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"QUAN inference package missing: {inference_dir}")
    sys.path.insert(0, str(inference_dir))
    from quan_transcribe import QuanTranscribe  # noqa: E402

    asr = QuanTranscribe.from_pretrained(str(args.model), device="cuda", lm=str(args.lm))
    todo = [ref for ref in requests if ref not in prior]
    started = time.perf_counter()
    for index, audio_ref in enumerate(todo, 1):
        request = requests[audio_ref]
        audio_path = resolve_audio(data_root, train_dir, by_id[request["sample_id"]], audio_ref)
        waveform = read_audio_16k(audio_path)
        text = asr.transcribe(waveform, sample_rate=16000, streaming=False, use_lm=True, beam_width=args.beam_width)
        prior[audio_ref] = {
            **request,
            "transcript": text,
            "audio_sha256": sha256_file(audio_path),
            "asr_model": str(args.model.resolve()),
            "beam_width": args.beam_width,
        }
        if index % 25 == 0 or index == len(todo):
            ordered = [prior[ref] for ref in requests if ref in prior]
            write_jsonl(args.output, ordered)
            print(f"transcribed {index}/{len(todo)} new history clips; total={len(ordered)}/{len(requests)}", flush=True)
    ordered = [prior[ref] for ref in requests if ref in prior]
    write_jsonl(args.output, ordered)
    print(json.dumps({
        "unique_history_audio": len(requests), "completed": len(ordered),
        "new_transcriptions": len(todo), "elapsed_seconds": round(time.perf_counter() - started, 2),
        "train_samples_with_history_audio": sum(any(t.get("audio") for t in by_id[s].get("history", [])) for s in train_ids),
        "val_samples_with_history_audio": sum(any(t.get("audio") for t in by_id[s].get("history", [])) for s in val_ids),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if len(ordered) == len(requests) else 2


if __name__ == "__main__":
    raise SystemExit(main())
