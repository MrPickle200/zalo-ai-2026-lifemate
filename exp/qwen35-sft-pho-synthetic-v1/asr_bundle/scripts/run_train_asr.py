from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.asr import PhoASRModel


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                   encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe only the fixed BTC train audio cohort with PhoWhisper Large.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model/pho-whisper-large-zaloai")
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets/train_audio_manifest.jsonl")
    parser.add_argument("--audio-root", type=Path, default=ROOT / "assets/audio")
    parser.add_argument("--output", type=Path, default=ROOT / "output/train_audio_transcripts.jsonl")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select a Colab GPU runtime such as L4.")
    if not (args.model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"PhoWhisper weights not found: {args.model_dir / 'model.safetensors'}")

    manifest = read_jsonl(args.manifest)
    keys = [row["key"] for row in manifest]
    assert len(manifest) == 4035 and len(keys) == len(set(keys)), "Expected 4,035 unique train audio entries"
    counts = Counter(row["kind"] for row in manifest)
    assert counts == {"main": 1236, "history": 2799}, counts
    expected = set(keys)
    prior = read_jsonl(args.output)
    prior_keys = [row["key"] for row in prior]
    assert len(prior_keys) == len(set(prior_keys)) and set(prior_keys) <= expected
    assert prior_keys == keys[:len(prior_keys)], "Checkpoint does not match this train manifest"
    saved = {row["key"]: row for row in prior}

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loaded checkpoint: {len(saved)}/{len(manifest)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="phowhisper-train-cache-") as cache_dir:
        asr = PhoASRModel(args.model_dir, cache_dir=cache_dir, device="cuda")
        load_start = time.perf_counter()
        asr._load()
        model_load_seconds = time.perf_counter() - load_start
        run_start = time.perf_counter()

        for number, item in enumerate(manifest, start=1):
            key = item["key"]
            if key in saved:
                continue
            audio = args.audio_root / item["audio"]
            if not audio.is_file():
                raise FileNotFoundError(f"Audio missing for {key}: {audio}")
            tick = time.perf_counter()
            transcript = asr.transcribe(str(audio)).strip()
            if not transcript:
                raise RuntimeError(f"Empty transcript for {key}; previous checkpoints are preserved")
            saved[key] = {**item, "transcript": transcript,
                          "inference_seconds": round(time.perf_counter() - tick, 4)}
            if number % args.checkpoint_every == 0 or number == len(manifest):
                ordered = [saved[row["key"]] for row in manifest if row["key"] in saved]
                write_jsonl(args.output, ordered)
                elapsed = time.perf_counter() - run_start
                print(f"PhoWhisper train {len(saved)}/{len(manifest)} elapsed={elapsed / 3600:.2f}h", flush=True)

        rows = [saved[row["key"]] for row in manifest]
        if len(rows) != len(manifest):
            raise RuntimeError(f"Incomplete run: {len(rows)}/{len(manifest)}")
        write_jsonl(args.output, rows)

    summary = {
        "status": "complete",
        "split": "fixed train IDs only",
        "clips": len(rows),
        "main": counts["main"],
        "history": counts["history"],
        "gpu": torch.cuda.get_device_name(0),
        "model_dir": str(args.model_dir),
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(sum(row["inference_seconds"] for row in rows), 3),
        "transcripts_file": str(args.output),
    }
    summary_path = args.output.with_name("train_audio_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
