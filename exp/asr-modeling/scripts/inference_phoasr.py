from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import audio_duration, load_validation, read_jsonl, score_rows, write_jsonl
from lifemate.asr import PhoASRModel
from lifemate.utils import find_data_root, resolve_audio


def main() -> int:
    project = _bootstrap.ROOT
    parser = argparse.ArgumentParser(description="Run PhoASR on the fixed LifeMate validation IDs")
    parser.add_argument("--val-ids", default=str(project / "data" / "splits" / "val_ids.json"))
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--model", default=str(project / "models" / "pho-asr-whisper-small"))
    parser.add_argument("--output-dir", default=str(project / "outputs" / "eval" / "phoasr-inference-val309"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--overwrite", action="store_true", help="ignore prior predictions and recompute every ID")
    args = parser.parse_args()

    root = find_data_root(args.data_root)
    samples = load_validation(root, Path(args.val_ids))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    saved = {} if args.overwrite else {row["id"]: row for row in read_jsonl(predictions_path)}
    valid_ids = {item["id"] for item in samples}
    if set(saved) - valid_ids:
        raise ValueError("Existing prediction file contains IDs outside this validation split")
    resumed_count = len(saved)
    rows_by_id = saved
    cache_dir = tempfile.TemporaryDirectory(prefix="lifemate-phoasr-cache-")
    asr = PhoASRModel(model_path=args.model, cache_dir=cache_dir.name, device=args.device)
    load_started = time.perf_counter()
    asr._load()
    model_load_seconds = time.perf_counter() - load_started

    todo = [item for item in samples if item["id"] not in rows_by_id]
    started = time.perf_counter()
    split_dir = root / "train"
    for index, item in enumerate(todo, 1):
        audio = resolve_audio(root, split_dir, item, item["audio"])
        duration = audio_duration(audio)
        inference_started = time.perf_counter()
        transcript = asr.transcribe(str(audio))
        inference_seconds = time.perf_counter() - inference_started
        rows_by_id[item["id"]] = {
            "id": item["id"], "reference": item.get("audio_text", ""),
            "prediction": transcript, "audio_seconds": duration,
            "inference_seconds": inference_seconds,
        }
        if index % 25 == 0 or index == len(todo):
            write_jsonl(predictions_path, [rows_by_id[sample["id"]] for sample in samples if sample["id"] in rows_by_id])
            print(f"new {index}/{len(todo)} (total {len(rows_by_id)}/{len(samples)})", flush=True)
    rows = [rows_by_id[item["id"]] for item in samples]
    write_jsonl(predictions_path, rows)
    elapsed = time.perf_counter() - started
    summary = {
        "model": str(Path(args.model).resolve()), "decoder": "deterministic greedy (num_beams=1)",
        "dataset": "all train examples" if Path(args.val_ids).name == "train_ids.json" else "validation IDs",
        "sample_ids_file": str(Path(args.val_ids).resolve()), "samples": len(samples), "metrics": score_rows(rows),
        "model_load_seconds": model_load_seconds, "new_inference_wall_seconds": elapsed,
        "resumed_predictions": resumed_count, "normalization":
            "lowercase and remove punctuation; WER uses Vietnamese whitespace tokens; CER removes whitespace",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    cache_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
