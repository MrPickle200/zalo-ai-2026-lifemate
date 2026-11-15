from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import (
    audio_duration, edit_distance, load_validation, normalize, read_audio_16k,
    read_jsonl, score_rows, write_jsonl,
)
from lifemate.utils import find_data_root, resolve_audio


def main() -> int:
    project = _bootstrap.ROOT
    model_dir = project / "exp" / "quan-transcribe-lifemate-domain-v1" / "output" / "model"
    inference_dir = project / "models" / "quan-transcribe" / "inference"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"QUAN inference code not found at {inference_dir}; clone qnguyen3/QUAN-Transcribe there")
    sys.path.insert(0, str(inference_dir))
    from quan_transcribe import QuanTranscribe

    parser = argparse.ArgumentParser(description="Run QUAN-Transcribe with KenLM beam search on fixed validation IDs")
    parser.add_argument("--val-ids", default=str(project / "data" / "splits" / "val_ids.json"))
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--model", default=str(model_dir))
    parser.add_argument("--lm", default=str(project / "models" / "quan-transcribe" / "lm"))
    parser.add_argument("--output-dir", default=str(project / "exp" / "quan-continuation-cot-v1" / "output" / "asr" / "epoch-01-val309"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--phoasr-predictions", default=str(project / "outputs" / "eval" / "phoasr-inference-val309" / "predictions.jsonl"))
    parser.add_argument("--overwrite", action="store_true", help="ignore prior predictions and recompute every ID")
    args = parser.parse_args()

    root = find_data_root(args.data_root)
    samples = load_validation(root, Path(args.val_ids))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    saved_rows = [] if args.overwrite else read_jsonl(predictions_path)
    rows_by_id = {row["id"]: row for row in saved_rows}
    valid_ids = {item["id"] for item in samples}
    if set(rows_by_id) - valid_ids:
        raise ValueError("Existing QUAN predictions contain IDs outside this validation split")
    resumed_count = len(rows_by_id)

    load_started = time.perf_counter()
    asr = QuanTranscribe.from_pretrained(args.model, device=args.device, lm=args.lm)
    model_load_seconds = time.perf_counter() - load_started
    lm_config_path = Path(args.lm) / "lm_config.json"
    lm_config = json.loads(lm_config_path.read_text(encoding="utf-8")) if lm_config_path.exists() else {}

    todo = [item for item in samples if item["id"] not in rows_by_id]
    started = time.perf_counter()
    split_dir = root / "train"
    for index, item in enumerate(todo, 1):
        audio = resolve_audio(root, split_dir, item, item["audio"])
        duration = audio_duration(audio)
        inference_started = time.perf_counter()
        waveform = read_audio_16k(audio)
        transcript = asr.transcribe(
            waveform, sample_rate=16000, streaming=False, use_lm=True, beam_width=args.beam_width
        )
        inference_seconds = time.perf_counter() - inference_started
        rows_by_id[item["id"]] = {
            "id": item["id"], "reference": item.get("audio_text", ""),
            "prediction": transcript, "audio_seconds": duration,
            "inference_seconds": inference_seconds,
        }
        if index % 25 == 0 or index == len(todo):
            write_jsonl(predictions_path, [rows_by_id[sample["id"]] for sample in samples if sample["id"] in rows_by_id])
            print(f"new {index}/{len(todo)} (total {len(rows_by_id)}/{len(samples)})", flush=True)

    rows = [rows_by_id[item["id"]] for item in samples if item["id"] in rows_by_id]
    write_jsonl(predictions_path, rows)
    elapsed = time.perf_counter() - started
    metrics = score_rows(rows)
    summary = {
        "model": str(Path(args.model).resolve()),
        "model_sha256": hashlib.sha256((Path(args.model) / "model.safetensors").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "units_sha256": hashlib.sha256((Path(args.model) / "units.json").read_bytes()).hexdigest(),
        "decoder": "KenLM CTC beam search", "lm": str(Path(args.lm).resolve()),
        "beam_width": args.beam_width, "lm_alpha": lm_config.get("alpha", 0.3),
        "lm_beta": lm_config.get("beta", 1.0),
        "dataset": "all train examples" if Path(args.val_ids).name == "train_ids.json" else "validation IDs",
        "sample_ids_file": str(Path(args.val_ids).resolve()), "samples": len(samples), "metrics": metrics,
        "model_load_seconds": model_load_seconds, "new_inference_wall_seconds": elapsed,
        "resumed_predictions": resumed_count,
        "normalization": "lowercase and remove punctuation; WER uses Vietnamese whitespace tokens; CER removes whitespace",
        "lm_config_sha256": hashlib.sha256(lm_config_path.read_bytes()).hexdigest() if lm_config_path.exists() else None,
        "inference_code_sha256": {
            name: hashlib.sha256((inference_dir / "quan_transcribe" / name).read_bytes()).hexdigest()
            for name in ("model.py", "transcriber.py", "features.py", "audio.py", "lm.py")
        },
    }

    pho_path = Path(args.phoasr_predictions)
    pho_rows = read_jsonl(pho_path)
    if pho_rows:
        quan_by_id = {row["id"]: row for row in rows}
        pho_by_id = {row["id"]: row for row in pho_rows}
        if set(quan_by_id) != set(valid_ids) or set(pho_by_id) != set(valid_ids):
            summary["comparison_note"] = (
                f"Skipped paired comparison until both files cover all validation IDs "
                f"(PhoASR {len(pho_by_id)}/{len(valid_ids)}, QUAN {len(quan_by_id)}/{len(valid_ids)})."
            )
        else:
            paired = [quan_by_id[item["id"]] for item in samples]
            pho_paired = [pho_by_id[item["id"]] for item in samples]
            deltas = []
            quan_wins = pho_wins = ties = 0
            for q, p in zip(paired, pho_paired):
                ref = normalize(q.get("reference") or "")
                qwer = edit_distance(ref, normalize(q.get("prediction") or "")) / max(1, len(ref))
                pwer = edit_distance(ref, normalize(p.get("prediction") or "")) / max(1, len(ref))
                deltas.append(qwer - pwer)
                if qwer < pwer:
                    quan_wins += 1
                elif pwer < qwer:
                    pho_wins += 1
                else:
                    ties += 1
            summary["comparison"] = {
                "phoasr_prediction_file": str(pho_path.resolve()),
                "paired_samples": len(paired), "phoasr": score_rows(pho_paired), "quan_transcribe": metrics,
                "quan_per_sample_wer_wins": quan_wins, "phoasr_per_sample_wer_wins": pho_wins,
                "ties": ties, "mean_paired_utterance_wer_delta_quan_minus_phoasr":
                    sum(deltas) / len(deltas) if deltas else None,
            }
            paired_dir = project / "outputs" / "eval" / "paired-analysis"
            paired_dir.mkdir(parents=True, exist_ok=True)
            (paired_dir / "phoasr-vs-quan-transcribe-val309.json").write_text(
                json.dumps(summary["comparison"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
