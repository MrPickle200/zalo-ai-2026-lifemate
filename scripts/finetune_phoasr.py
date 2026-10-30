from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.asr import PhoASRModel
from lifemate.utils import find_data_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Domain-adapt local PhoASR-whisper-small on the fixed LifeMate train IDs")
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--base-model", default="models/pho-asr-whisper-small")
    parser.add_argument("--output", default="outputs/models/phoasr-lifemate-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="train one pass over eight fixed train IDs")
    parser.add_argument("--audit-only", action="store_true", help="validate all train/validation audio and write manifest")
    parser.add_argument("--skip-audio-decode-audit", action="store_true", help="reuse the completed full-split decode audit")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PhoASR fine-tuning requires CUDA for this experiment")

    root = find_data_root(args.data_root)
    split_dir = Path(args.split_dir)
    train_ids = json.loads((split_dir / "train_ids.json").read_text(encoding="utf-8"))
    val_ids = json.loads((split_dir / "val_ids.json").read_text(encoding="utf-8"))
    if len(train_ids) != 1236 or len(val_ids) != 309 or set(train_ids) & set(val_ids):
        raise ValueError(f"Unexpected fixed split: train={len(train_ids)}, val={len(val_ids)}, overlap={len(set(train_ids)&set(val_ids))}")
    rows = {r["id"]: r for r in (json.loads(line) for line in (root / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    if not set(train_ids + val_ids) <= rows.keys():
        raise ValueError("Split IDs missing from train.jsonl")

    def path_for(sample_id: str) -> Path:
        audio_ref = rows[sample_id]["audio"]
        path = root / "train" / audio_ref
        if not path.is_file():
            raise FileNotFoundError(f"Missing audio for {sample_id}: {path}")
        if not rows[sample_id].get("audio_text", "").strip():
            raise ValueError(f"Empty transcript for {sample_id}")
        return path

    audit_started = time.time()
    all_ids = train_ids + val_ids
    audio_paths = {sample_id: path_for(sample_id) for sample_id in all_ids}
    # Decode every fixed-split item to verify readability, without reading environment files.
    if not args.skip_audio_decode_audit:
        for index, sample_id in enumerate(all_ids, 1):
            wave = PhoASRModel._read_audio(audio_paths[sample_id])
            if wave.size < 1600 or not np.isfinite(wave).all():
                raise ValueError(f"Unreadable/invalid audio for {sample_id}")
            if index % 250 == 0:
                print(f"audio audit {index}/{len(all_ids)}", flush=True)
        print(f"audio audit passed: {len(all_ids)} files, {time.time()-audit_started:.1f}s; train/val overlap=0", flush=True)
    else:
        print("reusing completed full-split audio decode audit; paths/transcripts are rechecked", flush=True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": "Qualcomm-AI-Research/PhoASR-whisper-small", "revision": "local checkpoint; revision metadata unavailable",
                "backend": "Transformers AutoModelForSpeechSeq2Seq / PyTorch", "train_ids": len(train_ids),
                "validation_ids": len(val_ids), "train_validation_overlap": 0,
                "train_ids_sha256": __import__("hashlib").sha256("\n".join(train_ids).encode()).hexdigest(),
                "validation_ids_sha256": __import__("hashlib").sha256("\n".join(val_ids).encode()).hexdigest(),
                "unicode_transcripts": "UTF-8 read; original text passed to tokenizer without normalization",
                "audit_audio_files": len(all_ids), "audio_decode_audit_reused": args.skip_audio_decode_audit}
    (out_dir / "dataset-audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        return 0

    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.base_model, local_files_only=True, torch_dtype=torch.float32).cuda()
    # Partial freeze: fixed encoder and lower decoder remain unchanged; train the top two decoder blocks.
    model.requires_grad_(False)
    for layer in model.model.decoder.layers[-2:]:
        layer.requires_grad_(True)
    model.model.decoder.layer_norm.requires_grad_(True)
    model.proj_out.requires_grad_(True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    ids = train_ids[:8] if args.smoke else train_ids
    if args.smoke:
        args.epochs = 1
        args.accumulation = 1
    model.train()
    losses = []
    best_val_loss = math.inf
    validation_ids = val_ids[:1] if args.smoke else val_ids
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    opt_steps = 0
    for epoch in range(args.epochs):
        epoch_loss_start = len(losses)
        order = list(ids)
        random.Random(args.seed + epoch).shuffle(order)
        for n, sample_id in enumerate(order, 1):
            waveform = PhoASRModel._read_audio(audio_paths[sample_id])
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            features = inputs.input_features.cuda()
            labels = processor.tokenizer(rows[sample_id]["audio_text"], return_tensors="pt").input_ids.cuda()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(input_features=features, labels=labels)
                loss = result.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch+1}, sample={sample_id}")
            losses.append(float(loss.detach()))
            scaler.scale(loss / args.accumulation).backward()
            should_step = n % args.accumulation == 0 or n == len(order)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
            if n % 50 == 0 or n == len(order):
                print(f"epoch={epoch+1}/{args.epochs} samples={n}/{len(order)} loss={losses[-1]:.4f} peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for val_index, sample_id in enumerate(validation_ids, 1):
                waveform = PhoASRModel._read_audio(audio_paths[sample_id])
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                features = inputs.input_features.cuda()
                labels = processor.tokenizer(rows[sample_id]["audio_text"], return_tensors="pt").input_ids.cuda()
                labels[labels == processor.tokenizer.pad_token_id] = -100
                with torch.autocast("cuda", dtype=torch.float16):
                    val_loss = model(input_features=features, labels=labels).loss
                if not torch.isfinite(val_loss):
                    raise FloatingPointError(f"Non-finite validation loss at sample={sample_id}")
                val_losses.append(float(val_loss))
                if val_index % 100 == 0:
                    print(f"validation {val_index}/{len(val_ids)}", flush=True)
        mean_val_loss = sum(val_losses) / len(val_losses)
        epoch_dir = out_dir / f"checkpoint-epoch-{epoch+1}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        processor.save_pretrained(epoch_dir)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            model.save_pretrained(out_dir / "best-checkpoint", safe_serialization=True)
            processor.save_pretrained(out_dir / "best-checkpoint")
        epoch_losses = losses[epoch_loss_start:]
        print(f"epoch={epoch+1} train_loss_mean={sum(epoch_losses)/len(epoch_losses):.4f} val_loss={mean_val_loss:.4f} best_val_loss={best_val_loss:.4f}", flush=True)
        model.train()
        if args.smoke:
            break
    # Verify that the selected checkpoint reloads and produces a deterministic transcript.
    del model
    torch.cuda.empty_cache()
    reloaded_processor = AutoProcessor.from_pretrained(out_dir / "best-checkpoint", local_files_only=True)
    reloaded_model = AutoModelForSpeechSeq2Seq.from_pretrained(out_dir / "best-checkpoint", local_files_only=True,
                                                               torch_dtype=torch.float16).cuda().eval()
    check_id = train_ids[0]
    check_wave = PhoASRModel._read_audio(audio_paths[check_id])
    check_inputs = reloaded_processor(check_wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        check_tokens = reloaded_model.generate(input_features=check_inputs.input_features.cuda().half(),
                                               language="vi", task="transcribe", do_sample=False, num_beams=1)
    inference_check = reloaded_processor.batch_decode(check_tokens, skip_special_tokens=True)[0].strip()
    if not inference_check:
        raise RuntimeError("Fine-tuned checkpoint inference returned an empty transcript")
    result = {**manifest, "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "per_device_batch_size": 1, "gradient_accumulation": args.accumulation,
              "freeze_strategy": "encoder frozen; decoder layers 0-9 frozen; decoder layers 10-11, final layer norm and output projection trainable",
              "trainable_parameters": sum(p.numel() for p in trainable), "optimizer_steps": opt_steps,
              "training_samples": len(ids), "train_loss_mean": sum(losses)/len(losses),
              "loss_first": losses[0], "loss_last": losses[-1],
              "training_seconds": round(time.time()-started, 1), "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
              "validation_loss": mean_val_loss, "best_validation_loss": best_val_loss,
              "inference_reload_check_id": check_id, "inference_reload_transcript": inference_check,
              "best_checkpoint": str((out_dir / "best-checkpoint").resolve()),
              "checkpoint_selection": "lowest full-validation teacher-forced loss across evaluated epochs"}
    (out_dir / "training-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
