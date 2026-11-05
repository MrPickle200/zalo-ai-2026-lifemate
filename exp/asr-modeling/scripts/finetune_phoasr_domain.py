from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.asr import PhoASRModel
from lifemate.utils import find_data_root


def main() -> int:
    project = _bootstrap.ROOT
    parser = argparse.ArgumentParser(description="Conservative PhoASR decoder LoRA adaptation using train_ids only")
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--base-model", default="models/pho-asr-whisper-small")
    parser.add_argument("--output", default="exp/phoasr-lifemate-domain-v1/output/model")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="train eight train IDs for one epoch")
    args = parser.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PhoASR fine-tuning requires CUDA")

    root = find_data_root(args.data_root)
    split_dir = Path(args.split_dir)
    train_ids = json.loads((split_dir / "train_ids.json").read_text(encoding="utf-8-sig"))
    val_ids = json.loads((split_dir / "val_ids.json").read_text(encoding="utf-8-sig"))
    if len(train_ids) != 1236 or len(val_ids) != 309 or set(train_ids) & set(val_ids):
        raise ValueError(f"Unexpected split sizes/overlap: train={len(train_ids)}, val={len(val_ids)}")
    train_set = set(train_ids)
    rows = {}
    manifest_path = root / "train" / "train.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in train_set:
            rows[row["id"]] = {"id": row["id"], "audio": row["audio"], "audio_text": row["audio_text"]}
    if set(rows) != train_set:
        raise ValueError(f"Training manifest misses {len(train_set - set(rows))} train IDs")
    audio_paths = {sid: root / "train" / rows[sid]["audio"] for sid in train_ids}
    for sid, path in audio_paths.items():
        if not path.is_file() or not rows[sid]["audio_text"].strip():
            raise ValueError(f"Missing training audio/transcript for {sid}: {path}")

    ids = train_ids[:8] if args.smoke else train_ids
    epochs = 1 if args.smoke else args.epochs
    accumulation = 1 if args.smoke else args.accumulation
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.base_model, local_files_only=True, torch_dtype=torch.float32
    ).cuda()
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    target_modules = [
        f"model.decoder.layers.{layer}.{attention}.{projection}"
        for layer in range(model.config.decoder_layers)
        for attention in ("self_attn", "encoder_attn")
        for projection in ("q_proj", "v_proj")
    ]
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=args.rank,
        lora_alpha=args.alpha, lora_dropout=0.05, target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not any("lora_" in name for name in trainable_names) or any(".encoder.layers." in name for name in trainable_names):
        raise RuntimeError("Expected decoder-only LoRA parameters; found no adapters or an encoder adapter")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    manifest = {
        "base_model": str(Path(args.base_model).resolve()),
        "method": "decoder-only LoRA on self- and cross-attention q_proj/v_proj",
        "rank": args.rank, "alpha": args.alpha, "dropout": 0.05,
        "train_ids": len(train_ids), "validation_ids": len(val_ids), "train_validation_overlap": 0,
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "validation_policy": "val309 IDs are used only for final inference evaluation; no val loss or checkpoint selection during training",
        "target_text": "original transcript; UTF-8; no transcript augmentation",
    }
    (out_dir / "dataset-audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    losses = []
    opt_steps = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = list(ids)
        random.Random(args.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for index, sample_id in enumerate(order, 1):
            waveform = PhoASRModel._read_audio(audio_paths[sample_id])
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            features = inputs.input_features.cuda()
            labels = processor.tokenizer(
                rows[sample_id]["audio_text"], return_tensors="pt", truncation=True, max_length=448
            ).input_ids.cuda()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            with torch.autocast("cuda", dtype=torch.float16):
                # Transformers 5 + PEFT's seq2seq wrapper duplicates input_ids
                # for Whisper; call the wrapped Whisper module directly. LoRA
                # layers remain installed and trainable on this path.
                loss = model.get_base_model()(input_features=features, labels=labels).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch+1}, sample={sample_id}")
            losses.append(float(loss.detach()))
            scaler.scale(loss / accumulation).backward()
            if index % accumulation == 0 or index == len(order):
                remainder = index % accumulation
                scaler.unscale_(optimizer)
                if remainder:
                    correction = accumulation / remainder
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
            if index % 50 == 0 or index == len(order):
                print(
                    f"epoch={epoch+1}/{epochs} samples={index}/{len(order)} "
                    f"loss={losses[-1]:.4f} peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}",
                    flush=True,
                )
        epoch_dir = out_dir / f"adapter-epoch-{epoch+1}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        processor.save_pretrained(epoch_dir)

    # Merge the adapter to a regular Whisper checkpoint consumable by the existing PhoASR inference path.
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    final_dir = out_dir / "final-model"
    merged.save_pretrained(final_dir, safe_serialization=True)
    processor.save_pretrained(final_dir)
    result = {
        **manifest, "seed": args.seed, "epochs": epochs, "learning_rate": args.learning_rate,
        "effective_train_samples": len(ids), "gradient_accumulation": accumulation,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "optimizer_steps": opt_steps, "train_loss_mean": sum(losses) / max(1, len(losses)),
        "loss_first": losses[0], "loss_last": losses[-1],
        "training_seconds": round(time.perf_counter() - started, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "final_checkpoint": str(final_dir.resolve()),
        "checkpoint_selection": "fixed final epoch; val309 was not read for selection",
    }
    (out_dir / "training-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
