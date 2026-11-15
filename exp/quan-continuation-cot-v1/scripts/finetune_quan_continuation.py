from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import _bootstrap  # noqa: F401
from asr_finetune_common import normalize_quan_text
from inference_common import read_audio_16k
from lifemate.utils import find_data_root


def trainable_forward(model, wav, wav_lens, grad_checkpoint: bool):
    """Training equivalent of QuanCTC.forward, which is decorated with torch.no_grad in inference code."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    feats, feat_mask = model.mel(wav, wav_lens)
    h, mask = model._encoder_hidden(feats, feat_mask)
    x = model.in_proj(model.in_norm(h))
    xc = (x * mask[..., None]).transpose(1, 2)
    xc = F.pad(xc, (model.cfg.conv_kernel - 1, 0))
    x = x + F.gelu(model.pos_conv(xc).transpose(1, 2))
    attention_mask = mask[:, None, None, :]
    for block in model.blocks:
        if grad_checkpoint and x.requires_grad:
            x = checkpoint(block, x, attention_mask, use_reentrant=False)
        else:
            x = block(x, attention_mask)
    projected = model.to_text(model.out_norm(x))
    rows = torch.cat([model.special_rows, model.unit_rows], dim=0)
    logits = projected.float() @ rows.float().t() + model.head_bias.float()
    return logits, mask.sum(1)


def main() -> int:
    project = _bootstrap.ROOT
    parser = argparse.ArgumentParser(description="Domain-adapt QUAN CTC reader on train_ids only")
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--base-model", default="exp/quan-transcribe-lifemate-domain-v1/output/model")
    parser.add_argument("--inference-code", default="models/quan-transcribe/inference")
    parser.add_argument("--output", default="exp/quan-continuation-cot-v1/output/checkpoints/epoch-01")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epoch-index", type=int, default=0,
                        help="offset the deterministic sample-order seed when chaining one-epoch runs")
    parser.add_argument("--reader-epochs", type=int, default=1)
    parser.add_argument("--encoder-epochs", type=int, default=1)
    parser.add_argument("--reader-learning-rate", type=float, default=5e-5)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-6)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="train eight train IDs with the reader only")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.nn import CTCLoss
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    inference_dir = Path(args.inference_code).resolve()
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"QUAN inference source is missing: {inference_dir}")
    sys.path.insert(0, str(inference_dir))
    from quan_transcribe import QuanTranscribe

    if not torch.cuda.is_available():
        raise RuntimeError("QUAN fine-tuning requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

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

    base_dir = Path(args.base_model)
    base_checkpoint_path = base_dir / "model.safetensors"
    if not base_checkpoint_path.is_file():
        raise FileNotFoundError(f"Input checkpoint is missing: {base_checkpoint_path}")
    base_checkpoint_sha256 = hashlib.sha256(base_checkpoint_path.read_bytes()).hexdigest()
    units = json.loads((base_dir / "units.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(units.get("model_id", "google/embeddinggemma-2"), local_files_only=True)
    unit_index = {token_id: index + 2 for index, token_id in enumerate(units["gemma_ids"])}
    targets: dict[str, list[int]] = {}
    unknown_units = total_units = 0
    for sid in train_ids:
        text = normalize_quan_text(rows[sid]["audio_text"])
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        target = [unit_index.get(token_id, 1) for token_id in token_ids]
        unknown_units += sum(unit == 1 for unit in target)
        total_units += len(target)
        if not target:
            raise ValueError(f"Empty CTC target after normalization for {sid}")
        targets[sid] = target
    unknown_rate = unknown_units / max(1, total_units)
    if unknown_rate > 0.01:
        raise ValueError(f"Too many target units are outside QUAN vocabulary: {unknown_rate:.2%}")

    ids = train_ids[:8] if args.smoke else train_ids
    reader_epochs = 1 if args.smoke else args.reader_epochs
    encoder_epochs = 0 if args.smoke else args.encoder_epochs
    accumulation = 1 if args.smoke else args.accumulation

    started_load = time.perf_counter()
    asr = QuanTranscribe.from_pretrained(args.base_model, device="cuda", lm=False)
    load_seconds = time.perf_counter() - started_load
    model = asr.model
    if model.cfg.unfreeze_top_k < 2:
        raise ValueError("Base QUAN checkpoint does not declare its top two encoder layers for fine-tuning")
    encoder_depth = len(model.encoder.layers)
    top_encoder_layers = list(model.encoder.layers[-2:])
    ctc_loss = CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint artifacts: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_model": str(base_dir.resolve()),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "method": "CTC reader adaptation, then reader plus top two EmbeddingGemma encoder layers",
        "reader_epochs": reader_epochs, "encoder_epochs": encoder_epochs,
        "reader_learning_rate": args.reader_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "train_ids": len(train_ids), "validation_ids": len(val_ids), "train_validation_overlap": 0,
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "ctc_target_normalization": "lowercase; punctuation removed; numbers written in Vietnamese words; common orthography normalized",
        "ctc_target_units": total_units, "unknown_target_units": unknown_units,
        "unknown_target_unit_rate": unknown_rate,
        "validation_policy": "val309 IDs are used only for final inference evaluation; no val loss or checkpoint selection during training",
    }
    (out_dir / "dataset-audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history = []
    stage_specs = [
        ("reader", reader_epochs, False),
        ("reader_plus_top2_encoder", encoder_epochs, True),
    ]
    opt_steps = 0
    training_started = time.perf_counter()
    for stage_name, stage_epochs, unfreeze_encoder in stage_specs:
        if not stage_epochs:
            continue
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        if unfreeze_encoder:
            for layer in top_encoder_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

        reader_params = [parameter for name, parameter in model.named_parameters()
                         if not name.startswith("encoder.") and parameter.requires_grad]
        groups = [{"params": reader_params, "lr": args.encoder_learning_rate if args.smoke else
                   args.reader_learning_rate if not unfreeze_encoder else args.reader_learning_rate / 5}]
        encoder_params = []
        if unfreeze_encoder:
            encoder_params = [parameter for layer in top_encoder_layers for parameter in layer.parameters()
                              if parameter.requires_grad]
            groups.append({"params": encoder_params, "lr": args.encoder_learning_rate})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

        for epoch in range(stage_epochs):
            model.train()
            model.encoder.eval()
            if unfreeze_encoder:
                for layer in top_encoder_layers:
                    layer.train()
            order = list(ids)
            random.Random(args.seed + args.epoch_index + len(history)).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            stage_losses = []
            for index, sample_id in enumerate(order, 1):
                waveform = read_audio_16k(audio_paths[sample_id])
                wav = torch.from_numpy(waveform.copy()).float().unsqueeze(0).cuda()
                wav_lens = torch.tensor([waveform.shape[0]], dtype=torch.long, device="cuda")
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits, output_lengths = trainable_forward(
                        model, wav, wav_lens, grad_checkpoint=not args.no_grad_checkpoint
                    )
                target = torch.tensor(targets[sample_id], dtype=torch.long, device="cuda")
                input_length = int(output_lengths[0].item())
                if input_length < target.numel():
                    raise ValueError(
                        f"CTC target is longer than model output for {sample_id}: "
                        f"{target.numel()} > {input_length}"
                    )
                log_probs = logits.float().log_softmax(-1).transpose(0, 1)
                loss = ctc_loss(
                    log_probs, target,
                    torch.tensor([input_length], dtype=torch.long, device="cuda"),
                    torch.tensor([target.numel()], dtype=torch.long, device="cuda"),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite CTC loss at {stage_name}/{epoch+1}, sample={sample_id}")
                stage_losses.append(float(loss.detach()))
                if use_scaler:
                    scaler.scale(loss / accumulation).backward()
                else:
                    (loss / accumulation).backward()

                should_step = index % accumulation == 0 or index == len(order)
                if should_step:
                    remainder = index % accumulation
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    if remainder:
                        correction = accumulation / remainder
                        for group in optimizer.param_groups:
                            for parameter in group["params"]:
                                if parameter.grad is not None:
                                    parameter.grad.mul_(correction)
                    active_params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
                    torch.nn.utils.clip_grad_norm_(active_params, 1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_steps += 1
                if index % 50 == 0 or index == len(order):
                    print(
                        f"stage={stage_name} epoch={epoch+1}/{stage_epochs} "
                        f"samples={index}/{len(order)} loss={stage_losses[-1]:.4f} "
                        f"peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}",
                        flush=True,
                    )
            history.append({
                "stage": stage_name, "epoch": epoch + 1,
                "train_loss_mean": sum(stage_losses) / len(stage_losses),
                "train_loss_last": stage_losses[-1], "samples": len(order),
            })

    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "units.json").write_text((base_dir / "units.json").read_text(encoding="utf-8"), encoding="utf-8")
    trainable_encoder_keys = {
        name for name, _ in model.named_parameters()
        if name.startswith("encoder.") and any(name.startswith(f"encoder.layers.{i}.") for i in range(encoder_depth - 2, encoder_depth))
    }
    state = {}
    for name, tensor in model.state_dict().items():
        if not name.startswith("encoder.") or name in trainable_encoder_keys:
            state[name] = tensor.detach().cpu().contiguous()
    save_file(state, str(out_dir / "model.safetensors"))

    result = {
        **manifest, "seed": args.seed, "effective_train_samples": len(ids),
        "optimizer_steps": opt_steps, "epochs_completed": len(history),
        "stage_history": history, "training_seconds": round(time.perf_counter() - training_started, 1),
        "model_load_seconds": round(load_seconds, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "final_checkpoint": str(out_dir.resolve()),
        "checkpoint_selection": "fixed final stage; val309 was not read for selection",
    }
    (out_dir / "training-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
