from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SEED = 20261031
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ActionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, limit: int | None = None):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if limit and len(self.rows) >= limit:
                        break
        self.examples = []
        self.truncated = 0
        for row in self.rows:
            messages = row["messages"]
            prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
                                                        add_generation_prompt=True, enable_thinking=False)
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                                       add_generation_prompt=False, enable_thinking=False)
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Chat template prompt mismatch at {row['sample_id']}:{row['step_index']}")
            target_ids = full_ids[len(prompt_ids):]
            # Match Agent._generate: reserve 128 generation tokens from the configured model context.
            prompt_budget = min(max_length - 128, max_length - len(target_ids))
            if len(prompt_ids) > prompt_budget:
                if prompt_budget <= 0:
                    raise ValueError(f"Target exceeds max sequence length at {row['sample_id']}:{row['step_index']}")
                # Match Agent._generate: right-truncate the rendered prompt to the model context budget.
                prompt_ids = prompt_ids[:prompt_budget]
                full_ids = prompt_ids + target_ids
                self.truncated += 1
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
            if len(labels) != len(full_ids) or not any(token != -100 for token in labels):
                raise ValueError(f"No assistant target tokens at {row['sample_id']}:{row['step_index']}")
            self.examples.append((full_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids, labels = self.examples[index]
        return {"input_ids": ids, "labels": labels}


def collate(rows, pad_id: int):
    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), length), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def run(args):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required; CUDA is unavailable")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "base_model": BASE_MODEL,
                      "seed": SEED, "quantization": "4-bit NF4, double quantization",
                      "lora_rank": 16, "lora_alpha": 32, "target_modules": TARGET_MODULES,
                      "learning_rate": 5e-5, "epochs": args.epochs,
                      "per_device_batch_size": 1,
                      "gradient_accumulation": 4 if args.smoke else args.gradient_accumulation,
                      "max_sequence_length": args.max_length, "smoke": args.smoke}, ensure_ascii=False), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = ActionDataset(Path(args.data), tokenizer, args.max_length,
                         limit=args.smoke_examples if args.smoke else None)
    if not data:
        raise RuntimeError("No SFT examples loaded")
    print(f"dataset_examples={len(data)} truncated={data.truncated}", flush=True)
    loader = DataLoader(data, batch_size=1, shuffle=True, collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
                               if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant,
                                                 device_map={"": 0}, torch_dtype="auto",
                                                 attn_implementation="sdpa", trust_remote_code=False)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, bias="none",
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    grad_accum = 4 if args.smoke else args.gradient_accumulation
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = args.max_steps if args.smoke else steps_per_epoch * args.epochs
    max_steps = total_steps
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.time()
    optimizer_step = 0
    while optimizer_step < max_steps:
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at optimizer step {optimizer_step}: {loss.item()}")
            (loss / grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % 10 == 0 or optimizer_step == 1:
                    print(json.dumps({"step": optimizer_step, "max_steps": max_steps,
                                      "loss_window_mean": sum(losses[-16:]) / len(losses[-16:]),
                                      "elapsed_seconds": round(time.time() - started, 1),
                                      "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
                if optimizer_step >= max_steps:
                    break
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    final_mean = sum(losses[-min(len(losses), grad_accum):]) / min(len(losses), grad_accum)
    training_info = {"gpu": torch.cuda.get_device_name(0), "base_model": BASE_MODEL,
                     "quantization": "4-bit NF4 double quantization", "lora_rank": 16,
                     "lora_alpha": 32, "target_modules": TARGET_MODULES,
                     "learning_rate": 5e-5, "epochs": args.epochs,
                     "per_device_batch_size": 1, "gradient_accumulation": grad_accum,
                     "max_sequence_length": args.max_length, "seed": SEED,
                     "optimizer_steps": optimizer_step, "dataset_examples": len(data),
                     "truncated_examples": data.truncated, "mean_final_loss": final_mean,
                     "mean_initial_loss": sum(losses[:min(len(losses), grad_accum)]) / min(len(losses), grad_accum),
                     "training_seconds": round(time.time() - started, 1), "smoke": args.smoke}
    if args.smoke:
        training_info["smoke_loss_decreased"] = training_info["mean_final_loss"] < training_info["mean_initial_loss"]
    (output / "training_manifest.json").write_text(json.dumps(training_info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(training_info, indent=2), flush=True)

    # Probe with a real production-format SFT prompt, not an unrelated chat question.
    probe_messages = data.rows[0]["messages"][:-1]
    del model, optimizer, quant, loader, data
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        device_map={"": 0}, torch_dtype="auto", attn_implementation="sdpa", trust_remote_code=False)
    reloaded = PeftModel.from_pretrained(base, output, is_trainable=False)
    encoded = tokenizer.apply_chat_template(probe_messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                             return_tensors="pt").to("cuda")
    reloaded.eval()
    with torch.inference_mode():
        generated = reloaded.generate(encoded, max_new_tokens=64, do_sample=False, num_beams=1,
                                      pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated[0, encoded.shape[1]:], skip_special_tokens=True).strip()
    from json import JSONDecoder
    start = decoded.find("{")
    if start < 0:
        raise RuntimeError(f"Reloaded adapter output had no JSON action: {decoded}")
    parsed, _ = JSONDecoder().raw_decode(decoded[start:])
    if not isinstance(parsed, dict) or parsed.get("action") not in {"tool_call", "stop"}:
        raise RuntimeError(f"Reloaded adapter output failed production action shape: {decoded}")
    if parsed["action"] == "tool_call":
        if not isinstance(parsed.get("tool"), str) or not isinstance(parsed.get("arguments", {}), dict):
            raise RuntimeError(f"Reloaded adapter produced an incomplete tool_call: {decoded}")
        parsed = {"action": "tool_call", "tool": parsed["tool"],
                  "arguments": parsed.get("arguments", {})}
    else:
        parsed = {"action": "stop"}
    print(json.dumps({"adapter_reload": "ok", "probe_action_parse": "ok", "probe": parsed}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/kaggle/input/lifemate-qwen-next-action-v1/sft_train.jsonl")
    p.add_argument("--output", default="/kaggle/working/qwen-next-action-lora-v1")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-examples", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
