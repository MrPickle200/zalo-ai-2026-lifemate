# Qwen3.5-4B SFT continuation adapter input

This is the exact final LoRA adapter from the completed two-epoch run, packaged as an input for the next one-epoch continuation. It is not a merged model and it is not the epoch-1 checkpoint.

## Upload to Kaggle

Upload this folder as a private Kaggle dataset. A suggested slug is `daoviet/lifemate-qwen35-sft-epoch2-adapter-v1`. In the updated notebook, attach the resulting dataset alongside the existing BTC competition data, SFT train dataset, base model, and offline wheel notebook `daoviet/lifemate-wheels`. Internet can remain disabled.

The notebook recursively locates `adapter_final/adapter_config.json` and `adapter_model.safetensors`, then verifies the weight SHA-256 before training. Keep the folder structure intact. `adapter_final/` must contain both files.

## Continuation recipe

- Parent: final adapter after 2 epochs, loaded from `results.zip`.
- Additional training: 1 epoch, learning rate `1e-5`, fresh optimizer and scheduler.
- Data and prompt: same 3,873 train rows and same action-only labels; no CoT targets and no validation labels.
- Sequence cap: 8,192; effective batch target: 16; expected accelerator: 4 × NVIDIA L4.
- Checkpoints: every 60 optimizer steps; final adapter is written to a separate Kaggle working directory.

After downloading the output, evaluate it using the fixed downstream cohort/val309 setup before deciding whether to merge, convert to GGUF, or promote it. This package does not include the base model or training dataset.
