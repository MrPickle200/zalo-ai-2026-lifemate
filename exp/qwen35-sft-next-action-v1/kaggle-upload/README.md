# LifeMate Qwen3.5 SFT dataset v1

This folder contains only the derived SFT package. The original BTC competition dataset is a separate, already-uploaded Kaggle input and should be attached to the notebook as-is; do not duplicate it into this dataset.

## Files

- `sft_train.jsonl`: 3,873 gold next-action examples from the fixed 1,236-trajectory training split. This is the only file used for gradient updates.
- `train_ids.json`: IDs in the training split.
- `val309_ids.json`: IDs reserved for validation.
- `val309_inputs.jsonl`: inference inputs for the 309 validation trajectories. It contains no target actions and is not read by the training cell; keep it for the later official local BTC evaluation workflow.
- `dataset_manifest.json`: counts, provenance, split checks, and SHA-256 hashes.

The train and validation trajectory IDs have zero overlap. Targets are one next action each (tool call, `sys.ask_missing_info`, or `STOP`); no chain-of-thought is included. Historical audio was transcribed with QUAN-Transcribe.

## Kaggle notebook inputs

Attach three inputs to `qwen35_lifemate_sft.ipynb`:

1. The existing BTC competition data input, preserving its supplied folder structure.
2. This folder uploaded as a private Kaggle Dataset, suggested slug `daoviet/lifemate-qwen35-sft-data-v1`.
3. The Qwen3.5-4B Transformers model input, suggested source `limamateus/qwen3-5-4b`.

The notebook discovers inputs by file/config structure, so Kaggle may assign different mount-folder names. It expects exactly one matching BTC tree, SFT bundle, and Qwen3.5-4B model. Training writes to `/kaggle/working/qwen35_lifemate_sft_v1`; it does not publish or submit anything.

After the run, download the adapter and training report. Merge with the same base model, convert and quantize to GGUF locally, then run the project's official BTC val309 evaluation before comparing/promoting the model.
