# LifeMate Qwen3.5 SFT — English system prompt variant v1

This is a separate prompt-language variant of `kaggle-upload`. The parent package is unchanged.

## What changed

Only `messages[0].content` (the system prompt) in `sft_train.jsonl` changed. It now uses the English branch from `Agent.next_action()` in `src/lifemate/agent.py`, with `/no_think` removed because the Qwen chat template is trained with thinking enabled. This is a language-alignment experiment; no CoT checklist or CoT target was added.

The Vietnamese user transcript, context, conversation history, available-tool schemas, ToolEnv observations, next-action targets, row ordering, split IDs, and val309 inference inputs are unchanged. The training set remains 3,873 action-only examples from the fixed 1,236 train trajectories; the 309 validation labels remain excluded.

## Files

- `sft_train.jsonl`: English-system-prompt SFT rows.
- `train_ids.json`, `val309_ids.json`: unchanged split IDs.
- `val309_inputs.jsonl`: unchanged label-free validation inputs.
- `dataset_manifest.json`: source/variant provenance and hashes.

## Kaggle

Upload this folder as a separate private Kaggle Dataset, suggested slug `daoviet/lifemate-qwen35-sft-data-en-v1`. When running the continuation notebook, attach this dataset and detach the Vietnamese `lifemate-qwen35-sft-data-v1` dataset so that only one `sft_train.jsonl` is discovered. The current Vietnamese continuation notebook pins the old dataset hash; its English variant must use the new manifest hash.
