# QUAN transcript normalization — val60

## Question and scope
Screen the effect of deterministic transcript normalization between current QUAN and Qwen on the fixed enriched 60-ID validation cohort. Raw QUAN remains the baseline. The requested numeric parser could not be scoped safely from available verified project code, so this artifact is explicitly the separate orthography-only fallback required by TASK.md; it is not the full numeric-plus-spelling treatment.

## Cohort and lock
Uses the existing 60 IDs, in source order, from `inference-latency-optimization-val60`, independently checked against the SFT benchmark cohort. Newline-joined ID SHA-256: `87b925a9671e6131c31ee3ee1288372289bda525037cd09f5796c9b1b7fc18e9`. (See `output/cohort_manifest.json`.) Branch order and the 20 A-repeat anchors were frozen in `output/execution_order.json`.

## Baseline/config lock
Current QUAN checkpoint SHA-256 `e31be8516152587134fb62e0d1690954221a0f02c53b6549cf79db58c8eb18ae`, KenLM beam 32, alpha 0.3, beta 1.0. Qwen SFT epoch 2 + CoT digest `973ae821f84016daad8db7ebf494fd2ddf98db7d1f14b5502ffb608cbfc8304d`; prompt/config fingerprints and source hashes are recorded in `output/input_lock.json`. Intended runtime remains thinking on, JSON, temperature 0, num_predict 768, num_ctx 8192, num_batch 512, max 5 action turns, seed 17, retrieval off, PhoASR history.

## Current status
Offline preflight, 309-transcript audit, and the separate spelling-only fallback benchmark are complete. Native Ollama was used outside the project .venv. The numeric parser was deferred, so the 60-ID cohort received no transcript changes and A/B are identical-input repeat runs. BTC and paired/bootstrap diagnostics are in `output/inference/summary.md`; they do not establish a normalization effect.

## Inventory
- `scripts/audit_transcripts.py`: pure NFC + frozen whole-token spelling transform and offline audit.
- `output/normalizer_spec.json`, `input_lock.json`, `execution_order.json`, `cohort_manifest.json`: frozen specification and pre-inference locks.
- `output/transcripts-raw.jsonl`, `transcripts-normalized.jsonl`: 309 raw/normalized transcript pairs.
- `output/audit-per-sample-309.jsonl`, `audit-summary.json`: lexical/slot audit.
- `output/inference/`: A/B/A-repeat predictions, ToolEnv traces, requests/telemetry, BTC per-sample rows, paired bootstrap, argument-field audit, and recovery record.
- `output/normalizer_spec-preinference.json` and `input_lock-preinference.json`: exact frozen pre-run artifacts.
- `output/audit-run.log`: script run record.
- `output/summary.md`: concise result and blockers.
