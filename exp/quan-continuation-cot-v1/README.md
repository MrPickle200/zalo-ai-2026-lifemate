# QUAN continuation and CoT screening

This experiment follows `TASK.md` in three stages: continue the existing QUAN LifeMate checkpoint for up to three one-epoch updates; score each update on the locked 309-ID validation split; compare downstream BTC on the pre-existing enriched train-only cohort60; then run a prompt-only CoT comparison after the ASR checkpoint is locked.

Fine-tuning uses only the fixed 1,236 train IDs. ASR evaluation uses the 309 held-out validation IDs; the downstream cohort is the pre-existing enriched, ordered 60-ID subset of those 309 and is never used for training. Public/private test data and all `*_environment/` files are excluded. No submission or Public LB upload is part of this experiment. Each downstream branch uses exactly one prediction ToolEnv per sample and gold replay only through the official ToolEnv.

## Locked starting point

- QUAN checkpoint: `exp/quan-transcribe-lifemate-domain-v1/output/model`.
- Starting weights SHA-256: `e31be8516152587134fb62e0d1690954221a0f02c53b6549cf79db58c8eb18ae`.
- Existing baseline: 309 validation IDs, QUAN KenLM CTC beam32, alpha=0.3, beta=1.0; corpus WER 0.1631, CER 0.1298. Predictions and summary are copied under `output/asr/` and left unchanged.
- Downstream cohort: exact order from `output/cohort_manifest.json`, SHA-256 of newline-joined IDs `87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9`. It is a fixed subset of held-out val309, enriched for prior agent failures, so it is not an unbiased val309 estimate.
- QUAN continuation will use the top-two encoder + reader stage with learning rates below the original encoder-stage rates. Each update is a separate checkpoint directory and is evaluated before deciding whether to continue.

## Scripts

- `scripts/finetune_quan_continuation.py`: one-epoch continuation from a supplied parent checkpoint; refuses to overwrite a populated output directory.
- `scripts/decode_validation.py`: same 309-ID order, preprocessing, and KenLM beam32 decoder for each checkpoint.
- `scripts/analyze_epoch.py`: corpus WER/CER, paired per-utterance deltas and bootstrap intervals, plus numeric/negation/time exactness diagnostics.
- `scripts/select_checkpoint.py`: selects a checkpoint by validation WER, using CER as a near-tie breaker; records early stopping.
- `scripts/run_downstream.py --prepare|--step2|--cot`: locks source/config/cohort/model fingerprints, runs the balanced A/B repeat, then the prompt-only CoT branch. Run with system Python outside `.venv` to reach native Ollama.
- `scripts/analyze_downstream.py --step2|--step3`: replays gold through official ToolEnv and writes BTC per-sample, aggregate, and paired bootstrap reports.

## Outcome

- Two additional QUAN epochs were run at lower learning rates. Both validation WER and CER increased at each epoch, so training stopped before epoch 3 and the existing checkpoint remained selected.
- Step 2 used A/B as repeated runs of the selected existing checkpoint, as required when no new checkpoint is selected. The initial payloads matched for all 60 IDs, but tool calls or state hashes differed on 26 samples, indicating unstable repeated generation.
- CoT was evaluated on the same fixed cohort. Its unmasked BTC point estimate was close to the A prompt control (+0.39 points; paired CI crosses zero), while the local 30-second-gated diagnostic fell and request cap hits increased. Do not promote CoT from this screen.
- Full metrics and limitations: `output/summary.md`.

## Artifact layout

- `output/preflight.json`: split, cohort, checkpoint, baseline, LM and source hashes.
- `output/checkpoints/epoch-XX/`: immutable checkpoint and training audit for each additional epoch.
- `output/asr/`: baseline and per-epoch validation predictions, summaries and paired analyses.
- `output/downstream/`: locked manifests, paired ToolEnv predictions/traces, gold replay, BTC per-sample scores, analyses and telemetry. Raw chain-of-thought is never stored.
- `output/summary.md`: per-step results, decisions, and limitations.
