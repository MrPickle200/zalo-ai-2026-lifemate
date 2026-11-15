# Qwen3.5 SFT epoch 3 + CoT on the fixed cohort60

Compare the continued epoch-3 LoRA adapter merged into the same Kaggle version-1 base against the already-completed epoch-2 SFT + CoT branch C from `qwen35-sft-benchmark-v1`. Reuse the exact old C per-sample BTC, predictions, and request telemetry; run only epoch 3 + CoT on the same ordered 60 held-out validation IDs.

The adapter archive is `exp/qwen35-sft-next-action-v1/results (1).zip`. The archive report says this is one epoch continued from the epoch-2 adapter, with fresh optimizer/scheduler state. Its recorded continuation adapter hash and SFT data hash are verified before merge. The remote report has no base-weight hash, so the user-attested Kaggle version-1 identity remains an explicit provenance limit.

## Pipeline

Run merge/conversion from the project `.venv`; run every Ollama command with system Python outside `.venv` because the local Ollama server is not reachable from `.venv`.

1. `scripts/audit_extract.py`: hash and inspect the new archive, verify epoch-2 continuation lineage and train-data hash, check old C fingerprints/artifact coverage, then extract only the adapter files and report.
2. `scripts/merge_adapter.py --base-dir models/kaggle-qwen3-5-4b-v1/Qwen3.5-4B`: merge in BF16 into the exact hashed Kaggle version-1 base and record a before/after logits check.
3. `scripts/convert_model.py`: use the benchmark's pinned llama.cpp converter and quantizer to produce the epoch-3 Q4_K_M GGUF.
4. `scripts/register_ollama.py`: register the unique local tag and record model digest/version.
5. `scripts/run_epoch3_cot.py --prepare`, `--smoke`, then `--run`: lock all settings, execute one ToolEnv per cohort ID, and preserve calls/state even on a caught failure.
6. `scripts/analyze_epoch3.py`: replay gold through official ToolEnv, compute BTC and paired bootstrap for epoch 3 C+CoT minus reused epoch 2 C+CoT.
7. `scripts/make_submission_epoch3.py --prepare`, then `--run` with system Python outside `.venv`: generate the full public-test candidate and run the official schema/order validator. This creates a local candidate; it does not submit it to Public LB.

## Locked comparison

- Cohort/order: exact 60 IDs in the completed benchmark `output/inference/cohort_manifest.json`.
- Old comparison: completed branch C (`COT`) in `exp/qwen35-sft-benchmark-v1/output/inference/`; old BTC is 26.00/100. It is not rerun.
- Shared inference: QUAN LifeMate FT transcripts, PhoASR history, English production prompt plus the exact CoT checklist, Ollama 0.40.15, thinking on, JSON, temperature 0, `num_predict=768`, `num_ctx=8192`, `num_batch=512`, five action turns, seed 17.
- This enriched cohort is a screening set, not full val309. RTX 4050 latency over 30 seconds is reported separately as a local diagnostic, not an A30 timing result.

## Artifacts

- `output/model-provenance/`: archive and lineage audit, adapter hashes, merge/conversion manifests, Ollama fingerprint, frozen input/config lock.
- `output/merged-hf/`, `output/gguf/`: epoch-3 merged model and GGUF artifacts.
- `output/inference/`: epoch-3 C+CoT predictions/traces/requests, gold ToolEnv scores, paired bootstrap, and analysis. Reused epoch-2 C files remain in the original benchmark experiment.
- `output/summary.md`: result, uncertainty, runtime, and provenance limits.
- `output/submission/`: public-test lock, run log, request telemetry, validator log, and candidate summary. The JSONL submission, prediction sidecar, and manifest are under `outputs/submissions/`.

## Result (2026-11-08)

- Reused epoch-2 C without rerunning it: BTC before runtime gate 26.00/100; local 30-second-gated diagnostic 23.67/100.
- New epoch-3 + CoT: BTC before runtime gate 23.28/100; local 30-second-gated diagnostic 23.28/100. Paired epoch3−epoch2 delta −2.72 points (95% bootstrap CI [−7.72,+1.33]); 4 improved, 5 worsened, 51 tied. The CI includes zero; this cohort does not establish that the extra epoch helps.
- Both branches used the exact same ordered cohort60 and first-request system-prompt hashes. New branch had 32/60 hash matches and M/U/G=51/64/108; zero runtime/gold replay failures and exactly one prediction ToolEnv per sample. Local >30-second samples were 9 vs 10 in the reused run; this is RTX 4050 timing, not an A30 result.
- No submission, public/private test, or Public LB was used. The remote training report still lacks an exact base-weight hash; the base identity is user-attested.

## Public-test candidate (2026-11-08)

- Generated and validated 634 rows using the epoch-3 + CoT tag, the exact TASK CoT checklist, cached QUAN FT/KenLM beam32 public-test transcripts, PhoASR history, thinking on, JSON, temperature 0, `num_predict=768`, context8192 and batch512. Exactly one prediction ToolEnv per sample; the official schema/order validator found 0 errors. No public/private labels were used and the candidate was not sent to Public LB.
- Submission: `outputs/submissions/qwen35-sft-epoch3-cot-quan-beam32-num-predict-768-num-batch-512-public-test.jsonl`; SHA-256 `3b697aeb2ce966e5bd8e6e18503d6388e40fdcdfde49d83f728f791d8f581195`. Predictions sidecar SHA-256 `d4a2ea830f6d4e1ed1be9c8ba109b7356a3b45b3d9556fde3d0de1e104d98549`.
- Inference had 0 runtime errors and took 15,279 seconds. Local RTX 4050 mean/p50/p95/max latency was 24.09/20.42/53.90/72.13s; 177 samples exceeded 30 seconds. This does not establish A30 runtime compliance. There were 2,181 Ollama requests; 63 reached `num_predict`.
