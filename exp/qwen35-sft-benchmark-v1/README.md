# Qwen3.5 SFT benchmark

Compare three inference branches on the locked enriched validation cohort60: A is the exact Kaggle version1 base, B is the final PEFT adapter merged into that base, and C is the same SFT model with the screened CoT checklist. This is a screening benchmark only; it does not use public/private test data or create a submission.

## Current status

The Kaggle API token was read from the project `.env` without printing or persisting its value. Kaggle model `limamateus/qwen3-5-4b/pytorch/default/version1` was downloaded; its API file inventory matched the extracted model files and all files were hashed. The final adapter is identical to epoch2 checkpoint486. The BF16 PEFT merge passed its before/after logits check, and the base and merged model were converted with the same pinned llama.cpp converter and quantized to Q4_K_M.

Separate Ollama tags exist for the base and SFT models. Both passed JSON parser/schema and real ToolEnv smoke on fixed official train sample `train_0050`. The smoke artifacts also retain a first failed attempt (`train_0001` returned valid `stop` JSON without a tool call); it was excluded from the pass gate. A/B/C inference and official gold ToolEnv replay are complete on the fixed cohort60. BTC was A `14.97`, B `22.84`, C `26.00`; the paired B−A and C−B 95% intervals include zero. A-repeat reproduced the first-request payload, calls, state hash and final state on all 20 fixed anchors. See `output/summary.md` for the decision and diagnostics.

Provenance limitation: the adapter config records the Kaggle path and version but no revision or base-weight hash. The user attested the model identity, and the local Kaggle version1 files are independently hashed; this still does not prove that the remote training run consumed those exact weight bytes. The SFT dataset used a Vietnamese system prompt; all benchmark branches use the English production prompt, so prompt-language mismatch limits interpretation.

## Scripts

- `scripts/audit_local_artifacts.py`: audits the immutable adapter archive, checkpoint correspondence, tokenizer files, and SFT dataset manifest.
- `scripts/fetch_kaggle_base.py`: downloads the exact Kaggle model version through `KAGGLE_API_TOKEN` from the project `.env`, streams a safe extraction, verifies API file inventory parity, and writes SHA-256 manifests.
- `scripts/merge_adapter.py`: verifies the base and adapter inputs, compares tokenizer vocabulary/template, performs the BF16 PEFT merge, and checks logits before/after merge.
- `scripts/build_llama_quantizer.py`: builds pinned llama.cpp `llama-quantize` for CPU-only conversion.
- `scripts/convert_gguf.py`: exports base and merged HF models to BF16 GGUF and quantizes both to Q4_K_M using the same converter and binary.
- `scripts/register_ollama_models.py`: registers separate Ollama tags and records GGUF and Ollama digests. Run with system Python.
- `scripts/smoke_ollama_models.py`: verifies JSON/schema and one real ToolEnv call for both tags. Run with system Python.
- `scripts/run_benchmark.py --prepare`: locks the fixed cohort, prompt/model mapping, source hashes, and balanced execution order.
- `scripts/run_benchmark.py --run`: runs A/B/C in paired 15-ID blocks, then repeats A on the first/last 10 cohort IDs. Run with system Python; Ollama cannot be reached from `.venv`.
- `scripts/run_benchmark.py --repeat`: runs only the pre-locked A-repeat anchors after validating completed A/B/C artifacts. Used for the audited recovery; it does not rerun or overwrite primary branches.
- `scripts/run_benchmark.py --analyze`: replays gold trajectories through official ToolEnv, computes BTC and paired bootstrap comparisons, and writes the summary.

## Artifact layout

- `output/model-provenance/`: source, adapter, merge, conversion, quantizer, and Ollama hashes/digests.
- `output/gguf/`: BF16 intermediates and Q4_K_M base/SFT models.
- `output/smoke/`: smoke attempts and gate manifest.
- `output/inference/`: locked cohort/config, predictions/traces/requests, gold replay scores, BTC metrics, paired analyses, and execution logs.
- `output/summary.md`: final score comparison and limitations.

Recovery note: the first A/B/C run completed 60/60 samples per branch, then a runner model-map key error stopped execution before A-repeat. The repeat-only recovery retained all primary artifacts, fixed the mapping, and ran exactly the 20 locked anchors. The recovery is recorded in `output/inference/benchmark_lock.json` and `output/inference/inference-run.json`.
