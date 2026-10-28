# WORKLOG.md

## 2026-10-28 — Initial baseline

### Done
- [x] Inspected competition rules and official dev kit interfaces without opening environment state files.
- [x] Created the project virtual environment and baseline project structure.
- [x] Implemented local ASR/LLM interfaces, schema validation, a bounded ToolEnv loop, and runner/evaluation/submission commands.
- [x] Inspected train sample layout and compiled all source and script files.
- [x] Installed runtime packages and a CUDA 13.2 PyTorch wheel into `.venv`; confirmed the RTX 4050 is visible to PyTorch.

### Result
- Model download was attempted with the `HF_TOKEN` from `.env`; local outbound connection to Hugging Face was refused. The 20-sample train evaluation therefore stopped before the first sample (0 samples run; no score).

### Files
- `AGENTS.md`
- `requirements.txt`
- `.gitignore`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-28 — CUDA and model setup

### Done
- [x] Installed the CUDA 13.2 PyTorch wheel and verified PyTorch and CTranslate2 both see the RTX 4050.
- [x] Added a development-only downloader that reads `HF_TOKEN` from `.env` without displaying it, plus pinned revisions and setup instructions.
- [x] Retried the 20-sample train evaluation after dependency setup.

### Result
- Model download failed because the local Hugging Face connection was refused despite the configured token. Evaluation remains at 0/20 samples; no score was produced.

### Files
- `requirements.txt`
- `requirements-torch-cu132.txt`
- `scripts/download_models.py`
- `src/lifemate/agent.py`
- `src/lifemate/asr.py`
- `README.md`
- `.gitignore`

## 2026-10-28 — Local model verification

### Done
- [x] Verified Whisper Small files and all three Qwen safetensors shards; Qwen config, tokenizer, and tensor indexes load locally.
- [x] Fixed PyAV compatibility by pinning 16.1.0 and added a CTranslate2 CUDA-to-CPU fallback for missing Windows DLLs.
- [x] Shortened duplicated prompt data and made LLM GPU memory allocation adapt to available VRAM.

### Result
- ASR transcribed and cached one sample using CPU fallback. Qwen weights loaded with CPU offload, but generation remained too slow on the 6 GB laptop after five minutes and was stopped; no end-to-end sample or evaluation score is available.

### Files
- `src/lifemate/asr.py`
- `src/lifemate/agent.py`
- `requirements.txt`
- `INSIGHTS.md`


## 2026-10-28 — Ollama inference trial

### Done
- [x] Added an optional Ollama chat API backend selected with `LIFEMATE_LLM_BACKEND=ollama`.
- [x] Downloaded `qwen3:4b-instruct-2507-q4_K_M` and ran `train_0001` through ASR, Ollama, and ToolEnv.

### Result
- End-to-end cached-ASR sample completed in 2.192 seconds. Ollama reported the 3.6 GB model fully on GPU at 6144 context. The prediction made two weather calls, while the gold label requires five calls; this single sample is not an accuracy benchmark.

### Files
- `src/lifemate/agent.py`

## 2026-10-28 — Ollama local CV

### Done
- [x] Completed the default 20-sample labeled train evaluation with Ollama and Qwen3-4B-Instruct-2507 Q4_K_M at 16384 context.
- [x] Saved the evaluation report to `outputs/eval/ollama-q4/cv20-summary.json`.

### Result
- 20/20 samples completed without runtime failures. Tool-call F1 was 0.0247, exact match 0.05, state match 0.25, and average latency 8.47 seconds per sample. LLM requests at 16384 context used a CPU/GPU split on this 6 GB GPU.

### Files
- `src/lifemate/agent.py`
- `outputs/eval/ollama-q4/cv20-summary.json`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-28 — Evaluation output organization

### Done
- [x] Grouped artifacts into `ollama-q4/`, `baseline-v1-val309/`, `oracle-asr-v1-val309/`, and `paired-analysis/` under `outputs/eval/`.
- [x] Updated evaluation and analysis scripts to read/write the experiment-specific directories and documented the layout and commands.

### Files
- `scripts/run_eval.py`
- `scripts/analyze_eval.py`
- `README.md`
- `WORKLOG.md`

## 2026-10-28 — Whisper CUDA runtime

### Done
- [x] Installed CUDA Runtime 12, cuBLAS 12, and cuDNN 9 wheels inside `.venv`.
- [x] Added Windows DLL path setup for NVIDIA wheel directories before loading CTranslate2.
- [x] Ran faster-whisper Small on the RTX 4050 and completed actual audio transcription with `device=cuda`.

### Result
- GPU transcription completed successfully. The model package fallback to CPU is no longer needed on this machine when these CUDA wheels are installed.

### Files
- `src/lifemate/asr.py`
- `requirements-torch-cu132.txt`
- `README.md`

## 2026-10-28 — Public test run paused

### Done
- [x] Generated 158 valid public-test submission rows before pausing at the user's request.
- [x] Added `--resume` to continue only from a matching valid output prefix.

### Result
- `outputs/submissions/submission.jsonl` has 158 unique, schema-valid rows. No public-test score is available because the split has no labels.

### Files
- `scripts/make_submission.py`
- `README.md`
- `outputs/submissions/submission.jsonl`

## 2026-10-28 — Public test submission complete

### Done
- [x] Completed all 634 public-test samples with Whisper Small and Ollama Qwen3-4B-Instruct-2507 Q4_K_M.
- [x] Validated the submission row count, schema, unique IDs, and order against the public manifest.

### Result
- `outputs/submissions/submission.jsonl` contains 634 valid rows in manifest order. The public split has no labels, so no score can be computed locally.

### Files
- `outputs/submissions/submission.jsonl`
- `scripts/make_submission.py`


## 2026-10-28 ? Public leaderboard score

### Done
- [x] Recorded the user-reported public score of 7.85 for the Ollama Qwen3-4B Q4_K_M baseline.

### Files
- `PROGRESS.md`

## 2026-10-28 — Fixed validation and oracle-ASR

### Done
- [x] Created a fixed 1,236/309 grouped train/validation split with seed 20261028; transcript near-duplicates at similarity >= 0.92 stay in one partition.
- [x] Added a deterministic local holdout runner with baseline and oracle-transcript modes, per-sample outputs, ToolEnv final-state checks, and paired error analysis.
- [x] Ran both modes on all 309 validation samples (0 runtime failures) and updated score tracking and insights.

### Result
- Baseline proxy mean 0.1919 (tool-call F1 0.1155; state match 0.2492).
- Oracle transcript proxy mean 0.2920 (tool-call F1 0.2429; state match 0.3301); paired gain +0.1001.
- Heuristic baseline errors were led by WRONG_TOOL (204) and MISSING_INFO_ERROR (51); oracle-corrected ASR cases: 16.

### Files
- `data/splits/train_ids.json`
- `data/splits/val_ids.json`
- `data/splits/split_report.json`
- `scripts/make_split.py`
- `scripts/run_eval.py`
- `scripts/analyze_eval.py`
- `src/lifemate/pipeline.py`
- `outputs/eval/baseline-v1-val309/predictions.jsonl`
- `outputs/eval/oracle-asr-v1-val309/predictions.jsonl`
- `outputs/eval/paired-analysis/summary.json`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-28 — Retrieval routing and new submission

### Done
- [x] Built a TF-IDF/cosine index from only the 1,236 fixed train IDs; added retrieved few-shots, tool shortlist/fallback, and train-derived missing-info examples.
- [x] Ran retrieval-v1 and a wider top-5/eight-tool variant across all 309 validation IDs (0 runtime failures each); compared paired metrics and error categories.
- [x] Generated the retrieval-v1 public-test submission and validated all rows, IDs/order, state-hash format, and tool schemas against the manifest/catalog without loading environment files.

### Result
- Retrieval-v1: proxy 0.2432, tool-call F1 0.2299, state match 0.2880; `WRONG_TOOL` 188 and `MISSING_INFO_ERROR` 53.
- Wider variant was lower (proxy 0.2392, F1 0.2287, state match 0.2783), so retrieval-v1 was selected.
- `outputs/submissions/retrieval-v1.jsonl` contains 634 valid rows. One transient Ollama CUDA OOM stopped the first pass at row 500; `--resume` completed the remaining prefix with unchanged context/model settings.
- User later reported Public LB 6.66 for retrieval-v1 (baseline Public LB: 7.85); recorded in `PROGRESS.md` and `INSIGHTS.md`.

### Files
- `src/lifemate/retrieval.py`
- `src/lifemate/agent.py`
- `src/lifemate/pipeline.py`
- `scripts/run_eval.py`
- `scripts/analyze_eval.py`
- `scripts/make_submission.py`
- `scripts/validate_submission.py`
- `outputs/eval/retrieval-v1-val309/`
- `outputs/eval/retrieval-v2-wide-val309/`
- `outputs/eval/paired-analysis/baseline-vs-retrieval-v1/`
- `outputs/eval/paired-analysis/baseline-vs-retrieval-v2-wide/`
- `outputs/submissions/retrieval-v1.jsonl`
- `PROGRESS.md`
- `INSIGHTS.md`
- `README.md`

- `src/lifemate/`
- `scripts/`
- `configs/baseline.yaml`
- `requirements-torch-cu132.txt`
