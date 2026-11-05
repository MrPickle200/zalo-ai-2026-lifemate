# WORKLOG.md

## 2026-11-02 — BamiBERT next-tool routing experiment

### Done
- Audited fixed 1,236 train / 309 validation split; no ID overlap. Reused train-only gold replay timesteps (4,043) and replayed all val309 gold trajectories once through ToolEnv to create 1,043 validation timesteps; zero replay failures.
- Built `tool-router-v1` dataset with transcript, context, transcript-resolved history, available tool names, prior calls/observations, and step index. Labels contain only tool name or STOP; current/future target arguments are excluded. Train step counts: step 0 1,148; step 1 850; step 2+ 809; STOP 1,236. Val counts: 286, 215, 233, 309.
- Downloaded `Qualcomm-AI-Research/BamiBERT` with `.env` HF token, revision `57bc1340debbe4e348ec549047a763caebe4a977`. Used encoder classification head, 103,064,980 parameters. The released tokenizer config names XLM-R Unigram, but tokenizer.json is BPE; loaded the BPE graph with Transformers' generic fast tokenizer.
- Checked the model card's stated BSD 3-Clause Clear and Qualcomm Responsible AI licenses; the card describes research/educational use. The model's July 2026 paper predates the competition cutoff.
- Trained deterministic fp16 RTX 4050 batch-8 runs: length 512 for 3 epochs (459s, peak 2.99 GiB); length 1024 for 3 epochs (930s, peak 4.67 GiB). Inverse-square-root class weighting, LR 2e-5, AdamW, weight decay 0.01. Best checkpoint by val accuracy.
- Evaluated exact next-tool classification on all 1,043 val timesteps; compared Qwen routing aligned to gold timestep, treating Qwen as diverged after wrong tool.

### Result
- Both lengths selected STOP on every validation timestep: top-1 0.2963 (STOP prevalence), top-3 0.3950, macro F1 0.0037, weighted F1 0.1354, non-STOP accuracy 0.0. STOP precision/recall/F1: 0.2963 / 1.0000 / 0.4571.
- First/second/third+ tool accuracy: 0 / 0 / 0. All gold tool targets were confused with STOP; largest were contacts.search (56), sys.ask_missing_info (47), smart_home.light_set (45), smart_home.get_status (31), reminders.create (26).
- Qwen aligned routing: first 0.2937 (n=286), second 0.0977 (n=215), later 0.0515 (n=233); STOP precision 0.3987, recall 0.7767.
- 512-token truncation was 93.73%; at 1024 it fell to 63.11% with no metric change. **Decision: NO-GO** for the decomposed routing policy as implemented. The classifier did not learn tool routing and was worse than Qwen on first and continuation actions; do not build the argument generator decomposition from this result.
- Classification metrics only; no competition score was calculated. Production agent and submission pipeline were not changed, and no public test was run.

### Files
- `src/lifemate/tool_router.py`
- `exp/tool-router-v1/scripts/build_tool_router_dataset.py`
- `exp/tool-router-v1/scripts/train_tool_router.py`
- `exp/tool-router-v1/scripts/eval_tool_router.py`
- `exp/tool-router-v1/output/training-data/`
- `exp/tool-router-v1-val309/output/`
- `exp/tool-router-v1-val309-1024/output/`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-31 — PhoASR paired error diversity and hybrid

### Done
- Created `exp/asr-paired-val309/output/paired_transcripts.jsonl` and summary from the existing public and fine-tuned PhoASR validation outputs; all 309 fixed IDs align.
- Quantified WER and transcript overlap: 203 identical / 106 different; public lower WER on 19, FT lower on 37, tied on 253; both wrong on 301.
- Confirmed stored ASR output has no confidence metadata, so confidence selection was not implemented.
- Added an opt-in dual-transcript prompt path to the baseline evaluator and agent. Evaluated full val309 using baseline Qwen3-4B, full schemas, no retrieval, and ToolEnv; 309/309 completed with zero failures.
- Wrote paired metric and regression artifacts under `exp/paired-analysis/phoasr-hybrid-v1/output/`. Paired comparison: 28 improved, 49 worsened, 232 unchanged vs PhoASR public.
- Compiled the changed Python files and did not create a public submission because hybrid metrics regressed.

### Result
- Dual transcript: proxy 0.2064, micro tool-call F1 0.1509, final-state match 0.2718 vs PhoASR public 0.2286 / 0.1693 / 0.2751.
- 37 of the 49 regressions had identical transcripts, with examples of unnecessary tool calls and missed correct calls; the added prompt itself shifted agent behavior. Keep PhoASR public, stop this ASR hybrid line, and focus on agent/tool selection.

### Files
- `src/lifemate/agent.py`
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/asr-modeling/scripts/make_paired_asr.py`
- `exp/asr-modeling/scripts/analyze_phoasr_hybrid.py`
- `exp/asr-paired-val309/output/`
- `exp/phoasr-dual-transcript-v1-val309/output/`
- `exp/paired-analysis/phoasr-hybrid-v1/output/`
- `PROGRESS.md`
- `INSIGHTS.md`

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

## 2026-10-29 — PhoASR controlled ASR ablation

### Done
- Added a separate PhoASR Transformers backend using the official Qualcomm-AI-Research PhoASR-whisper-small local checkpoint; retained the faster-whisper Whisper Small backend as baseline.
- Added `--asr` selection to local validation and submission runners, plus a PhoASR baseline config and paired ASR/downstream analysis script.
- Smoke-tested 5 fixed validation samples with non-empty Vietnamese transcripts and downstream Qwen/ToolEnv execution.
- Ran the full 309-sample fixed validation with baseline Qwen3-4B, full schemas, and retrieval disabled. No runtime failures.
- Generated the 634-row public-test candidate at `outputs/submissions/phoasr-small-baseline-v1.jsonl`. Two Ollama runner resource failures were recovered using `--resume`, with the same baseline configuration.

### Result
- PhoASR corpus WER/CER: 0.2804 / 0.2013; Whisper Small: 0.5108 / 0.3657.
- PhoASR downstream proxy/F1/state: 0.2286 / 0.1693 / 0.2751; baseline: 0.1919 / 0.1155 / 0.2492.
- Paired: 55 improved, 24 worsened, 230 unchanged. Proxy oracle gap closed: 36.7%.
- Mean latency 11.17s; one validation sample exceeded 30s locally (34.08s max), so A30 runtime remains to be checked.

### Files
- `src/lifemate/asr.py`
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/lifemate-pipeline/scripts/make_submission.py`
- `exp/asr-modeling/scripts/analyze_asr.py`
- `configs/phoasr-small-baseline.yaml`
- `exp/phoasr-small-baseline-val309/output/`
- `exp/paired-analysis/baseline-vs-phoasr-small-baseline/output/`
- `PROGRESS.md`
- `INSIGHTS.md`

### Public candidate status
- Complete: 634 rows, 634 unique IDs in manifest order, zero schema errors via `exp/lifemate-pipeline/scripts/validate_submission.py`.
- Candidate passed validation. User-reported Public LB: 9.79.
- `exp/lifemate-pipeline/scripts/download_models.py`
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
- [x] Saved the evaluation report to `exp/ollama-q4/output/cv20-summary.json`.

### Result
- 20/20 samples completed without runtime failures. Tool-call F1 was 0.0247, exact match 0.05, state match 0.25, and average latency 8.47 seconds per sample. LLM requests at 16384 context used a CPU/GPU split on this 6 GB GPU.

### Files
- `src/lifemate/agent.py`
- `exp/ollama-q4/output/cv20-summary.json`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-28 — Evaluation output organization

### Done
- [x] Grouped artifacts into `exp/ollama-q4/`, `exp/baseline-v1-val309/`, `exp/oracle-asr-v1-val309/`, and `exp/paired-analysis/`.
- [x] Updated evaluation and analysis scripts to read/write the experiment-specific directories and documented the layout and commands.

### Files
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/lifemate-pipeline/scripts/analyze_eval.py`
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
- `exp/lifemate-pipeline/scripts/make_submission.py`
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
- `exp/lifemate-pipeline/scripts/make_submission.py`


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
- `exp/lifemate-pipeline/scripts/make_split.py`
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/lifemate-pipeline/scripts/analyze_eval.py`
- `src/lifemate/pipeline.py`
- `exp/baseline-v1-val309/output/predictions.jsonl`
- `exp/oracle-asr-v1-val309/output/predictions.jsonl`
- `exp/paired-analysis/historical-reports/output/summary.json`
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
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/lifemate-pipeline/scripts/analyze_eval.py`
- `exp/lifemate-pipeline/scripts/make_submission.py`
- `exp/lifemate-pipeline/scripts/validate_submission.py`
- `exp/retrieval-v1-val309/output/`
- `exp/retrieval-v2-wide-val309/output/`
- `exp/paired-analysis/baseline-vs-retrieval-v1/output/`
- `exp/paired-analysis/baseline-vs-retrieval-v2-wide/output/`
- `outputs/submissions/retrieval-v1.jsonl`
- `PROGRESS.md`
- `INSIGHTS.md`
- `README.md`

## 2026-10-29 — Soft retrieval

### Done
- [x] Added a full-tool soft-retrieval mode: TF-IDF top-3 demonstrations stay prompt-only, with explicit anti-copy and missing-required-argument instructions.
- [x] Added a deterministic curated-fewshot mode based on diverse train-only examples; kept it available but skipped its validation run after soft retrieval performed well.
- [x] Measured hard-shortlist recall from previous validation predictions and evaluated soft retrieval on all 309 validation samples.

### Result
- Soft retrieval achieved proxy 0.2558, aggregate tool-call F1 0.2435, and state match 0.2977 with zero runtime failures. Paired vs baseline: 86 improved, 32 worsened, 191 tied. Error counts: WRONG_TOOL 194 and MISSING_INFO_ERROR 49.
- Selected soft retrieval; full public inference completed in 6,914.7 seconds and created `outputs/submissions/soft-retrieval-v1.jsonl` with 634 unique rows, correct manifest order, and zero schema errors. Empty tool-call rows: 139 (retrieval-v1: 156). Public leaderboard result is pending external upload.
- User-reported Public LB: 7.51, improving over retrieval-v1 (6.66) but below baseline (7.85).

### Files
- `src/lifemate/agent.py`
- `src/lifemate/pipeline.py`
- `src/lifemate/retrieval.py`
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/lifemate-pipeline/scripts/make_submission.py`
- `exp/soft-retrieval-v1-val309/output/`
- `exp/paired-analysis/baseline-vs-soft-retrieval-v1/output/`
- `outputs/submissions/soft-retrieval-v1.jsonl`
- `INSIGHTS.md`
- `PROGRESS.md`

- `src/lifemate/`
- `exp/lifemate-pipeline/scripts/`
- `configs/baseline.yaml`
- `requirements-torch-cu132.txt`

## 2026-10-30 — PhoASR LifeMate domain fine-tuning

### Done
- Confirmed the local base is `Qualcomm-AI-Research/PhoASR-whisper-small`; local model metadata did not include a revision commit. Backend: Hugging Face Transformers `AutoModelForSpeechSeq2Seq` with PyTorch; model config reports Transformers 4.48.0, installed runtime is Transformers 5.21.0.
- Added `exp/asr-modeling/scripts/finetune_phoasr.py` to verify the fixed split, audio readability, transcript presence/UTF-8 handling, run smoke/full training, select by validation loss, and reload/infer the checkpoint. Added `exp/asr-modeling/scripts/analyze_phoasr_ft.py` for paired WER/CER and downstream/oracle-gap analysis. Made PhoASR model path configurable through `LIFEMATE_ASR_MODEL`.
- Audited and decoded all 1,236 train and 309 validation audio files; zero missing/unreadable audio, empty transcripts, or train/validation ID overlaps.
- Smoke run trained the fixed eight-ID subset, completed eight optimizer steps with finite loss, saved/reloaded weights and generated a transcript. Peak GPU memory 1.94 GB.
- Full run: seed 17; one epoch; LR 2e-6; batch 1; accumulation 8; encoder and decoder blocks 0–9 frozen, decoder blocks 10–11 plus layer norm/output head trainable. 58,734,336 trainable params, 155 optimizer steps, 523.8s training, peak GPU memory 2.03 GB. Train loss mean 2.776, validation loss 2.4969. Best checkpoint: `exp/phoasr-lifemate-v1/output/model/best-checkpoint`.
- Evaluated exactly the fixed 309 validation samples using the baseline Ollama Qwen3-4B Q4_K_M agent, full tool schemas, existing prompt/decoding, one ToolEnv/sample, and retrieval off. All 309 completed without runtime failures.
- Direct validation WER/CER: PhoASR public 0.2804/0.2013; LifeMate FT 0.2757/0.1966. Downstream proxy/F1/state: PhoASR public 0.2286/0.1693/0.2751; FT 0.2222/0.1634/0.2686. Paired score counts: 9 improved, 12 worsened, 288 unchanged.

### Decision
- WER/CER improved slightly, but all downstream metrics regressed. Remaining oracle gaps are proxy 0.0698, tool F1 0.0795, final-state 0.0615; the FT widened the public-ASR oracle gap on all three downstream measures. No public submission was created. No Public LB is available.

### Files
- `src/lifemate/asr.py`
- `exp/asr-modeling/scripts/finetune_phoasr.py`
- `exp/asr-modeling/scripts/analyze_phoasr_ft.py`
- `exp/phoasr-lifemate-v1/output/model/`
- `exp/phoasr-lifemate-v1-smoke/output/model/`
- `exp/phoasr-lifemate-ft-v1-val309/output/`
- `exp/paired-analysis/phoasr-public-vs-ft-v1/output/summary.json`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-31 — PhoASR fine-tune public-test candidate

### Done
- Generated `outputs/submissions/phoasr-lifemate-ft-v1.jsonl` on all 634 public-test samples using the fine-tuned ASR checkpoint, baseline Ollama Qwen3-4B Q4_K_M, full tool schemas, and no retrieval.
- Inference completed in 6,886.8 seconds (10.86 seconds/sample average).
- Official submission validator passed: 634 rows, 634 unique IDs in manifest order, zero tool-schema errors. There were zero malformed calls and 108 empty tool-call rows.

### Result
- Candidate is ready at `outputs/submissions/phoasr-lifemate-ft-v1.jsonl`. User-reported Public LB: 9.78, 0.01 below the PhoASR public candidate's 9.79; this is effectively a tie with no measurable public-score gain despite the small WER reduction.

### Files
- `outputs/submissions/phoasr-lifemate-ft-v1.jsonl`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-31 — Qwen tool-calling SFT preparation

### Done
- Added `exp/qwen-next-action-v1/scripts/prepare_qwen_sft.py` and generated the train-only dataset at `exp/qwen3-4b-lifemate-lora-v1/output/model/sft_train.jsonl` with a companion audit summary.
- Verified the fixed split contains 1,236 train IDs and 309 held-out validation IDs with zero overlap. Dataset contains 1,236 rows; official `zaic_tools.schema_validate` reported zero invalid gold calls.
- Attempted to provision the PEFT/quantization backend; package installation was unavailable because pip is configured with `PIP_NO_INDEX=1` and the packages are not installed.

### Result
- Stopped before smoke training under the task fallback. The 6 GB RTX 4050 cannot support full fine-tuning of the 4B model. No adapter or validation outputs exist, and no public submission was created.
- Training configuration/time/checkpoint and downstream metrics are N/A. Whether SFT produces a gain on PhoASR transcripts remains undetermined.

### Files
- `exp/qwen-next-action-v1/scripts/prepare_qwen_sft.py`
- `exp/qwen3-4b-lifemate-lora-v1/output/model/sft_train.jsonl`
- `exp/qwen3-4b-lifemate-lora-v1/output/model/dataset-audit.json`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-31 — Agent architecture audit and step-cap diagnostic

### Done
- Audited `LifeMatePipeline`, `Agent`, and official ToolEnv interface. Confirmed one ToolEnv per sample, one action per Qwen turn, observation feedback to later prompts, persistent session state, and a five-turn default cap.
- Audited gold call-count distribution over the 1,545 labeled train examples: 111 zero-call, 369 one-call, 428 two-call, 359 three-call, 161 four-call, 107 five-call, and 10 six-call.
- Added `exp/agent-architecture-audit/scripts/audit_agent_architecture.py`; selected 30 complex fixed-validation samples with existing PhoASR baseline predictions and emitted gold trajectories, tool results, dependency reasons, baseline action/observation traces, and failure labels.
- Added optional `--sample-ids-file` and `--max-steps` support to `exp/lifemate-pipeline/scripts/run_eval.py`; evaluated the same 30 IDs at caps 6 and 7 using PhoASR public + Ollama Qwen3-4B, full schemas, no retrieval. Baseline cap5 predictions were already available.
- Added `exp/agent-architecture-audit/scripts/compare_agent_architecture_audit.py` and paired cap5/6/7 metrics under `exp/paired-analysis/agent-architecture-cap5-vs-cap7/output/`.

### Result
- Gold trajectory heuristic found 30/30 read-before-write, observation-dependent cases (25 primarily categorized observation-dependent and five missing-info; categories overlap). Baseline cap5 missed the first exact action on all 30, final-state match 0/30, tool-call F1 0.1073.
- Caps 6 and 7 did not improve first-action, full-trajectory, or state match; both had 29 unchanged and one worsened paired proxy outcomes, with slightly lower F1 and higher mean latency. Cap7 reached its turn limit on 10/30 samples.
- Kept production MAX_STEPS at 5. Existing code is already iterative; the measured bottleneck on this hard subset is wrong first-tool selection, not absent observation plumbing. No new prototype or public submission was created.

### Files
- `exp/lifemate-pipeline/scripts/run_eval.py`
- `exp/agent-architecture-audit/scripts/audit_agent_architecture.py`
- `exp/agent-architecture-audit/scripts/compare_agent_architecture_audit.py`
- `exp/agent-architecture-audit-v1-val309/output/`
- `exp/agent-architecture-audit-v1-iterative-cap6-val30/output/`
- `exp/agent-architecture-audit-v1-iterative-cap7-val30/output/`
- `exp/paired-analysis/agent-architecture-cap5-vs-cap7/output/`
- `PROGRESS.md`
- `INSIGHTS.md`

## 2026-10-31 — First-action oracle diagnostic

### Done
- Added diagnostic-only controls to `Agent.next_action` and `LifeMatePipeline.run`: constrain the first tool name while generating its arguments, inject the exact first gold call, or inject the first two gold calls. All injected calls are schema-validated and executed on the sample's ToolEnv; later observations feed normal Qwen turns.
- Added `exp/agent-architecture-audit/scripts/run_first_action_oracles.py` for the fixed 30 IDs, using the exact stored PhoASR public transcript and cap 5. Added `exp/agent-architecture-audit/scripts/analyze_oracle_continuation.py` to label the first post-oracle failure in Oracle B.
- Added `exp/agent-architecture-audit/scripts/validate_oracle_artifacts.py` to verify the 30 rows, transcript identity, injected-call execution, observations, and zero runtime failures.
- Completed Oracle A, B, and optional C across 30/30 samples each, with zero runtime failures. Oracle C was run because Oracle B improved tool F1 but still had 0/30 final-state matches.
- Saved per-sample trajectories, observations, states, summaries, continuation taxonomy, and paired outcomes under the requested experiment directories.

### Results
- Baseline: first full action 0/30; F1 0.1073; state 0/30; exact trajectory 0/30; proxy 0.0518.
- Oracle A: forced tool name 30/30; Qwen-generated first full call exact 3/30; F1 0.1304; state 0/30; proxy 0.0596. Paired: 5 improved, 19 unchanged, 6 worsened.
- Oracle B: injected first call exact 30/30; F1 0.3431; state 0/30; proxy 0.1719. Paired: 27 improved, 0 unchanged, 3 worsened. Continuation failures: wrong second tool 24; observation ignored 0; wrong later argument 2; premature stop 1; extra call 0; turn cap 0; other 3.
- Oracle C: injected first two calls exact 30/30; F1 0.5191; state 1/30; proxy 0.2783. Paired: 29 improved, 0 unchanged, 1 worsened. Exact full trajectories remained 0/30.
- Conclusion: both first-call argument grounding and subsequent next-tool policy matter. The large Oracle A→B gain indicates first-call arguments are difficult; Oracle B continuation is usually wrong at tool 2, and Oracle C still rarely reaches the final state. Prioritize SFT on next full action with observation-conditioned iterative trajectories, not only tool-name routing. No fine-tuning or submission was done.

### Files
- `src/lifemate/agent.py`
- `src/lifemate/pipeline.py`
- `exp/agent-architecture-audit/scripts/run_first_action_oracles.py`
- `exp/agent-architecture-audit/scripts/analyze_oracle_continuation.py`
- `exp/oracle-first-tool-name-30/output/`
- `exp/oracle-first-full-call-30/output/`
- `exp/oracle-first-two-calls-30/output/`
- `exp/paired-analysis/oracle-first-action/output/`
- `INSIGHTS.md`

## 2026-10-31 — Qwen next-action trajectory dataset

### Done
- Added `exp/qwen-next-action-v1/scripts/build_qwen_next_action_dataset.py` to construct one ToolEnv per fixed train trajectory, replay each gold call, and serialize each next full action plus a terminal STOP with real preceding observations.
- Kept the fixed 309 validation IDs out of the dataset. Prompts use the local PhoASR public model on train audio and history audio, and mirror the production `Agent.next_action` system prompt, compact tool schemas, context/history payload, and observation structure.
- Replayed all 1,236 train trajectories successfully. Generated 4,043 examples, average 2.271 tool actions (maximum six), 1,236 STOP targets, 170 `sys.ask_missing_info` targets, zero replay failures, and zero duplicate timesteps.
- Added reproducibility hashes to `exp/qwen-next-action-v1/output/training-data/manifest.json`; validated all 4,043 targets through the production action parser and checked train/validation membership, tool names, and argument keys.
- Added a QLoRA training runner and private Kaggle GPU notebook configured for a 30-step smoke run before the full run.

### Current status
- The first upload attempt was blocked by automatic approval review because the dataset contains derived PhoASR transcripts and ToolEnv observations. After the user explicitly confirmed, uploaded the private Kaggle dataset `daoviet/lifemate-qwen-next-action-v1` (v1), then published v2 with the reproducibility manifest. Verified the private dataset file list.
- Kaggle kernel v1 reached ERROR because the notebook hard-coded an input path absent from the Kaggle mount. Dynamic input discovery fixed the path for v2.
- Kaggle kernel v2 loaded the dataset/model but ran out of memory on a 6,144-token sequence: the T4 had 551 MiB free and SDPA needed another 864 MiB. Reduced the configured context to 4,096, enabled PyTorch expandable allocator segments, and aligned the local inference context setting with training.
- Published dataset v3 and submitted the T4 smoke kernel. The run completed 30 optimizer steps (loss 2.693 to 0.224; peak VRAM 13.38 GiB), then ended ERROR in the post-training probe because the probe used an unrelated greeting prompt and required action JSON. Kaggle exposed no output artifacts for the failed run.
- Added `exp/qwen-next-action-v1/scripts/monitor_kaggle_kernel.py` to poll status/logs every five minutes by default and print the terminal error trace. The watcher stopped after the kernel reached ERROR.
- Fixed the smoke probe to use a real SFT prompt, validate the generated tool/argument shape, and use gradient accumulation 4 during smoke only. Regenerated the private notebook and staged dataset runner/manifest locally; syntax and staged hashes pass. No upload or new model run was started, per the pause instruction.

### Files
- `exp/qwen-next-action-v1/scripts/build_qwen_next_action_dataset.py`
- `exp/qwen-next-action-v1/scripts/finalize_qwen_next_action_manifest.py`
- `exp/qwen-next-action-v1/scripts/train_qwen_next_action.py`
- `exp/qwen-next-action-v1/scripts/prepare_qwen_kaggle_kernel.py`
- `exp/qwen-next-action-v1/scripts/analyze_qwen_next_action.py`
- `src/lifemate/agent.py`
- `exp/qwen-next-action-v1/output/training-data/`

## PhoASR N-best hard-30 diagnostic (2026-11-01)

- Added `exp/phoasr-nbest-v1/scripts/build_phoasr_nbest_cache.py`; generated five candidates/sample for the existing fixed hard-30 IDs: production greedy, beam 3, beam 5, and temperature sampling at 0.2/0.35. Cache contains original text, decode config, rank, null confidence fields, audio ref/hash, gold text for diagnostics, manifest and generation config.
- Generation ran on local RTX 4050 in 423.7s. All 30 IDs produced five candidates; no empty transcripts or runtime failures; duplicate transcript rate 56.7%. Production greedy matched the canonical PhoASR cache for all samples. Initial attempt to compute transition log probabilities triggered a CUDA assertion; removed the unsupported confidence calculation and left scores null.
- Added `exp/phoasr-nbest-v1/scripts/run_nbest_downstream_eval.py` for cached current, WER-oracle and consensus-medoid transcript modes. Completed WER-oracle and automatic consensus runs on all 30 samples, each using one existing ToolEnv per sample, fixed Qwen/Ollama configuration, production prompt/schema/parser, and five-turn cap. Both had zero runtime failures. Full downstream-oracle scoring was not run because testing each alternative would require multiple ToolEnv trajectories per sample, conflicting with the competition's exactly-one-ToolEnv rule.
- ASR WER/CER current: 0.3048/0.1778; N-best oracle: 0.2890/0.1652 (minimum per-sample WER and CER separately). WER oracle improved 12, tied 18, worsened zero. WER-oracle downstream F1/proxy rose 0.1073→0.1441 / 0.0518→0.0672, but state match and first full-call accuracy remained 0/30. Consensus medoid produced WER 0.2986 and proxy 0.0518, no meaningful downstream gain.
- Artifacts: `exp/phoasr-nbest-v1/output/` and `exp/phoasr-nbest-v1/output/hard30/summary.md`. No production behavior or submission artifacts changed.
- Paired WER-oracle downstream outcomes: 6 improved, 5 worsened, 19 tied. Semantic differences include recovering “khóa lại” on a garage-door request; names, payment intent, and conditional/negation phrases remain fragile.

## PhoASR N-best expansion — 100 additional fixed validation IDs (2026-11-01)

- Selected the first 100 IDs from the existing fixed validation order after excluding the prior hard-30. This was deterministic and independent of ASR/downstream results. Generated 500 alternatives with the same five decode configs; no empty text or generation failures; 61% within-sample duplicate candidate rate; 1,008s on RTX 4050.
- Production vs candidate oracle WER/CER: 0.3638/0.2492 vs 0.3364/0.2279. WER oracle improved 29 samples, tied 71, worsened zero.
- Replayed 100 WER-oracle and 100 automatic consensus trajectories with the existing Qwen/Ollama settings, prompt, schemas, parser, ToolEnv and five-turn cap; zero runtime failures. Baseline 1-best metrics were paired from the existing PhoASR val309 run; all 100 stored transcripts exactly match the generated production candidates.
- Current 1-best: F1 0.1657, state 0.31, proxy 0.2464. WER oracle: 0.1671/0.32/0.2567. Consensus: 0.1605/0.32/0.2542. Paired proxy outcomes vs baseline were WER oracle 2 up/2 down/96 tied, consensus 1 up/2 down/97 tied.
- Full result: `exp/phoasr-nbest-v1/output/val100/summary.md`. Downstream oracle over every alternative remains unrun under the exactly-one-ToolEnv-per-sample constraint. No submission or production behavior changed.
- Recomputed with the BTC screenshot formula and gold hashes replayed through ToolEnv: score estimate 6.1071 current, 7.2738 WER oracle, 7.2738 consensus (0–100 scale). Only 10–11 samples receive nonzero scores because of the hash and matched-call gates. Matching used exact name+arguments with multiplicity; per-sample details are in `btc_formula_analysis.json`.

## 2026-11-02 - Qwen next-action LoRA validation on Kaggle

### Run
- Uploaded the adapter as private Kaggle dataset `daoviet/lifemate-qwen-next-action-lora-v1-adapter`; attached it and the user-provided private LifeMate dataset to kernel `daoviet/lifemate-qwen-next-action-lora-v1-val309-v3` (version 1).
- Kernel completed on Tesla T4. Ran PhoASR public + iterative ToolEnv on fixed train-only val309, five-turn cap, 4,096-token context. All 309 samples completed with zero harness runtime failures. Elapsed kernel time was 16,158.9 seconds; mean latency was 52.25 seconds/sample and 220 samples exceeded 30 seconds on T4.
- Replayed each gold trajectory through ToolEnv to compute `hash_gold`; all 309 replays matched labeled final states, with zero replay failures. Matching uses exact tool name and arguments with duplicate-call multiplicity. No `*_environment/` files were read directly.

### BTC formula results
- Qwen LoRA: **2.3193 / 100**; 65/309 state-hash matches; 25 exact matched calls, 84 submitted calls, 734 gold calls. Aggregate exact-call precision 0.2976, recall 0.0341, F1 diagnostic 0.0611. Local proxy mean 0.1635 is a separate diagnostic.
- PhoASR baseline on the same val309 cohort: **4.9145 / 100**; 72/309 hash matches; 100 exact matched calls, 692 submitted calls, 734 gold calls. Aggregate exact-call precision 0.1445, recall 0.1362, F1 diagnostic 0.1403; proxy mean 0.2286.
- Adapter scored 2.5952 points (52.8%) below baseline; no public submission was made. Whole-list exact-match rate rose (25/309 vs 20/309), but 19 adapter exact matches were empty gold trajectories (baseline: 12); BTC score is zero when no calls match. Non-empty exact matches were 6 vs 8, and lower call recall outweighed higher precision.
- T4 timing is a risk signal only; competition runtime is evaluated on A30. Remeasure timing on target hardware before making a runtime claim.

### Artifacts
- `exp/btc-scoring/scripts/score_btc.py`
- `exp/qwen-next-action-lora-v1-kaggle-val309/output/official_summary.json`
- `exp/qwen-next-action-lora-v1-kaggle-val309/output/official_scores.jsonl`
- `exp/qwen-next-action-lora-v1-kaggle-val309/output/validation_report.md`
- `exp/paired-analysis/historical-reports/output/qwen-next-action-lora-v1-kaggle-val309-vs-phoasr-baseline-official.json`
- `exp/qwen-next-action-lora-v1-kaggle-val309/output/kaggle-output-targeted/`
- `PROGRESS.md`
- `WORKLOG.md`


## 2026-11-02 — PhoASR N-best consensus public submission

- Generated and validated all 634 public-test rows using the PhoASR five-candidate consensus-medoid transcript and Qwen3-4B-Instruct-2507 via Ollama (context 8192, five ToolEnv steps maximum).
- Public leaderboard result recorded from the user report: **9.49**. This is the leaderboard score, separate from local BTC-formula validation scores.
- Submission: outputs/submissions/phoasr-nbest-consensus-v1.jsonl; validator: 634 rows, 634 unique IDs, zero schema errors.

## 2026-11-02 — Session 4: prompt, language, model, and cap A/B

### Setup and execution
- Used only the fixed train-only val309 split, PhoASR public transcripts, and one ToolEnv per predicted sample. No public/private test tuning, submission, or Public LB run.
- Fixed a seeded 60-sample cohort stratified by gold action count; it includes the only six-action val sample and 16 missing-info samples. IDs and composition are in `exp/session4-prompt-ab/output/cohort_ids.json` and `cohort_manifest.json`.
- Ran four cap-5 branches with the observation-grounded-v2 prompt: Qwen3-4B Q4_K_M in Vietnamese/English and Qwen3.5-4B Q4_K_M in Vietnamese/English. Temperature 0, thinking off, JSON mode, and Ollama context 8192 were held fixed. All four completed 60/60 without runtime failures; all transcripts exactly matched the previous Qwen3 baseline on these IDs.
- Ollama 0.40.13. Qwen3.5 tag `qwen3.5:4b-q4_K_M` loaded successfully via Ollama at 4.21B parameters, Q4_K_M, 3.3 GB. Text-only JSON smoke passed in 13.42s; a real LifeMate sample (`train_1265`) completed. `ollama ps` showed full GPU placement at context 8192; VRAM snapshots were 3820/6141 MiB (Qwen3) and 3962/6141 MiB (Qwen3.5). The Qwen3.5 model card lists Apache-2.0; Ollama listed the tag as four weeks old on this run, suggesting it predates the 2026-10-26 cutoff (relative-age inference). References: https://huggingface.co/Qwen/Qwen3.5-4B and https://ollama.com/library/qwen3.5/tags.

### Official BTC results
- Scored by replaying each gold trajectory through ToolEnv; exact tool name and arguments are matched with duplicate-call multiplicity. The official per-sample hash gate and zero-score rule for no matched calls were applied. All gold replays matched labeled final states with zero replay failures.
- On the fixed 60-ID cohort, previous Qwen3 predictions scored 7.4802/100. Prompt v2 scored Qwen3-Vietnamese 2.7778, Qwen3-English 4.0000, Qwen3.5-Vietnamese 1.3889, and Qwen3.5-English 3.1667. The new prompt/model branches did not beat the old prediction reference. English was the strongest pilot language for both models; this is screening evidence only.
- Cap comparison on the same 60 IDs, Qwen3-English: cap 5 scored 4.0000; cap 6 scored 4.8333 (+0.8333). Cap 6 improved one sample and tied 59, while 23 samples emitted more than five calls; the one gold trajectory requiring more than five actions was present. This is a small-cohort signal with substantial extra-call risk.
- Full fixed val309 confirmation of Qwen3-English: cap 5 scored **3.0868/100** and cap 6 **3.1702/100**, both below the prior PhoASR/Qwen3 baseline **4.9145/100**. Cap 6 gained only 0.0835 points over cap 5: 1 sample improved, 6 worsened, 302 tied; hash matches stayed 64/309. Cap 6 submitted 87 more calls (818 vs 736 unmatched calls), and 87 samples emitted more than five calls although only one gold sample requires over five actions. Mean latency was 5.22s vs 4.97s; neither had latency above 30s among completed samples.
- Both full runs had two model-request failures on `train_1431` and `train_1725`: prompts required 8,537 and 8,265 tokens against the fixed 8,192 context. They remain failed samples in the official score. Context was not raised. Local proxy and aggregate call-F1 are retained as diagnostics and were not used as official scores.

### Decision and artifacts
- Do not change the production default. Qwen3.5 is NO-GO for the next confirmation based on its weaker pilot BTC results. Qwen3-English with cap 6 is the highest full-val score among tested branches, but its 0.0835-point gain over cap 5 comes with marked over-calling and does not beat baseline; retain it only as a candidate for a focused follow-up, with cap 5 as the conservative operating choice.
- Per-run predictions, summaries, configs, and logs: `exp/session4-prompt-ab/output/`. Paired BTC reports: `exp/paired-analysis/session4-prompt-ab/output/`. Runtime/model record: `runtime-smoke.json`. No production code or submission was changed.

## 2026-11-02 — Full val309 Qwen3 vs Qwen3.5 comparison

- Re-ran both models sequentially over the same 309 fixed train-only validation IDs, with PhoASR public, observation-grounded-v2 English prompt, five-step cap, context 8192, temperature 0, thinking off, JSON mode, and one ToolEnv per sample. No public/private test or submission was used.
- Official BTC scoring used exact tool name and arguments, duplicate multiplicity, and gold-state hash replay through ToolEnv. Qwen3 scored **3.0868/100**; Qwen3.5 (`qwen3.5:4b-q4_K_M`) scored **1.3916/100**. Qwen3 was higher on 17 samples, Qwen3.5 on 7, and 285 tied; mean paired difference was +1.6952 BTC points for Qwen3.
- Qwen3.5 had more hash matches (74 vs 64) and no runtime failures, but only 38 exact matched calls vs Qwen3's 109; it submitted 82 calls vs 845. Its aggregate exact-call precision/recall were 0.463/0.052 vs Qwen3's 0.129/0.149. Hash match alone did not overcome the official matched-call/recall gates. Both scores remain below the previous Qwen3 baseline 4.9145/100.
- Mean latency was 2.45s for Qwen3.5 and 4.96s for Qwen3. Qwen3 had two context-overflow failures (`train_1431`, `train_1725`, 8,537/8,265 tokens against 8,192); Qwen3.5 completed all 309. The Qwen3 score exactly reproduced the prior cap-5 full-val run.
- Decision: Qwen3 remains the better of these two tested agents by official BTC score. Qwen3.5 is faster and has more hash matches but under-calls substantially; do not replace the model default. Artifacts: `exp/session4-qwen3-vs-qwen35-val309/output/` and `exp/paired-analysis/session4-qwen3-vs-qwen35-val309/output/paired-report.json`.

## 2026-11-02 — Local Qwen3/Qwen3.5 prompt-v2 submissions

- Generated both full `public_test` files sequentially with PhoASR public, observation-grounded-v2 English prompt, cap 5, Q4_K_M models, temperature 0, thinking off, JSON mode, and Ollama context 8192.
- Official order/schema validation passed for both: 634/634 rows, 634 unique IDs, zero schema errors. These are local submission candidates only; neither was sent to the Public LB.
- Files: `outputs/submissions/qwen3-4b-promptv2-en-phoasr-v1.jsonl` and `outputs/submissions/qwen35-4b-promptv2-en-phoasr-v1.jsonl`; configs, validation summaries, and logs are under the matching `exp/<experiment>/output/submission-artifacts/` folders. Mean generation latency was 4.85s/sample for Qwen3 and 2.40s/sample for Qwen3.5.

## 2026-11-03 — Qwen3.5 thinking/tool-protocol evaluation (paused at user request)

- Added opt-in Ollama controls for `think=true/false` and native function calling. Native mode converts sample tool schemas, reads `message.tool_calls`, executes at most the first call, logs over-calls/schema extensions, and returns observations with the matching `role=tool`. Native history is truncated to the executed call so every call has one tool result. Production defaults remain unchanged.
- Ran the 12-ID stratified smoke across A/B/C/D. No runtime/context exceptions. Thinking smoke was slow; D smoke recorded 2 extra native calls in a response. Full D was not started. Smoke summaries and IDs are stored under `exp/qwen35-thinking-tool-protocol-val309/output/`.
- Full A completed 309/309 using `qwen3.5:4b-q4_K_M` Q4_K_M, Ollama 0.40.13, PhoASR public (including audio in history), prompt-v2 English, temperature 0, context 8192, `num_predict=512`, cap 5, one prediction ToolEnv/sample. Official BTC: **1.3916/100**; the 30-second runtime gate did not change the formula score. Gold replay matched all 309 labels; no replay/schema/runtime failures. Mean/p95/max latency 2.59/4.86/8.22s.
- Full B completed the same 309 IDs with native tools and role=`tool` observations. Official BTC after the runtime gate: **6.3484/100**; BTC hash/call formula before the gate: **6.4779/100**. Gold replay matched all 309 labels; 2 samples exceeded 30s, 12 extra same-response calls were not executed, 4 schema repair turns (2 failed), zero ToolEnv schema rejections and zero context/runtime exceptions. Mean/p95/max latency 9.66/19.75/44.41s. Paired bootstrap remains pending.
- Full C (thinking enabled, JSON protocol) was stopped after 102/309 samples at the user's request. Partial mean/p95/max latency was 31.45/56.61/87.20s; 43 samples exceeded 30s. No committed sample had a context/runtime exception. **No official val309 C score was calculated.** The next fixed ID is `train_0157`; use `--resume` to continue the committed prefix if desired.
- D full was not run. Therefore no paired thinking effect under native calling and no factorial interaction is reported. No production change, submission, public/private test run, or Public LB request occurred.
- Artifacts: `exp/qwen35-thinking-tool-protocol-val309/output/`; partial C summary: `C-think-json/val309/partial_summary.json`; full BTC scores are adjacent to A and B predictions. The paired-analysis folder is reserved; do not interpret smoke/partial results as full val309.


## 2026-11-03 — Qwen3.5 thinking/tool-protocol evaluation completed

- Resumed C from the original 102 valid samples after a sandbox-blocked Ollama connection attempt produced 207 connection-error rows; those invalid rows were removed and the original 102 rows preserved. Completed C 309/309, then D 309/309. All gold trajectories replayed successfully through ToolEnv and matched all 309 labels; no prediction runtime exceptions or ToolEnv schema rejections.
- BTC scoring (exact tool name+arguments, duplicates by multiplicity, hash gate; score in parentheses is before the per-sample 30-second local runtime mask): A **1.3916 (1.3916)**; B **6.3484 (6.4779)**; C **5.6095 (9.2079)**; D **7.2168 (7.4326)**. Local RTX 4050 over-30s samples: A 0, B 2, C 140, D 12. A30 was not measured, so pre-mask scores are only diagnostics.
- Paired bootstrap, 10,000 resamples, fixed seed 20261102: B−A +4.9569 points (95% CI 2.8155, 7.3950); C−A +4.2179 (2.0065, 6.6775); D−C +1.6073 (−0.8093, 4.0887); D−B +0.8684 (−1.5264, 3.3496); interaction (D−C)−(B−A) −3.3495 (−6.5210, −0.2697).
- C mean/p95/max runtime 31.87/63.11/87.20s, 140 above 30s, 171 failed parse/schema repairs. D 15.14/28.11/52.66s, 12 above 30s, one failed parse/schema repair, 22 native extra calls not executed. No production default or submission changed.
- Full details: `exp/qwen35-thinking-tool-protocol-val309/output/`; paired report: `exp/paired-analysis/qwen35-thinking-tool-protocol-val309/output/paired-bootstrap-report.json`.

## 2026-11-04 — Track C Ollama defaults and PhoASR/QUAN fine-tune pilot

- Kept the process-wide backend default as Transformers. The Ollama route now defaults to `qwen3.5:4b-q4_K_M`, thinking on, JSON protocol, native tools off, context 8192 and 512 generated tokens; environment variables still select another model, backend, thinking mode, protocol, context, or token limit. README documents the explicit Track C setup and English prompt.
- Track C smoke passed against installed Ollama tag `qwen3.5:4b-q4_K_M`: JSON output parsed, LifeMate `stop` action parsed, and sample `train_1144` ran through one ToolEnv with a schema-valid first call. Actual run config and output: `exp/qwen35-track-c-quan-vs-phoasr-val309/output/smoke/track-c-smoke.json`. Local latency was 30.40s.
- Verified both final ASR checkpoint audits: PhoASR `exp/phoasr-lifemate-domain-v1/output/model/final-model` and QUAN `exp/quan-transcribe-lifemate-domain-v1/output/model`; both trained on exactly 1,236 train IDs, overlap 0, and fixed-final checkpoint selection. QUAN used the existing KenLM beam32 decoder. Existing transcript caches covered all 309 validation IDs; experiment copies and SHA-256 manifests are saved under `pilot60/`.
- Pre-run estimate for two full Track C branches was 19,694 seconds (5.47h), above the 4h limit, so ran fixed stratified paired pilot60 only. Both branches ran sequentially, same IDs, one ToolEnv/sample, Qwen3.5 Track C configuration, and no test/public submission.
- Official formula score (exact tool name+arguments, duplicate multiplicity, gold replay/hash gate; **before runtime mask**) was PhoASR FT **10.000/100**, QUAN FT + KenLM beam32 **15.730/100**. Paired QUAN−PhoASR difference was **+5.730 points**, bootstrap 95% CI **[−1.444, +13.405]** over 10,000 resamples. Outcomes: 8 improved, 4 worsened, 48 tied. The CI includes zero, so this pilot is not sufficient evidence to switch the transcript default.
- Corpus WER/CER: PhoASR **0.2597/0.1945**, QUAN **0.1786/0.1522**. Exact matched calls were 26 vs 34; aggregate exact-call precision/recall 0.321/0.241 vs 0.378/0.315; hash-match rates 0.500 vs 0.533. Both had zero runtime exceptions and ToolEnv schema rejections; parse/schema repair failures were 33 vs 32. Mean latency 32.34 vs 32.35s, p95 59.15 vs 55.20s, max 80.50 vs 71.47s; samples over 30s were 33 vs 28.
- Conclusion: QUAN improved WER/CER and the pilot BTC point estimate, but the paired uncertainty includes zero and only 8/60 samples improved. Do not change ASR default from this pilot. Full val309 and Public LB were not run. Detailed per-sample transcripts, calls, hashes, themes, latency and scores: `exp/qwen35-track-c-quan-vs-phoasr-val309/output/pilot60/paired-analysis.json`.
- Examples: `train_1144` (currency/amount wording) moved to the two gold `exchange_rate.convert` calls under QUAN, while PhoASR prompted five `exchange_rate.get_rate` calls; `train_0241` recovered the gold weather → calendar list → cancel sequence where PhoASR created an event. Conversely `train_0013` lost BTC under QUAN after extra contacts/SMS calls beyond the gold weather call. These illustrate argument/tool/over-call changes, not a causal proof from transcript alone.

## 2026-11-05 — QUAN FT + KenLM beam32 Qwen3.5 Track C submission

- Generated the full local `public_test` submission with the fixed QUAN LifeMate fine-tune checkpoint and KenLM CTC beam search (width 32, α=0.3, β=1.0, 4-gram), then Qwen3.5 4B Q4_K_M on Ollama using observation-grounded-v2 English, thinking enabled, JSON protocol, temperature 0, context 8192, `num_predict=512`, and a five-action cap. Exactly one ToolEnv session was used per sample.
- QUAN transcripts were cached for all 634 public test samples; transcript SHA-256 `882d85b20c7f2c5b34921f61a6a1012055f4efb78b6ff566f64568e2600f7c2f`. Fixed ASR checkpoint weights SHA-256 `e31be8516152587134fb62e0d1690954221a0f02c53b6549cf79db58c8eb18ae`; no train/validation ID overlap was recorded in the transcript manifest.
- Submission validation passed: 634 manifest rows, 634 submission rows, 634 unique IDs, zero schema errors. Runtime failures: 0. Inference took 21,363.1 seconds (about 5.93 hours).
- Local candidate at generation time; later submitted to the Public LB and scored 26.5. Submission: `outputs/submissions/qwen35-track-c-quan-ft-beam32-public-test.jsonl`; predictions, manifests, and transcript cache are under `exp/qwen35-track-c-quan-vs-phoasr-val309/output/submission-artifacts/`.

## 2026-11-05 — Experiment directory restructuring

- Moved generated evaluation, benchmark, training, checkpoint, and N-best artifacts from `outputs/` into experiment-scoped directories under `exp/<experiment>/output/`. Cross-experiment reports are grouped under `exp/paired-analysis/<comparison>/output/`.
- Moved all 49 Python scripts from the root `scripts/` directory into experiment-local `scripts/` folders. Added per-folder bootstrap helpers so scripts can still find the project source tree, and rebased script defaults that targeted the old evaluation/model/training paths.
- Kept only submission-format JSONL files in `outputs/submissions/`; moved manifests, prediction/transcript sidecars, and run logs into matching experiment `output/submission-artifacts/` directories. Updated the QUAN FT + KenLM beam32 manifest to record the confirmed 26.5 Public LB score.
- Added a README inventory for each experiment, updated `AGENTS.md`, `README.md`, `TASK.md`, and `.gitignore` to describe and enforce the new layout. Shared pretrained model weights remain in `models/`; reusable transcript cache remains in `cache/`.
