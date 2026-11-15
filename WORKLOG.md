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

## 2026-11-05 — Full Track C configuration error analysis on val309

- Used the submission experiment at `exp/qwen35-track-c-quan-vs-phoasr-val309/`, its manifest and runner settings, and the user-confirmed English prompt in `src/lifemate/agent.py`. Runtime fix: launched native Windows `ollama.exe serve` outside `.venv`; verified Ollama 0.40.14 and installed `qwen3.5:4b-q4_K_M`. Added a reusable native launcher under the analysis experiment.
- Ran fixed train-only val309 sequentially, one prediction ToolEnv/sample. The run was paused by the user after 187/309 and resumed only after validating prediction/trace prefixes; completed all 309 without replaying prior IDs. No public/private test data, submission generation, or Public LB request was used.
- Replayed all 309 gold trajectories through ToolEnv; zero replay failures. BTC uses exact tool name+arguments, duplicate multiplicity and state-hash gate. Formula score before the separate 30-second runtime gate: **12.9851/100**; with the competition timeout zero rule applied: **7.3571/100**. `M/U/|G|=233/293/734`; 114 hash matches; aggregate exact-call P/R/F1 diagnostic 0.4430/0.3174/0.3698.
- Runtime on local RTX 4050: mean/p50/p95/max 35.35/33.23/64.65/104.85 seconds; 177 samples exceeded 30 seconds; no runtime exceptions or ToolEnv schema rejections. This is not an A30 benchmark. Paired reproducibility check against the existing same-config pilot60 found identical calls and state hashes on all 60 IDs; formula difference 0.000 (95% bootstrap CI [0,0]).
- Main observed layers: 147 sample-level token-cap truncations, 23 invalid action-protocol outputs, 65 argument-grounding cases, 16 missing-info policy cases, 14 other early/under-calls, 8 tool selection/order, 9 over-calls/duplicates, 27 with no observed action error. These primary categories sum to 309. JSON repair failed on 170 samples; gold asked for `sys.ask_missing_info` in 47 cases and the model predicted none. QUAN transcript WER/CER was 0.1631/0.1298; numeric token exactness 60.2%, negation polarity 96.4%, temporal-marker exactness 77.3%. ASR/action causality was not inferred from correlation.
- Artifacts and ranked follow-up proposals: `exp/current-config-error-analysis-val309/output/summary.md`. Per-sample table, traces, scorer details, taxonomy, grouped metrics, paired pilot report and reproducibility hashes are in the same `output/` directory. No product prompt/model/protocol was changed.

## 2026-11-05 — `num_predict` 512 vs 768 diagnostic pilot

- Locked the existing 60-ID fixed train-only pilot cohort before branch B. It is enriched for prior token-cap/JSON repair cases and includes multi-step, low-step, and near-context prompts; scores are diagnostic, not an unbiased val309 estimate. Ran 512 then 768 sequentially with one prediction ToolEnv per sample/branch, fixed transcripts, QUAN FT + KenLM beam32, PhoASR for history, Qwen3.5 4B Q4_K_M, English prompt, thinking on, JSON protocol, temperature 0, context8192, and cap5. Native Ollama 0.40.14 ran outside `.venv`.
- BTC formula (exact tool name+arguments, duplicate multiplicity, ToolEnv gold replay/hash gate, M>0; no local runtime mask): 512 **14.5820/100**, 768 **17.5159/100**, paired +**2.9339** points (95% bootstrap CI **[−0.7407, +7.2725]**); 4 improved, 2 worsened, 54 tied. `M/U/G=32/57/108` vs `35/66/108`; hash matches 28 vs 30. Exact-call micro F1 0.3249 vs 0.3349 is a separate diagnostic.
- Sample count hitting `num_predict` or requiring JSON repair fell 39→18 (53.8%). Failed-repair samples were 29 vs 14. The pilot screening gate passed for mechanism/point estimate, but uncertainty includes zero. Local RTX 4050 sample mean/p50/p95/max latency 29.65/25.04/54.83/77.94s vs 30.25/27.32/54.84/69.06s; 26 vs 28 samples exceeded 30s. These are not A30 measurements and did not mask the formula score.
- Investigation found the rerun 512 differed from the archived baseline on ordered calls 34/60, call multisets 33/60, and hashes 20/60 while current transcripts matched 60/60. Checked source hash before instrumentation, English prompt hash, Ollama/model fingerprint, and request settings; root cause remains unresolved. Do not treat the paired pilot as reproducibility-confirmed.
- No full-val 768 run: pre-run projection was 4.55 hours (>4h), and the 512 reproducibility gate failed. The observed 768 pilot latency projects to 2.60h post hoc; it does not override the pre-run rule. No production default, prompt, submission, or Public LB was changed.
- Artifacts: `exp/num-predict-512-vs-768-val/output/`; experiment scripts/notes are in the same directory. Per-request telemetry uses exact payload SHA-256 and records no raw chain-of-thought.

## 2026-11-05 — Sequential inference latency optimization, fixed pilot60

- Added opt-in Ollama telemetry for prompt/eval counts and durations, cached prompt count, load/total duration, request wall time, tokens/s, model digest, and effective batch metadata. Added an opt-in `LIFEMATE_OLLAMA_NUM_BATCH`; absent by default, so existing request payloads remain unchanged. Raw thinking is not stored. The native Windows Ollama server runs outside `.venv`; verified Ollama 0.40.14 and the locked Qwen3.5 model digest.
- Re-ran the 60-sample 768-token baseline using the current server default `num_batch=1024`, then performed a balanced 512/1024 block crossover on the same 60 IDs. Exactly one prediction ToolEnv/sample/branch; gold replay used separate ToolEnv instances. No full val309, test, submission, or fine-tune run.
- Resolved the earlier baseline reproducibility mismatch: historical `llama-server` used `-b 512 -ub 512`, current omitted-option default uses `-b 1024 -ub 1024`. Explicit512 exactly reproduced archived ordered calls/call multisets/state hashes on all 60 IDs; explicit1024 exactly matched current default on all 60. Current implicit1024 differed from archived output on 36/60 calls and 21/60 hashes. Identical first-request payload hashes were confirmed on all 60 IDs.
- Current-default baseline (effective1024): BTC formula 14.0635/100; mean/p50/p95/max 32.58/29.32/58.71/75.03s; 30 samples >30s; 187 requests. Explicit512: BTC 17.5159/100; 30/60 hashes; M/U/G=35/66/108; mean/p50/p95/max 30.64/27.42/54.66/78.85s; 29 samples >30s; 177 requests. Explicit1024: score and calls/hashes matched current default; mean/p50/p95 33.52/30.03/60.23s.
- 512 improves the median by 6.5%, p95 by 6.9%, and samples >30s by one versus current default, with zero errors or schema rejects. The paired latency and BTC bootstrap intervals include zero, so keep 512 as an experiment candidate; no production default changed. Scores are diagnostic on an enriched train-only pilot, not val309 estimates.
- `ollama ps` reported 100% GPU. Across 1,118 five-second samples, VRAM averaged 3,931 MiB and peaked at 3,980/6,141 MiB; utilization averaged 79.72%. Runner used `--flash-attn auto` and matching server logs report it enabled. No model reload occurred within a continuous cohort after warm-up. Therefore num_gpu, Flash Attention/KV q8_0, and keep_alive comparisons were skipped at their task gates.
- Full report, scripts, hashes, per-request telemetry, predictions/traces, BTC guardrails and paired deltas: `exp/inference-latency-optimization-val60/`.

## 2026-11-05 — Set explicit Ollama `num_batch` default

- Per user request, changed the Ollama request builder to send `options.num_batch=512` by default instead of relying on the server's implicit 1024 default. `LIFEMATE_OLLAMA_NUM_BATCH` remains an explicit override.
- This makes the observed/reproducible 512 setting part of the application request configuration and should stabilize outputs across Ollama server defaults. It applies only to the Ollama backend; the Transformers backend is unchanged.
- Updated the README, progress table, and insights. No inference run, test, full-val evaluation, submission generation, or A30 benchmark was performed as part of this configuration change.

## 2026-11-06 — Qwen3.5 thinking budget pilot60

- Ran the fixed enriched train-only pilot60 at `num_predict=512/768/1024/1536`, thinking on, JSON, `num_ctx=8192`, explicit request `num_batch=512`, same model digest/version, prompt, transcripts, and pipeline. Used balanced 15-ID rounds, one warm-up per session, one prediction ToolEnv/sample/cap, and native Python outside `.venv` to reach the native Ollama server.
- BTC formula scores before any local runtime mask were 16.857/14.460/15.905/14.516. Paired deltas vs 512 were −2.397 [−7.952,+1.778], −0.952 [−6.230,+4.048], and −2.341 [−6.667,+0.317] points. These are screening metrics only because exact control reproducibility failed.
- Repeated cap512 with identical first-request payload SHA-256 on 60/60 IDs; calls differed on 17/60 and state hashes on 11/60 versus the primary current run. The repeat matched the archived 512 artifact on all 60 calls/hashes, while the primary run differed on the same 17 calls and 11 hashes. Current source hash differs from the earlier pilot; the variance is not resolved. No quality cap recommendation is made.
- At cap1536, 4/172 requests (3/60 samples) reached `eval_count >= num_predict`; 1 additional length case was classified as a context-limited candidate. Paired 1536−1024 score delta was −1.389 points [−6.111,+2.778], with 2 improved/2 worsened/56 tied. Did not run conditional cap2048.
- Recovered an initial runner prefix-flush duplication: all repeated sample/request rows were exactly identical, raw files were retained, canonical files were deduplicated, and the runner writer was corrected. Four cap branches have 60 unique predictions each; the repeated 512 control is a separate 60-sample artifact.
- GPU sampler captured 1,102 five-second samples over partial run coverage: utilization mean/p95/max 76.1/87/100%; VRAM mean/max 3,877/3,877 MiB; temperature mean/max 77.5/87°C. RTX 4050 only, not an A30 benchmark. No test set, fine-tuning, submission, or Public LB was used.
- Full manifests, prediction/trace/request telemetry, ToolEnv scores, paired analysis, recovery audit, GPU observations, and decision are in `exp/qwen35-thinking-budget-val60/`.

## 2026-11-07 — QUAN continuation and CoT screen

- Continued the existing QUAN LifeMate checkpoint on the fixed 1,236 train IDs for two epochs with reader LR `5e-6` and top-two encoder LR `2e-7`. Decoded the same 309 held-out validation IDs after each epoch with the existing KenLM CTC beam32 configuration. Existing-checkpoint rerun matched archived text 309/309.
- Baseline WER/CER was `0.1630547 / 0.1298079`; epoch 1 was `0.1632021 / 0.1299904`; epoch 2 was `0.1633496 / 0.1305836`. Both metrics worsened consecutively, so stopped before epoch 3 and retained the existing checkpoint. Paired bootstrap intervals crossed zero at both epochs; no candidate was selected.
- Re-ran 768-token downstream branches A/B as same-checkpoint controls on the exact pre-existing enriched cohort60, in balanced 15-ID blocks. Ollama 0.40.15 and Qwen3.5 4B Q4_K_M digest were locked; thinking on, JSON, temperature 0, context8192, explicit batch512, five action turns. Used system Python outside `.venv` for Ollama. Each sample had one prediction ToolEnv; gold trajectories were replayed through the official ToolEnv scorer.
- BTC formula before any local runtime mask: A `14.7937/100`, B `15.7381/100`; paired B−A `+0.944` points (95% CI `[−1.667,+4.222]`), with 2 improved, 1 worsened, 57 tied. Hash matches were 31/60 and 35/60; M/U/G were 33/72/108 and 33/66/108. Gold replay failures and runtime errors were zero.
- The 60/60 first-request payload hashes matched between A and B, but 26/60 samples differed in submitted tool calls or state hash. Treat the repeat as evidence of material run-to-run output variance; small single-run deltas are inconclusive. Local 30-second-gated scores were 7.50 and 8.17, separately labeled diagnostics; 30 and 28 samples exceeded 30 seconds on the RTX 4050.
- Ran prompt-only CoT on the same cohort after locking the existing checkpoint. Added only the prescribed ordered checklist before output-format instructions; raw thinking was not recorded. CoT scored `15.1825/100` unmasked vs A `14.7937`, paired delta `+0.389` (95% CI `[−7.444,+8.286]`), with 7 improved, 6 worsened, 47 tied. Hash matches were 33 vs 31; M/U/G were 35/69/108 vs 33/72/108; gold replay/runtime failures were zero.
- CoT local 30-second-gated diagnostic was `3.6111/100` vs A `7.5000` (paired delta `−3.889`, 95% CI `[−11.111,+2.778]`). CoT had 36 samples over 30 seconds vs 30 for A, cap-hit requests 23 vs 19, and repair requests/failures 35/18 vs 29/14. Prompt tokens rose 6.5%, generated tokens including thinking rose 8.0%. The unmasked point gain is uncertain and comes with worse local runtime/truncation diagnostics; do not promote CoT.
- Cohort60 is enriched and screening-only; no full-val BTC or A30 runtime claim follows. No public/private test, submission, or Public LB was used; no `*_environment/` file was read. Full summary: `exp/quan-continuation-cot-v1/output/summary.md`.


## 2026-11-07 — Qwen3.5 SFT benchmark cohort60

- Read KAGGLE_API_TOKEN from the project .env without printing or persisting it. Downloaded and hashed the exact Kaggle Qwen3.5-4B version1 files; merged the epoch2 adapter into that base in BF16, converted base and SFT with the pinned llama.cpp toolchain, quantized both to Q4_K_M, and registered separate Ollama tags. The training report lacks a base-weight input hash, so local provenance cannot prove which exact bytes the remote training job consumed; the user attested the Kaggle model identity.
- Passed the real ToolEnv smoke gate for both tags. Ran the locked enriched held-out cohort60 in balanced 15-ID blocks: A base/no-CoT 14.97 BTC, B SFT/no-CoT 22.84, C SFT+CoT 26.00. Gold scores use exact tool name+arguments with duplicate multiplicity and the official ToolEnv hash gate; no local runtime mask. Gold replay had 0 failures.
- Paired 20,000-bootstrap results: B−A +7.87 [−1.40,+17.37], C−B +3.16 [−4.03,+10.74], C−A +11.03 [+3.22,+19.69]. The first two intervals include zero; the positive combined change cannot isolate SFT and CoT effects. A-repeat matched first-request payload hashes, calls, state hashes and final states on all 20/20 anchors; repeat BTC delta was exactly zero.
- Local RTX 4050 diagnostics: mean latency A/B/C 33.53/18.45/17.90s; >30s samples 31/8/10; no runtime/parser errors or schema rejections. These are not A30 timing results. Kept this as screening only; no full-val309, submission, public/private test or Public LB.
- An initial runner mapping error occurred only after A/B/C completed; the recovery retained those artifacts and ran only A-repeat. No raw CoT or secret was saved. Details: exp/qwen35-sft-benchmark-v1/output/summary.md and output/inference/analysis.json.
## 2026-11-07 — Qwen3.5 SFT + CoT branch C public-test candidate

- Generated all 634 `public_test` rows using the locked branch C setup: Qwen3.5 SFT epoch 2 + CoT, QUAN LifeMate fine-tuned ASR with KenLM beam32 cached transcripts, PhoASR history ASR, thinking enabled, JSON actions, `num_predict=768`, `num_ctx=8192`, `num_batch=512`, temperature 0, five action turns. Used system Python outside `.venv`; local Ollama was reachable there.
- The official submission validator passed: 634 rows, 634 unique IDs in manifest order, zero schema/order errors. Submission SHA-256 `37237f3651686e040dea4ad7334df763ef36d7069e661f94b8c2304f613f9e90`; matching predictions sidecar SHA-256 `51db84d76b5a1e3a615afda068b11f69a32131fd365f7f48befee080b17ef004`. Exactly one official ToolEnv was created per sample; the sidecar preserves its executed calls and state hash.
- One Ollama request failed with CUDA out of memory on `public_0527`; the runtime error is recorded in the prediction sidecar and run summary. The schema-constrained submission row contains only the ToolEnv calls and state hash. The candidate has 1 runtime error. Local RTX 4050 latency was mean/p50/p95/max 22.70/19.62/46.74/86.73 seconds, with 158 samples over 30 seconds; these are local diagnostics, not A30 measurements or an A30 timeout determination.
- No raw chain-of-thought, labels, or `*_environment/` files were saved/read. The generation pipeline did not submit to Public LB. The user later reported a Public LB score of **39.16** for this candidate; Codex did not independently verify the score. Artifact and provenance details are in `exp/qwen35-sft-c-public-submission/README.md` and its `output/summary.json`.

## 2026-11-08 — Qwen3.5 SFT epoch-3 continuation + CoT screen

- Audited `exp/qwen35-sft-next-action-v1/results (1).zip`: it contains one continuation epoch from the epoch-2 adapter, with fresh optimizer/scheduler state. The continuation-source adapter SHA-256 matches epoch 2; the 3,873-row SFT data hash matches its manifest. Adapter SHA-256: `d6f410595341ed47970b2264a522dd8000508f4e95587c15e27b446da9f2fd54`; merged and quantized model registered as `lifemate-qwen35-kagglev1-sft-epoch3-q4km:latest`, Ollama digest `e54c164122873adbbeb4407566e24bb3e056f7e51915dba77d5d0204b89d82b0`.
- Reused epoch-2 branch C predictions, per-sample scores and requests; did not rerun C. Locked the same ordered enriched held-out cohort60, prompt/CoT hashes, QUAN FT transcripts, ASR history, Ollama 0.40.15 and settings. New C+CoT inference completed 60/60 in 1,161 seconds, exactly one prediction ToolEnv per sample, zero inference errors and zero gold replay errors.
- BTC formula before local runtime gate: epoch2 C **26.00**, epoch3+CoT **23.28**; paired delta **−2.72** (20,000-bootstrap 95% CI **[−7.72,+1.33]**), 4 improved / 5 worsened / 51 tied. Hash matches 35→32; M/U/G 47/70/108→51/64/108. The interval includes zero and the point estimate is lower; this screen does not show benefit from the extra epoch.
- Local 30-second-gated diagnostic: 23.67→23.28 (paired delta −0.39, 95% CI [−5.78,+4.72]); local >30s samples 10→9; mean latency 17.90→18.98s. Local RTX 4050 times do not establish A30 compliance. New branch micro exact-call F1 0.4574; final-state matches 46/60; local proxy 50/50 mean reported separately.
- Limits: fixed enriched cohort60, reused rather than repeated epoch2 control, SFT prompt language Vietnamese vs evaluated production English, and no exact base-weight hash in remote training report. No public/private test, submission or Public LB was used; no raw CoT or `*_environment/` file was read/saved. Full report and artifacts: `exp/qwen35-sft-epoch3-cot-val60/`.

## 2026-11-08 — Qwen3.5 SFT epoch-3 + CoT public-test candidate

- Reused the validated epoch-3 merged Q4_K_M Ollama model and the locked QUAN LifeMate FT + KenLM beam32 cached public-test transcripts. Kept the branch C English prompt/CoT checklist, PhoASR history, thinking+JSON, cap768, ctx8192, batch512 and seed17. Used system Python outside `.venv` for local Ollama.
- Generated 634/634 rows in manifest order, each with exactly one prediction ToolEnv and all executed calls preserved. The official submission validator accepted 634 rows/634 unique IDs with zero schema/order errors. There were zero inference runtime errors. No labels were used and the candidate was not submitted to Public LB.
- Submission SHA-256 `3b697aeb2ce966e5bd8e6e18503d6388e40fdcdfde49d83f728f791d8f581195`; predictions sidecar SHA-256 `d4a2ea830f6d4e1ed1be9c8ba109b7356a3b45b3d9556fde3d0de1e104d98549`. Manifest and detailed logs are in `exp/qwen35-sft-epoch3-cot-val60/output/submission/`.
- Local RTX 4050 inference took 15,279s; latency mean/p50/p95/max 24.09/20.42/53.90/72.13s, with 177 samples over 30s. These are local diagnostics, not A30 compliance results. 2,181 Ollama requests; 63 reached the 768-token cap. The candidate is ready as a local artifact only; no Public LB action was taken.
- 2026-11-09: Completed `exp/quan-transcript-normalization-val60` as a separate spelling-only fallback after the numeric parser could not be safely scoped. Audited all 309 QUAN transcripts; zero exact-token spelling changes and no out-of-scope edits. Native Ollama 0.40.15 ran 60 A / 60 B / 20 A-repeat with the locked SFT epoch2+CoT config; official ToolEnv gold replay and 20,000 paired bootstrap completed. B−A +2.78 BTC points (95% CI −0.28,+7.22), consistent with repeat-run variance. One telemetry merge exception was recovered from staging without repeating inference. No submission or Public LB. The original numeric-plus-spelling treatment remains incomplete.

## 2026-11-10 — QUAN number-normalized train-map benchmark on val60

- Ran 3 branches on the same fixed enriched held-out val60 cohort: raw QUAN epoch-06 transcript; same QUAN transcript with Vietnamese cardinal normalization before the train-only phrase map; and the BTC reference script as an oracle transcript. The map was built from 1,236 train IDs only; val labels were used only for evaluation.
- All branches used the same Qwen SFT epoch-2 + CoT configuration, PhoASR history, ToolEnv and action cap. Native system Python reached Ollama 0.40.15 outside .venv. Completed 60 samples/branch (180 total) in 3,208.9s, with zero inference errors and zero gold ToolEnv replay failures.
- Project WER A/B/C: 0.1757/0.2046/0.0000. Number-value WER: 0.1540/0.1784/0.0000. Number parser normalized 35 QUAN spans; the map applied 28 corrections across 22 tasks and exactly matched the aligned reference 0/28 times.
- Official BTC formula A/B/C: 23.29/19.83/32.33. B−A was −3.45 points (95% paired bootstrap CI [−10.00,+1.83], 3 improved / 3 worsened / 54 tied). The 60-ID cohort is enriched and the BTC interval includes zero; do not generalize it as a full-val estimate.
- Report: exp/quan-map-correction-val60-v1/output/benchmark/summary.md; full predictions, inputs, map applications, scores, traces and locks are in the matching output directory.


## 2026-11-10 — QUAN continuation epoch-02 downstream branch

- Added branch D on the fixed enriched held-out val60 cohort using the raw QUAN-transcribe two-epoch continuation checkpoint and the same Qwen SFT epoch-2 + CoT setup. System Python outside the project .venv reached the local Ollama model. All 60 predictions completed; inference runtime errors and official gold ToolEnv replay failures were both zero.
- Transcript WER was 0.1795 project-normalized / 0.1569 number-value-normalized. BTC formula score was 24.62/100; local 30s-gated diagnostic was 22.29.
- Added paired D-vs-A/B/C analysis and updated the benchmark summary, lock, README, progress and insights records. This remains a one-run enriched-cohort screen; no full-val309 or submission was run.


## 2026-11-10 — QUAN four-branch comparison on full val309

- Completed A/B/C/D downstream inference for all 309 validation IDs with the locked Qwen SFT epoch-2 + CoT model and shared ToolEnv configuration. A/B used epoch-06 QUAN transcripts, C used the original BTC script, and D used the raw epoch-02 QUAN continuation. The branches ran in rotating 20-ID blocks.
- Official BTC per-sample replay, full-split WER/map analysis, and six paired 20,000-bootstrap comparisons are saved under exp/quan-map-correction-val309-v1/output/benchmark/. Runtime counts: {'A': 1, 'B': 1, 'C': 1, 'D': 1}.
- The complete split was used for evaluation only; map rules remained frozen from train data. No public/private test or submission was used.

## 2026-11-13 — Qwen3.5 Pho synthetic targeted adapter Public LB result

- User reported Public LB score **51.19** for `phowhisper-large-v2-qwen35-val309-benchmark-winner-focused-public-test`; this user-provided score is recorded as ground truth per the user's later instruction and is superseded by the 55.92 ask-stop result. The generated submission contains 634 rows and its BTC schema validation passed with 634 unique IDs, zero schema errors, and zero runtime errors. Submission SHA-256: `91e6e3c239f0f20d757f6e4625694fbfaa240b2f3cc97c9db738528db11e36f6`.
- The same adapter scored **50.5423/100** under the official BTC formula on fixed val309; this local validation score is distinct from Public LB.
- Inference used fixed thinking-on (`thinking: true`) on every model request, JSON protocol, `num_predict=768`, temperature 0, and the focused post-lookup prompt. It did not use adaptive thinking or turn thinking off based on elapsed time. Lock and manifest: `exp/qwen35-sft-pho-synthetic-v1/output/submission-adapter-benchmark-winner/` and `outputs/submissions/`.

## 2026-11-13 — Qwen3.5 Pho synthetic adapter 3 epoch benchmark

- Đã merge adapter `adapter_final-3-epochs.zip` vào Qwen/Qwen3.5-4B, lượng tử hóa Q4_K_M và benchmark toàn bộ 309 mẫu PhoWhisper val309. Adapter archive SHA-256: `883e0759d552dacda9b4c52c4f93d1bf16a7bc6562be2768f33e06e9bc3e3ca`; Ollama digest: `1c347b01a6efe66c9571bf56dc1038ee7f032679d85e9db12d68d099b24617f0`.
- Dùng cùng prompt focused post-lookup, transcript PhoWhisper Large v2 đã cache, thinking bật cố định, JSON, temperature 0, `num_predict=768`, context8192, batch512, seed17 và tối đa 5 action turns. Scoring theo BTC ToolEnv; báo điểm sau runtime gate 30 giây và điểm trước gate riêng. Dùng lại predictions đã khóa cho baseline tốt nhất, checkpoint step120 và adapter 1 epoch; chỉ adapter 3 epoch được infer mới.
- Official BTC val309 sau gate: baseline `42.7067`, step120 `46.6729`, adapter 1 epoch `50.5423`, adapter 3 epoch `50.1293`. Adapter 3 epoch so với baseline +`7.4226` điểm [95% CI `+3.0690,+11.8231`]; so với adapter 1 epoch −`0.4130` [`−4.2690,+3.3449`], khoảng tin cậy gồm 0. Không có bằng chứng rằng tiếp tục đến epoch 3 cải thiện điểm tổng thể so với epoch 1.
- Chẩn đoán gold ask: adapter 3 epoch chính xác `30/47` so với `23/47` ở epoch 1. Có một lỗi context overflow trên `train_1725` (8.265 tokens so với context 8.192); lỗi tương tự đã có ở các nhánh đối chiếu. Độ trễ trung bình 18.50s, đo trên RTX 4050 nên không suy rộng thành thời gian A30.
- Không tạo submission. Báo cáo, paired scores và predictions: `exp/qwen35-sft-pho-synthetic-v1/output/adapter-final-3epoch-benchmark/val309/`.

## 2026-11-13 — Adaptive thinking trên hai adapter Pho synthetic

- Chạy adaptive inference đủ 120/120 mẫu trên mỗi adapter, dùng lại chính cohort stress 120 mẫu của thí nghiệm step120 (42 mẫu step120 >30s và 78 fast controls ghép theo turn/gold-ask). Predictions fixed-thinking của adapter 1/3 epoch được lấy từ benchmark val309 đã khóa, rồi chấm paired theo cùng ID.
- Policy: mỗi mẫu bắt đầu `think=true`; khi cộng dồn latency các request Ollama của mẫu vượt 20s, mọi request tiếp theo dùng `think=false`. Giữ focused post-lookup prompt, PhoWhisper Large v2 cached main/history, JSON, temperature 0, cap768, context8192, batch512, seed17, tối đa 5 turns.
- Adapter 1 epoch: score sau local 30s gate fixed/adaptive `46.0677/52.8772`, paired `+6.8095` điểm (95% bootstrap CI `[+2.8095,+11.3690]`), wins/ties/losses `10/109/1`; trước gate `56.7443/54.5439`; >30s `30/13`; mean latency `23.88/22.47s`.
- Adapter 3 epoch: score sau gate `41.5042/47.1239`, paired `+5.6197` (95% CI `[+2.1991,+9.6336]`), wins/ties/losses `9/111/0`; trước gate `57.3243/53.5882`; >30s `37/17`; mean latency `25.35/22.49s`.
- Không có inference runtime errors. Policy chuyển thinking-off ở 54/120 mẫu; tổng request off là 70/498 cho adapter 1 epoch và 71/502 cho adapter 3 epoch. Exact gold ask giữ ở 5/120 cho cả fixed và adaptive ở mỗi adapter.
- Kết quả là screening trên cohort cố ý chọn nhiều mẫu chậm, không thể xem như BTC full val309 hoặc public score. Điểm trước gate giảm ở cả hai adapter, còn điểm sau gate tăng do ít mẫu bị mask bởi ngưỡng runtime. Chưa tạo submission; nếu muốn dùng adaptive, chạy full val309 cho adapter được chọn.
- Báo cáo: `exp/qwen35-sft-pho-synthetic-v1/output/runtime-budget-adaptive-adapters/summary.md`; paired details, predictions, locks và checksums nằm cạnh đó. Script tái lập: `exp/qwen35-sft-pho-synthetic-v1/scripts/runtime_budget_adaptive_adapters.py`.

## 2026-11-14 — Qwen Pho synthetic adapter 4 epochs and public candidate

- Merged adapter_final-4-epochs.zip into the Qwen3.5-4B base, converted it to Q4_K_M GGUF, registered it in Ollama, and verified the registered GGUF digest. Adapter archive SHA-256: dc963ba280ab399419c593a479fc65fc3413d898d1881ff5beef0c0fc249ce5b; GGUF SHA-256: 796ad8bd495637ee3448bd5be4779774d0c89502b9ed16052367182d1feb3728; Ollama digest: 528597f6f48cae627a54057b424b48cc479330d5b0721d2a840578caf20f40bf.
- Completed all 309 fixed PhoWhisper val IDs with thinking fixed on, focused post-lookup prompt, JSON, temperature 0, cap768, context8192, batch512, seed17, and max five action turns. Final4 scored 60.1475 before the local 30s gate / 53.7904 after; final3 scored 56.8124 / 50.1293. Paired final4−final3: +3.3351 raw points [95% bootstrap CI +0.1577,+6.5020], 39 wins/248 ties/22 losses; after gate +3.6611 [+0.2368,+7.1146]. Exact gold asks 38/47 vs 30/47. All branches had the same 8,265-token overflow on train_1725 under the 8,192 context.
- Selected final4 by raw full-thinking val309 score. Generated the local 634-row public-test candidate through the PhoWhisper focused submission runner with Ollama serving the Q4 GGUF. Context was raised to the competition maximum 16,384 to accommodate the known overflow; thinking stayed fixed on and other settings stayed unchanged.
- BTC validator passed: 634 rows, 634 unique IDs, zero schema errors, zero runtime errors; one ToolEnv session per sample. Submission SHA-256: 4f0bc5953cd79c76b6444c297d845927eb8debdac8ea3fc6096456a006b6c2f5. Local Qwen+ToolEnv mean 20.90s, 117 samples >30s; estimated ASR+Qwen mean 26.50s, 216 >30s. At candidate generation, no Public LB submission had been made; the then-highest known score was 51.19 for the 1-epoch candidate. The user later reported Public LB **53.47** for this final4 full-loop candidate; it is recorded as ground truth per the user's later instruction and is superseded by the 55.92 ask-stop result.
- Updated INFERENCE_DECISIONS.md, PROGRESS.md, and exp/qwen35-sft-pho-synthetic-v1/README.md. Full report: exp/qwen35-sft-pho-synthetic-v1/output/adapter-final-4epoch-benchmark/val309/summary.md; submission lock/manifest: exp/qwen35-sft-pho-synthetic-v1/output/submission-final3-vs-final4/.

## 2026-11-14 — Ask-stop and adaptive thinking pilot60 on final4

- Ran a paired 2x2 policy screen on the exact existing enriched fixed pilot60 using the targeted synthetic 4-epoch Qwen adapter (lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest, digest 528597f6f48cae627a54057b424b48cc479330d5b0721d2a840578caf20f40bf). The four branches were fixed full, fixed + stop after successful sys.ask_missing_info, adaptive 20s + full loop, and adaptive 20s + ask-stop. Each branch completed 60/60 with zero inference runtime errors.
- Kept PhoWhisper Large v2 cached main/history, focused post-lookup prompt, JSON, temperature 0, num_predict=768, context8192, batch512, seed17, and max five action turns. Adaptive policy starts thinking on and switches it off after cumulative per-sample request latency exceeds 20s. Scoring used official BTC ToolEnv exact-call multiplicity and state-hash gate; 10,000 paired bootstrap draws; local 30s runtime gate and pre-gate scores reported separately.
- BTC after local gate: fixed full 47.28, ask-stop 63.33, adaptive full 51.17, combined 60.83. Before gate: 49.90, 63.33, 51.17, 60.83 respectively. Ask-stop vs fixed full was +16.05 points after gate (95% CI [+8.23,+24.80]) and +13.43 before gate. Adaptive alone was +3.89 after gate ([+0.56,+8.33]) but +1.27 before gate (CI [−2.14,+5.56], includes zero). Adding adaptive on top of ask-stop changed score −2.50 ([−7.50,+1.67], includes zero).
- The fresh fixed control reproduced archived final4 on 54/60 ordered call lists and 56/60 state hashes. Cohort is deliberately enriched, not an unbiased estimate of full val309 or Public LB; timings are local RTX 4050 diagnostics, not A30. Treat ask-stop as a candidate for later full-val confirmation; do not promote either policy from this screen alone. No inference defaults or submission were changed.
- Report, predictions, traces, request telemetry, and paired details: exp/qwen35-sft-pho-synthetic-v1/output/two-policy-val60/; rerun script: exp/qwen35-sft-pho-synthetic-v1/scripts/run_two_policy_val60.py.

## 2026-11-14 — Public candidate from the fixed ask-stop pilot branch

- Generated a separate 634-row public-test candidate using the highest-scoring branch from the enriched policy pilot60: final4 adapter, focused post-lookup prompt, thinking fixed on, stop the action loop after a successful sys.ask_missing_info call. The pilot branch scored 63.3333 BTC after and before the local 30-second gate on its selected 60-task cohort; this is not a Public LB estimate.
- Kept PhoWhisper Large v2 cached main/history, JSON, temperature 0, num_predict=768, num_ctx=16384, num_batch=512, seed17, max five action turns. The ask-stop behavior was applied in memory to the pipeline; src/lifemate/pipeline.py was not modified.
- Official public-test schema validator passed: 634 rows, 634 unique IDs, zero schema errors. The one-ToolEnv-per-sample audit passed for all 634 tasks. Inference recorded one runtime error on public_0418: local Ollama returned HTTP 500 with CUDA out of memory; the other 633 completed without runtime errors.
- Local Qwen+ToolEnv mean latency was 18.5926s, with 86/634 over 30s. Estimated ASR+Qwen mean was 24.1975s, with 172/634 over 30s. These are local diagnostics, not NVIDIA A30 timings. Candidate SHA-256: c65cbf01ff162d3cbcec76769cc94e18e295d123ef703cf93d8dfea206bf220e. Public LB was not submitted.
- Candidate and prediction sidecar: outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-ask-stop-pilot60-public-test.jsonl and outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-ask-stop-pilot60-public-test_predictions.jsonl. Manifest and full request traces: exp/qwen35-sft-pho-synthetic-v1/output/submission-final4-ask-stop-pilot60/. Runner: exp/qwen35-sft-pho-synthetic-v1/scripts/make_submission_final4_ask_stop.py.


## 2026-11-14 — User-reported Public LB ground truth for ask-stop candidate

- The user reported a Public LB score of **55.92** for the fixed_ask_stop final4 candidate and explicitly instructed that reported scores be treated as ground truth. Recorded 55.92 as the authoritative score, superseding the previous 53.47 final4 full-loop score and the older 51.19 1-epoch score.
- Updated the candidate winner selection, experiment summary, outputs/submissions manifest, PROGRESS.md, and INFERENCE_DECISIONS.md. Candidate SHA-256 remains c65cbf01ff162d3cbcec76769cc94e18e295d123ef703cf93d8dfea206bf220e. The local CUDA OOM on public_0418 remains documented separately from the reported Public LB score.
## 2026-11-14 — Three prompt arms screened after ask-stop

- Ran three prompt-only arms on the same enriched fixed pilot60 with the targeted final4 adapter: ask precision, completion audit, and argument grounding. Fixed thinking, focused prompt, PhoWhisper cached transcripts, decoding settings, five-turn limit, and the successful sys.ask_missing_info stop guard were held constant. Each arm completed 60/60; 180 predictions total; zero inference runtime errors; one ToolEnv per inference sample.
- Official BTC ToolEnv scores before/after the local 30-second gate: ask precision 63.89/62.22; completion audit 60.78/60.78; argument grounding 62.22/62.22. The archived ask-stop baseline is 63.33/63.33. Paired gated deltas and 95% bootstrap intervals: ask precision −1.11 [−10.00,+7.78]; completion audit −2.56 [−10.00,+5.00]; argument grounding −1.11 [−9.44,+7.22]. All intervals include zero; no prompt arm is promoted.
- Descriptive subgroup deltas (not inferential): on gold-ask tasks (n=25), ask precision +5.33, completion audit +0.53, argument grounding +9.33; on no-gold-ask tasks (n=35), respectively −5.71, −4.76, and −8.57. These conflicting directions and the enriched cohort argue against using the subgroup gains as a full-validation estimate.
- Local RTX 4050 timing, not A30. The best known Public LB remains 55.92, user-reported ground truth for the existing final4 ask-stop candidate. No submission or inference policy changed.
- Script: exp/qwen35-sft-pho-synthetic-v1/scripts/run_policy_prompt_screen_val60.py. Report, predictions, request logs, paired details, and lock: exp/qwen35-sft-pho-synthetic-v1/output/policy-prompt-screen-val60/.

## 2026-11-14 — Final4 Q8_0 merge and paired pilot60

- Reused the locally verified Qwen3.5-4B base and adapter_final-4-epochs.zip, merged adapter, converted BF16 to Q8_0 with pinned llama.cpp quantizer, and registered a separate Ollama tag. Q8 GGUF SHA-256 ed0f2be012e71e94775be5dfa7375bfb6205d29c336e7f8ebb9ccfd99f2f231a; tag lifemate-qwen35-phosynth-targeted-4epoch-q8-0:latest; digest 522187f399af365cead2dc73796ec6484c73ac43e7f8ce74c7b961a69cc58f4a. Existing Q4 tag and submission were untouched.
- Ran fresh Q8 inference on all 60 IDs of the existing enriched pilot, with cached PhoWhisper Large v2 main/history, focused prompt, fixed thinking, successful sys.ask_missing_info stop guard, JSON, temperature 0, cap768, context8192, batch512, seed17, and five-turn limit. Compared against archived Q4 ask-stop predictions for the exact same IDs. Official BTC ToolEnv exact-call multiplicity and state-hash gate; one ToolEnv per Q8 sample. Zero inference runtime errors.
- Q8 scored 59.22 before / 38.89 after the local 30-second gate, versus Q4 63.33/63.33. Paired Q8−Q4 raw −4.11 points [95% bootstrap CI −14.00,+5.33]; gated −24.44 [−36.67,−12.78]. Q8 state hashes 50/60, exact gold asks 19/60. Q8 mean local latency 29.38s with 23/60 over 30s; Q4 11.75s and 2/60.
- Ollama loaded Q8 with approximately 74% GPU / 26% CPU residency on the local RTX 4050 at context8192. The local gate does not estimate A30 performance. The enriched pilot and archived Q4 control limit quality inference; the ungated confidence interval includes zero. Q8 is not promoted or submitted. Q4 final4 fixed-thinking ask-stop remains the current public baseline with user-reported ground-truth Public LB 55.92.
- Build: exp/qwen35-sft-pho-synthetic-v1/scripts/build_final4_q8.py; registration: exp/qwen35-sft-pho-synthetic-v1/scripts/register_ollama_final4_q8.py; benchmark: exp/qwen35-sft-pho-synthetic-v1/scripts/benchmark_final4_q8_pilot60.py; report, per-ID details, predictions, logs, comparison lock, and model provenance: exp/qwen35-sft-pho-synthetic-v1/output/adapter-final-4epoch-q8-benchmark/.
## 2026-11-14 — Selective ASR error rescue pilot60

- Tested the second experiment on the exact enriched fixed pilot60 with the final4 Q4 adapter and fixed ask-stop control. Routing is label-free: compare cached PhoWhisper Large v2 and QUAN epoch-06 transcripts; expose the secondary only when normalized action, negation/polarity, number, or time tokens disagree. Unflagged treatment rows reuse the archived ask-stop predictions. The route selected 27/60; only those 27 were freshly inferred.
- Kept focused prompt, thinking fixed on, successful sys.ask_missing_info stop guard, JSON, temperature 0, num_predict=768, num_ctx=8192, num_batch=512, seed17, and max five action turns. Official BTC exact-call multiplicity and ToolEnv hash gate; one ToolEnv per fresh inference sample.
- BTC before/after local 30-second gate: Q4 ask-stop baseline 63.33/63.33; selective rescue 60.83/60.83. Paired delta −2.50 points (95% bootstrap CI [−9.17,+4.17]); 2 wins/55 ties/3 losses. The five changed sample scores were train_1499 (+1.0), train_1430 (−1.0), train_0345 (−1.0), train_0066 (−1.0), and train_1183 (+0.5). Zero runtime errors.
- Treatment Qwen+ToolEnv mean latency was 11.12s vs 11.75s baseline, but cached QUAN transcripts exclude incremental ASR compute, memory residency, and total parameter-count validation. Pilot is enriched and not full-val/Public LB evidence; do not promote the route or change current public policy. Public LB leader remains 55.92, user-reported ground truth for final4 Q4 fixed-thinking ask-stop.
- Script: exp/selective-asr-error-rescue-val60/scripts/benchmark_selective_asr_rescue.py. Report, routing decisions, branch predictions/traces/requests, paired details, and lock: exp/selective-asr-error-rescue-val60/output/.
## 2026-11-14 — Ask-stop with a six-turn cap on pilot60

- After the previously tested successful sys.ask_missing_info stop guard, raised the cap from five to six and reran the same enriched pilot60 on final4 Q4 with fixed thinking, focused prompt, cached PhoWhisper, JSON/temp0/cap768/context8192/batch512/seed17. One ToolEnv per sample, official BTC scorer.
- All 60 completed with zero runtime errors. The runner confirmed max_action_turns=6, but none of the 60 outputs used a sixth turn; lengths were 1–5. Therefore this screen does not measure whether an extra turn helps.
- BTC before/after local 30s gate: archived max5 ask-stop 63.33/63.33; fresh max6 61.67/61.67. Paired delta −1.67 [95% bootstrap CI −5.00,0.00], wins/ties/losses 0/59/1. Ordered tool calls matched on 57/60 and hashes on 59/60. Mean local latency 13.14s vs 11.75s; 2/60 over30s in max6. Differences reflect fresh-run variation because no sixth turn occurred.
- Do not promote max6 from this pilot. Need a cohort containing samples that actually reach a fifth successful action without stop/ask to evaluate turn six. Public Q4 ask-stop remains unchanged at user-reported ground-truth LB 55.92.
- Script: exp/qwen35-sft-pho-synthetic-v1/scripts/benchmark_final4_askstop_max6_pilot60.py. Report, fresh predictions, traces, request logs, paired details and lock: exp/qwen35-sft-pho-synthetic-v1/output/ask-stop-max6-pilot60/.
## 2026-11-14 — Max-six follow-up on eligible val309 samples

- The first ask-stop max6 pilot60 did not exercise a sixth turn. To test the condition directly while keeping the prior 60-task preference, selected all 25 archived final4 val309 samples whose first five actions were successful tool calls, with no ask and no runtime error. For this cohort, the ask-stop guard leaves the max-five archived prefix unchanged. Fresh max6 inference ran only on those 25.
- Fixed thinking, focused prompt, cached PhoWhisper, Q4 final4, ask-stop retained; max_action_turns=6; JSON/temp0/cap768/context8192/batch512/seed17. One ToolEnv per fresh sample; official BTC exact-call multiplicity and state-hash scoring. 25/25 actually used turn six; zero runtime errors.
- BTC before/after the local 30-second gate: max-five archived baseline 67.84/35.09; max-six 73.66/13.58. Paired raw delta +5.82 [95% bootstrap CI −6.40,+18.04]; gated delta −21.51 [−37.51,−7.20]; wins/ties/losses 3/21/1. Raw CI includes zero.
- Mean latency rose 33.07s→38.88s on local RTX 4050; 21/25 max6 samples exceeded 30s. This is a selected continuation cohort, not full-val/Public LB; local runtime is not A30 timing. Do not promote without confirming A30 runtime and broader quality.
- Script: exp/qwen35-sft-pho-synthetic-v1/scripts/benchmark_final4_askstop_max6_eligible_val309.py. Report and per-ID details, inference, lock: exp/qwen35-sft-pho-synthetic-v1/output/ask-stop-max6-eligible-val309/.
## 2026-11-14 — Adaptive thinking plus max-six ask-stop on eligible val309

- Tested adaptive 20-second thinking with the successful `sys.ask_missing_info` terminal guard and a six-action cap on the same 25 eligible val309 samples as the fixed max-six run. Final4 Q4, focused prompt, cached PhoWhisper, JSON/temp0/cap768/context8192/batch512/seed17. Adaptive starts thinking on and turns it off for later requests when cumulative model-request latency reaches 20 seconds.
- All 25 completed without runtime errors. Adaptive switched on 24/25 samples and disabled thinking for 44 requests; 23/25 used turn six (ask-stop ended the others earlier).
- Official BTC before/after local 30s gate: archived max5 fixed-thinking 67.84/35.09; fresh fixed max6 73.66/13.58; adaptive max6 67.08/40.69. Adaptive−max5: raw −0.76 [95% CI −12.00,+10.31], gated +5.60 [−6.40,+18.40]. Adaptive−fixed6: raw −6.58 [−20.38,+6.33], gated +27.11 [+11.56,+44.89].
- Mean local latency: max5 33.07s, fixed6 38.88s, adaptive6 27.91s; >30s: 14/25, 21/25, 9/25. Timings are local RTX 4050 with cached ASR, not A30. The continuation cohort is selected and the adaptive-vs-max5 interval includes zero; do not change the public candidate or max-five policy based on this screen.
- Script: `exp/qwen35-sft-pho-synthetic-v1/scripts/benchmark_final4_askstop_max6_adaptive20_eligible_val309.py`. Artifacts: `exp/qwen35-sft-pho-synthetic-v1/output/ask-stop-max6-adaptive20-eligible-val309/`.
## 2026-11-15 — Adaptive20 max6 ask-stop full val309 and selected submission

- Reconstructed the fixed-thinking max5 ask-stop control from the archived final4 val309 run: truncated 47 rows after the first successful `sys.ask_missing_info`, replayed truncated tool calls through official ToolEnv to recompute hashes/final state, and subtracted discarded post-ask request latency. Reused archived max5 trajectories for the remaining rows.
- Ran fresh final4 Q4 inference on all 309 canonical val IDs with focused prompt, cached PhoWhisper main/history, adaptive thinking (on initially, off after cumulative request latency reaches 20s), max six action turns, and stop after a successful `sys.ask_missing_info`. One context overflow on train_1725 per branch; no other runtime errors.
- BTC raw/local-gated: reconstructed fixed max5 ask-stop 63.80/57.88; adaptive max6 ask-stop 63.11/60.34. Paired adaptive−control raw −0.69 (95% CI [−1.94,+0.45]); gated +2.46 ([+0.96,+4.27]). Over 30s fell 31→15; mean local latency 16.49s→15.85s. State hashes matched 259/309 vs 256/309. Adaptive switched 54/309 samples, 83 requests; 7/309 used six tool calls.
- Full val supports adaptive20+max6 as a runtime-oriented option, not as raw BTC winner; local RTX 4050 with cached ASR is not A30 performance. Keep fixed-thinking max5 ask-stop for the user-requested raw-score selection. Highest known Public LB remains 55.92 (ground truth per user).
- Created `outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-val309-max5-fixed-askstop-public-test.jsonl` by byte-identical reuse of the existing 55.92 ask-stop candidate, with matching predictions sidecar and selection manifest. Official schema validator: 634 rows, 634 unique IDs, 0 errors. SHA-256: `c65cbf01ff162d3cbcec76769cc94e18e295d123ef703cf93d8dfea206bf220e`.
- Benchmark script: `exp/qwen35-sft-pho-synthetic-v1/scripts/benchmark_final4_askstop_max6_adaptive20_val309.py`. Full report, locks, telemetry and paired details: `exp/qwen35-sft-pho-synthetic-v1/output/ask-stop-max6-adaptive20-val309/`.

## 2026-11-15 — Public candidate: fixed full thinking, max six, ask-stop

- Generated the requested separate public-test candidate with the final4 Q4 model, fixed thinking enabled on every request, a six-action-turn cap, and automatic loop stop after a successful sys.ask_missing_info. Used the focused post-lookup prompt and PhoWhisper Large v2 for main and history audio.
- Completed all 634 public tasks. Official schema validation: 634 rows, 634 unique IDs, zero schema errors. There were zero runtime errors and exactly one ToolEnv session per sample. The predictions sidecar has 39 samples with zero tool calls; tool-call count distribution: 0:39, 1:169, 2:203, 3:109, 4:62, 5:37, 6:15.
- Local Qwen+ToolEnv mean latency was 18.67s, with 96/634 over 30s; this uses precomputed ASR and excludes ASR inference. Estimated combined ASR+Qwen mean was 24.28s, with 171/634 over 30s. These are local diagnostics, not A30 timing or a Public LB score.
- Submission SHA-256: 71122d12e3ef52910e6f4b6b8e9a0dfc53f936aa1db6cc6f9ab96f2d789d8dde. Predictions sidecar SHA-256: f842af4bff3bdd4087fd0158b875fba3a1a17e2ea21929f9e44d0b6c69c82137.
- At generation time this candidate had not been submitted and its Public LB score was unknown. The user later reported **56.16 Public LB** for this fixed-thinking max6 candidate and instructed that user-reported scores are ground truth; this supersedes the prior 55.92 max5 ask-stop result. Fixed-thinking max6 has not been benchmarked on full val309. The full-val raw BTC reference remains 63.80 for fixed-thinking max5 versus 63.11 for adaptive20 max6; adaptive20 is the only max6 configuration tested on full val309.
- Candidate: outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-ask-stop-max6-full-thinking-public-test.jsonl. Predictions: outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-ask-stop-max6-full-thinking-public-test_predictions.jsonl. Run summary, lock, and schema log: exp/qwen35-sft-pho-synthetic-v1/output/submission-final4-ask-stop-max6-full-thinking/.
- Generator: exp/qwen35-sft-pho-synthetic-v1/scripts/make_submission_final4_askstop_max6_fixed.py.

## 2026-11-15 — User-reported Public LB 56.16 and inference configurations

- The user reports 56.16 Public LB for the fixed-thinking max6 ask-stop public candidate created above. Record this as ground truth. Candidate SHA-256: 71122d12e3ef52910e6f4b6b8e9a0dfc53f936aa1db6cc6f9ab96f2d789d8dde. The previous max5 fixed-thinking ask-stop score of 55.92 remains valid historically but is no longer the highest-known result.
- Public-scoring configuration associated with 56.16: final4 Q4, PhoWhisper Large v2 for main/history, focused post-lookup prompt, fixed thinking enabled on every request, maximum six action turns, stop after sys.ask_missing_info succeeds. The output has 634 unique IDs, zero schema/runtime errors, and one ToolEnv session per sample. The 56.16 score is the user's report.
- Keep the adaptive-thinking max6 policy as a separate runtime-oriented inference option: start each sample with thinking enabled; after cumulative model-request latency reaches 20 seconds, disable thinking for subsequent requests in that sample; allow at most six action turns; stop after successful sys.ask_missing_info. On full val309, adaptive20 max6 scored 63.11 raw / 60.34 after the local 30-second gate, versus fixed-thinking max5 ask-stop at 63.80 / 57.88. Local RTX 4050 timing with cached ASR is not A30 timing. The 56.16 Public LB result is not attributed to the adaptive branch.

## 2026-11-15 — Adaptive total ASR+Qwen budget and max-turn full val309

- Compared end-to-end adaptive thresholds 15s and 20s on the same 309 final4 Q4 validation tasks. The threshold includes per-task PhoWhisper main/history inference time plus cumulative Qwen request latency; first Qwen request uses thinking. Thinking turns off before subsequent requests once the total reaches the threshold. A successful `sys.ask_missing_info` stops the harness. One rollout to seven action turns was scored at caps 4–7; repair requests in retained action turns count toward latency.
- Official BTC raw / after local 30s gate: threshold15 cap5 61.34/60.27, cap6 61.64/60.56, cap7 61.64/60.56; threshold20 cap5 61.53/58.30, cap6 61.83/58.30, cap7 61.83/58.30. No runtime errors. Threshold15−20 at cap6: +2.26 gated points (paired bootstrap 95% CI [+0.13,+4.48]), −0.19 raw. At 15s, cap6−cap5 +0.29 gated (CI [+0.00,+0.88]); cap7−cap6 0.00.
- Recommend adaptive total budget 15s and max6 action turns for the runtime-oriented final4 candidate. Max7 adds no measured score. Fixed-thinking max5 remains the raw-score winner; keep the user's 56.16 fixed-thinking public candidate unchanged. PhoWhisper cache timing averages 6.09s/task (median 5.22, p95 16.67; 309 main/692 history clips), excluding model-load startup. Qwen timing is local RTX 4050, not A30; repeat on the new adapter/target GPU.
- Runner: `exp/qwen35-sft-pho-synthetic-v1/scripts/run_adaptive_totalbudget_maxturn_fullval309.py`. Summary and paired artifacts: `exp/qwen35-sft-pho-synthetic-v1/output/adaptive-totalbudget-maxturn-fullval309/`.

## 2026-11-15 — Final5 20k adapter merge and adaptive15/max6 pilot60

- Merged `adapter_final-5-epochs-20k-rows.zip` into the verified Qwen3.5-4B base and converted to Q4_K_M GGUF. GGUF SHA-256: `ae5c220611b6641307aa10c0173ef7431ccbf2e214ad1886710c7bfbf8543365`; Ollama tag `lifemate-qwen35-phosynth-final5-20k-q4km:latest`. The local benchmark used RTX 4050, not A100.
- Ran final4 control and final5 treatment on the same enriched fixed cohort60 using adaptive 15s total ASR+Qwen budget, max6 action turns, successful `sys.ask_missing_info` stop, and official BTC ToolEnv scoring. Both branches completed 60/60 with zero runtime errors; each matched hashes on 50/60.
- BTC raw/gated: final4 63.00/61.33; final5 59.67/59.67. Paired final5−final4: −3.33 raw (95% CI [−11.67,+5.00]); −1.67 after gate (CI [−10.00,+6.67]); wins/ties/losses 3/53/4. Mean end-to-end time was 16.15s vs 15.65s; cached ASR mean 4.98s/task. The interval includes zero and this cohort is enriched, so this is diagnostic evidence only. Keep final4 and the 56.16 Public LB candidate unchanged; do not treat pilot60 as full-val/public performance.
- Report, paired details, requests and predictions: `exp/qwen35-sft-pho-synthetic-v1/output/adapter-final5-20k-adaptive15-pilot60/`; benchmark entry point: `exp/qwen35-sft-pho-synthetic-v1/scripts/benchmark_adapter_final5_20k_adaptive15_pilot60.py`.
