# Insights

## PhoASR controlled ASR ablation (2026-10-29)

- Kept the baseline Qwen3-4B Q4_K_M agent, prompt, full tool schemas, ToolEnv, evaluator, and fixed 309-ID split. Only replaced Whisper Small with the official Qualcomm-AI-Research PhoASR-whisper-small checkpoint; retrieval and all demo/shortlist modes were off.
- Checkpoint: `Qualcomm-AI-Research/PhoASR-whisper-small` (Whisper Small fine-tune, 0.2B parameters), released under BSD-3-Clause-Clear and Qualcomm Responsible AI License. Used local Transformers inference on CUDA with deterministic greedy decoding; no CTranslate2 conversion.
- Corpus WER improved from 0.5108 to 0.2804 and CER from 0.3657 to 0.2013. PhoASR still made errors with numerals, names, and place names, but reduced Whisper repetition failures substantially. Examples are in `outputs/eval/paired-analysis/baseline-vs-phoasr-small-baseline/asr-analysis.json`.
- Downstream proxy improved from 0.1919 to 0.2286, tool-call F1 from 0.1155 to 0.1693, and final-state match from 0.2492 to 0.2751. Paired counts: 55 improved, 24 worsened, 230 unchanged. PhoASR closed 36.7% of the oracle proxy gap, 42.3% of tool-F1 gap, and 32.0% of state-match gap.
- Full validation had zero runtime failures; mean latency was 11.17s and one sample exceeded 30s (max 34.08s) on the local RTX 4050. This is a runtime risk to recheck on A30. Public inference hit two Ollama runner resource failures, resumed from the ordered prefix each time, and completed. The 634-row candidate passed the official schema/order validator. User-reported Public LB: 9.79.

## Dev kit

- `ToolEnv` records every attempted call, including schema-rejected calls; submit exactly `env.get_tool_calls()`.
- ToolEnv is constructed from the split data root, with one session per sample.
- The package contains no standalone evaluator; a local evaluation harness must replay predictions against labeled train examples.
- On Windows, CUDA-enabled PyTorch does not guarantee CTranslate2 has its CUDA 12/cuBLAS DLLs; the ASR backend needs a CPU fallback when those DLLs are missing.


## Ollama inference

- The Q4_K_M model fits entirely on the 6 GB RTX 4050 at 6144 context, but a long prompt (6264 tokens before generation) fails with HTTP 400. At 16384 context it runs split across CPU/GPU (23%/77%) and remains slower; measure with the actual prompt distribution.
- The 20-sample Ollama CV completed without runtime errors but achieved only 0.0247 tool-call F1, so fast inference alone does not make this model setup a usable baseline.
- On Windows, `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, and `nvidia-cudnn-cu12` can be installed in the project venv; add their package `bin` directories to the DLL loader path before importing CTranslate2. Whisper Small then transcribes on the RTX 4050.

## Fixed validation and oracle-ASR (2026-10-28)

- A deterministic grouped split of 1,545 train examples produced 1,236 train IDs and 309 validation IDs (seed 20261028). Grouping normalized transcripts at character similarity >= 0.92 kept near-duplicates together. Validation proportions were close to train for single/multi-tool (30.4%/69.6% vs 31.2%/68.8%), missing-info (15.2% vs 13.8%), and out-of-scope (7.4% vs 7.1%). No audio-type metadata was available.
- On all 309 validation samples, Whisper Small + Qwen3-4B Q4_K_M had local proxy mean 0.1919, tool-call F1 0.1155, and final-state match 0.2492. Oracle transcripts scored 0.2920, 0.2429, and 0.3301 respectively; paired proxy gain was +0.1001 (101 improved, 27 worsened, 181 tied).
- The proxy is the per-sample mean of set-based tool-call F1 and labeled final-state exact match. The local kit does not expose the official combined score formula, so these values are diagnostic rather than directly comparable to the public LB.
- Heuristic baseline error breakdown among 290 non-perfect samples: WRONG_TOOL 204 (70.3%), MISSING_INFO_ERROR 51 (17.6%), ASR_SLOT_ERROR 16 (5.5%), WRONG_ARGUMENT 13 (4.5%), OUT_OF_SCOPE_ERROR 6 (2.1%), PLANNING_ERROR 0, OTHER 0. ASR_SLOT_ERROR requires baseline failure and oracle exact calls plus state match; this likely undercounts partial ASR effects.
- Current evidence points to agent/tool selection and missing-info handling as the primary bottleneck, with a meaningful secondary ASR contribution. Next, inspect WRONG_TOOL and MISSING_INFO cases and try prompt/few-shot changes using training data only; target numeric/time/entity transcription checks separately. Do not fine-tune until these simpler experiments are evaluated.

## Retrieval-assisted routing (2026-10-28)

- Retrieval corpus uses only the fixed 1,236 `train_ids.json` samples. The deterministic retriever uses word and adjacent-word-bigram TF-IDF/cosine, top-3 examples, up to five voted tools plus `sys.ask_missing_info`, and falls back to the full installed tool set below similarity 0.06. Qwen sees three retrieved demonstrations plus train-derived recipient/time/location/amount missing-info examples.
- Retrieval-v1 improved validation over baseline: proxy 0.2432 vs 0.1919, aggregate tool-call F1 0.2299 vs 0.1155, and final-state match 28.8% vs 24.9%. Paired outcomes: 79 improved, 38 worsened, 192 tied.
- Error counts changed: `WRONG_TOOL` 204→188 (-16); `MISSING_INFO_ERROR` 51→53 (+2). The missing-info rule/examples did not reduce this error group. `WRONG_ARGUMENT` rose 13→24, so inspect copy-over from demonstrations.
- In 20 of the 38 retrieval-v1 regression samples, the shortlist omitted at least one gold tool. A wider variant (top-5 retrieval, up to eight tools) did not help: proxy 0.2392, F1 0.2287, state match 27.8%. Keep v1 for the current submission; the broader shortlist also increased missing-info errors (55).
- Retrieval-v1 produced a 634-row public-test submission at `outputs/submissions/retrieval-v1.jsonl`; manifest order, unique IDs, row shape, state hashes, and tool schemas validated with zero errors. The user reported Public LB 6.66, below the baseline's 7.85 despite better local validation, so the local gain did not generalize to the public metric.
- Before another submission, inspect the public-score gap and reduce `MISSING_INFO_ERROR` and demonstration argument copying on the fixed validation split. Do not widen the shortlist further without a clearer local gain.

## Soft retrieval session (2026-10-29)

- Validation shortlist recall, treating `candidate_tools: null` as the documented full-tool fallback: retrieval-v1 hit at least one gold tool on 78.3% of samples and contained all gold tools on 65.4%; retrieval-v2-wide reached 79.6% and 66.7% respectively.
- Compared with baseline, retrieval-v1 had 38 lower-scoring samples; 20 had at least one gold tool excluded by its shortlist. Retrieval-v2-wide had 36 lower-scoring samples; 18 had a gold tool excluded. Representative misses include `train_0625` (gold `music.play`, shortlist contained calendar/chat tools) and `train_1334` (gold contacts/medication tools missing while ewallet transfer was retained).
- Implemented a soft mode in which the same top-3 TF-IDF examples are prompt references and the full available tool schema remains visible. The prompt explicitly says examples do not determine routing or authorize copying arguments; the missing-required-argument rule is stated directly.
- Soft retrieval completed on all 309 validation samples with zero runtime failures: proxy 0.2558, aggregate tool-call F1 0.2435, and state match 29.77%. Versus baseline: 86 improved, 32 regressed, 191 tied (baseline proxy 0.1919, F1 0.1155, state 24.92%). Versus retrieval-v1: 63 improved, 48 regressed, 198 tied; mean paired proxy gain +0.0126.
- Soft retrieval lowered the heuristic `WRONG_TOOL` count from 204 to 194 and `MISSING_INFO_ERROR` from 51 to 49 relative to baseline. `WRONG_ARGUMENT` rose 13→16 and out-of-scope errors 6→10; inspect these separately if doing another prompt iteration.
- Selected soft retrieval over hard shortlist: local proxy, F1, and final-state match all exceed retrieval-v1 (0.2558/0.2435/0.2977 vs 0.2432/0.2299/0.2880), while regression count vs baseline is lower (32 vs 38). Curated-fewshot validation was skipped to protect time for full public inference.
- Full public inference completed in 6,914.7 seconds (10.91 seconds/sample average). `outputs/submissions/soft-retrieval-v1.jsonl` has 634 rows in manifest order, 634 unique IDs, and zero official tool-schema validation errors. It contains 139 empty tool-call lists (21.9%); retrieval-v1 had 156 (24.6%), so this is not an increase relative to the prior submission.
- Public LB score was reported as 7.51. This is +0.85 over retrieval-v1 (6.66) but remains 0.34 below the baseline (7.85), despite the stronger local proxy. Soft retrieval reduced the hard-shortlist generalization gap but did not fully recover baseline public performance.
