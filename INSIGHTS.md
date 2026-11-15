# Insights

## Ollama `num_batch` reproducibility and latency pilot (2026-11-05)

- Reproduction was unstable because the effective server batch setting was not recorded in the prior experiment manifest. The archived server runner used `-b 512 -ub 512`; the current default uses `-b 1024 -ub 1024`, despite the same Ollama version, model digest, request payload, prompt and transcript cache. Explicit `num_batch=512` exactly matched all archived calls and state hashes on 60/60 samples. Explicit1024 matched the current implicit-default output on 60/60. The old-vs-current mismatch (36 ordered calls, 21 hashes) is therefore attributed to effective `num_batch` drift, not telemetry instrumentation.
- On the enriched pilot60, explicit512 vs current default1024 reduced the sample median 6.5% (29.32→27.42s), p95 6.9% (58.71→54.66s), and samples over 30 seconds from 30 to 29. BTC formula was 17.5159 vs 14.0635/100 under exact tool-name/arguments matching, duplicate multiplicity, gold-replayed state-hash gate and M>0; 512 reproduced the archived score. Paired bootstrap intervals for latency and score include zero. Set the application-side Ollama API default explicitly to 512 for reproducibility and faster local iteration; this pilot does not establish an A30 runtime benefit or a val309 score gain. Override with `LIFEMATE_OLLAMA_NUM_BATCH` when testing another value.
- Ollama prompt/decode duration telemetry shows decode dominated total request work on this run. GPU placement was already 100% GPU; peak VRAM was 3,980/6,141 MiB, and Flash Attention was active in auto mode. No repeated model reload occurred in continuous inference. `num_gpu`, KV q8_0 and keep_alive were not justified by the observed bottlenecks.
- Limitations: 60 enriched train-only samples, local RTX 4050, and configuration switches can add runner-load time. These scores do not estimate val309 and do not establish A30 <=30-second compliance. Details: `exp/inference-latency-optimization-val60/output/summary.md`.

## Qwen3.5 thinking/tool-protocol experiment (complete 2026-11-03)

- Full fixed train-only val309 BTC scores (official after local 30s runtime gate; pre-mask diagnostic in parentheses): A no-think JSON **1.3916 (1.3916)**; B no-think native tools **6.3484 (6.4779)**; C thinking JSON **5.6095 (9.2079)**; D thinking native tools **7.2168 (7.4326)**. Calls matched by exact tool name and arguments, duplicates by multiplicity, with gold-state hash gate and ToolEnv replay.
- Paired bootstrap (10,000 samples, seed 20261102): B−A **+4.96** points (95% CI 2.82 to 7.39); C−A **+4.22** (2.01 to 6.68); D−C **+1.61** (−0.81 to 4.09); D−B **+0.87** (−1.53 to 3.35); interaction (D−C)−(B−A) **−3.35** (−6.52 to −0.27). Native protocol improved the no-thinking branch over JSON; evidence for an additional thinking gain with native tools is inconclusive. The negative interaction suggests a smaller protocol gain with thinking enabled.
- Runtime mask used local RTX 4050 measurements, not an A30 benchmark. Samples over 30 seconds: A 0, B 2, C 140, D 12. Pre-mask values are diagnostics; A30 may reduce runtime masking, but its effect was not measured. All four gold replays matched the 309 labels, with zero replay failures. C mean/p95/max latency: 31.87/63.11/87.20s; D: 15.14/28.11/52.66s. C recorded 171 failed parse/schema repairs; D recorded one and 22 unexecuted extra native calls.
- Do not change production defaults based on this local cohort alone. No submission was created and Public LB was not used. Full predictions, diagnostics, BTC details, configs, and paired report are under `exp/qwen35-thinking-tool-protocol-val309/output/` and `exp/paired-analysis/qwen35-thinking-tool-protocol-val309/output/paired-bootstrap-report.json`.

## Session 4 prompt and cap A/B (2026-11-02)

- On the fixed train-only 60-ID val309 cohort, observation-grounded-v2 did not improve over the prior Qwen3 prediction reference: BTC was 7.4802/100 old vs 2.7778 Qwen3-Vietnamese and 4.0000 Qwen3-English. Both Qwen3.5 branches were lower (1.3889 Vietnamese, 3.1667 English). These are official BTC scores using exact name+arguments with duplicate multiplicity and gold-state hash replay through ToolEnv; the local proxy remains a separate diagnostic.
- English was the strongest prompt language in this pilot for Qwen3 and Qwen3.5, but neither model/prompt branch exceeded the old reference. Qwen3.5 is NO-GO for the next full confirmation.
- On full fixed val309, Qwen3-English prompt v2 scored 3.0868 at cap 5 and 3.1702 at cap 6, both below the existing 4.9145 baseline. Cap 6's +0.0835 BTC points came with 87 extra submitted calls, 82 more unmatched calls, no additional hash-matched samples, and 87/309 predictions exceeding five calls while only one gold sample needs over five actions. The score gain is too small to justify changing the default; cap 6 is only a follow-up candidate.
- Both full runs had the same two context-overflow failures (8,537 and 8,265 tokens vs the fixed 8,192 context). Context was kept fixed; those samples remain score zero. Artifacts and detailed paired analysis are under `exp/session4-prompt-ab/output/` and `exp/paired-analysis/session4-prompt-ab/output/`.

## Qwen3 vs Qwen3.5 full val309 comparison (2026-11-02)

- Held PhoASR, prompt-v2 English, cap 5, context 8192, deterministic JSON decoding, and the full fixed train-only val309 cohort constant. Official BTC was 3.0868 for Qwen3 and 1.3916 for Qwen3.5; paired scores favored Qwen3 on 17 samples, Qwen3.5 on 7, with 285 ties. Neither exceeded the 4.9145 baseline.
- Qwen3.5 matched more final-state hashes (74 vs 64) and ran faster (2.45s vs 4.96s mean), but produced only 38 exact calls vs Qwen3's 109. Its low call recall explains why more hash matches did not yield a higher BTC score. Qwen3 had two fixed-context overflows; Qwen3.5 had none.
- Keep Qwen3 as the stronger tested model by BTC; do not change the production default based on this comparison. Full predictions and paired scoring are in `exp/session4-qwen3-vs-qwen35-val309/output/` and `exp/paired-analysis/session4-qwen3-vs-qwen35-val309/output/paired-report.json`.

## BamiBERT tool-router experiment (2026-11-02)

- Routing did not become easier in this setup: BamiBERT predicted STOP for every fixed val309 timestep at both 512 and 1024 tokens. Top-1 was 29.63%, exactly the STOP share; non-STOP, first-action, second-action, and later-action accuracy were all 0%. The result is a clear NO-GO for continuing this decomposed policy as implemented.
- Errors were not concentrated on one tool: all tool classes collapsed to STOP. Most frequent were contacts.search (56), sys.ask_missing_info (47), smart_home.light_set (45), smart_home.get_status (31), and reminders.create (26).
- STOP is the source of the apparent top-1 score, with recall 1.0 but precision 0.296. Class weighting did not correct the collapse. Increasing input length reduced truncation from 93.7% to 63.1% but did not change predictions, so truncation alone does not explain the failure.
- Qwen's aligned next-tool accuracy was 29.4% on first tools, 9.8% on second tools, and 5.2% on later tools (excluding STOP targets). BamiBERT failed to improve any tool-routing step; this also does not support the idea that a small classifier resolves the downstream continuation weakness.
- The experiment used gold ToolEnv observations and arguments only in previous observations; no target arguments entered the classifier label or future context. STOP imbalance was present, but the collapse across every tool means more argument-generation work is not justified by these results.

## Qwen next-action trajectory dataset (2026-10-31)

- Replayed all 1,236 fixed train trajectories through one fresh official ToolEnv each and generated 4,043 timestep examples. Mean gold trajectory length was 2.271 tool actions (maximum six); each successful sample has one STOP target, and 170 targets call `sys.ask_missing_info`.
- Replay failures and duplicate timesteps: zero. All targets parse through the production action parser; tool names and argument keys were checked against the installed tool schemas. The fixed 309 validation IDs have zero overlap and are excluded.
- Training prompts use local PhoASR public transcripts, production schemas/context/history serialization, and actual ToolEnv observations. No state snapshot is included in the prompt. This is dataset integrity evidence only; no QLoRA training or downstream gain has been measured yet.
- The private payload was uploaded after explicit user confirmation as Kaggle dataset `daoviet/lifemate-qwen-next-action-v1`; its manifest is included in v2. The T4 smoke run reached 30/30 optimizer steps with loss 2.693 ? 0.224 and peak VRAM 13.38 GiB. Adapter loading/generation worked, but the post-training parser check failed because it used an unrelated greeting prompt; Kaggle exposed no artifact because the notebook ended ERROR. A locally regenerated notebook uses an actual SFT prompt; it has not been uploaded or rerun while model execution is paused.
- Smoke kernel v1 failed before model load because the dataset runner was absent at the hard-coded path. Dynamic input discovery fixed the mount path in v2.
- Smoke v2 loaded the data/model and failed during SDPA attention with CUDA OOM at sequence length 6,144 (T4: 551 MiB free, 864 MiB additional allocation requested). Kernel v3 now uses a 4,096-token context and expandable CUDA allocator segments; result is pending.

## PhoASR paired error diversity and dual-transcript experiment (2026-10-31)

- Paired artifacts cover all 309 fixed validation IDs. Public and FT transcripts were identical on 203 samples and differed on 106. Per-sample WER favored public on 19, FT on 37, and tied on 253; both transcripts had nonzero WER on 301 samples. Corpus WER was 0.2804 public and 0.2757 FT. Differences include number formatting and recovery (`train_0373`: FT preserves `30` and `26`), money/time/name details (`train_0244`: FT better recovers “sáu giờ rưỡi” but still changes the recipient to “Minh Trúc”), and app/domain phrase errors (`train_0055`).
- Existing PhoASR artifacts only contain decoded text, with no comparable confidence fields. Confidence selector and selector downstream evaluation were skipped; no synthetic confidence proxy was introduced.
- The dual-transcript prompt completed 309/309 ToolEnv samples with zero runtime failures. Against PhoASR public: proxy 0.2064 vs 0.2286, micro tool-call F1 0.1509 vs 0.1693, final-state match 0.2718 vs 0.2751. Paired outcomes: 28 improved, 49 worsened, 232 unchanged. It also underperformed FT on proxy and tool F1, although state match was slightly higher than FT.
- Regression inspection found 37/49 worsened samples had identical ASR transcripts, so transcript disagreement was not required for the failure. Examples include `train_0559` (public made the correct light call; hybrid stopped) and `train_1161` (hybrid invented a reminder for an out-of-scope utterance). This points to the extra formatting/instructions shifting Qwen behavior and causing over/under-calling; in `train_0038`, the differing “lắp/lắc bếp” phrase coincided with irrelevant kitchen status and missing-info calls. The dual prompt did not exploit diversity enough to offset this.
- No selector or hybrid candidate qualifies for submission. Keep PhoASR public as the current ASR choice; treat this PhoASR branch as near saturation and shift effort to agent/tool selection and prompt reliability. The existing PhoASR public submission remains the best of these candidates at reported Public LB 9.79; no new submission or leaderboard probe was made.

## PhoASR LifeMate fine-tune (2026-10-30)

- Fine-tuned the existing local `Qualcomm-AI-Research/PhoASR-whisper-small` checkpoint (local revision hash unavailable) with Transformers/PyTorch. Fixed split audit passed for all 1,545 audio files; 1,236 train and 309 validation IDs had no overlap. UTF-8 Vietnamese transcripts were passed without normalization.
- Configuration: seed 17, one epoch, LR 2e-6, per-device batch 1, gradient accumulation 8; encoder and decoder blocks 0–9 frozen, blocks 10–11 plus final layer norm/output projection trainable (58.7M params). Train loss mean 2.776 (first item 5.541, last item 1.822), validation teacher-forced loss 2.497, training 523.8s, peak VRAM 2.03 GB. Checkpoint reload and greedy inference succeeded.
- Direct ASR on the fixed 309 validation samples: PhoASR public WER/CER 0.2804/0.2013; fine-tuned WER/CER 0.2757/0.1966. This is a small absolute WER gain of 0.0047. Some utterances improved on numeric formatting and phrase recovery (for example `train_0373`, 30 and 26 degrees; `train_0994`, preserving the hesitation phrase), but proper names remain unreliable (`train_0244` turns Minh into Minh Trúc; `train_0166` still hears Lan as Lenin).
- Controlled downstream run completed 309/309 with the same Ollama Qwen3-4B Q4_K_M, prompt, full schemas, ToolEnv, and no retrieval; no runtime failures. Proxy/F1/state changed from 0.2286/0.1693/0.2751 to 0.2222/0.1634/0.2686. Paired sample scores: 9 improved, 12 worsened, 288 unchanged. Mean latency 11.69s vs 11.17s for PhoASR public.
- Oracle gap remaining after fine-tune: proxy 0.0698, tool F1 0.0795, final-state match 0.0615. Relative to the PhoASR public gap, the FT closed -10.2% / -8.0% / -11.8% respectively (that is, widened downstream gaps despite modestly lower WER). The checkpoint did not qualify under the original local-validation submission criterion. At the user's later request, a 634-row public-test candidate was generated and validated; user-reported Public LB: 9.78.
- Conclusion: one conservative domain fine-tune improves lexical WER/CER slightly but not local downstream validation. Its Public LB of 9.78 is 0.01 below the PhoASR public candidate's 9.79, so it effectively matched that result without measurable public-score headroom. More ASR tuning is not justified without a changed training strategy or stronger validation evidence.

## PhoASR controlled ASR ablation (2026-10-29)

- Kept the baseline Qwen3-4B Q4_K_M agent, prompt, full tool schemas, ToolEnv, evaluator, and fixed 309-ID split. Only replaced Whisper Small with the official Qualcomm-AI-Research PhoASR-whisper-small checkpoint; retrieval and all demo/shortlist modes were off.
- Checkpoint: `Qualcomm-AI-Research/PhoASR-whisper-small` (Whisper Small fine-tune, 0.2B parameters), released under BSD-3-Clause-Clear and Qualcomm Responsible AI License. Used local Transformers inference on CUDA with deterministic greedy decoding; no CTranslate2 conversion.
- Corpus WER improved from 0.5108 to 0.2804 and CER from 0.3657 to 0.2013. PhoASR still made errors with numerals, names, and place names, but reduced Whisper repetition failures substantially. Examples are in `exp/paired-analysis/baseline-vs-phoasr-small-baseline/output/asr-analysis.json`.
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

## Qwen tool-calling SFT preparation (2026-10-31)

- Prepared a JSONL chat SFT corpus from exactly the 1,236 fixed `train_ids.json` rows; the 309 validation IDs were excluded and train/validation overlap is zero. Inputs use gold `audio_text`, the current baseline Vietnamese system prompt, sample context/history, and the same compact full installed-tool schemas generated by `Agent._tool_specs`. Targets preserve the official ordered `{name, arguments}` tool-call representation, including empty lists and `sys.ask_missing_info` calls.
- Dataset audit: 850 multi-call, 298 single-call, 88 no-action, and 170 missing-info samples. All 1,236 rows contain the requested fields. Official `zaic_tools.schema_validate` found zero invalid gold calls; call metadata is stripped and unknown tool/argument keys are rejected during preparation.
- SFT could not proceed: `peft`, `bitsandbytes`, `trl`, and `datasets` are absent, and the configured pip environment has `PIP_NO_INDEX=1`; a normal dependency install returned “No matching distribution found.” Full fine-tune is not an option on this 6 GB RTX 4050. No smoke run, adapter, LoRA validation, error comparison, or submission was produced.
- Conclusion for the scientific question is **undetermined**. The training-vs-validation experiment remains unrun, so no claim of gain or regression is supported. Kept PhoASR public and all inference/evaluator settings untouched. Training target is the full ordered production call list; wiring a trained model into the existing iterative action loop needs a compatible decoding design before inference evaluation.

## Agent architecture audit (2026-10-31)

- Code audit confirms the baseline is already interactive. `LifeMatePipeline.run` creates exactly one ToolEnv, then loops up to five times. Each `Agent.next_action` emits one `tool_call` or `stop`; after a valid call, `env.call` returns the observation and the next Qwen prompt serializes the accumulated observations. The same ToolEnv and in-memory app state persist through the sample; final calls/state/hash come from that trajectory. Qwen request count is 1–2 per action turn because an invalid action can trigger one repair generation; actual repair counts are not stored in historical predictions.
- ToolEnv exposes `validate`, `call`, `get_tool_calls`, `get_state`, and `get_state_hash`. Calls update the same session state, and observations are JSON dictionaries. Read/search/list tools can expose intermediate IDs and state. Rejected calls are logged; the current pipeline then stops after one ToolEnv rejection. The sample has a five-turn cap including the explicit stop turn.
- The full labeled train split has 111 zero-call, 369 one-call, 428 two-call, 359 three-call, 161 four-call, 107 five-call, and 10 six-call examples (maximum six tool calls). Thus the five-turn cap cannot represent all six-call sequences, even before reserving a stop turn.
- Inspected 30 highest-complexity paired holdout examples with at least two gold calls (27 had five calls, one had six, two had four). In all 30 a read/query preceded a write; the gold results contained a later argument value absent from transcript/context/history, so these trajectories are observation-dependent by a concrete value-match heuristic. Five also contain `sys.ask_missing_info`. These are deliberately difficult selected examples, not estimates of class prevalence in the whole dataset.
- Existing PhoASR baseline failed to match the first gold action on all 30 and final state on all 30: 24 made a wrong first action, six emitted no calls, and 23 missed the gold first read/query. Paired subset metrics: cap5 proxy 0.0518, micro tool F1 0.1073, state match 0/30. Examples: `train_1265` requires lottery lookup and contact searches before transfers using returned contact IDs, but baseline emitted no calls; `train_0065` requires forecast → calendar lookup → conditional cancellation → contact lookup, while baseline diverted into weather and unrelated light calls.
- Re-ran the same 30 with caps 6 and 7. Neither changed state match (0/30), first-action exact match (0/30), or full-trajectory exact match (0/30). Cap6 proxy/F1 was 0.0512/0.1058; cap7 was 0.0512/0.1053. Each had 29 unchanged and one worsened versus cap5. Recorded mean latencies were 15.00s for the existing cap5 run, 12.40s at cap6, and 13.46s at cap7; run-to-run timing does not show a consistent latency effect. Ten cap7 runs used all seven turns and nine emitted seven calls, indicating a larger cap can permit over-calling without solving routing.
- Kept the production cap at five after this diagnostic; the cap extension is available as an evaluation option only. Architecture conclusion: **the required interactive loop exists; absence of observation feedback is not the missing piece**. Gold trajectories often need interaction, but failures occur before the correct first read/query, implicating Qwen tool selection/prompt behavior more than loop architecture. This hard 30-example subset does not establish why the public leaderboard differs from top teams; it does not support a cap change or public submission.

## First-action oracle diagnostics (2026-10-31)

- Reused exactly the same 30 hard holdout IDs and each sample's stored PhoASR public transcript. Baseline reference was the existing cap-5 Qwen prediction. No gold transcript went to the model. Oracle A constrained the first prompt to the gold tool schema/name but still asked Qwen to produce the ordinary action JSON and arguments; Oracle B executed only gold call 1; optional Oracle C executed only gold calls 1–2. In B/C, actual ToolEnv observations were passed to all subsequent baseline turns. One ToolEnv was constructed per sample per experiment; cap remained five.
- Baseline: first full action accuracy 0/30, micro tool-call F1 0.1073, final-state match 0/30, exact trajectory 0/30, proxy 0.0518.
- Oracle A (tool name only): first tool correct 30/30 by constraint, but first full call correct only 3/30. F1 0.1304, state 0/30, exact trajectory 0/30, proxy 0.0596. Paired proxy: 5 improved, 19 unchanged, 6 worsened.
- Oracle B (full first call): F1 0.3431, state 0/30, exact trajectory 0/30, proxy 0.1719. Paired proxy: 27 improved, 0 unchanged, 3 worsened. Relative to Oracle A, exact first arguments add +0.2127 tool-call F1 and +0.1122 proxy, while state match remains unchanged.
- Oracle B continuation first-error taxonomy: wrong second tool 24; ignored observation value 0; wrong later argument 2; premature STOP 1; extra call 0; turn cap 0; other 3. All 30 still missed final state. For example, `train_1265` receives the exact `tv.control(off)` call, then Qwen turns the TV back on instead of looking up lottery results. This is a policy/tool-selection error after the injected first observation.
- Oracle C (full first two calls): F1 0.5191, state 1/30, exact trajectory 0/30, proxy 0.2783. Paired proxy: 29 improved, 0 unchanged, 1 worsened. For `train_1265`, once the gold lottery query is also injected, Qwen correctly searches Bà Sáu but then calls `phone.call` with the returned contact ID instead of searching Thảo and making the requested transfers.
- Oracle A→B deltas are +0.2127 tool-call F1, +0.1122 proxy, and 0 final-state rate. Oracle B→C adds +0.1760 F1, +0.1065 proxy, and +0.0333 final-state rate.
- Interpretation: first-call argument grounding is a major bottleneck (name-only constraint yields just 3 exact calls; gold arguments add substantial F1). Multi-step policy is also a major bottleneck: after the exact first call, the model chose the wrong second tool in 24/30; even injecting two calls produced only 1 final-state match. The recommended training target is therefore the **next full action conditioned on transcript, history, schemas, and accumulated observations**, using iterative trajectories—not a standalone first-tool router. Do not treat these selected hard examples as a validation-wide or Public LB estimate; no full val309 rerun or public submission was made.

## PhoASR N-best hard-30 diagnostic (2026-11-01)

- Five decode configurations generated 5 candidates for each of the same 30 fixed hard IDs. Candidate strings were preserved as decoded; production greedy matched the established PhoASR cache on all 30. Duplicate rate was 56.7%; no empty strings or runtime failures.
- Current 1-best WER/CER was 0.3048/0.1778. Per-sample N-best oracle reached 0.2890/0.1652; 12 improved, 18 tied, none worsened. This is a modest lexical search-space gain.
- WER-oracle transcript selection improved micro tool F1 from 0.1073 to 0.1441 and proxy from 0.0518 to 0.0672, but final-state match and first full-call accuracy remained 0/30. The deterministic consensus medoid did not improve proxy. The better candidate set therefore has measurable call-level upside under gold selection, but the simple no-gold selector did not find it reliably.
- Paired proxy outcomes under WER-oracle selection were 6 improved, 5 worsened, and 19 unchanged. Semantically important differences include recovering “khóa lại” for a garage door (`train_0571`); remaining confusions include payment intent, negation/conditionals, names, and organization terms. A small WER gain does not guarantee improved tool trajectories.
- Downstream-oracle selection across all alternatives was skipped to honor the exact-one-ToolEnv-per-sample competition constraint. Most visible source errors remain intent/negation and entity/relation wording (e.g. “huỷ lịch” vs unrelated words, “gọi báo cho Minh” vs “đừng có báo cho mình”); the N-best examples do not establish that these are consistently recoverable.
- Conclusion: candidate generation has a small but real WER headroom; simple consensus selection is not useful, and no evidence supports a production reranker/fusion yet. Qwen action selection, argument grounding, and multi-step policy remain the primary bottleneck on this hard subset.

## QUAN continuation and CoT screen (2026-11-07)

- Two lower-LR QUAN continuation epochs slightly worsened corpus WER and CER on the same val309 IDs; early stopping retained the existing checkpoint. The extra training did not establish a better ASR candidate, so downstream A/B was a repeated-control check, not a base-vs-fine-tuned comparison.
- On the fixed enriched cohort60, identical A/B first-request payload hashes still produced differing calls/state hashes on 26/60 samples. This confirms meaningful run-to-run output variance under the current Ollama setup; near-zero paired BTC differences should not drive a model or checkpoint decision.
- CoT's BTC point estimate was +0.39 over A, but the 95% paired interval `[−7.44,+8.29]` spans zero widely. The local 30-second diagnostic moved −3.89 points, samples over 30 seconds rose 30→36, cap-hit requests 19→23, and generated tokens including thinking rose 8%. Keep the existing prompt; this screen does not justify CoT promotion.
- The cohort is enriched and local RTX 4050 timing is not an A30 benchmark. See `exp/quan-continuation-cot-v1/output/summary.md` for exact scoring, paired controls, and artifacts.

## PhoASR N-best expansion — 100 additional validation IDs (2026-11-01)

- The cohort was fixed as the first 100 IDs in validation order after removing the prior hard-30, avoiding any selection based on observed outcomes. Production WER/CER was 0.3638/0.2492; N-best oracle reached 0.3364/0.2279. Oracle WER improved 29/100, tied 71, worsened none.
- Downstream 1-best vs WER oracle: tool-call F1 0.1657→0.1671, final-state 0.31→0.32, proxy 0.2464→0.2567. Paired proxy changed on only four samples (2 improved, 2 worsened); 96 tied. Consensus proxy was 0.2542, with F1 below baseline (0.1605) and the same 0.32 state match.
- This broader result confirms PhoASR's candidate set contains useful lexical alternatives, but even gold WER ranking brings little downstream improvement. Consensus remains non-beneficial. The evidence favors focusing on Qwen policy and argument grounding; an ASR reranker is not a priority.
- Applying the supplied BTC scoring formula directly on the same 100 samples gives mean final scores of 6.107 current, 7.274 WER oracle, and 7.274 consensus (0–100 scale). The gain is concentrated in two samples, with 98 unchanged; hash match and at least one exact matched call are both required for a positive per-sample score.

## Track C PhoASR FT vs QUAN FT pilot (2026-11-04)

- On the same fixed stratified pilot60, QUAN FT + KenLM beam32 reduced corpus WER/CER from 0.2597/0.1945 to 0.1786/0.1522. BTC formula score before runtime mask rose from 10.000 to 15.730/100, but paired bootstrap difference +5.730 points had a wide 95% CI [−1.444, +13.405]; 8 improved, 4 worsened and 48 tied.
- The pilot meets the point-estimate threshold, but not the CI criterion. QUAN had slightly fewer >30s samples (28 vs 33), with similar mean latency (~32.35s); parse/schema repair failures (32 vs 33), runtime exceptions (0/0) and ToolEnv schema rejections (0/0) were comparable. This does not establish a reliable downstream gain or full-val effect.
- ASR metrics do not fully determine downstream behavior: a large WER/CER improvement coincided with only eight BTC improvements and four regressions. Inspect per-sample number/unit, time, negation, entity, tool-name/argument, early-stop and over-call fields in `exp/qwen35-track-c-quan-vs-phoasr-val309/output/pilot60/paired-analysis.json` before a larger run. Keep the default transcript unchanged until a larger paired evaluation resolves the uncertainty.
- Full val309 was not run because the pre-run Track C estimate exceeded four hours (about 5.47h for two branches). These are pilot-only results, not Public LB or A30 runtime estimates.
- Concrete paired cases include `train_1144`, where QUAN's currency/amount transcript accompanied the two gold `exchange_rate.convert` calls while PhoASR led to five `exchange_rate.get_rate` calls, and `train_0241`, where QUAN produced the gold weather/calendar/cancel sequence while PhoASR created a calendar event. `train_0013` regressed when QUAN produced extra contacts/SMS calls beyond the gold weather call. Treat these as observed paired examples; the transcripts alone do not establish causality.

## Full Track C error analysis — QUAN FT + Qwen3.5 thinking/JSON (2026-11-05)

- Fixed train-only val309, one prediction ToolEnv/sample, exact name+arguments and duplicate-multiplicity matching, hash gate, gold trajectory replay through ToolEnv. BTC formula was **12.9851/100 before the separate 30-second gate**; after applying the competition timeout zero rule it was **7.3571/100**. 114/309 hashes matched; M/U/G=233/293/734; aggregate call F1 diagnostic 0.3698. The 15 exact no-action/hash-match samples still score zero under BTC's M>0 requirement.
- Generation/runtime dominates this local run: 147/309 samples hit the `num_predict=512` length limit during a failed JSON repair; 23 more had invalid action shape. No ToolEnv schema rejections or runtime exceptions. 177 samples exceeded 30s; mean/p50/p95/max on RTX 4050 were 35.35/33.23/64.65/104.85s. Do not treat local timing as an A30 benchmark.
- No missing-info calls were predicted despite 47 gold `sys.ask_missing_info` cases. Argument mismatch and under/over-call errors remain substantial. Exact transcript token checks (numeric 60.2%, temporal markers 77.3%, negation 96.4%) are diagnostic only; WER/slot correlation does not establish action causality.
- Reproduction check against the earlier same-config pilot60: ordered tool calls and state hashes matched 60/60; formula-score delta was exactly zero. The small runtime-gated paired change is timing-only.
- Follow-up priority: isolate generation-budget vs native protocol because this run has many cap failures and prior Qwen3.5 protocol ablation showed fewer repairs with native calls; then test missing-info behavior; audit post-observation next-action errors; use oracle transcripts only in clearly marked diagnostic branches. Prior retrieval examples did not reduce missing-info errors, cap 5/6/7 did not fix tool routing, and the previous QLoRA next-action run scored below baseline, so these are not recommended as automatic fixes.
- Full tables, examples, uncertainty labels, paired pilot comparison, limits, and GO/NO-GO criteria are in `exp/current-config-error-analysis-val309/output/summary.md`.

## `num_predict` 512 vs 768 pilot (2026-11-05)

- On an enriched fixed pilot60, increasing the cap reduced the sample count that hit a generation cap or needed JSON repair from 39 to 18 (53.8%). BTC formula point estimate moved from 14.5820 to 17.5159/100 (+2.9339 points), but paired bootstrap CI [−0.7407,+7.2725] includes zero and only 4 samples improved, 2 worsened, 54 tied. This supports a mechanism effect in the selected cohort, not a val309 score claim.
- The instrumented 512 replay did not reproduce the archived 512 artifact: 34 ordered call differences, 33 call-multiset differences, 20 hash differences out of 60, with transcripts matching 60/60. Source, prompt, model digest/version and request settings were checked; the output variance remains unexplained. Treat single-run A/B gains cautiously and prioritize resolving reproducibility before broad confirmation.
- No context-limited requests reached `num_ctx=8192`; all `done_reason=length` requests also reached the configured `num_predict`. Component estimates (UTF-8 characters/4) undercounted Ollama's authoritative prompt count by about 970 tokens at the median, so retain them only as rough relative breakdowns. Thinking remains part of the shared `eval_count`; do not interpret extra generated tokens as a separate reasoning budget.
- Keep the production default at 512 for now. The pilot cohort was error-enriched, the paired interval crossed zero, and the pre-run full-768 estimate was 4.55h, above the task's four-hour cap. The separate post-run projection of 2.60h does not change the screening-only decision.

## Qwen3.5 thinking generation budget pilot60 (2026-11-06)

- `num_predict` is the total generation cap for each request, including thinking and JSON answer; multiple ReAct requests can occur per sample. Across caps 512/768/1024/1536, the pilot BTC formula means were 16.857/14.460/15.905/14.516 on an enriched train-only cohort. Paired intervals were wide and all included zero; these are not val309 estimates.
- Raising the cap reduced actual request cap hits from 31 samples at 512 to 3 at 1536, while 1536 had 4/172 requests reach the cap and one separate context-limited candidate. The paired 1536−1024 score trend was negative (−1.389 points, 95% CI [−6.111,+2.778]); there was no reason to open 2048.
- A same-setting 512 repeat had identical initial request payload hashes for all 60 IDs but differed from the primary 512 run on 17 ordered call sequences and 11 state hashes. It matched the archived pilot exactly, so the primary output is not reproducible under the observed runtime. The quality-selection gate failed; do not pick a cap from the point estimates until this output variance is understood and the controlled comparison is rerun.
- Prompt/model/version fingerprints matched the prior pilot, but the `agent.py` source hash differs and old per-request telemetry did not record `num_batch`. The new requests explicitly used 512. Do not infer a single cause from these differences; request hashes show identical initial inputs, while generation output still diverged.
- The initial runner repeated flushed prefixes in artifact files; all duplicates were exact copies, archived, and deduplicated by ID. This was an artifact-writing issue, not repeated ToolEnv inference. Keep the recovery audit with the experiment.
- Full detail and uncertainty: `exp/qwen35-thinking-budget-val60/output/summary.md` and `output/reproducibility.json`. RTX 4050 timings/GPU readings are local diagnostics only; no A30 compliance claim follows.


## Qwen3.5 SFT downstream screen — 2026-11-07

- On the fixed enriched cohort60, the official BTC point estimates were base/no-CoT A 14.97, SFT/no-CoT B 22.84, and SFT+CoT C 26.00. SFT-only paired delta B−A was +7.87 points (95% bootstrap CI [−1.40,+17.37]); adding CoT to SFT C−B was +3.16 [−4.03,+10.74]. Both intervals include zero, so this screen does not establish either isolated effect. C−A was +11.03 [3.22,19.69], which is a combined change and cannot attribute the gain.
- The fixed 20-anchor A repeat exactly reproduced initial payload hashes, ordered calls, ToolEnv state hashes and final states (20/20). This run was reproducible on these anchors under the locked setup; generalize only after larger evaluation.
- B/C were faster than A on local RTX 4050 and had fewer >30s samples, but local timings are diagnostics, not A30 compliance. Keep the SFT+CoT branch as a candidate for locked full-val309 evaluation; do not promote based only on this enriched screen or submit it.
- Training data used a Vietnamese prompt while the benchmark uses the English production prompt, and the remote training report has no input-weight hash. Both remain interpretation limits.

## Qwen3.5 SFT epoch-3 continuation screen (2026-11-08)

- On the same enriched cohort60, the continued adapter + CoT scored 23.28 BTC before local runtime gating, versus 26.00 for the reused epoch-2 C result. Paired difference −2.72 points, 95% CI [−7.72,+1.33]; 4 improved, 5 worsened, 51 tied. The interval crosses zero and the point estimate is lower; do not promote the extra epoch from this screen.
- Exact calls rose 47→51 matched and unmatched calls fell 70→64, but hash matches fell 35→32; per-sample BTC also requires state-hash match. Local 30s-gated scores were 23.67→23.28, also uncertain. Local run timing is not an A30 result.
- The archive confirms continuation from epoch2 and the SFT data hash, but not the exact remote base-weight bytes. The reused epoch2 result also means there is no new control-repeat variance estimate. See `exp/qwen35-sft-epoch3-cot-val60/output/summary.md` and its paired per-ID file.

## Qwen3.5 SFT epoch-3 + CoT public-test candidate (2026-11-08)

- The local candidate passed the official validator: 634 rows, 634 unique IDs in manifest order, zero schema/order errors, zero inference runtime errors, exactly one ToolEnv per sample. It was not submitted to Public LB, and no public/private labels were used.
- Local RTX 4050 mean latency was 24.09s and 177/634 samples exceeded 30s; p95 was 53.90s. This does not demonstrate A30 compliance, so runtime remains the main deployment uncertainty. 63/2,181 requests reached the 768-token cap.
- The public-test artifact has no score absent labeled evaluation. The 60-ID enriched screen scored 23.28 BTC before runtime gating, below reused epoch-2 C at 26.00 with a paired CI crossing zero; do not infer a public-test gain. Candidate hashes and validator result are in `exp/qwen35-sft-epoch3-cot-val60/output/submission/summary.json`.
## Qwen3.5 SFT + CoT branch C public-test candidate (2026-11-07)

- The full candidate is schema-valid and ordered: 634 rows, 634 unique public-test IDs, zero schema/order errors. This confirms packaging and ToolEnv output shape only; public-test labels were not used, so there is no quality score from this run.
- One request on `public_0527` failed with CUDA out of memory. Keep the recorded error visible; do not describe this candidate as runtime-failure-free. It was produced locally on RTX 4050, so its 158 samples above 30 seconds and measured latency distribution cannot establish the official A30 timeout rate.
- The generation pipeline did not submit to the Public LB. The user later reported a Public LB score of **39.16** for this candidate; this score was not independently verified. Candidate and predictions sidecar hashes, model/prompt/ASR locks, and per-sample ToolEnv audit are recorded in `exp/qwen35-sft-c-public-submission/output/summary.json` and `submission-lock.json`.
## QUAN transcript-normalization spelling-only fallback (2026-11-09)

- The requested fixed cohort60 was confirmed against both existing manifests and its newline-joined ID hash. Native Ollama 0.40.15 was reachable from system Python/PowerShell, outside the project `.venv`; the user’s environment note did not mean inference should be abandoned.
- A safe numeric parser was not available, so the benchmark was split to the explicitly permitted spelling-only fallback. The frozen whole-token NFC mapping changed 0/309 hypotheses and 0/60 cohort transcripts; all other treatment inputs were identical.
- On identical A/B prompts and transcripts, official BTC was 23.22 vs 26.00; paired +2.78 points, 95% CI [−0.28,+7.22], 56 tied / 3 improved / 1 worsened. A-repeat scored 30.67 on its fixed 20 anchors; A-repeat−A was +2.50 [0.00,+7.50], while A-repeat−B was 0.00 [0.00,0.00]. First-request payload hashes matched on every A/B sample, yet outputs differed on 7 samples, showing material run variance. Do not attribute the A/B point difference to spelling normalization.
- All gold ToolEnv replays passed, no inference runtime exceptions or schema rejections occurred, and no raw CoT was stored. RTX 4050 local 30-second masks are diagnostic only, not A30 compliance. No submission/Public LB was used.
- This does not complete the requested numeric-plus-spelling treatment. The full task remains pending a safely scoped numeric parser and a separately locked confirmatory run; do not promote from this screening fallback. Details: `exp/quan-transcript-normalization-val60/output/inference/summary.md`.

## QUAN train-map correction on held-out val60 (2026-11-10)

- On the fixed enriched val60 screen, number parsing ran before map matching and normalized 35 QUAN spans. The train-only map applied 28 candidate corrections across 22 tasks; none matched the BTC reference at the aligned span (0/28 exact precision). Number-value WER rose from 0.1540 raw to 0.1784 with the map.
- With the same Qwen SFT epoch-2 + CoT setup, BTC formula was 23.29 raw QUAN, 19.83 after number normalization + map, and 32.33 with the BTC script as oracle input. B−A was −3.45 points (95% paired CI [−10.00,+1.83], 3 improved / 3 worsened / 54 tied). This is a diagnostic enriched cohort; the BTC interval includes zero, while the map's held-out exact correction precision is 0%.
- Treat the medium/high candidate map as unsafe for global replacement on this screen. Details and per-application evidence: exp/quan-map-correction-val60-v1/output/benchmark/summary.md.


## QUAN-transcribe epoch-02 continuation on val60 (2026-11-10)

- The raw two-epoch continuation transcript scored project WER 0.1795 and number-value WER 0.1569 on the fixed enriched 60-ID screen; downstream BTC formula score was 24.62/100, with zero inference runtime errors and zero gold replay failures.
- Compared with raw QUAN epoch-06 branch A, D changed downstream BTC by +1.33 points (paired 95% CI [-2.17,+5.83]). This single-run enriched screen is diagnostic; do not infer a general epoch effect from it.
- D used the locked Qwen SFT epoch-2 + CoT model/prompt and raw epoch-02 QUAN transcript, with no map. Details: exp/quan-map-correction-val60-v1/output/benchmark/summary.md.


## QUAN transcript map comparison on full val309 (2026-11-10)

- Ran all four transcript inputs on all 309 held-out validation IDs using the same Qwen SFT epoch-2 + CoT setup. A/B use epoch-06 QUAN transcripts (B normalizes Vietnamese number phrases before the frozen train-only map), C uses the BTC-provided reference script as an oracle, and D uses the raw QUAN-transcribe epoch-02 continuation.
- Full metrics, map audit and paired bootstrap comparisons are recorded at exp/quan-map-correction-val309-v1/output/benchmark/summary.md. On this full split, map precision was 3/192 (1.6%); paired D−A BTC difference was -1.36 points (95% CI [-4.09,+1.31]).
- Validation labels were used for evaluation/reporting only; train-only map rules were frozen before the run. No submission was produced.
- The Vietnamese cardinal parser composes tens as tens-unit × 10 + units, then handles hundreds and larger scale words recursively; it converts complete word spans to digits before matching the frozen map. It normalized 232 transcript spans on full val309.
- The combined parser+map branch changed 140 utterances and only 3/192 map applications exactly matched the aligned reference (1.6%). B had worse Project WER (19.77% vs A 16.20%), number-value WER (16.95% vs 14.54%), and BTC formula score (29.80 vs 31.59); do not use this map globally without rebuilding it.
- Ollama was 0.40.15 on rounds 1–10 and 0.40.16 on rounds 11–16; model digest and prompt were unchanged and all four branches within each ID block used the same version. train_1725 exceeded the 8,192-token context in all branches and received zero BTC sample score.

## Qwen SFT epoch 2 vs epoch 3 English + PhoWhisper large v2 (2026-11-11)

- Trên mẫu ngẫu nhiên cố định 80/309 task, giữ PhoWhisper large v2 cho audio chính và history, prompt English + CoT, ToolEnv và cấu hình suy luận như nhau. BTC sau ngưỡng local 30 giây: epoch 2 **38.82**, epoch 3 English **43.78**; trước ngưỡng lần lượt **42.22** và **50.23**.
- Chênh lệch ghép cặp epoch 3−epoch 2 **+4.96 điểm** (bootstrap 95% CI **[−3.46,+13.48]**), 16 task nghiêng về epoch 3, 9 về epoch 2, 55 hòa. CI chứa 0 nên chưa đủ bằng chứng đổi sang epoch 3.
- Epoch 3 có 57 hash trạng thái khớp so với 53, nhưng latency trung bình cao hơn (22.32 vs 19.31 giây) và số task local vượt 30 giây nhiều hơn (15 vs 9). Cả hai nhánh có 0 lỗi runtime.
- Mẫu 80 task là screening; ngưỡng 30 giây đo trên máy local, không chứng minh thời gian trên A30. PhoWhisper large v2 có thể có chồng lấn với dữ liệu validation. Chi tiết: exp/qwen-epoch2-vs-epoch3en-phowhisper-val80/output/summary.md.

## Ask-missing-info prompt audit on val309 (2026-11-11)

- Post-hoc classification of the 47 gold ask tasks found 24 person/conversation IDs, 13 other object IDs, and 10 genuinely absent or underspecified user values. Exact gold asks were baseline 10/47 vs replacement prompt 1/47.
- Contact lookup is the clearest recurring subproblem: contacts.search returned multiple candidates in 20/21 baseline searches and 18/19 variant searches among 23 contact_id tasks. Exact asks were 7/23 vs 1/23. A narrow rule should condition on lookup cardinality and context, rather than merely say to ask when a field is missing.
- Several object IDs are resolvable from a descriptive reference or user criterion, while others are truly ambiguous (e.g. one of two deliveries, one of four orders). Do not ask for internal IDs when lookup can uniquely resolve the intended object. ASR can separately erase the name needed for lookup.
- The broad prompt replacement had no reliable BTC gain: local gated +0.07 points (95% CI −3.07,+3.08); ungated −1.00 (−4.19,+2.36). Do not promote it. A contact-ambiguity rule remains a hypothesis for a fresh-cohort test; the 47 audit cases are diagnostic only. Full taxonomy: exp/ask-ambiguity-prompt-val309/output/ask_error_taxonomy.md.


## Post-lookup ambiguity prompt follow-up (2026-11-11)

- Thử câu lệnh hẹp: chỉ dùng ID khi yêu cầu và lịch sử xác định một ứng viên; nếu còn nhiều ứng viên hợp lý thì hỏi đúng trường ID chưa rõ. So với prompt baseline trên toàn val309, BTC sau ngưỡng local 30 giây là 30.08 vs 28.69; chênh lệch ghép cặp +1.39 điểm (95% CI [−1.57,+4.39]), chưa chứng minh được cải thiện.
- Tập xác nhận 262 task ngoài 47 task dùng để hình thành giả thuyết có BTC 30.31 vs 27.84, Δ +2.46 [−0.70,+5.55]; tập này không có gold ask nên chỉ đo ảnh hưởng phụ và câu hỏi thừa. Trên 47 task chẩn đoán, exact gold ask giảm từ 10 xuống 6 và BTC đổi −4.59 [−12.90,+3.44]. Không nên thay prompt hiện tại dựa trên kết quả này.
- Cả nhánh baseline và focused có một lỗi vượt context ở `train_1725` (8,250 > 8,192 token), được tính 0 qua local gate. Baseline dùng lại prediction đã khóa; không tạo submission. Báo cáo: `exp/ask-lookup-disambiguation-followup-val309/output/summary.md`.


## Focused prompt: QUAN/PhoASR versus PhoWhisper Large on val309 (2026-11-11)

- Đính chính nguồn của benchmark prompt trước: nhánh đạt 30.08 dùng QUAN checkpoint-verification-rerun cho transcript chính và PhoASRModel cho history. Giữ nguyên Qwen epoch 3, prompt focused, ToolEnv và decoding; thay ASR cả audio chính lẫn history bằng PhoWhisper Large v2 thì BTC local-gated đạt 42.71, chênh **+12.62 điểm** (paired bootstrap 95% CI **[+7.68,+17.46]**, 82 thắng / 33 thua / 194 hòa).
- Trên 262 task ngoài nhóm gold-ask, PhoWhisper hơn **+13.19** điểm [ +7.90,+18.43 ]. Trong 47 task chẩn đoán, exact ask tăng từ **6/47 lên 14/47**; BTC chênh +9.46 có CI sau local gate [−2.31,+21.89]. Đây là chẩn đoán vì 47 task đã được dùng để hình thành giả thuyết prompt.
- Cả hai nhánh có một task vượt context 8,192 token (`train_1725`; PhoWhisper 8,253 token). PhoWhisper nhánh chạy một lần; baseline ghép cặp dùng lại prediction đã khóa. Khoảng tin cậy không tính biến thiên giữa các lần chạy. PhoWhisper validation có thể trùng dữ liệu huấn luyện; ngưỡng 30 giây là thời gian local, không chứng minh A30. Không tạo public submission.
- Chi tiết: `exp/ask-lookup-phowhisper-asr-val309/output/summary.md`; per-ID và lock chứa trong cùng thư mục output.


## PhoWhisper Large + Qwen epoch 3 focused-prompt public candidate (2026-11-11)

- Generated `outputs/submissions/phowhisper-large-v2-qwen35-epoch3-en-cot-focused-public-test.jsonl` from 1,933 cached PhoWhisper Large V2 main/history transcripts, Qwen epoch-3 English CoT, and the focused post-lookup prompt. The candidate has 634/634 unique IDs, passes official schema validation, and has 0 runtime errors.
- Local Qwen + ToolEnv mean was 21.53s/sample (130 above 30s); estimated combined ASR + Qwen mean 27.14s (230 above 30s). These are local timing measurements, not A30 measurements. The setup scored 42.71 BTC on val309 in the paired comparison. User subsequently submitted the candidate and reported Public LB score **47.320** on 2026-11-11.
- Manifest and hashes: `outputs/submissions/phowhisper-large-v2-qwen35-epoch3-en-cot-focused-public-test_manifest.json`; predictions sidecar: `outputs/submissions/phowhisper-large-v2-qwen35-epoch3-en-cot-focused-public-test_predictions.jsonl`.


## Public LB score update: PhoWhisper Large focused-prompt candidate (2026-11-11)

- User reported Public LB score **47.320** for `outputs/submissions/phowhisper-large-v2-qwen35-epoch3-en-cot-focused-public-test.jsonl`.
- The score is marked user-reported in the candidate manifest; submission and prediction hashes are unchanged.
