# `num_predict=512` vs `768` on fixed validation data

Controlled Ollama experiment for the current QUAN FT + KenLM beam32, Qwen3.5 4B, English JSON-prompt configuration. Only the per-request `num_predict` setting changes. The primary run is a fixed paired diagnostic pilot of 60 train-only validation IDs; it is enriched for prior truncation/repair failures and is not an unbiased val309 estimate.

## Scripts

- `scripts/run_branch.py`: sequentially runs one branch with one prediction ToolEnv per sample and opt-in request telemetry. It halts on an archived-baseline mismatch by default; `--continue-after-baseline-divergence` is available only after recording an investigation, as done for this pilot.
- `scripts/analyze_pilot.py`: replays each gold trajectory through ToolEnv, computes the official BTC formula, paired bootstrap, output differences, latency and request token summaries.

## Artifacts

All experiment data is under `output/`: fixed cohort and input hashes, configs, predictions, traces, `token_usage.jsonl`, prompt component estimates, per-sample BTC scores, paired comparison, and `summary.md`. Raw thinking and model prompt payloads are not stored; payload SHA-256 and size estimates are stored instead.

Prompt-component sizes use UTF-8 character count divided by four as an approximate tokenizer because a matching Ollama tokenizer/template counter is not exposed locally. `prompt_eval_count` is authoritative. Any full-val 768 run is a separate, gated phase and must not be launched unless the pilot criteria in the root `TASK.md` pass.

## Pilot outcome (2026-11-05)

- Fixed enriched pilot60 completed sequentially for both branches; exactly one prediction ToolEnv per sample/branch. Each branch has 60 predictions, 60 traces, zero runtime exceptions, and zero ToolEnv schema rejections. Gold replay had zero failures.
- BTC formula before any local runtime mask: 512 **14.5820/100**, 768 **17.5159/100**, paired difference **+2.9339** (95% bootstrap CI **[−0.7407, +7.2725]** points); 4 improved, 2 worsened, 54 tied. Matching uses exact tool name+arguments, duplicate multiplicity, gold replay hash gate, and `M>0`.
- Samples hitting token cap or requiring JSON repair fell from 39 to 18 (53.8%). This passes the pilot's mechanism screen and the point estimate did not decline, but the CI includes zero and the cohort is enriched, so this is not a val309 effect estimate.
- The instrumented 512 run differed from the archived baseline on ordered calls for 34/60 IDs, call multisets for 33/60, and state hashes for 20/60 despite matching all 60 current transcripts and matching source/prompt/model/version/config fingerprints. The cause remains unresolved; paired results are screening evidence only.
- No 768 full-val run was made. The pre-run estimate was 4.55 hours, over the 4-hour task limit, and the baseline reproducibility gate failed. The observed pilot B mean latency projects to 2.60 hours for 309 samples, but this post-run projection does not replace the pre-run estimate required by the task.
- Full per-request counters, component estimates, exact request payload SHA-256, sample-level scores, paired outcomes, timing, context headroom, and decision rationale are under `output/summary.md` and `output/paired_comparison.json`. Thinking content is never stored.

## Pilot outcome (2026-11-05)

- Fixed enriched pilot60 completed sequentially for both branches; exactly one prediction ToolEnv per sample/branch. Each branch has 60 predictions, 60 traces, zero runtime exceptions, and zero ToolEnv schema rejections. Gold replay had zero failures.
- BTC formula before any local runtime mask: 512 **14.5820/100**, 768 **17.5159/100**, paired difference **+2.9339** (95% bootstrap CI **[−0.7407, +7.2725]** points); 4 improved, 2 worsened, 54 tied. Matching uses exact tool name+arguments, duplicate multiplicity, gold replay hash gate, and `M>0`.
- Samples hitting token cap or requiring JSON repair fell from 39 to 18 (53.8%). This passes the pilot's mechanism screen and the point estimate did not decline, but the CI includes zero and the cohort is enriched, so this is not a val309 effect estimate.
- The instrumented 512 run differed from the archived baseline on ordered calls for 34/60 IDs, call multisets for 33/60, and state hashes for 20/60 despite matching all 60 current transcripts and matching source/prompt/model/version/config fingerprints. The cause remains unresolved; paired results are screening evidence only.
- No 768 full-val run was made. The pre-run estimate was 4.55 hours, over the 4-hour task limit, and the baseline reproducibility gate failed. The observed pilot B mean latency projects to 2.60 hours for 309 samples, but this post-run projection does not replace the pre-run estimate required by the task.
- Full per-request counters, component estimates, exact request payload SHA-256, sample-level scores, paired outcomes, timing, context headroom, and decision rationale are under `output/summary.md` and `output/paired_comparison.json`. Thinking content is never stored.
