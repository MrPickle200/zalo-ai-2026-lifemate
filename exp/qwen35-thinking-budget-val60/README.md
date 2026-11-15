# Qwen3.5 thinking budget — fixed pilot60

Controlled screening experiment for choosing the total Ollama generation cap (`num_predict`) before fine-tuning. Uses the existing enriched, train-only 60-ID validation cohort. It is not an unbiased estimate of val309.

## Locked configuration

- Model `qwen3.5:4b-q4_K_M`, thinking enabled, JSON output, temperature 0.
- `num_ctx=8192`, explicit `num_batch=512` in every measured and warm-up request, and at most five action turns.
- English `observation-grounded-v2` prompt; QUAN FT + KenLM beam32 cached current transcripts; PhoASR for conversation history; same parser, repair policy, schemas, ToolEnv, seed 17, and source as the existing pilot pipeline.
- Caps 512, 768, 1024, and 1536. The 2048 branch is conditional on the pre-registered criteria in `TASK.md`.
- Four rounds of 15 cohort IDs use a balanced rotating cap order. One excluded warm-up is issued before each cap/round session. Runs stop on the first runtime error and preserve completed prefixes.
- The runner uses native Python outside `.venv` to reach the native Windows Ollama server. No raw model thinking is saved.

## Entry points

- `scripts/run_pilot.py --only-preflight`: verifies the fixed cohort, transcript/sample coverage, Ollama version, model digest, and writes input/config fingerprints.
- `scripts/run_pilot.py`: executes all four registered branches sequentially.
- `scripts/analyze_pilot.py`: replays gold trajectories with ToolEnv, calculates the BTC formula and paired bootstrap diagnostics, and writes `output/summary.md`.
- `scripts/repeat_control_512.py`: repeats the 512 control on the same IDs after a reproducibility mismatch.
- `scripts/recover_prefix_duplicates.py`: audits and recovers duplicated flush prefixes, preserving raw files.
- `scripts/sample_gpu.py`: records 5-second local GPU telemetry while an inference session runs.
- `scripts/finalize_report.py`: records control reproducibility, GPU summaries, per-cap run configs, and the final cap decision.

## Artifacts

`output/` contains cohort/config manifests, balanced execution order, per-round warm-ups, per-cap run metadata, predictions, traces, per-request Ollama telemetry (including payload SHA-256 and prompt estimates), ToolEnv BTC per-sample results, paired comparisons, and the summary. No submission is generated.

The initial runner flushed prior samples in each in-memory block prefix more than once. The complete run was preserved; identical duplicate rows were audited, the original files retained with `-raw-prefix-flush.jsonl`, and canonical files deduplicated by sample/request ID. The runner was corrected. A repeated same-setting control found output divergence, so quality cap selection is intentionally withheld; see `output/reproducibility.json` and `output/summary.md`.

Legacy 512/768 prediction artifacts were not reused because they did not provide sufficient verifiable evidence for the required explicit `num_batch=512` request setting. The new 512 run is also checked against the archived pilot for reproducibility before interpreting quality differences.
