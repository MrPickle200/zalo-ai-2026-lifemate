# current-config-error-analysis-val309

Completed full fixed train-only val309 error analysis for the QUAN FT + Qwen3.5 Track C configuration. The inference run was paused/resumed after 187 samples and finished all 309 IDs without duplicating predictions. Gold trajectories were replayed through ToolEnv and scored.

## Scripts

- `scripts/run_val309.py` runs or resumes the exact ordered val309 prefix and appends only missing IDs.
- `scripts/compare_pilot60.py` compares the 60 overlapping samples with the previous same-configuration pilot.
- `scripts/analyze_val309.py` writes the per-sample table, cohort breakdowns, slot metrics, and error taxonomy.
- `scripts/start_ollama_native.ps1` starts native Windows Ollama outside `.venv` and verifies the local model tag.

## Results and outputs

- Formula BTC before the separate 30s runtime gate: **12.9851/100**; after the competition timeout rule: **7.3571/100**.
- Predictions/traces: 309/309. Gold replay failures, runtime exceptions, schema rejections: 0. Samples over 30s on local RTX 4050: 177 (not an A30 benchmark).
- `output/summary.md`
- `output/config_manifest.json`
- `output/predictions.jsonl`, `output/step-traces.jsonl`
- `output/btc-scores.jsonl`, `output/official_summary.json`
- `output/per_sample_error_table.csv`, `output/per_sample_error_table.jsonl`
- `output/error_taxonomy.json`, `output/error_analysis_summary.json`
- `output/paired_pilot60_comparison.json`
- `output/transcript_cache_val309.jsonl`, `output/run_config.json`
