# Inference latency optimization, fixed pilot60

Screen sequential Ollama latency for the existing enriched, train-only 60-ID cohort. This is a local RTX 4050 diagnostic; it is not a val309 estimate or an A30 runtime benchmark. The locked baseline is Qwen3.5 4B Q4_K_M, Ollama 0.40.14, English observation-grounded prompt, thinking on, JSON, temperature 0, `num_predict=768`, `num_ctx=8192`, five ToolEnv action turns, and cached QUAN transcripts with PhoASR history support.

## Entry points and artifacts

- `scripts/run_pilot.py --branch baseline` runs the reproducibility/latency baseline, including one excluded warm-up request.
- `scripts/run_pilot.py --branch batch-pair` runs the balanced 512/1024 `num_batch` crossover only after baseline and server-state review.
- `output/` stores the copied cohort/config manifests, per-request Ollama telemetry, per-sample predictions and traces, GPU samples, paired scoring, and decision notes.
- Request telemetry stores payload hashes and Ollama counters; it never stores raw thinking.

Each measured branch uses exactly one fresh prediction ToolEnv per cohort sample. Gold replay is separate and follows the project scorer. No `*_environment/` files are read. No full val309, test, submission, or fine-tuning run is in scope.

## Result

Completed the baseline and explicit `num_batch` 512/1024 crossover. At experiment time, the current server default was 1024; explicit512 was about 6.5% faster at the sample median and 6.9% faster at p95 than that baseline, and exactly reproduced the archived 768 pilot calls/state hashes. Paired intervals still include zero. In a later user-directed change, the application Ollama request default was set explicitly to 512; `LIFEMATE_OLLAMA_NUM_BATCH=1024` preserves the comparison baseline. GPU placement, Flash Attention/KV, and keep-alive comparisons were skipped at their documented gates. Full metrics, BTC score guardrails, reproducibility evidence, and limitations are in `output/summary.md`.
