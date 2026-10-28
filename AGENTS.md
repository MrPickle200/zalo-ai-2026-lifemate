# Agent instructions

## Project goal
Build a Vietnamese voice agent for Zalo AI Challenge 2026 — LifeMate that turns speech into tool calls in the simulated environment.

## Competition rules
Read `COMPETITION.md` before decisions about models, inference architecture, data access, ToolEnv, runtime, or submission format. It is the source of truth; do not duplicate the full rules here.

## Development principles
Prioritize correctness, reproducibility, a simple end-to-end pipeline, and local evaluation, in that order before optimizing models, latency, or accuracy. Keep the baseline simple and avoid over-engineering before it runs.

## External research
When implementation ideas are needed, consult relevant GitHub repositories and Kaggle notebooks/solutions. For Kaggle use the `nvidia-kaggle-skill`. Check license and compatibility before adapting code. Useful topics include Vietnamese ASR, tool/function calling, structured generation, lightweight inference, and agent evaluation.

## Project structure
- `configs/`: baseline configuration.
- `src/lifemate/`: ASR, agent, validation, ToolEnv loop, and pipeline.
- `scripts/`: sample runner, local evaluation, dataset inspection, and submission writer.
- `cache/transcripts/`: reusable transcripts keyed by audio content.
- `outputs/eval/` and `outputs/submissions/`: generated local artifacts (ignored by git).
- `tests/`: focused regression checks.
- `train+public_test/`: competition data and official `zaic_tools` packages; do not inspect or read any `*_environment/` file directly. Use ToolEnv only.

