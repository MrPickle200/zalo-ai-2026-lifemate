# Agent instructions

## Project goal
Build a Vietnamese voice agent for Zalo AI Challenge 2026 — LifeMate that turns speech into tool calls in the simulated environment.

## Competition rules
Read `COMPETITION.md` before decisions about models, inference architecture, data access, ToolEnv, runtime, or submission format. It is the source of truth; do not duplicate the full rules here.

## Development principles
Prioritize correctness, reproducibility, a simple end-to-end pipeline, and local evaluation, in that order before optimizing models, latency, or accuracy. Keep the baseline simple and avoid over-engineering before it runs.

## Official scoring rule
Use the BTC scoring formula below for every reported competition score; do not use the local 50/50 F1/state proxy as an official score. For each sample, let `M` be the number of exactly matched tool calls, `U` the number of unmatched submitted calls (including schema-rejected calls), and `|G|` the number of gold calls. Match calls by exact tool name and arguments; account for duplicate calls by multiplicity. If `hash_sys != hash_gold` or `M == 0`, sample score is 0. Otherwise compute `P = M / (M + U)`, `R = M / |G|`, and `Score_i = 2*P*R/(P+R)`. The final score on a 0–100 scale is `(100/N) * sum(Score_i)` over all `N` samples. Obtain `hash_gold` by replaying the labeled gold trajectory through the official ToolEnv; compare it with the prediction's ToolEnv hash. Report the matching convention and split/cohort with the score. Treat local proxy, aggregate call F1, and final-state match as separate diagnostics.

## External research
When implementation ideas are needed, consult relevant GitHub repositories and Kaggle notebooks/solutions. For Kaggle use the `nvidia-kaggle-skill`. Check license and compatibility before adapting code. Useful topics include Vietnamese ASR, tool/function calling, structured generation, lightweight inference, and agent evaluation.

## Project structure
- `configs/`: baseline configuration.
- `src/lifemate/`: ASR, agent, validation, ToolEnv loop, and pipeline.
- `scripts/`: sample runner, local evaluation, dataset inspection, and submission writer.
- `cache/transcripts/`: reusable transcripts keyed by audio content.
- `outputs/eval/` and `outputs/submissions/`: generated local artifacts (ignored by git).
- Organize `outputs/eval/` artifacts into one directory per experiment (for example, `baseline-v1-val309/` or `oracle-asr-v1-val309/`), keeping that experiment's predictions and summary together. Put cross-experiment comparisons in `outputs/eval/paired-analysis/`; don't leave experiment result files loose in the `outputs/eval/` root.
- `tests/`: focused regression checks.
- `train+public_test/`: competition data and official `zaic_tools` packages; do not inspect or read any `*_environment/` file directly. Use ToolEnv only.

