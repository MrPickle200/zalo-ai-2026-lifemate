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
- `exp/<experiment>/scripts/`: scripts used by that experiment. Do not add experiment scripts to a project-wide `scripts/` directory.
- `exp/<experiment>/output/`: predictions, summaries, checkpoints, logs, and other generated experiment artifacts.
- `exp/<experiment>/README.md`: concise experiment note with purpose, script entry points, important settings, and artifact locations.
- `exp/paired-analysis/<comparison>/`: cross-experiment comparisons, each with its own `output/` and `README.md`.
- Put shared pipeline and submission utilities under `exp/lifemate-pipeline/scripts/`; keep experiment-specific helpers with their experiment.
- `cache/transcripts/`: reusable transcripts keyed by audio content.
- `outputs/`: submission files only, including any matching prediction sidecars and manifests. Do not store evaluation results, checkpoints, logs, or experiment notes here.
- `models/`: shared downloaded pretrained weights; experiment-trained checkpoints belong under that experiment's `exp/<experiment>/output/`.
- Keep `exp/` experiment directories self-contained: place the scripts, outputs, and notes for an experiment together. Use a clear stable experiment name and avoid loose artifacts directly under `exp/`.
- `tests/`: focused regression checks.
- `train+public_test/`: competition data and official `zaic_tools` packages; do not inspect or read any `*_environment/` file directly. Use ToolEnv only.

