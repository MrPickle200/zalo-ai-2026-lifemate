# QUAN transcript map comparison — full val309

## Scope

This is a paired comparison on all 309 held-out validation IDs, with one Qwen SFT epoch-2 + CoT run per branch:

- **A:** raw QUAN epoch-06 transcript.
- **B:** epoch-06 transcript, Vietnamese number normalization, then the frozen train-only exact phrase map.
- **C:** original BTC reference script as an oracle transcript.
- **D:** raw QUAN-transcribe continuation checkpoint at epoch 02.

All branches share the same Qwen model/prompt, PhoASR history, ToolEnv configuration, and 5-action limit. Branch order rotates across 20-ID blocks. The map rules were frozen using train-only candidates; validation labels were not used to select rules.

Ollama versions: rounds 1-10: Ollama 0.40.15; rounds 11-16: Ollama 0.40.16. The Qwen model digest is unchanged, and all four branches for each ID block used the same version. The same task, train_1725, exceeded the configured 8,192-token context in all four branches (8,236–8,239 tokens); no tool calls were generated, and its official BTC sample score is 0 in each branch.

## Results

| Branch | Project WER | Number-value WER | BTC formula /100 | Local 30s diagnostic |
|---|---:|---:|---:|---:|
| A — epoch-06 QUAN raw | 0.1620 | 0.1454 | 31.59 | 27.95 |
| B — epoch-06 + number parser/map | 0.1977 | 0.1695 | 29.80 | 25.76 |
| C — BTC oracle script | 0.0000 | 0.0000 | 47.12 | 42.49 |
| D — QUAN epoch 02 raw | 0.1633 | 0.1467 | 30.24 | 26.91 |

Train-map audit: 3/192 exact corrections (1.6%) across 140 IDs.

## Reproduction

Run from the project root with native system Python outside the virtual environment:

1. python exp/quan-map-correction-val309-v1/scripts/prepare_val309.py
2. python exp/quan-map-correction-val309-v1/scripts/run_val309.py
3. python exp/quan-map-correction-val309-v1/scripts/analyze_val309.py

The run script resumes completed blocks and staged per-ID work. Inference artifacts, per-sample BTC scores, paired bootstrap outputs, source hashes and run progress are under output/benchmark/.
