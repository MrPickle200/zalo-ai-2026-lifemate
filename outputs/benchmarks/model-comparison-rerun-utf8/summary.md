# Model benchmark summary ? hard-30 (PhoASR UTF-8 cache)

| Model | Samples | Tool-call F1 | State match | Proxy | First tool | First full call | Exact trajectory | Runtime failures | Avg. trajectory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M | 30/30 | 0.1139 | 0.0% | 0.0569 | 16.7% | 0.0% | 0.0% | 0 | 3.20 |
| Phi-4-mini | 30/30 | 0.0000 | 0.0% | 0.0000 | 3.3% | 0.0% | 0.0% | 0 | 2.33 |
| Qwen2.5-3B Q4_K_M | 30/30 | 0.0162 | 0.0% | 0.0081 | 3.3% | 0.0% | 0.0% | 0 | 2.80 |
| Sailor2-1B | 30/30 | 0.0000 | 0.0% | 0.0000 | 0.0% | 0.0% | 0.0% | 0 | 0.17 |
| Gemma 4 E4B | 29/30 | 0.0086 | 0.0% | 0.0043 | 17.2% | 0.0% | 0.0% | 1 | 0.59 |

## MiniCPM3-4B

Excluded: benchmark was stopped because inference was too slow for the 30-second per-task runtime target. No complete summary was produced.

Gemma 4 E4B is included as an exploratory run; it was released after the competition cutoff and is not competition-eligible.
