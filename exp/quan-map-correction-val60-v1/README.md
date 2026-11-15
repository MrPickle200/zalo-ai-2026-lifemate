# QUAN map correction benchmark — fixed val60

## Scope

This experiment runs four transcript inputs through the same Qwen downstream agent on the same fixed 60-task held-out validation cohort:

- **A — raw QUAN:** selected QUAN 10-epoch checkpoint, epoch 6.
- **B — QUAN + map:** the same transcript, Vietnamese cardinal phrases normalized to integer tokens first, then the train-only exact phrase map applied.
- **C — BTC script:** original BTC audio_text reference, as an oracle transcript.
- **D — QUAN epoch 02:** raw transcript from the two-epoch QUAN continuation checkpoint, without the map.

The cohort is the existing enriched val60 screen, a subset of val309 selected for prior failure cases. Treat it as a diagnostic screen, not an unbiased val309 estimate.

All branches used Qwen SFT epoch 2 + CoT, the same prompt, PhoASR history path, ToolEnv setup, and 5-action limit. Ollama 0.40.15 was called with native system Python outside the project .venv. The A/B/C run completed 180 downstream tasks in 3,208.9 seconds. The added D run completed 60 tasks in 1100.9 seconds. All four branches had zero inference runtime failures and zero gold replay failures.

## Results

| Branch | Project WER | Number-value WER | BTC formula /100 | Local 30s-gated diagnostic |
|---|---:|---:|---:|---:|
| A — raw QUAN | 0.1757 | 0.1540 | 23.29 | 20.12 |
| B — number normalization + train map | 0.2046 | 0.1784 | 19.83 | 15.83 |
| C — BTC oracle script | 0.0000 | 0.0000 | 32.33 | 32.33 |
| D — QUAN continuation epoch 02 | 0.1795 | 0.1569 | 24.62 | 22.29 |

The map applied 28 corrections across 22 tasks; **0/28 exactly matched the BTC reference at the aligned span**. The number parser normalized 35 QUAN spans before map matching. Paired BTC difference B−A was −3.45 points (95% bootstrap CI [−10.00, +1.83]; 3 improved, 3 worsened, 54 tied).

The map branch was worse by point estimate for both WER and BTC; the paired BTC interval includes zero. The D epoch-02 continuation measured project WER 0.1795, number-value WER 0.1569, and downstream BTC 24.62/100 on this screen. The zero exact-correction precision is a strong warning against applying these candidate rules globally. The oracle branch's BTC score is an upper-bound diagnostic, not a deployable ASR result.

## Reproduction

Run these commands from the project root with the native system Python, not .venv:

1. python exp/quan-map-correction-val60-v1/scripts/prepare_benchmark.py
2. python exp/quan-map-correction-val60-v1/scripts/run_benchmark.py
3. python exp/quan-map-correction-val60-v1/scripts/analyze_benchmark.py
4. python exp/quan-map-correction-val60-v1/scripts/run_epoch02_branch.py
5. python exp/quan-map-correction-val60-v1/scripts/analyze_epoch02_branch.py

The frozen cohort, selected rules, transcript inputs, per-application audit, predictions, ToolEnv traces, official BTC per-sample scores, metrics and paired bootstrap outputs are in output/benchmark/.
