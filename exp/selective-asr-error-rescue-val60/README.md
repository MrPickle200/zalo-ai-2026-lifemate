# Selective ASR error rescue — pilot60

## Method

Use the final4 Q4 adapter with the existing fixed-thinking ask-stop policy. Compare cached PhoWhisper Large v2 and QUAN epoch-06 transcripts without using labels. Expose the secondary transcript to Qwen only when normalized action, polarity, number, or time tokens differ. Reuse archived Q4 ask-stop outputs for unselected IDs; infer only selected IDs. The cohort is the existing enriched pilot60.

## Result

- Selected 27/60 samples; 27 fresh Qwen inferences; zero runtime errors.
- Official BTC before/after the local 30-second gate: Q4 ask-stop baseline 63.33/63.33; selective rescue 60.83/60.83.
- Paired delta −2.50 points (95% bootstrap CI [−9.17,+4.17]); 2 wins, 55 ties, 3 losses. Do not promote the route.
- Local Qwen+ToolEnv mean 11.12s vs 11.75s baseline. The secondary ASR was cached; its runtime, memory use, and parameter budget are excluded. This enriched screen is not full-val or Public LB evidence.

## Artifacts

- Entry point: scripts/benchmark_selective_asr_rescue.py
- Report and summary: output/summary.md, output/summary.json
- Routing, cached transcript inputs, selected inference traces and paired per-ID scores: output/
- Current public Q4 ask-stop candidate and policy are unchanged; user-reported Public LB ground truth remains 55.92.