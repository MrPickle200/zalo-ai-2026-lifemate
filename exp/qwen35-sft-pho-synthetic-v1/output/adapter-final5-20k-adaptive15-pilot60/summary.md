# Final5 20k-row adapter — adaptive15/max6 enriched pilot60

Both adapters used the same enriched 60-task cohort, focused prompt, adaptive total ASR+Qwen budget, successful ask-stop, and six-action-turn cap. First Qwen request uses thinking; later requests turn thinking off once cached per-task PhoWhisper time plus prior Qwen latency reaches 15 seconds. The local 30-second gate includes cached ASR timing; ASR model-load startup is excluded.

| Adapter | BTC raw | BTC after ASR+Qwen 30s gate | Mean end-to-end s | >30s | Errors | Hash match |
|---|---:|---:|---:|---:|---:|---:|
| final4 control | 63.00 | 61.33 | 16.15 | 2/60 | 0 | 50/60 |
| final5 20k | 59.67 | 59.67 | 15.65 | 2/60 | 0 | 50/60 |

Paired final5−final4: raw -3.33 (95% CI [-11.666666666666666, 5.0]); after gate -1.67 (95% CI [-10.0, 6.666666666666667]); gated wins/ties/losses [3, 53, 4].
ASR mean 4.98s/task; mean Qwen: final4 11.17s, final5 10.68s. This cohort is enriched, not full val/public; local RTX 4050 timing is not A30.

See paired_details.jsonl for per-ID scores, latency, ToolEnv hashes, turns, and thinking switches.
