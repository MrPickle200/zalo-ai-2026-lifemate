# PhoWhisper large v2 versus QUAN epoch 06

Main val309: `output/{pho,quan}/predictions.jsonl` and `summary.json`. History val692: `scripts/run_history.py --branch pho|quan` writes `history_predictions.jsonl` and `history_summary.json`. `scripts/analyze_asr.py` produces paired metrics and rule-based verb-error events. All reference text comes from train.jsonl; public-test labels are unused.

Both decoders are Vietnamese, 16 kHz mono. PhoWhisper uses greedy generation with max_new_tokens=256 from its generation_config.json. QUAN uses KenLM beam 32. Per-clip timers encompass fresh audio decoding and inference, with no transcript cache hits; model load is timed separately. The QUAN main run resumed 200 saved rows, so its per-clip timing combines two sessions. PhoWhisper release data may overlap val309, so WER/CER are diagnostic rather than a clean generalization comparison.
