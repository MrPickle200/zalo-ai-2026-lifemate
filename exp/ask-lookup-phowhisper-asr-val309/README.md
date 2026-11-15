# Focused prompt: QUAN/PhoASR versus PhoWhisper Large

Compare the focused post-lookup prompt on the same 309 validation tasks with the Qwen 3.5 epoch-3 checkpoint. The archived QUAN-main/PhoASR-history predictions are reused; the paired treatment uses cached PhoWhisper Large v2 transcripts for all 309 main clips and 692 history clips. This measures the ASR stack change with fixed downstream settings.

Run `compare.py --prepare`, `compare.py --run`, and `compare.py --analyze` with native system Python so Ollama is reachable. The run can resume completed IDs. Scoring reports the official BTC metric, local 30-second gate, pre-gate diagnostic, and paired bootstrap. No public-test submission is created.
