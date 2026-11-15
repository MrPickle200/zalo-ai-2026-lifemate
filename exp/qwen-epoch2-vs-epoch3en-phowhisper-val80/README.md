# Qwen epoch 2 vs epoch 3 English with PhoWhisper

Compare the two locked Qwen SFT models on a seeded 80-task sample of val309. Both use the same English + CoT baseline prompt, PhoWhisper large v2 transcripts for the main utterance and all history audio, ToolEnv, decoding settings, and 5-step cap. Models run in alternating 20-task blocks.

Run scripts/benchmark.py --prepare to lock the cohort and fingerprints, then --run with native system Python for local Ollama. BTC is scored with official ToolEnv gold replay. PhoWhisper val data may overlap its training corpus; this is a paired diagnostic rather than a clean generalization estimate.
