# Qwen3.5 model-size screen on Colab L4

Compare the strongest saved public-test configuration's Qwen3.5 4B SFT epoch-3 English model against a larger Qwen3.5 model while keeping PhoWhisper large v2 transcripts, English + CoT prompt, JSON protocol, and ToolEnv loop fixed. This is a diagnostic only: 9B exceeds the competition's LLM and total-parameter limits.

Upload `output/colab-resources/lifemate-colab-qwen-scale/` to Google Drive. The folder contains a 4.69 MB validation resource ZIP, the exact 2.71 GB Q4_K_M 4B SFT GGUF, SHA-256 checksums, and Colab instructions. Ollama pulls 9B directly on Colab. The ZIP contains cached main/history PhoWhisper transcripts for val309, fixed pilot60 IDs, official train ToolEnv resources, and the runnable comparison script. Audio and ASR weights are excluded.

Experiment-specific sources are in `scripts/`: `run_compare.py` runs the paired inference and BTC gold replay after extraction, `Modelfile` registers the 4B SFT GGUF, and `build_bundle.py` rebuilds the upload package from the project. Generated predictions and score summaries are written to `output/` inside the extracted Colab bundle; copy them back to this experiment's `output/` if retaining results in the project.

Settings: thinking on, temperature 0, JSON protocol, context 8192, generation cap 768, batch 512, maximum five action turns. The runner reports ungated BTC, paired delta and a 30-second LLM+ToolEnv diagnostic. The latter excludes ASR time and is not a competition runtime score. PhoWhisper validation may overlap its training corpus, so interpret paired deltas rather than absolute scores as the main screen.
