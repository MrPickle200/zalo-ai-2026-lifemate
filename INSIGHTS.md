# Insights

## Dev kit

- `ToolEnv` records every attempted call, including schema-rejected calls; submit exactly `env.get_tool_calls()`.
- ToolEnv is constructed from the split data root, with one session per sample.
- The package contains no standalone evaluator; a local evaluation harness must replay predictions against labeled train examples.
- On Windows, CUDA-enabled PyTorch does not guarantee CTranslate2 has its CUDA 12/cuBLAS DLLs; the ASR backend needs a CPU fallback when those DLLs are missing.


## Ollama inference

- The Q4_K_M model fits entirely on the 6 GB RTX 4050 at 6144 context, but a long prompt (6264 tokens before generation) fails with HTTP 400. At 16384 context it runs split across CPU/GPU (23%/77%) and remains slower; measure with the actual prompt distribution.
- The 20-sample Ollama CV completed without runtime errors but achieved only 0.0247 tool-call F1, so fast inference alone does not make this model setup a usable baseline.
- On Windows, `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, and `nvidia-cudnn-cu12` can be installed in the project venv; add their package `bin` directories to the DLL loader path before importing CTranslate2. Whisper Small then transcribes on the RTX 4050.
