# LifeMate

Baseline for Zalo AI Challenge 2026: Vietnamese speech recognition with faster-whisper, followed by a Qwen3 tool-calling agent executed through the official ToolEnv.

## Setup

Install Python dependencies in `.venv`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-torch-cu132.txt
```

Place the official data kit in `train+public_test/train+public_test/` and the local model weights in `models/`. The inference pipeline uses local models only. The Windows CUDA requirements include CUDA 12, cuBLAS, and cuDNN runtime wheels for faster-whisper GPU inference.

## Run

Select Ollama as the LLM backend, then generate the public test submission:

```powershell
$env:LIFEMATE_LLM_BACKEND = "ollama"
$env:LIFEMATE_OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
$env:LIFEMATE_OLLAMA_CONTEXT = "16384"
.venv\Scripts\python scripts\make_submission.py --split public_test
```

Run one sample with `scripts/run_sample.py --sample train_0001`, or local CV with `scripts/run_eval.py --split train --limit 20`.
