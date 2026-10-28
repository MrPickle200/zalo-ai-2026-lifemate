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

If interrupted, add `--resume` to continue from the existing valid output prefix. Run one sample with `scripts/run_sample.py --sample train_0001`.

Run the fixed train-only validation and oracle-ASR comparison with:

```powershell
.venv\Scripts\python scripts\make_split.py
.venv\Scripts\python scripts\run_eval.py --mode baseline
.venv\Scripts\python scripts\run_eval.py --mode oracle
.venv\Scripts\python scripts\run_eval.py --mode retrieval
.venv\Scripts\python scripts\analyze_eval.py --retrieval-experiment retrieval-v1-val309
```

Evaluation artifacts are grouped under `outputs/eval/<experiment>/`; the paired report is in `outputs/eval/paired-analysis/`.
Generate a retrieval-assisted public-test submission with `.venv\Scripts\python scripts\make_submission.py --split public_test --retrieval`; add `--resume` to continue from a matching valid prefix.
