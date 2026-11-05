# LifeMate

Baseline for Zalo AI Challenge 2026: Vietnamese speech recognition, followed by a Qwen tool-calling agent executed through the official ToolEnv.

## Setup

Install Python dependencies in `.venv`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-torch-cu132.txt
```

Place the official data kit in `train+public_test/train+public_test/` and the local model weights in `models/`. The inference pipeline uses local models only. The Windows CUDA requirements include CUDA 12, cuBLAS, and cuDNN runtime wheels for faster-whisper GPU inference.

## Run

The default inference backend remains local Transformers. To run the selected Track C
configuration with Ollama (Qwen3.5 4B Q4_K_M, thinking enabled, JSON protocol), set:

```powershell
$env:LIFEMATE_LLM_BACKEND = "ollama"
$env:LIFEMATE_OLLAMA_MODEL = "qwen3.5:4b-q4_K_M"
$env:LIFEMATE_OLLAMA_THINK = "true"
$env:LIFEMATE_OLLAMA_PROTOCOL = "json"
$env:LIFEMATE_OLLAMA_CONTEXT = "8192"
$env:LIFEMATE_PROMPT_LANGUAGE = "en"
.venv\Scripts\python exp\lifemate-pipeline\scripts\make_submission.py --split public_test
```

Override `LIFEMATE_OLLAMA_MODEL`, `LIFEMATE_OLLAMA_THINK`,
`LIFEMATE_OLLAMA_PROTOCOL` (`json` or `native`), `LIFEMATE_OLLAMA_CONTEXT`, or
`LIFEMATE_OLLAMA_NUM_PREDICT` as needed. The local Transformers model remains selectable with `LIFEMATE_LLM_MODEL`;
choosing Ollama is explicit and does not change the backend default.

If interrupted, add `--resume` to continue from the existing valid output prefix. Run one sample with `exp/lifemate-pipeline/scripts/run_sample.py --sample train_0001`.

Run the fixed train-only validation and oracle-ASR comparison with:

```powershell
.venv\Scripts\python exp\lifemate-pipeline\scripts\make_split.py
.venv\Scripts\python exp\lifemate-pipeline\scripts\run_eval.py --mode baseline
.venv\Scripts\python exp\lifemate-pipeline\scripts\run_eval.py --mode oracle
.venv\Scripts\python exp\lifemate-pipeline\scripts\run_eval.py --mode retrieval
.venv\Scripts\python exp\lifemate-pipeline\scripts\analyze_eval.py --retrieval-experiment retrieval-v1-val309
```

Evaluation artifacts and experiment notes are grouped under `exp/<experiment>/`; cross-experiment reports are under `exp/paired-analysis/<comparison>/`. The `outputs/` directory is reserved for competition submissions and their sidecars.
Generate a retrieval-assisted public-test submission with `.venv\Scripts\python exp\lifemate-pipeline\scripts\make_submission.py --split public_test --retrieval`; add `--resume` to continue from a matching valid prefix.
