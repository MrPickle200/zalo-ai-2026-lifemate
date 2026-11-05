from __future__ import annotations

import json
import shutil
from pathlib import Path
import _bootstrap


ROOT = _bootstrap.ROOT
EXPERIMENT = ROOT / "exp/qwen-next-action-v1/output/training-data"
KAGGLE_DIR = EXPERIMENT / "kaggle"
DATA_DIR = EXPERIMENT / "kaggle-data"


def main():
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "exp/qwen-next-action-v1/scripts/train_qwen_next_action.py", DATA_DIR / "train_qwen_next_action.py")
    shutil.copy2(EXPERIMENT / "manifest.json", DATA_DIR / "manifest.json")
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# LifeMate Qwen3-4B next-action QLoRA\n",
            "\n",
            "Uses the fixed train-only, replayed trajectory dataset. The smoke run trains on 128 examples for 30 optimizer steps with gradient accumulation 4, then reloads the saved adapter and probes it with a real SFT trajectory prompt.\n",
            "\n",
            "Keep SMOKE_RUN=True until the smoke adapter is saved, reloaded, and its action output parses. Set it to False only for full training.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import os, subprocess, sys\n",
            "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
            "import torch\n",
            "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'MISSING')\n",
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
            "    'transformers==4.51.3', 'peft==0.15.2', 'bitsandbytes==0.45.5', 'accelerate==1.7.0'])\n",
            "",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from pathlib import Path\n",
            "SMOKE_RUN = True\n",
            "input_root = Path('/kaggle/input')\n",
            "print('Kaggle inputs:', [p.name for p in input_root.iterdir()])\n",
            "trainer_path = next(input_root.rglob('train_qwen_next_action.py'), None)\n",
            "if trainer_path is None:\n",
            "    raise FileNotFoundError('train_qwen_next_action.py is missing from attached Kaggle datasets')\n",
            "data_path = trainer_path.with_name('sft_train.jsonl')\n",
            "if not data_path.is_file():\n",
            "    raise FileNotFoundError(f'SFT JSONL missing beside runner: {data_path}')\n",
            "print('Training input directory:', trainer_path.parent)\n",
            "args = [sys.executable, str(trainer_path), '--data', str(data_path),\n",
            "        '--output', '/kaggle/working/qwen-next-action-lora-v1', '--epochs', '2', '--max-length', '4096']\n",
            "if SMOKE_RUN:\n",
            "    args += ['--smoke', '--smoke-examples', '128', '--max-steps', '30']\n",
            "env = os.environ.copy()\n",
            "env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
            "subprocess.check_call(args, env=env)\n",
            "",
        ]},
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python", "version": "3"}},
                "nbformat": 4, "nbformat_minor": 5}
    notebook_path = KAGGLE_DIR / "qwen_next_action.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {"id": "daoviet/lifemate-qwen-next-action-qlora-v1", "title": "LifeMate Qwen next-action QLoRA v1",
                "code_file": "qwen_next_action.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": ["daoviet/lifemate-qwen-next-action-v1"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    dataset_metadata = {"title": "LifeMate Qwen next-action trajectory SFT v1",
                        "id": "daoviet/lifemate-qwen-next-action-v1",
                        "licenses": [{"name": "other"}], "keywords": ["lifemate", "sft", "qwen"]}
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kernel_dir": str(KAGGLE_DIR), "dataset_dir": str(DATA_DIR),
                      "kernel_ref": metadata["id"], "dataset_ref": dataset_metadata["id"]}, indent=2))


if __name__ == "__main__":
    main()
