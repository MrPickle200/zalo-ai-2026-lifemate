from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "exp/qwen35-sft-pho-synthetic-v1/scripts"


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Select a Colab GPU runtime (L4 recommended).")
    model_dir = Path(os.environ.get("LIFEMATE_PHOWHISPER_MODEL", ""))
    if not model_dir.is_dir() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            "Set LIFEMATE_PHOWHISPER_MODEL to the PhoWhisper Large model directory "
            "mounted in Google Drive. The model weights are intentionally not in the data ZIP."
        )

    for name in ("transcribe_train.py", "build_direct_dataset.py"):
        subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
