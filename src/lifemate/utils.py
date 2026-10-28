from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


def find_data_root(explicit: str | Path | None = None) -> Path:
    value = explicit or os.environ.get("LIFEMATE_DATA_ROOT") or "train+public_test/train+public_test"
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}; pass --data-root or set LIFEMATE_DATA_ROOT")
    return root


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def resolve_audio(root: Path, split_dir: Path, item: dict[str, Any], audio_ref: str) -> Path:
    candidates = [split_dir / audio_ref, root / split_dir.name / audio_ref, root / audio_ref]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Audio not found for {item.get('id')}: {audio_ref}")

