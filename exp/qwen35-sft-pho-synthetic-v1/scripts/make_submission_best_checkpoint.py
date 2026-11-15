from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
EVAL = EXP / "output/val309-checkpoint-eval"
OUT = EXP / "output/submission-best-checkpoint"
FOCUSED_WRAPPER = ROOT / "exp/phowhisper-large-qwen-focused-public-v1/scripts/make_submission.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load submission workflow: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_winner():
    summary = json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((EVAL / "comparison_lock.json").read_text(encoding="utf-8"))
    branches = summary["branches"]
    candidates = []
    for key in ("step60", "step120"):
        branch = branches[key]
        if branch.get("status") != "complete" or branch.get("samples_completed") != 309:
            raise RuntimeError(f"{key} val309 evaluation is incomplete")
        candidates.append((summary["groups"]["all_val309"][key]["btc_0_100"], key))
    score, key = max(candidates)
    checkpoint = lock["checkpoints"][key]
    return key, score, checkpoint["model"], checkpoint["digest"], summary


def configure():
    key, score, model, digest, summary = select_winner()
    module = load_module("focused_submission_for_best_sft_checkpoint", FOCUSED_WRAPPER)
    module.__file__ = str(Path(__file__).resolve())
    module.MODEL = model
    module.DIGEST = digest
    module.STEM = f"phowhisper-large-v2-qwen35-phosynth-{key}-focused-public-test"
    module.EXP = EXP
    module.OUT = OUT

    # The shared runner sets Ollama's model from its own module globals.
    module.base.MODEL = model
    module.base.DIGEST = digest
    module.base.EXP = EXP
    module.base.OUT = OUT

    comparison = summary["groups"]["all_val309"]
    module.SELECTED_CHECKPOINT = key
    module.SELECTED_VAL309_SCORE = score
    module.STEP60_VAL309_SCORE = comparison["step60"]["btc_0_100"]
    module.STEP120_VAL309_SCORE = comparison["step120"]["btc_0_100"]
    return module


def main():
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in ("--prepare", "--run"):
        raise SystemExit("Use --prepare or --run")
    module = configure()
    key = module.SELECTED_CHECKPOINT
    score = module.SELECTED_VAL309_SCORE
    print(json.dumps({
        "selected_checkpoint": key,
        "val309_btc": score,
        "step60_btc": module.STEP60_VAL309_SCORE,
        "step120_btc": module.STEP120_VAL309_SCORE,
        "model": module.MODEL,
        "output_dir": str(module.OUT),
    }, ensure_ascii=False), flush=True)
    if sys.argv[1] == "--prepare":
        module.prepare()
    else:
        module.run()


if __name__ == "__main__":
    main()
