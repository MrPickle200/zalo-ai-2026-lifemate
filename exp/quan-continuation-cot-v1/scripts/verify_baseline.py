from __future__ import annotations

import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import read_jsonl


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = _bootstrap.ROOT
    experiment = root / "exp" / "quan-continuation-cot-v1"
    original = experiment / "output" / "asr" / "current_checkpoint_val309_predictions.jsonl"
    replay = experiment / "output" / "asr" / "checkpoint-verification-rerun-val309" / "predictions.jsonl"
    old, new = read_jsonl(original), read_jsonl(replay)
    if [r["id"] for r in old] != [r["id"] for r in new] or len(old) != 309:
        raise RuntimeError("Baseline replay does not have exact ordered val309 coverage")
    mismatches = [
        {"id": a["id"], "archived": a.get("prediction"), "replay": b.get("prediction")}
        for a, b in zip(old, new) if a.get("prediction") != b.get("prediction")
    ]
    result = {
        "samples": len(old), "ordered_id_match": True,
        "exact_prediction_matches": len(old) - len(mismatches),
        "exact_prediction_mismatches": mismatches,
        "archived_prediction_file_sha256": digest(original),
        "replay_prediction_file_sha256": digest(replay),
        "model_sha256": json.loads((experiment / "output" / "asr" / "checkpoint-verification-rerun-val309" / "summary.json").read_text(encoding="utf-8"))["model_sha256"],
        "reproducible": not mismatches,
    }
    out = experiment / "output" / "asr" / "checkpoint-verification.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
