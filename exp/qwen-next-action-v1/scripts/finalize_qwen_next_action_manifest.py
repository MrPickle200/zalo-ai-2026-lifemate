from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path("exp/qwen-next-action-v1/output/training-data")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path("train+public_test/train+public_test")
    train_ids = Path("data/splits/train_ids.json")
    val_ids = Path("data/splits/val_ids.json")
    trajectories = manifest["successful_trajectories_replayed"]
    examples = manifest["generated_timestep_examples"]
    manifest["average_steps_per_trajectory_including_stop"] = examples / trajectories if trajectories else 0
    manifest["reproducibility_inputs_sha256"] = {
        "train_ids": sha256(train_ids), "validation_ids_excluded": sha256(val_ids),
        "official_train_manifest": sha256(root / "train/train.jsonl"),
        "tool_catalog": sha256(root / "train/train_tools_catalog.json"),
        "dataset_builder": sha256(Path("exp/qwen-next-action-v1/scripts/build_qwen_next_action_dataset.py")),
    }
    manifest["pho_asr_model"] = {"path": "models/pho-asr-whisper-small",
                                 "config_sha256": sha256(Path("models/pho-asr-whisper-small/config.json"))}
    manifest["training_runner_sha256"] = sha256(Path("exp/qwen-next-action-v1/scripts/train_qwen_next_action.py"))
    data_path = out / "sft_train.jsonl"
    manifest["training_data"]["sha256"] = sha256(data_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "target_counts"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
