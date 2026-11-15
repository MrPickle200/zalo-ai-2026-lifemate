from __future__ import annotations

import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = _bootstrap.ROOT
    experiment = root / "exp" / "quan-continuation-cot-v1"
    checkpoint = root / "exp" / "quan-transcribe-lifemate-domain-v1" / "output" / "model"
    summary_path = root / "exp" / "quan-transcribe-lifemate-domain-v1-val309" / "output" / "summary.json"
    predictions_path = root / "exp" / "quan-continuation-cot-v1" / "output" / "asr" / "current_checkpoint_val309_predictions.jsonl"
    replay_predictions_path = experiment / "output" / "asr" / "checkpoint-verification-rerun-val309" / "predictions.jsonl"
    verification_path = experiment / "output" / "asr" / "checkpoint-verification.json"
    train_ids_path, val_ids_path = root / "data" / "splits" / "train_ids.json", root / "data" / "splits" / "val_ids.json"
    cohort_path = experiment / "output" / "cohort_manifest.json"
    lm_config = root / "models" / "quan-transcribe" / "lm" / "lm_config.json"

    train_ids = json.loads(train_ids_path.read_text(encoding="utf-8-sig"))
    val_ids = json.loads(val_ids_path.read_text(encoding="utf-8-sig"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8-sig"))
    audit = json.loads((checkpoint / "dataset-audit.json").read_text(encoding="utf-8-sig"))
    training_summary = json.loads((checkpoint / "training-summary.json").read_text(encoding="utf-8-sig"))
    baseline_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    old_val_ids = json.loads(val_ids_path.read_text(encoding="utf-8-sig"))
    baseline_rows = read_jsonl(predictions_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if len(train_ids) != 1236 or len(val_ids) != 309 or set(train_ids) & set(val_ids):
        raise RuntimeError("The fixed 1,236/309 train/validation split failed audit")
    if audit.get("train_ids") != 1236 or audit.get("validation_ids") != 309 or audit.get("train_validation_overlap") != 0:
        raise RuntimeError("Current QUAN checkpoint audit does not match the fixed split")
    if audit.get("train_ids_sha256") != hashlib.sha256("\n".join(train_ids).encode()).hexdigest():
        raise RuntimeError("Current checkpoint train ID hash differs from the fixed train split")
    if audit.get("validation_ids_sha256") != hashlib.sha256("\n".join(val_ids).encode()).hexdigest():
        raise RuntimeError("Current checkpoint validation ID hash differs from the fixed validation split")
    if cohort.get("ids") is None or len(cohort["ids"]) != 60 or len(set(cohort["ids"])) != 60:
        raise RuntimeError("Locked cohort manifest must contain 60 unique IDs")
    if hashlib.sha256("\n".join(cohort["ids"]).encode()).hexdigest() != cohort.get("ids_sha256_newline_joined"):
        raise RuntimeError("Locked cohort hash mismatch")
    if set(cohort["ids"]) - set(val_ids):
        raise RuntimeError("The locked downstream cohort must be a subset of the held-out val309 IDs")
    if [row["id"] for row in baseline_rows] != old_val_ids:
        raise RuntimeError("Copied current-checkpoint baseline must cover val309 in exact order")
    if baseline_summary.get("samples") != 309 or baseline_summary.get("decoder") != "KenLM CTC beam search":
        raise RuntimeError("Current-checkpoint validation summary is incompatible")
    if not verification.get("reproducible") or verification.get("exact_prediction_matches") != 309:
        raise RuntimeError("Archived validation predictions did not reproduce exactly from the extant checkpoint")

    required_paths = [
        train_ids_path, val_ids_path, cohort_path, checkpoint / "model.safetensors",
        checkpoint / "config.json", checkpoint / "units.json", checkpoint / "dataset-audit.json",
        checkpoint / "training-summary.json", predictions_path, replay_predictions_path,
        verification_path, summary_path, lm_config,
        root / "models" / "quan-transcribe" / "inference" / "quan_transcribe" / "model.py",
        root / "models" / "quan-transcribe" / "inference" / "quan_transcribe" / "transcriber.py",
        root / "models" / "quan-transcribe" / "inference" / "quan_transcribe" / "features.py",
        root / "models" / "quan-transcribe" / "inference" / "quan_transcribe" / "audio.py",
        root / "models" / "quan-transcribe" / "inference" / "quan_transcribe" / "lm.py",
    ]
    hashes = {str(p.relative_to(root)): sha256(p) for p in required_paths}
    cohort_id_hash = hashlib.sha256("\n".join(cohort["ids"]).encode()).hexdigest()
    result = {
        "experiment": "quan-continuation-cot-v1", "status": "preflight_passed",
        "data_policy": "Fixed train_ids and val_ids only; no public/private test; no direct access to any *_environment/ files",
        "split": {"train_ids": len(train_ids), "validation_ids": len(val_ids), "overlap": 0,
                  "train_ids_sha256_newline_joined": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
                  "validation_ids_sha256_newline_joined": hashlib.sha256("\n".join(val_ids).encode()).hexdigest()},
        "starting_checkpoint": str(checkpoint.relative_to(root)),
        "starting_checkpoint_sha256": hashes[str((checkpoint / "model.safetensors").relative_to(root))],
        "original_training_summary_checkpoint_path": training_summary.get("final_checkpoint"),
        "existing_val_metric_checkpoint_reference": baseline_summary.get("model"),
        "checkpoint_path_note": "The historical summaries point at a former outputs/models path; the extant archived checkpoint in the experiment directory is used and separately hashed here.",
        "baseline_validation": {"predictions": str(predictions_path.relative_to(root)), "samples": len(baseline_rows),
                                "corpus_wer": baseline_summary["metrics"]["corpus_wer"],
                                "corpus_cer": baseline_summary["metrics"]["corpus_cer"],
                                "decoder": baseline_summary["decoder"], "beam_width": baseline_summary["beam_width"],
                                "lm_alpha": baseline_summary["lm_alpha"], "lm_beta": baseline_summary["lm_beta"],
                                "exact_reproduced_predictions": verification["exact_prediction_matches"]},
        "cohort": {"source": "exp/inference-latency-optimization-val60/output/cohort_manifest.json",
                   "samples": len(cohort["ids"]), "ids_sha256_newline_joined": cohort_id_hash,
                   "subset_of_heldout_val309": True, "enriched_screening_cohort": True,
                   "used_for_training": False},
        "planned_continuation": {"stage": "reader_plus_top2_encoder", "epochs": "up to 3 chained one-epoch updates",
                                  "reader_group_lr": 1e-6, "top2_encoder_lr": 2e-7,
                                  "original_reader_stage2_lr": 1e-5, "original_encoder_lr": 1e-6,
                                  "seed": 17, "separate_checkpoint_per_epoch": True},
        "hashes": hashes,
        "no_public_submission_or_leaderboard_upload": True,
    }
    path = experiment / "output" / "preflight.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
