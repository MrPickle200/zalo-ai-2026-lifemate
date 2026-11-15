from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/data"
DATA_ROOT = ROOT / "train+public_test/train+public_test"
TRAIN_IDS = ROOT / "data/splits/train_ids.json"
VAL_IDS = ROOT / "data/splits/val_ids.json"
SOURCE_SFT = ROOT / "exp/qwen35-sft-next-action-v1/kaggle-upload-prompt-en-v1/sft_train.jsonl"
ASR = EXP / "output/asr/train_audio_transcripts.jsonl"
VAL_PHO = ROOT / "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho"
SFT_ROWS_EXPECTED = 3873

sys.path.insert(0, str(EXP / "scripts"))
from build_dataset import read_jsonl, rewrite_phowhisper, sha256, transcript_maps, write_jsonl  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from lifemate.tool_executor import available_tool_names, create_tool_env  # noqa: E402


def update_synthetic_text(base: dict, utterance: dict, index: int) -> dict:
    row = copy.deepcopy(base)
    payload = json.loads(row["messages"][1]["content"])
    payload["transcript"] = utterance["synthetic_transcript"]
    row["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    row["source"] = "assistant_authored_direct_synthetic_paraphrase_v1"
    row["synthetic_id"] = f"direct_synthetic_{index + 1:04d}"
    row["synthetic_parent"] = {"sample_id": base["sample_id"], "step_index": base["step_index"]}
    row["synthetic_text_id"] = base["sample_id"]
    row["synthetic_kind"] = "direct_text_paraphrase"
    return row


def replay_official_train(items: dict, train_ids: list[str]) -> list[dict]:
    records = []
    failures = []
    for sample_id in train_ids:
        item = items[sample_id]
        env = create_tool_env("train", sample_id, DATA_ROOT)
        calls = item.get("label", {}).get("tool_calls", [])
        available = set(available_tool_names(item))
        error = None
        for step, call in enumerate(calls):
            name = call.get("name")
            arguments = call.get("arguments", {})
            if name not in available:
                error = f"gold tool unavailable at step {step}: {name}"
                break
            validation = env.validate(name, arguments)
            if not validation.get("ok"):
                error = f"gold call rejected at step {step}: {validation}"
                break
            result = env.call(name, arguments)
            if not result.get("ok"):
                error = f"gold call failed at step {step}: {result}"
                break
        expected = item.get("label", {}).get("final_state", {})
        actual = env.get_state()
        state_match = all(actual.get(app) == value for app, value in expected.items())
        record = {
            "sample_id": sample_id,
            "gold_tool_calls": len(calls),
            "calls_validated_and_executed": error is None,
            "final_state_matches_btc_label": state_match,
            "state_hash": env.get_state_hash(),
            "error": error,
        }
        records.append(record)
        if error or not state_match:
            failures.append(record)
        if len(records) % 100 == 0 or len(records) == len(train_ids):
            print(f"ToolEnv gold replay {len(records)}/{len(train_ids)}", flush=True)
    write_jsonl(OUT / "toolenv_replay_train_1236.jsonl", records)
    if failures:
        raise RuntimeError(f"ToolEnv replay failures: {len(failures)}/{len(train_ids)}; see toolenv_replay_train_1236.jsonl")
    return records


def main() -> None:
    train_ids = json.loads(TRAIN_IDS.read_text(encoding="utf-8-sig"))
    val_ids = json.loads(VAL_IDS.read_text(encoding="utf-8-sig"))
    assert len(train_ids) == 1236 and len(set(train_ids)) == 1236
    assert len(val_ids) == 309 and not (set(train_ids) & set(val_ids))

    batch_paths = sorted(OUT.glob("synthetic_direct_batch_*.jsonl"))
    consolidated = OUT / "synthetic_utterances_1236.jsonl"
    if batch_paths:
        utterances = [row for path in batch_paths for row in read_jsonl(path)]
    elif consolidated.is_file():
        utterances = read_jsonl(consolidated)
    else:
        raise FileNotFoundError("No directly authored synthetic utterances found")
    by_synthetic_id = {row["sample_id"]: row for row in utterances}
    duplicate_ids = [sid for sid, count in Counter(row["sample_id"] for row in utterances).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate synthetic sample IDs: {duplicate_ids[:10]}")
    if set(by_synthetic_id) != set(train_ids):
        missing = sorted(set(train_ids) - set(by_synthetic_id))
        extra = sorted(set(by_synthetic_id) - set(train_ids))
        raise ValueError(f"Synthetic coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    main, history = transcript_maps()
    bad_source = [sid for sid in train_ids if by_synthetic_id[sid].get("source_transcript") != main[sid]]
    empty_text = [sid for sid in train_ids if not by_synthetic_id[sid].get("synthetic_transcript", "").strip()]
    unchanged_text = [sid for sid in train_ids if by_synthetic_id[sid]["synthetic_transcript"].strip() == main[sid].strip()]
    if bad_source or empty_text or unchanged_text:
        raise ValueError(f"Synthetic text QA failed: source={len(bad_source)}, empty={len(empty_text)}, unchanged={len(unchanged_text)}")

    items = {row["id"]: row for row in read_jsonl(DATA_ROOT / "train/train.jsonl")}
    source_rows = read_jsonl(SOURCE_SFT)
    assert len(source_rows) == SFT_ROWS_EXPECTED
    assert {row["sample_id"] for row in source_rows} <= set(train_ids)
    originals = [rewrite_phowhisper(row, items[row["sample_id"]], main, history) for row in source_rows]
    synthetic = [update_synthetic_text(row, by_synthetic_id[row["sample_id"]], index)
                 for index, row in enumerate(originals)]
    assert len(originals) == SFT_ROWS_EXPECTED == len(synthetic)
    assert all(row["target"] == originals[index]["target"] for index, row in enumerate(synthetic))
    assert all(row["sample_id"] in train_ids and row["sample_id"] not in val_ids for row in originals + synthetic)

    # Interleave source and synthetic examples deterministically to avoid presenting
    # the two halves as separate curriculum blocks during QLoRA fine-tuning.
    combined = originals + synthetic
    random.Random(20261112).shuffle(combined)

    btc_train = [items[sid] for sid in train_ids]
    write_jsonl(OUT / "synthetic_utterances_1236.jsonl", [by_synthetic_id[sid] for sid in train_ids])
    write_jsonl(OUT / "btc_train_1236.jsonl", btc_train)
    write_jsonl(OUT / "original_phowhisper_sft_3873.jsonl", originals)
    write_jsonl(OUT / "synthetic_direct_sft_3873.jsonl", synthetic)
    write_jsonl(OUT / "combined_sft_7746.jsonl", combined)

    val_main = {row["id"]: row for row in read_jsonl(VAL_PHO / "predictions.jsonl")}
    val_history = read_jsonl(VAL_PHO / "history_predictions.jsonl")
    assert len(val_main) == 309 and len(val_history) == 692
    write_jsonl(OUT / "validation_only_phowhisper_main_309.jsonl", [val_main[sid] for sid in val_ids])
    write_jsonl(OUT / "validation_only_phowhisper_history_692.jsonl", val_history)

    replay = replay_official_train(items, train_ids)
    targets = Counter("STOP" if row["target"]["action"] == "stop" else row["target"]["tool"] for row in originals)
    manifest = {
        "experiment": "qwen35-sft-pho-synthetic-v1",
        "status": "prepared_for_sft_after_toolenv_validation",
        "btc_train_trajectories": len(btc_train),
        "pho_main_transcripts": len(main),
        "pho_history_transcripts": len(history),
        "direct_synthetic_utterances": len(utterances),
        "original_phowhisper_sft_rows": len(originals),
        "synthetic_direct_sft_rows": len(synthetic),
        "combined_sft_rows": len(combined),
        "synthetic_to_original_sft_ratio": len(synthetic) / len(originals),
        "toolenv_gold_replay": {
            "trajectories": len(replay),
            "validated_and_executed": sum(row["calls_validated_and_executed"] for row in replay),
            "final_state_matches_btc_label": sum(row["final_state_matches_btc_label"] for row in replay),
            "failures": 0,
        },
        "original_target_counts": dict(sorted(targets.items())),
        "validation_ids_excluded_from_training": len(val_ids),
        "train_validation_overlap": 0,
        "synthetic_method": "LLM-authored Vietnamese paraphrases and controlled rewrites, entered as complete utterances. No programmatic text generation and no generic politeness-prefix augmentation. One synthetic utterance per BTC train trajectory; that utterance is substituted into each corresponding gold SFT step while all gold targets and prior observations remain unchanged.",
        "semantic_validation_note": "Synthetic utterances were manually reviewed against train context and BTC references. ToolEnv replay verifies only the inherited labeled action sequence and final state; ToolEnv does not score natural-language semantic equivalence.",
        "input_asr": "PhoWhisper large v2 greedy Vietnamese transcription for train main/history audio",
        "training_split": "fixed 1236 BTC train IDs only",
        "validation_transcripts_for_eval_only": True,
        "public_private_test_used": False,
        "train_ids_sha256": sha256(TRAIN_IDS),
        "validation_ids_sha256": sha256(VAL_IDS),
        "source_sft_sha256": sha256(SOURCE_SFT),
        "phowhisper_transcripts_sha256": sha256(ASR),
        "btc_train_file_sha256": sha256(DATA_ROOT / "train/train.jsonl"),
        "synthetic_utterances_sha256": sha256(OUT / "synthetic_utterances_1236.jsonl"),
        "original_dataset_sha256": sha256(OUT / "original_phowhisper_sft_3873.jsonl"),
        "synthetic_dataset_sha256": sha256(OUT / "synthetic_direct_sft_3873.jsonl"),
        "combined_dataset_sha256": sha256(OUT / "combined_sft_7746.jsonl"),
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
