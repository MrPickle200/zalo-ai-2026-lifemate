from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.schemas import parse_action  # noqa: E402
from lifemate.tool_executor import available_tool_names, create_tool_env  # noqa: E402

DATA_ROOT = ROOT / "train+public_test" / "train+public_test"
TRAIN_IDS_PATH = ROOT / "data" / "splits" / "train_ids.json"
VAL_IDS_PATH = ROOT / "data" / "splits" / "val_ids.json"
MAIN_QUAN_PATH = ROOT / "exp" / "phoasr-vs-quan-transcribe-train1545" / "output" / "quan" / "predictions.jsonl"
HISTORY_QUAN_PATH = ROOT / "exp" / "qwen35-sft-next-action-v1" / "output" / "data" / "quan_history_transcripts.jsonl"
OUT_DIR = ROOT / "exp" / "qwen35-sft-next-action-v1" / "kaggle-upload"

# This is Agent.next_action's current default Vietnamese policy with /no_think
# removed because the production Ollama request has thinking enabled.
SYSTEM_PROMPT = (
    "Bạn là agent LifeMate, xử lý yêu cầu bằng tiếng Việt thông qua các tool được cung cấp. "
    "Mục tiêu là hoàn thành đúng toàn bộ yêu cầu hiện tại, không chỉ thực hiện bước đầu tiên. "
    "Căn cứ vào transcript hiện tại, context/profile, history, available tools và mọi observations từ các lượt trước. "
    "Chỉ dùng tool có trong schema của mẫu; đọc kỹ mô tả, required fields, kiểu dữ liệu và enum. "
    "Không gọi tool gần giống nếu không đáp ứng yêu cầu; nếu không có tool phù hợp hoặc yêu cầu không cần hành động thì trả stop. "
    "Trước mỗi lượt, xác định yêu cầu nào còn chưa hoàn thành và chọn đúng bước tiếp theo cần thiết. "
    "Sau mỗi tool call, đọc observation mới nhất: kiểm tra thành công/thất bại, lấy chính xác ID, giá trị và dữ liệu trả về để quyết định bước sau hoặc điền arguments. "
    "Không đoán, tự tạo hoặc thay đổi ID/giá trị mà tool đã trả về; không lặp lại action đã hoàn thành. "
    "Không dừng sau một bước tra cứu nếu yêu cầu còn cần hành động tiếp theo. Chỉ stop khi mọi phần yêu cầu đã hoàn tất, hoặc không thể tiến hành an toàn bằng các tool hiện có. "
    "Nếu thiếu required argument, trước hết kiểm tra transcript, context/profile, history và observations xem dữ kiện đã có chưa. "
    "Chỉ gọi sys.ask_missing_info khi dữ kiện thực sự còn thiếu và không thể suy ra an toàn; hỏi đúng các trường còn thiếu, không hỏi lại dữ kiện đã có. "
    "Nếu transcript có vẻ nhận dạng sai, dùng ngữ cảnh để hiểu ý định nhưng không tự bịa chi tiết quan trọng; hỏi thêm nếu cách hiểu làm thay đổi tool hoặc arguments. "
    "Tránh tool call thừa, trùng lặp hoặc gây thay đổi trạng thái ngoài yêu cầu. "
    'Mỗi lượt chỉ trả đúng một JSON object, không markdown, không giải thích: '
    '{"action":"tool_call","tool":"app.tool","arguments":{}} hoặc {"action":"stop"}.'
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_ids(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def main() -> int:
    train_dir = DATA_ROOT / "train"
    items = {row["id"]: row for row in read_jsonl(train_dir / "train.jsonl")}
    train_ids = json.loads(TRAIN_IDS_PATH.read_text(encoding="utf-8-sig"))
    val_ids = json.loads(VAL_IDS_PATH.read_text(encoding="utf-8-sig"))
    assert len(items) == 1545
    assert len(train_ids) == 1236 and len(val_ids) == 309
    assert len(set(train_ids)) == len(train_ids) and len(set(val_ids)) == len(val_ids)
    assert not (set(train_ids) & set(val_ids)) and set(train_ids) | set(val_ids) == set(items)

    main_quan = {row["id"]: row["prediction"] for row in read_jsonl(MAIN_QUAN_PATH)}
    history_rows = read_jsonl(HISTORY_QUAN_PATH)
    history_quan = {row["audio_ref"]: row["transcript"] for row in history_rows}
    assert set(main_quan) == set(items), "QUAN main transcript IDs do not cover the official train split exactly"

    catalog = json.loads((train_dir / "train_tools_catalog.json").read_text(encoding="utf-8"))
    examples: list[dict] = []
    failures: list[dict] = []
    target_counts: Counter[str] = Counter()
    successful_samples = 0
    step_counts: list[int] = []
    history_audio_uses = 0

    def history_for(sample: dict) -> list[dict]:
        nonlocal history_audio_uses
        history = []
        for turn in sample.get("history", []):
            converted = dict(turn)
            audio_ref = converted.get("audio")
            if audio_ref:
                if audio_ref not in history_quan:
                    raise KeyError(f"Missing QUAN history transcript for {sample['id']} {audio_ref}")
                converted["text"] = history_quan[audio_ref]
                converted.pop("audio", None)
                history_audio_uses += 1
            history.append(converted)
        return history

    def input_messages(sample: dict, transcript: str, history: list[dict], available: list[str], observations: list[dict]) -> list[dict]:
        compact_specs = __import__("lifemate.agent", fromlist=["Agent"]).Agent._tool_specs(available, catalog)
        context = {key: value for key, value in sample.get("context", {}).items() if key != "installed_apps"}
        payload = {
            "transcript": transcript,
            "context": context,
            "history": history,
            "available_tools": compact_specs,
            "observations": observations,
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)},
        ]

    for sample_id in train_ids:
        sample = items[sample_id]
        gold = sample.get("label", {}).get("tool_calls")
        if not isinstance(gold, list):
            failures.append({"sample_id": sample_id, "reason": "label.tool_calls is not a list"})
            continue
        transcript = main_quan[sample_id]
        try:
            history = history_for(sample)
            available = available_tool_names(sample)
            env = create_tool_env("train", sample_id, DATA_ROOT)
            observations: list[dict] = []
            sample_examples: list[dict] = []
            for step_index, call in enumerate(gold):
                if not isinstance(call, dict) or not isinstance(call.get("arguments", {}), dict):
                    raise ValueError(f"malformed gold call at step {step_index}")
                name = call.get("name", "")
                if name not in available:
                    raise ValueError(f"gold tool not available at step {step_index}: {name}")
                arguments = call.get("arguments", {})
                validation = env.validate(name, arguments)
                if not validation.get("ok"):
                    raise ValueError(f"ToolEnv validation failed at step {step_index}: {validation}")
                target = {"action": "tool_call", "tool": name, "arguments": arguments}
                if parse_action(json.dumps(target, ensure_ascii=False)) != target:
                    raise ValueError(f"production parser round-trip failed at step {step_index}")
                sample_examples.append({
                    "sample_id": sample_id,
                    "step_index": step_index,
                    "messages": input_messages(sample, transcript, history, available, observations),
                    "target": target,
                    "source": "official_gold",
                })
                result = env.call(name, arguments)
                observations.append({"tool": name, "arguments": arguments, "result": result})
                if name == "sys.ask_missing_info":
                    break
            else:
                target = {"action": "stop"}
                sample_examples.append({
                    "sample_id": sample_id,
                    "step_index": len(gold),
                    "messages": input_messages(sample, transcript, history, available, observations),
                    "target": target,
                    "source": "official_gold",
                })
            examples.extend(sample_examples)
            successful_samples += 1
            step_counts.append(len(gold))
            for row in sample_examples:
                target = row["target"]
                target_counts["STOP" if target["action"] == "stop" else target["tool"]] += 1
        except Exception as exc:
            failures.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}: {exc}"})

    if failures:
        write_jsonl(OUT_DIR / "replay_failures.jsonl", failures)
        raise RuntimeError(f"Dataset build failed on {len(failures)} train trajectories; see replay_failures.jsonl")
    if successful_samples != len(train_ids):
        raise RuntimeError(f"Expected {len(train_ids)} replayed train samples, got {successful_samples}")

    # Evaluation inputs contain no target labels: they only pair val309 IDs with
    # QUAN transcripts for the current utterance and audio turns in its history.
    val_inputs = []
    for sample_id in val_ids:
        sample = items[sample_id]
        history = history_for(sample)
        val_inputs.append({"sample_id": sample_id, "transcript": main_quan[sample_id], "history": history})

    write_jsonl(OUT_DIR / "sft_train.jsonl", examples)
    write_jsonl(OUT_DIR / "val309_inputs.jsonl", val_inputs)
    (OUT_DIR / "train_ids.json").write_text(json.dumps(train_ids, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "val309_ids.json").write_text(json.dumps(val_ids, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment": "qwen35-sft-next-action-v1",
        "source": "official BTC train split; existing fixed grouped 1236/309 split",
        "train_trajectories": len(train_ids),
        "validation_trajectories_excluded_from_training": len(val_ids),
        "train_validation_overlap": 0,
        "replayed_train_trajectories": successful_samples,
        "sft_examples": len(examples),
        "gold_tool_calls_mean": sum(step_counts) / len(step_counts),
        "gold_tool_calls_max": max(step_counts),
        "target_counts": dict(sorted(target_counts.items())),
        "quan_main_transcripts": len(main_quan),
        "quan_history_audio_transcripts": len(history_quan),
        "train_history_audio_turns_used": sum(1 for sid in train_ids for turn in items[sid].get("history", []) if turn.get("audio")),
        "val_history_audio_turns_in_inference_only_file": sum(1 for sid in val_ids for turn in items[sid].get("history", []) if turn.get("audio")),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "tool_catalog_sha256": digest_file(train_dir / "train_tools_catalog.json"),
        "official_train_jsonl_sha256": digest_file(train_dir / "train.jsonl"),
        "train_ids_sha256": digest_ids(train_ids),
        "val309_ids_sha256": digest_ids(val_ids),
        "sft_train_sha256": digest_file(OUT_DIR / "sft_train.jsonl"),
        "val309_inputs_sha256": digest_file(OUT_DIR / "val309_inputs.jsonl"),
        "transcript_source": "QUAN-Transcribe with KenLM, beam_width=32; existing train1545 main predictions plus locally decoded history audio",
        "input_serialization": "Agent.next_action current Vietnamese prompt; context, history, compact available tool specs, and ToolEnv observations",
        "target_serialization": "one JSON next action per row; no chain-of-thought targets",
        "training_contains_val_labels": False,
    }
    (OUT_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
