from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.asr import PhoASRModel
from lifemate.schemas import parse_action
from lifemate.tool_executor import available_tool_names, create_tool_env
from lifemate.utils import find_data_root, load_jsonl, resolve_audio


SYSTEM_PROMPT = (
    "Bạn là agent LifeMate tiếng Việt. Dùng transcript cuối cùng, context, history và tool schema để thực hiện yêu cầu. "
    "Chỉ gọi tool đã cài và có trong schema. Không bịa dữ liệu; thiếu thông tin thiết yếu thì gọi sys.ask_missing_info. "
    "Nếu tool phù hợp yêu cầu required argument nhưng request hiện tại không cung cấp và không thể suy ra an toàn "
    "từ context/history, không đoán giá trị; gọi sys.ask_missing_info. "
    "Sau khi hoàn tất, trả action stop. Mỗi lần chỉ trả đúng một JSON object, không giải thích. "
    'Ví dụ: {"action":"tool_call","tool":"app.tool","arguments":{}} hoặc {"action":"stop"}. /no_think'
)


def compact_tool_specs(available: list[str], catalog: dict) -> list[dict]:
    selected = set(available)
    compact = []
    for app in catalog.get("apps", []):
        for spec in app.get("tools", []):
            if spec.get("name") not in selected:
                continue
            params = spec.get("parameters", {})
            properties = {
                name: {key: prop[key] for key in
                       ("type", "description", "enum", "format", "minimum", "maximum",
                        "exclusiveMinimum", "exclusiveMaximum", "x-ref", "x-match", "items")
                       if key in prop}
                for name, prop in params.get("properties", {}).items()
            }
            compact.append({"name": spec["name"], "description": spec.get("description", "")[:160],
                            "required": params.get("required", []), "one_of": params.get("x-one-of", []),
                            "args": properties})
    return compact


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay gold train trajectories and build timestep-level SFT examples")
    parser.add_argument("--data-root")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output-dir", default="outputs/training/qwen-next-action-v1")
    parser.add_argument("--asr-model", default="models/pho-asr-whisper-small")
    args = parser.parse_args()

    root = find_data_root(args.data_root)
    train_dir = root / "train"
    all_items = {row["id"]: row for row in load_jsonl(train_dir / "train.jsonl")}
    train_ids = json.loads((Path(args.split_dir) / "train_ids.json").read_text(encoding="utf-8"))
    val_ids = json.loads((Path(args.split_dir) / "val_ids.json").read_text(encoding="utf-8"))
    if len(train_ids) != 1236 or len(val_ids) != 309 or set(train_ids) & set(val_ids):
        raise SystemExit("Expected the fixed 1,236/309 split with no overlap")
    if (set(train_ids) | set(val_ids)) - set(all_items):
        raise SystemExit("Fixed split contains IDs missing from official train manifest")

    catalog = json.loads((train_dir / "train_tools_catalog.json").read_text(encoding="utf-8"))
    catalog_tools = {t["name"]: t for app in catalog.get("apps", []) for t in app.get("tools", [])}
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    failure_rows: list[dict] = []
    examples_written = 0
    step_counts: list[int] = []
    target_counts: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()
    duplicate_timesteps = 0
    asr = PhoASRModel(model_path=args.asr_model)
    output_path = outdir / "sft_train.jsonl"

    def transcript_for(item: dict, audio_ref: str) -> str:
        audio_path = resolve_audio(root, train_dir, item, audio_ref)
        return asr.transcribe(str(audio_path))

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for trajectory_index, sample_id in enumerate(train_ids, 1):
            item = all_items[sample_id]
            gold = item.get("label", {}).get("tool_calls")
            if not isinstance(gold, list):
                failure_rows.append({"sample_id": sample_id, "reason": "label.tool_calls is not a list"})
                continue
            available = available_tool_names(item)
            bad_shape = None
            for step, call in enumerate(gold):
                name = call.get("name") if isinstance(call, dict) else None
                args_obj = call.get("arguments", {}) if isinstance(call, dict) else None
                spec = catalog_tools.get(name)
                if name not in available or spec is None or not isinstance(args_obj, dict):
                    bad_shape = f"step {step}: unknown/unavailable tool or malformed arguments"
                    break
                unknown = sorted(set(args_obj) - set(spec.get("parameters", {}).get("properties", {})))
                if unknown:
                    bad_shape = f"step {step}: unknown argument keys {unknown}"
                    break
            if bad_shape:
                failure_rows.append({"sample_id": sample_id, "reason": bad_shape})
                continue

            try:
                transcript = transcript_for(item, item["audio"])
                history = []
                for turn in item.get("history", []):
                    converted = dict(turn)
                    if converted.get("audio"):
                        converted["text"] = transcript_for(item, converted["audio"])
                        converted.pop("audio", None)
                    history.append(converted)
            except Exception as exc:
                failure_rows.append({"sample_id": sample_id, "reason": f"PhoASR transcript: {type(exc).__name__}: {exc}"})
                continue

            # One official ToolEnv per trajectory. No state or environment file is read here.
            try:
                env = create_tool_env("train", sample_id, root)
                observations: list[dict] = []
                step_records: list[dict] = []
                replay_error = None
                compact_specs = compact_tool_specs(available, catalog)
                context = {key: value for key, value in item.get("context", {}).items() if key != "installed_apps"}
                for step, call in enumerate(gold):
                    if (sample_id, step) in seen:
                        duplicate_timesteps += 1
                        replay_error = f"duplicate timestep key at step {step}"
                        break
                    target = {"action": "tool_call", "tool": call["name"], "arguments": call.get("arguments", {})}
                    parsed_target = parse_action(json.dumps(target, ensure_ascii=False))
                    if parsed_target != target:
                        replay_error = f"step {step}: target did not round-trip through production parser"
                        break
                    args_obj = target["arguments"]
                    validation = env.validate(target["tool"], args_obj)
                    if not validation.get("ok"):
                        replay_error = f"step {step}: ToolEnv rejected gold call: {validation.get('error')}"
                        break
                    prompt_payload = {"transcript": transcript, "context": context, "history": history,
                                      "available_tools": compact_specs, "observations": observations}
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False,
                                                                     separators=(",", ":"), default=str)},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
                    ]
                    step_records.append({"sample_id": sample_id, "step_index": step,
                                         "messages": messages, "target": target})
                    result = env.call(target["tool"], args_obj)
                    observations.append({"tool": target["tool"], "arguments": args_obj, "result": result})
                if replay_error is None:
                    expected_state = item.get("label", {}).get("final_state", {})
                    replay_state = env.get_state()
                    mismatched_apps = [app for app, state in expected_state.items() if replay_state.get(app) != state]
                    if mismatched_apps:
                        replay_error = f"final state mismatch for apps: {mismatched_apps}"
                if replay_error:
                    failure_rows.append({"sample_id": sample_id, "reason": replay_error})
                    continue

                stop_target = {"action": "stop"}
                stop_message = {"role": "assistant", "content": json.dumps(stop_target, separators=(",", ":"))}
                stop_payload = {"transcript": transcript, "context": context, "history": history,
                                "available_tools": compact_specs, "observations": observations}
                step_records.append({"sample_id": sample_id, "step_index": len(gold),
                                     "messages": [
                                         {"role": "system", "content": SYSTEM_PROMPT},
                                         {"role": "user", "content": json.dumps(stop_payload, ensure_ascii=False,
                                                                                      separators=(",", ":"), default=str)},
                                         stop_message,
                                     ], "target": stop_target})
                for record in step_records:
                    key = (sample_id, record["step_index"])
                    if key in seen:
                        duplicate_timesteps += 1
                    seen.add(key)
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    examples_written += 1
                    target_counts["STOP" if record["target"]["action"] == "stop" else record["target"]["tool"]] += 1
                step_counts.append(len(gold))
            except Exception as exc:
                failure_rows.append({"sample_id": sample_id, "reason": f"ToolEnv replay: {type(exc).__name__}: {exc}"})
            if trajectory_index % 100 == 0:
                print(f"replayed {trajectory_index}/{len(train_ids)} trajectories; examples={examples_written}; failures={len(failure_rows)}", flush=True)

    failures_path = outdir / "replay_failures.jsonl"
    with failures_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in failure_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "qwen-next-action-v1",
        "data_root_split": "train",
        "original_train_trajectories": len(train_ids),
        "fixed_validation_trajectories_excluded": len(val_ids),
        "train_validation_overlap": len(set(train_ids) & set(val_ids)),
        "successful_trajectories_replayed": len(step_counts),
        "generated_timestep_examples": examples_written,
        "average_gold_tool_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
        "max_gold_tool_steps": max(step_counts, default=0),
        "stop_examples": target_counts.get("STOP", 0),
        "ask_missing_info_examples": target_counts.get("sys.ask_missing_info", 0),
        "replay_failures": len(failure_rows),
        "duplicate_timestep_count": duplicate_timesteps,
        "target_counts": dict(sorted(target_counts.items())),
        "transcript_source": "PhoASR public (local inference/cache; no train audio_text labels)",
        "prompt_serialization": "Agent.next_action system prompt and user payload; observations retain tool, arguments, result",
        "training_data": {"path": output_path.name, "sha256": digest(output_path)},
        "replay_failures_file": failures_path.name,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
