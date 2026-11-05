from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.tool_executor import available_tool_names
from lifemate.utils import find_data_root, load_jsonl


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


def normalize_calls(item: dict, catalog_tools: dict) -> list[dict]:
    calls = item.get("label", {}).get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{item.get('id')}: label.tool_calls must be a list")
    allowed = set(available_tool_names(item))
    result = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or set(call) - {"name", "arguments"}:
            raise ValueError(f"{item.get('id')} call {index}: unexpected call metadata/shape")
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in allowed or name not in catalog_tools:
            raise ValueError(f"{item.get('id')} call {index}: unknown or unavailable tool {name!r}")
        if not isinstance(args, dict):
            raise ValueError(f"{item.get('id')} call {index}: arguments must be an object")
        spec = catalog_tools[name].get("parameters", {})
        unknown = set(args) - set(spec.get("properties", {}))
        if unknown:
            raise ValueError(f"{item.get('id')} call {index}: unknown argument keys {sorted(unknown)}")
        result.append({"name": name, "arguments": args})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and audit the fixed train-only Qwen SFT corpus")
    parser.add_argument("--data-root")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output-dir", default="exp/qwen3-4b-lifemate-lora-v1/output/model")
    args = parser.parse_args()

    root = find_data_root(args.data_root)
    all_items = {item["id"]: item for item in load_jsonl(root / "train" / "train.jsonl")}
    train_ids = json.loads((Path(args.split_dir) / "train_ids.json").read_text(encoding="utf-8"))
    val_ids = json.loads((Path(args.split_dir) / "val_ids.json").read_text(encoding="utf-8"))
    if len(train_ids) != 1236 or len(val_ids) != 309 or set(train_ids) & set(val_ids):
        raise SystemExit("Expected the fixed 1,236/309 split with no overlap")
    catalog = json.loads((root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    tools = {tool["name"]: tool for app in catalog.get("apps", []) for tool in app.get("tools", [])}
    tools["sys.ask_missing_info"] = {"parameters": {"properties": {"tool_name": {}, "params": {}}}}

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    output_path = outdir / "sft_train.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample_id in train_ids:
            item = all_items[sample_id]
            calls = normalize_calls(item, tools)
            bucket = "no_action" if not calls else "single_call" if len(calls) == 1 else "multi_call"
            counts[bucket] += 1
            if any(c["name"] == "sys.ask_missing_info" for c in calls):
                counts["missing_info_samples"] += 1
            schema_context = {"system_prompt": SYSTEM_PROMPT,
                              "available_tools": compact_tool_specs(available_tool_names(item), catalog)}
            target = {"tool_calls": calls}
            record = {
                "sample_id": sample_id,
                "user_transcript": item.get("audio_text", ""),
                "tool_schema_context": schema_context,
                "assistant_target": target,
                # Standard chat SFT shape; target uses the exact ordered submission call representation.
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT + " Trả về JSON theo dạng {\"tool_calls\":[{\"name\":\"...\",\"arguments\":{}}]} ."},
                    {"role": "user", "content": json.dumps({
                        "transcript": item.get("audio_text", ""),
                        "context": {k: v for k, v in item.get("context", {}).items() if k != "installed_apps"},
                        "history": item.get("history", []),
                        "available_tools": schema_context["available_tools"],
                        "observations": [],
                    }, ensure_ascii=False, separators=(",", ":"))},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
                ],
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {"training_samples": len(train_ids), "validation_samples_excluded": len(val_ids),
               "split_overlap": 0, "target_format": "ordered production {tool_calls:[{name,arguments}]} list",
               "counts": dict(counts), "dataset": str(output_path)}
    (outdir / "dataset-audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
