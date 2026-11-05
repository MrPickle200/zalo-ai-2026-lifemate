from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect allowed sample metadata, audio paths, and tool catalog")
    parser.add_argument("--split", choices=["train", "public_test"], default="train")
    parser.add_argument("--data-root")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    root = find_data_root(args.data_root)
    split_dir = root / args.split
    manifest = split_dir / f"{args.split}.jsonl"
    catalog = json.loads((split_dir / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    count = 0
    for item in load_jsonl(manifest):
        if count < args.limit:
            installed = item.get("context", {}).get("installed_apps", {})
            print(json.dumps({"id": item.get("id"), "keys": list(item), "audio": item.get("audio"),
                              "audio_exists": (split_dir / item.get("audio", "")).is_file(),
                              "context_keys": list(item.get("context", {})), "history_turns": len(item.get("history", [])),
                              "installed_apps": len(installed), "available_tools": sum(len(a.get("tools", [])) for a in installed.values())},
                             ensure_ascii=False))
        count += 1
    print(json.dumps({"split": args.split, "samples": count, "catalog_apps": len(catalog.get("apps", [])),
                      "catalog_tools": sum(len(a.get("tools", [])) for a in catalog.get("apps", []))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

