from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate submission order/schema using manifest and official pure schema checker")
    parser.add_argument("--split", choices=("public_test", "train"), default="public_test")
    parser.add_argument("--data-root")
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()
    root = find_data_root(args.data_root)
    split_dir = root / args.split
    items = list(load_jsonl(split_dir / f"{args.split}.jsonl"))
    catalog = json.loads((split_dir / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))
    from zaic_tools import schema_validate

    path = Path(args.submission)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    if len(rows) != len(items):
        errors.append(f"row count {len(rows)} != manifest count {len(items)}")
    seen = set()
    for index, (row, item) in enumerate(zip(rows, items)):
        expected_id = item.get("id")
        sample_id = row.get("id") if isinstance(row, dict) else None
        if sample_id in seen:
            errors.append(f"row {index + 1} duplicates id {sample_id}")
        seen.add(sample_id)
        if sample_id != expected_id:
            errors.append(f"row {index + 1} id {sample_id!r} != manifest id {expected_id!r}")
        if not isinstance(row, dict) or set(row) != {"id", "tool_calls", "state_hash"}:
            errors.append(f"row {index + 1} has invalid submission keys")
            continue
        if not isinstance(row["tool_calls"], list):
            errors.append(f"row {index + 1} tool_calls is not a list")
            continue
        if not isinstance(row["state_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["state_hash"]):
            errors.append(f"row {index + 1} state_hash is not a lowercase SHA-256 hex digest")
        allowed = {"sys.ask_missing_info"}
        for app in item.get("context", {}).get("installed_apps", {}).values():
            allowed.update(app.get("tools", []))
        for call_index, result in enumerate(schema_validate(row["tool_calls"], catalog, allowed)):
            if not result["ok"]:
                errors.append(f"{sample_id} call {call_index} {result['name']}: {result['error']}")
    print(json.dumps({"split": args.split, "manifest_rows": len(items), "submission_rows": len(rows),
                      "unique_ids": len(seen), "schema_errors": sum(" call " in x for x in errors),
                      "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
