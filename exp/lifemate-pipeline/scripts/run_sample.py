from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline LifeMate pipeline on one sample")
    parser.add_argument("--sample", required=True, help="sample id or path to a one-line/sample JSON file")
    parser.add_argument("--split", default="train", choices=["train", "public_test"])
    parser.add_argument("--data-root")
    parser.add_argument("--output", help="optional output JSON path")
    args = parser.parse_args()
    seed_everything()
    root = find_data_root(args.data_root)
    candidate = Path(args.sample)
    if candidate.is_file():
        item = json.loads(candidate.read_text(encoding="utf-8"))
        split = args.split
    else:
        split = args.split
        manifest = root / split / f"{split}.jsonl"
        item = next((row for row in load_jsonl(manifest) if row["id"] == args.sample), None)
        if item is None:
            raise SystemExit(f"Sample {args.sample!r} not found in {manifest}")
    result = LifeMatePipeline(root, split).run(item)
    output = {k: result[k] for k in ("id", "tool_calls", "state_hash")}
    encoded = json.dumps(output, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    print(f"transcript={result['transcript']} latency={result['latency_seconds']:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

