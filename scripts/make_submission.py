from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a competition-format submission JSONL")
    parser.add_argument("--split", choices=["public_test", "train"], default="public_test")
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="outputs/submissions/submission.jsonl")
    parser.add_argument("--limit", type=int, help="optional small smoke run")
    parser.add_argument("--resume", action="store_true", help="continue after an existing valid output prefix")
    args = parser.parse_args()
    seed_everything()
    root = find_data_root(args.data_root)
    items = list(load_jsonl(root / args.split / f"{args.split}.jsonl"))
    pipeline = LifeMatePipeline(root, args.split)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as existing:
            for index, line in enumerate(existing):
                row = json.loads(line)
                if index >= len(items) or row.get("id") != items[index].get("id"):
                    raise SystemExit(f"Existing output is not a matching prefix at row {index + 1}")
                if set(row) != {"id", "tool_calls", "state_hash"}:
                    raise SystemExit(f"Existing output has an invalid schema at row {index + 1}")
                count += 1
        if args.limit is not None and count > args.limit:
            raise SystemExit("Existing output is longer than --limit")
    total_latency = 0.0
    new_count = 0
    started = time.perf_counter()
    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8", newline="\n") as out:
        for item in items[count:]:
            if args.limit is not None and count >= args.limit:
                break
            result = pipeline.run(item)
            row = {"id": result["id"], "tool_calls": result["tool_calls"], "state_hash": result["state_hash"]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            out.flush()
            total_latency += result["latency_seconds"]
            new_count += 1
            if new_count % 25 == 0:
                elapsed = time.perf_counter() - started
                print(f"{count}/{len(items)} rows; avg {total_latency / new_count:.2f}s/sample for this run; elapsed {elapsed:.1f}s", flush=True)
    elapsed = time.perf_counter() - started
    average_latency = total_latency / new_count if new_count else 0.0
    print(f"{count} rows -> {out_path}; added {new_count}; elapsed {elapsed:.1f}s; avg {average_latency:.2f}s/sample for this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

