from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.utils import find_data_root, load_jsonl

SEED = 20261028
VAL_FRACTION = 0.2


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").casefold()))


def strata(item: dict) -> tuple[str, ...]:
    calls = item.get("label", {}).get("tool_calls", [])
    intent = calls[0].get("name", "out_of_scope") if calls else "out_of_scope"
    primary_app = intent.split(".", 1)[0]
    return (intent, primary_app,
            "multi" if len(calls) > 1 else "single",
            "missing" if any(c.get("name") == "sys.ask_missing_info" for c in calls) else "complete",
            "out_of_scope" if not calls else "in_scope")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a fixed, grouped train/validation split")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    args = parser.parse_args()
    root = find_data_root(args.data_root)
    items = list(load_jsonl(root / "train" / "train.jsonl"))
    # Candidate pairs share a normalized 3-token shingle; near matches are then verified.
    postings: dict[str, list[int]] = defaultdict(list)
    normalized = [normalize(item.get("audio_text", "")) for item in items]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    for i, text in enumerate(normalized):
        if text in exact:
            union(i, exact[text])
        else:
            exact[text] = i
        words = text.split()
        shingles = set(" ".join(words[k:k + 3]) for k in range(max(1, len(words) - 2)))
        candidates = set(j for shingle in shingles for j in postings[shingle])
        for j in candidates:
            if SequenceMatcher(None, text, normalized[j], autojunk=False).ratio() >= args.near_duplicate_threshold:
                union(i, j)
        for shingle in shingles:
            postings[shingle].append(i)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    # Deterministic randomized greedy allocation, balancing each task attribute.
    rng = random.Random(args.seed)
    strata_by_group = {g: Counter(s for i in inds for s in strata(items[i])) for g, inds in groups.items()}
    group_ids = list(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda g: -len(groups[g]))
    target = args.val_fraction * len(items)
    totals = Counter(s for item in items for s in strata(item))
    val_counts: Counter = Counter()
    val_groups: set[int] = set()
    val_size = 0
    for g in group_ids:
        n = len(groups[g])
        if val_size >= target:
            break
        # Incremental change in normalized squared distribution/size error.
        delta = (((val_size + n - target) ** 2 - (val_size - target) ** 2) / max(target, 1) ** 2) * len(totals)
        for feature, count in strata_by_group[g].items():
            desired = max(totals[feature] * args.val_fraction, 1)
            delta += (2 * (val_counts[feature] - totals[feature] * args.val_fraction) * count + count ** 2) / desired
        if delta < 0:
            val_groups.add(g)
            val_counts.update(strata_by_group[g])
            val_size += n
    val_items = [i for g in val_groups for i in groups[g]]
    train_items = [i for i in range(len(items)) if find(i) not in val_groups]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, indices in (("train_ids.json", train_items), ("val_ids.json", val_items)):
        (out / name).write_text(json.dumps([items[i]["id"] for i in indices], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"seed": args.seed, "method": "grouped greedy stratification over first intent, primary app, call count, missing-info and out-of-scope; normalized transcript groups at character similarity >= threshold", "near_duplicate_threshold": args.near_duplicate_threshold,
              "train_count": len(train_items), "validation_count": len(val_items),
              "audio_type_metadata_available": False,
              "unique_groups": len(groups), "multi_sample_groups": sum(len(v) > 1 for v in groups.values()),
              "validation_distributions": {field: dict(Counter(strata(items[i])[k] for i in val_items)) for k, field in enumerate(("first_intent", "primary_app", "single_multi", "missing_info", "scope"))},
              "train_distributions": {field: dict(Counter(strata(items[i])[k] for i in train_items)) for k, field in enumerate(("first_intent", "primary_app", "single_multi", "missing_info", "scope"))}}
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
