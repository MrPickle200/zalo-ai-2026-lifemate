from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/qwen35-thinking-budget-val60/output"
CAPS = (512, 768, 1024, 1536)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def deduplicate(path: Path, key: str) -> dict[str, Any]:
    rows = read_rows(path)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_key = str(row[key])
        previous = unique.get(item_key)
        if previous is not None and previous != row:
            raise RuntimeError(f"Conflicting duplicate {key}={item_key} in {path.name}; leaving all files untouched")
        unique.setdefault(item_key, row)
    result = list(unique.values())
    return {"path": path, "rows": rows, "unique": result,
            "duplicates_removed": len(rows) - len(result), "key": key}


def main() -> int:
    audit_path = OUT / "recovery_audit.json"
    if audit_path.exists():
        raise FileExistsError("Recovery audit already exists; refusing to rewrite outputs")
    ids = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    all_results = []
    for cap in CAPS:
        for kind, key in (("predictions", "id"), ("step-traces", "id"), ("requests", "request_id")):
            path = OUT / f"{kind}-{cap}.jsonl"
            all_results.append(deduplicate(path, key))

    # Audit every branch before mutating any artifact.
    by_name = {(x["path"].name.split("-")[0], int(x["path"].stem.split("-")[-1])): x for x in all_results}
    branches = {}
    for cap in CAPS:
        pred = by_name[("predictions", cap)]
        trace = by_name[("step", cap)] if ("step", cap) in by_name else next(
            x for x in all_results if x["path"].name == f"step-traces-{cap}.jsonl")
        req = by_name[("requests", cap)]
        if len(pred["unique"]) != 60 or {x["id"] for x in pred["unique"]} != set(ids):
            raise RuntimeError(f"cap={cap} unique predictions do not match the 60-ID cohort")
        if len(trace["unique"]) != 60 or {x["id"] for x in trace["unique"]} != set(ids):
            raise RuntimeError(f"cap={cap} unique traces do not match the 60-ID cohort")
        if {x["sample_id"] for x in req["unique"]} != set(ids):
            raise RuntimeError(f"cap={cap} unique request telemetry does not cover cohort IDs")
        branches[str(cap)] = {"predictions_raw_rows": len(pred["rows"]), "predictions_unique": 60,
            "prediction_duplicates_removed": pred["duplicates_removed"],
            "traces_raw_rows": len(trace["rows"]), "traces_unique": 60,
            "trace_duplicates_removed": trace["duplicates_removed"],
            "requests_raw_rows": len(req["rows"]), "requests_unique": len(req["unique"]),
            "request_duplicates_removed": req["duplicates_removed"],
            "all_duplicate_contents_identical": True,
            "request_ids_unique_after_recovery": len({x["request_id"] for x in req["unique"]}) == len(req["unique"])}

    # Keep the flushed files for audit, and replace canonical filenames only after all checks pass.
    for result in all_results:
        original: Path = result["path"]
        archive = original.with_name(original.stem + "-raw-prefix-flush" + original.suffix)
        if archive.exists():
            raise FileExistsError(f"Raw archive already exists: {archive}")
        original.rename(archive)
        write_rows(original, result["unique"])
    for cap in CAPS:
        meta_path = OUT / f"run-{cap}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update({"samples_completed": 60, "status": "complete_after_exact_duplicate_recovery",
                    "output_recovery": "repeated flushed prefixes were byte-semantically identical by sample/request ID; original files retained with -raw-prefix-flush suffix"})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"reason": "Initial runner flushed the entire current block prefix after every sample, duplicating earlier rows in each block output.",
        "recovery": "Verified all repeated IDs/request IDs had exactly identical JSON content, archived raw JSONL files, and retained the first occurrence per ID/request ID.",
        "inference_calls_repeated": False, "prediction_toolenvs_per_sample_per_cap": 1,
        "cohort_ids": len(ids), "branches": branches,
        "canonical_artifacts": "predictions-*.jsonl, step-traces-*.jsonl, requests-*.jsonl",
        "raw_archives_suffix": "-raw-prefix-flush.jsonl"}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
