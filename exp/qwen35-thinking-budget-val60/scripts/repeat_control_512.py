from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/qwen35-thinking-budget-val60/output"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from run_pilot import DIGEST, HOST, MODEL, OUT as RUN_OUT, api_json, read_jsonl, run_block, warmup, write_json  # noqa: E402


def main() -> int:
    if RUN_OUT != OUT:
        raise RuntimeError("Experiment output path mismatch")
    label = "512-repeat"
    paths = [OUT / f"{kind}-{label}.jsonl" for kind in ("predictions", "step-traces", "requests")]
    paths += [OUT / f"run-{label}.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Repeat-control outputs exist; refusing overwrite")
    ids = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    source_a = json.loads((ROOT / "exp/num-predict-512-vs-768-val/output/cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    source_b = json.loads((ROOT / "exp/inference-latency-optimization-val60/output/cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    if ids != source_a or ids != source_b or len(ids) != 60:
        raise RuntimeError("Fixed cohort mismatch")
    version = api_json("/api/version").get("version")
    model = next((m for m in api_json("/api/tags").get("models", []) if m.get("name") == MODEL), None)
    if version != "0.40.14" or not model or model.get("digest") != DIGEST:
        raise RuntimeError(f"Ollama fingerprint mismatch: {version=} {model=}")
    source = ROOT / "exp/current-config-error-analysis-val309/output/transcript_cache_val309.jsonl"
    transcript_rows = read_jsonl(source)
    transcripts = {x["id"]: x["transcript"] for x in transcript_rows}
    from lifemate.utils import find_data_root, load_jsonl
    data_root = find_data_root("train+public_test/train+public_test")
    items = {x["id"]: x for x in load_jsonl(data_root / "train/train.jsonl")}
    if set(ids) - transcripts.keys() or set(ids) - items.keys():
        raise RuntimeError("Fixed cohort input coverage failed")
    started = time.time()
    for i in range(4):
        block = ids[i * 15:(i + 1) * 15]
        warm = warmup(512, 512)
        write_json(OUT / f"warmup-{label}-round{i + 1}.json", warm)
        run_block(512, i + 1, block, transcripts, items, 512, label=label)
    predictions = read_jsonl(OUT / f"predictions-{label}.jsonl")
    requests = read_jsonl(OUT / f"requests-{label}.jsonl")
    if len(predictions) != 60 or {x["id"] for x in predictions} != set(ids):
        raise RuntimeError("Repeat control did not finish the exact cohort")
    write_json(OUT / f"run-{label}.json", {"branch": label, "num_predict": 512,
        "num_ctx": 8192, "num_batch": 512, "effective_num_batch": 512,
        "model": MODEL, "model_digest": DIGEST, "ollama_version": version,
        "seed": 17, "samples_completed": len(predictions), "requests": len(requests),
        "started_epoch": started, "ended_epoch": time.time(),
        "purpose": "same-setting control repeat after archived-control divergence"})
    print(json.dumps({"branch": label, "samples": len(predictions), "requests": len(requests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
