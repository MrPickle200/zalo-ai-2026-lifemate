from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\zalo-ai-challenge-2026\lifemate")
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "exp/btc-scoring/scripts"),
    str(ROOT / "exp/quan-continuation-cot-v1/scripts"),
]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/adapter-final-4epoch-q8-benchmark/pilot60"
Q8_PROV = EXP / "output/adapter-final-4epoch-q8-benchmark/model-provenance"
BASE_OUT = EXP / "output/two-policy-val60"
TAG = "lifemate-qwen35-phosynth-targeted-4epoch-q8-0:latest"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def boot_ci(differences, seed=20261115, draws=10000):
    rng = random.Random(seed)
    n = len(differences)
    values = sorted(100 * statistics.mean(rng.choices(differences, k=n)) for _ in range(draws))
    return [values[int(.025 * draws)], values[int(.975 * draws) - 1]]


def run():
    base = load_module("two_policy_for_q8_pilot", EXP / "scripts/run_two_policy_val60.py")
    ev, compare, prompt, runner, ids, cohort, items, main, history, data, model_record, final4, cohort_manifest = base.build()
    q8 = read_json(Q8_PROV / "ollama-models.json")
    if q8["tag"] != TAG or q8["status"] != "registered":
        raise RuntimeError("Q8 Ollama tag differs from registered manifest.")
    gguf = ROOT / q8["gguf"]["path"]
    if sha(gguf) != q8["gguf"]["sha256"]:
        raise RuntimeError("Q8 GGUF hash differs from registration manifest.")
    runner.MODEL = TAG
    runner.EXPECTED_DIGEST = q8["ollama_digest"]
    version, digest = runner.model_fingerprint()
    if version != q8["ollama_version"] or digest != q8["ollama_digest"]:
        raise RuntimeError("Q8 Ollama version or digest changed.")
    runner.inject_cot = prompt.focused_inject
    branch_out = OUT / "inference/q8_fixed_ask_stop"
    branch_out.mkdir(parents=True, exist_ok=True)
    runner.OUT = branch_out
    baseline_rows = read_jsonl(BASE_OUT / "inference/fixed_ask_stop/predictions-COT.jsonl")
    baseline_details = read_json(BASE_OUT / "paired_details.json")
    if [row["id"] for row in baseline_rows] != cohort or [row["id"] for row in baseline_details] != cohort:
        raise RuntimeError("Archived Q4 ask-stop pilot60 differs from locked cohort.")
    inputs = [
        Path(__file__).resolve(),
        EXP / "scripts/run_two_policy_val60.py",
        EXP / "scripts/build_final4_q8.py",
        EXP / "scripts/register_ollama_final4_q8.py",
        Q8_PROV / "merge-manifest.json",
        Q8_PROV / "conversion-manifest.json",
        Q8_PROV / "ollama-models.json",
        EXP / "output/two-policy-val60/comparison_lock.json",
        EXP / "output/two-policy-val60/paired_details.json",
        EXP / "output/two-policy-val60/inference/fixed_ask_stop/predictions-COT.jsonl",
        ROOT / "src/lifemate/pipeline.py",
        ROOT / "src/lifemate/agent.py",
        ROOT / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        ROOT / "exp/qwen35-thinking-budget-val60/output/cohort_manifest.json",
    ]
    lock = {
        "experiment": "final4-q8-vs-q4-ask-stop-pilot60",
        "cohort": {"ids": cohort, "n": 60, "source": "existing enriched fixed pilot60"},
        "q8": {"model": TAG, "digest": digest, "gguf_sha256": sha(gguf)},
        "q4": {"model": model_record["tag"] if "tag" in model_record else base.MODEL,
               "digest": model_record["digest"]},
        "settings": {"thinking": True, "temperature": 0, "protocol": "JSON",
                     "num_predict": 768, "num_ctx": 8192, "num_batch": 512,
                     "seed": 17, "max_steps": 5,
                     "asr": "cached PhoWhisper Large v2 main/history",
                     "prompt_sha256": prompt.prompt_hash(prompt.focused_inject),
                     "stop_after_successful_ask": True},
        "inputs_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in inputs
        },
        "caveat": "Enriched pilot60. Local RTX 4050 Q8 offload and latency are not A30 runtime."
    }
    lock_path = OUT / "comparison_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Existing Q8 pilot lock differs from inputs.")
    write_json(lock_path, lock)
    import lifemate.pipeline as pipeline_module
    original_source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
    try:
        base.patch_pipeline(pipeline_module, original_source, True)
        for round_no, start in enumerate(range(0, 60, 15), 1):
            block = cohort[start:start + 15]
            done = {row["id"] for row in read_jsonl(branch_out / "predictions-COT.jsonl")}
            todo = [sid for sid in block if sid not in done]
            if not todo:
                continue
            runner.warmup("COT", round_no)
            runner.run_block("COT", round_no, todo, main, items, cot=True)
            print(f"Q8 completed block {round_no}/4 ({start + len(block)}/60)", flush=True)
    finally:
        base.patch_pipeline(pipeline_module, original_source, False)
    analyze(cohort, items, data, baseline_details, branch_out, lock)


def analyze(cohort, items, data, baseline_details, branch_out, lock):
    from score_btc import score_sample, call_key
    rows = read_jsonl(branch_out / "predictions-COT.jsonl")
    if [row["id"] for row in rows] != cohort:
        raise RuntimeError("Q8 predictions incomplete or not canonical order.")
    baseline = {row["id"]: row["fixed_ask_stop"] for row in baseline_details}
    details = []
    for row in rows:
        sid = row["id"]
        result = score_sample(sid, items[sid], row, data)
        calls = row.get("tool_calls") or []
        asks = [call for call in calls if call.get("name") == "sys.ask_missing_info"]
        gold_asks = [call for call in items[sid].get("label", {}).get("tool_calls", [])
                     if call.get("name") == "sys.ask_missing_info"]
        details.append({
            "id": sid,
            "q4": baseline[sid],
            "q8": {
                "after_gate": result["official_sample_score"],
                "before_gate": result["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match": result["hash_match"],
                "over_30s": result["runtime_over_30s"],
                "runtime_failure": result["runtime_failure"],
                "matched_calls": result["matched_calls"],
                "unmatched_calls": result["unmatched_submitted_calls"],
                "latency_seconds": row.get("latency_seconds"),
                "ask_calls": len(asks),
                "exact_gold_ask": any(call_key(a) == call_key(g) for a in asks for g in gold_asks),
            },
        })
    q8 = [detail["q8"] for detail in details]
    q4 = [detail["q4"] for detail in details]
    after_diffs = [x["after_gate"] - y["btc_after_runtime_gate"] for x, y in zip(q8, q4)]
    before_diffs = [x["before_gate"] - y["btc_before_runtime_gate"] for x, y in zip(q8, q4)]
    summary = {
        "experiment": lock["experiment"],
        "cohort": lock["cohort"],
        "q8": {
            "btc_after_gate": 100 * statistics.mean(x["after_gate"] for x in q8),
            "btc_before_gate": 100 * statistics.mean(x["before_gate"] for x in q8),
            "over_30s": sum(x["over_30s"] for x in q8),
            "runtime_errors": sum(x["runtime_failure"] for x in q8),
            "mean_latency_seconds": statistics.mean(x["latency_seconds"] for x in q8),
            "hash_matches": sum(x["hash_match"] for x in q8),
            "exact_gold_asks": sum(x["exact_gold_ask"] for x in q8),
            "ask_calls": sum(x["ask_calls"] for x in q8),
        },
        "q4_baseline": {
            "btc_after_gate": 100 * statistics.mean(x["btc_after_runtime_gate"] for x in q4),
            "btc_before_gate": 100 * statistics.mean(x["btc_before_runtime_gate"] for x in q4),
            "over_30s": sum(x["over_30s"] for x in q4),
            "mean_latency_seconds": statistics.mean(x["latency_seconds"] for x in q4),
        },
        "paired": {
            "after_gate_points": 100 * statistics.mean(after_diffs),
            "after_gate_ci95": boot_ci(after_diffs),
            "before_gate_points": 100 * statistics.mean(before_diffs),
            "before_gate_ci95": boot_ci(before_diffs, seed=20261116),
            "wins_ties_losses": [sum(x > 0 for x in after_diffs),
                                 sum(x == 0 for x in after_diffs),
                                 sum(x < 0 for x in after_diffs)],
        },
        "caveat": lock["caveat"],
    }
    write_json(OUT / "paired_details.json", details)
    write_json(OUT / "summary.json", summary)
    s = summary
    lines = [
        "# Final4 Q8_0 vs Q4_K_M with ask-stop — enriched pilot60",
        "",
        "Same final4 adapter, focused prompt, fixed thinking, cached PhoWhisper, JSON, temperature 0, cap768, context8192, batch512, seed17, five turns and terminal successful sys.ask_missing_info guard. The Q4 baseline is archived from the same enriched cohort; Q8 ran fresh. Local RTX 4050 Q8 offload and timing are not A30 performance.",
        "",
        "| Model | BTC before gate | BTC after local 30s gate | >30s | Mean sec | Hashes | Exact asks |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Q4_K_M baseline | {s['q4_baseline']['btc_before_gate']:.2f} | {s['q4_baseline']['btc_after_gate']:.2f} | {s['q4_baseline']['over_30s']}/60 | {s['q4_baseline']['mean_latency_seconds']:.2f} | - | - |",
        f"| Q8_0 | {s['q8']['btc_before_gate']:.2f} | {s['q8']['btc_after_gate']:.2f} | {s['q8']['over_30s']}/60 | {s['q8']['mean_latency_seconds']:.2f} | {s['q8']['hash_matches']}/60 | {s['q8']['exact_gold_asks']}/60 |",
        "",
        f"Paired Q8−Q4 after gate: {s['paired']['after_gate_points']:+.2f} points (95% CI {s['paired']['after_gate_ci95']}); before gate: {s['paired']['before_gate_points']:+.2f} (95% CI {s['paired']['before_gate_ci95']}). Wins/ties/losses: {s['paired']['wins_ties_losses']}.",
        "",
        "This is a diagnostic screen, not an unbiased full-val or Public LB estimate. A30 runtime must be measured separately before submission.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()

