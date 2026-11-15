from __future__ import annotations
import hashlib, importlib.util, json, random, statistics, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
BASE_EVAL = EXP / "output/val309-checkpoint-eval"
OUT = EXP / "output/adapter-final-1epoch-benchmark/val309"
MODEL = "lifemate-qwen35-phosynth-targeted-1epoch-q4km:latest"
SEED = 17
BLOCK = 15

def load_eval():
    path = EXP / "scripts/evaluate_val309_checkpoints.py"
    spec = importlib.util.spec_from_file_location("locked_qwen35_checkpoint_eval_for_adapter_final1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

ev = load_eval()

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_jsonl(path: Path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if path.exists() else []

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def build():
    compare, prompt, runner, ids, items, main, history, data, prior_payload = ev.context()
    q4 = EXP / "output/adapter-final-1epoch-benchmark/gguf/adapter-final-1epoch-Q4_K_M.gguf"
    conversion = read_json(EXP / "output/adapter-final-1epoch-benchmark/model-provenance/conversion-manifest.json")
    if not q4.is_file() or sha(q4) != conversion["q4_k_m_gguf"]["sha256"]:
        raise RuntimeError("Adapter Q4 GGUF does not match its conversion manifest")
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=30) as response:
        tags = {row["name"]: row["digest"] for row in json.load(response)["models"]}
    if MODEL not in tags:
        raise RuntimeError(f"Benchmark Ollama model is missing: {MODEL}")
    script = Path(__file__).resolve()
    files = [script, q4, EXP / "output/adapter-final-1epoch-benchmark/model-provenance/conversion-manifest.json",
             EXP / "output/adapter-final-1epoch-benchmark/model-provenance/adapter-archive-manifest.json",
             EXP / "output/adapter_final-1-epoch.zip", BASE_EVAL / "comparison_lock.json",
             BASE_EVAL / "inference/step120/predictions-COT.jsonl", ev.BASELINE]
    inputs = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}
    lock = {
        "experiment": "qwen35-sft-pho-synthetic-v1-adapter-final-1epoch-benchmark",
        "cohort": prior_payload["cohort"], "prior_evaluation_lock_sha256": sha(BASE_EVAL / "comparison_lock.json"),
        "baseline": prior_payload["baseline"],
        "step120": {"model": prior_payload["checkpoints"]["step120"]["model"],
                     "digest": prior_payload["checkpoints"]["step120"]["digest"],
                     "predictions_sha256": sha(BASE_EVAL / "inference/step120/predictions-COT.jsonl")},
        "adapter_final1": {"model": MODEL, "digest": tags[MODEL], "q4_sha256": sha(q4),
                           "adapter_archive_sha256": sha(EXP / "output/adapter_final-1-epoch.zip"),
                           "conversion_manifest_sha256": sha(EXP / "output/adapter-final-1epoch-benchmark/model-provenance/conversion-manifest.json")},
        "prompt": prior_payload["prompt"], "settings": prior_payload["settings"],
        "asr": "Cached PhoWhisper Large v2 main val309 + history val692",
        "input_sha256": inputs,
    }
    return compare, prompt, runner, ids, items, main, history, data, lock

def prepare():
    *_, lock = build()
    path = OUT / "comparison_lock.json"
    if path.exists() and read_json(path) != lock:
        raise RuntimeError("Existing adapter-final1 evaluation lock differs from current model or inputs")
    if not path.exists():
        write_json(path, lock)
    write_json(OUT / "cohort_manifest.json", lock["cohort"])
    print(json.dumps({"samples": 309, "model": lock["adapter_final1"], "prompt": lock["prompt"], "settings": lock["settings"]}, ensure_ascii=False, indent=2), flush=True)

def run():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use system Python so the evaluation reaches the local Ollama server")
    compare, prompt, runner, ids, items, main, history, data, lock = build()
    if read_json(OUT / "comparison_lock.json") != lock:
        raise RuntimeError("Evaluation lock no longer matches the model, prompt, or input files")
    runner.MODEL = MODEL
    runner.EXPECTED_DIGEST = lock["adapter_final1"]["digest"]
    runner.OUT = OUT / "inference/adapter_final1"
    runner.inject_cot_original = runner.inject_cot
    runner.inject_cot = prompt.focused_inject
    import lifemate.asr as asr_module
    asr_module.PhoASRModel = compare.CachedPhoWhisper
    asr_module.PhoASRModel.lookup = history
    runner.seed_everything(SEED)
    runner.OUT.mkdir(parents=True, exist_ok=True)
    pred_path = runner.OUT / "predictions-COT.jsonl"
    prior = read_jsonl(pred_path)
    if [r["id"] for r in prior] != ids[:len(prior)]:
        raise RuntimeError("Saved adapter predictions are not a canonical val309 prefix")
    for round_no, start in enumerate(range(0, len(ids), BLOCK), 1):
        block = ids[start:start + BLOCK]
        done = {r["id"] for r in read_jsonl(pred_path)}
        todo = [sid for sid in block if sid not in done]
        if not todo:
            continue
        runner.warmup("COT", round_no)
        while todo:
            before = len(read_jsonl(pred_path))
            try:
                runner.run_block("COT", round_no, todo, main, items, cot=True)
            except Exception:
                rows = read_jsonl(pred_path)
                if len(rows) <= before or not rows[-1].get("error"):
                    raise
            done = {r["id"] for r in read_jsonl(pred_path)}
            todo = [sid for sid in block if sid not in done]
        print(f"adapter_final1: block {round_no}/21 complete; {len(read_jsonl(pred_path))}/309", flush=True)
    rows = read_jsonl(pred_path)
    if [r["id"] for r in rows] != ids:
        raise RuntimeError("Adapter inference did not save all 309 IDs in canonical order")
    print("adapter_final1: inference complete 309/309", flush=True)

def pct(values):
    return 100 * statistics.mean(values) if values else None

def label(item):
    value = item.get("label", {})
    return json.loads(value) if isinstance(value, str) else value

def analyze():
    compare, prompt, runner, ids, items, main, history, data, lock = build()
    if read_json(OUT / "comparison_lock.json") != lock:
        raise RuntimeError("Evaluation lock no longer matches current inputs")
    predictions = {
        "best_baseline": read_jsonl(ev.BASELINE),
        "step120": read_jsonl(BASE_EVAL / "inference/step120/predictions-COT.jsonl"),
        "adapter_final1": read_jsonl(OUT / "inference/adapter_final1/predictions-COT.jsonl"),
    }
    by_id = {}
    for key, rows in predictions.items():
        if [r["id"] for r in rows] != ids:
            raise RuntimeError(f"{key} does not have all canonical val309 predictions")
        by_id[key] = {r["id"]: r for r in rows}
    asks = {sid for sid in ids if any(c.get("name") == "sys.ask_missing_info" for c in label(items[sid]).get("tool_calls", []))}
    details = []
    for sid in ids:
        gold_asks = [c for c in label(items[sid]).get("tool_calls", []) if c.get("name") == "sys.ask_missing_info"]
        row = {"id": sid, "group": "audited_47_diagnostic" if sid in asks else "fresh_262_primary", "gold_ask": sid in asks}
        for key in predictions:
            pred = by_id[key][sid]
            score = ev.score_sample(sid, items[sid], pred, data)
            predicted_asks = [c for c in (pred.get("tool_calls") or []) if c.get("name") == "sys.ask_missing_info"]
            exact = any(ev.call_key(a) == ev.call_key(g) for a in predicted_asks for g in gold_asks)
            row[key] = {"btc": score["official_sample_score"],
                "btc_before_runtime_gate": score["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match": score["hash_match"], "runtime_error": bool(pred.get("error")),
                "error": pred.get("error"), "latency_seconds": pred.get("latency_seconds"),
                "ask": bool(predicted_asks), "ask_exact": exact}
        details.append(row)
    summaries = {}
    groups = {"all_val309": ids, "fresh_262_primary": [s for s in ids if s not in asks],
              "audited_47_diagnostic": [s for s in ids if s in asks]}
    for name, subset in groups.items():
        current = [r for r in details if r["id"] in set(subset)]
        sub = {}
        for key in predictions:
            lat = [r[key]["latency_seconds"] for r in current if r[key]["latency_seconds"] is not None]
            sub[key] = {"btc_0_100": pct([r[key]["btc"] for r in current]),
                "btc_before_runtime_gate_0_100": pct([r[key]["btc_before_runtime_gate"] for r in current]),
                "hash_matches": sum(r[key]["hash_match"] for r in current),
                "runtime_errors": sum(r[key]["runtime_error"] for r in current),
                "mean_latency_seconds": statistics.mean(lat) if lat else None,
                "gold_ask_exact": sum(r[key]["ask_exact"] for r in current),
                "ask_count": sum(r[key]["ask"] for r in current)}
        for key in ("step120", "adapter_final1"):
            ds = [r[key]["btc"] - r["best_baseline"]["btc"] for r in current]
            ungated = [r[key]["btc_before_runtime_gate"] - r["best_baseline"]["btc_before_runtime_gate"] for r in current]
            seed = 20261114 + (120 if key == "step120" else 1)
            sub[f"paired_{key}_minus_best_baseline"] = {"btc_delta_points": pct(ds),
                "bootstrap_95pct_CI_points": ev.boot_ci(ds, seed), "checkpoint_wins": sum(x > 0 for x in ds),
                "baseline_wins": sum(x < 0 for x in ds), "ties": sum(x == 0 for x in ds),
                "ungated_delta_points": pct(ungated), "ungated_bootstrap_95pct_CI_points": ev.boot_ci(ungated, seed + 1)}
        summaries[name] = sub
    summary = {"experiment": lock["experiment"], "cohort": {"count": 309, "fresh_primary": len(groups["fresh_262_primary"]),
        "audited_gold_ask_diagnostic": len(groups["audited_47_diagnostic"])},
        "branches": {"best_baseline": {"model": lock["baseline"]["model"], "predictions_reused": True, "status": "complete", "samples_completed": 309},
            "step120": {"model": lock["step120"]["model"], "predictions_reused": True, "status": "complete", "samples_completed": 309},
            "adapter_final1": {"model": MODEL, "digest": lock["adapter_final1"]["digest"], "predictions_reinferred": True, "status": "complete", "samples_completed": 309}},
        "code_sha256": {"benchmark_script": sha(Path(__file__).resolve())},
        "prompt_and_settings": {**lock["prompt"], **lock["settings"]}, "groups": summaries,
        "method": {"scoring": "Official BTC ToolEnv per sample; local runtime-gated score and pre-gate diagnostic reported separately",
            "baseline": "Reused exact prior baseline and step120 predictions; only adapter_final1 was newly inferred",
            "paired_by_id": True, "bootstrap_draws": 10000, "bootstrap_seed": 20261114},
        "caveat": "The adapter archive identifies Qwen/Qwen3.5-4B but does not include a hash of the training base snapshot. Inference latency was measured on local RTX 4050, not the competition A30."}
    write_json(OUT / "summary.json", summary)
    write_jsonl(OUT / "paired_details.jsonl", details)
    all_group = summaries["all_val309"]
    names = {"best_baseline": "Best baseline", "step120": "Checkpoint step 120", "adapter_final1": "Synthetic adapter after 1 epoch"}
    lines = ["# Adapter final 1 epoch: PhoWhisper val309", "", "All branches use the focused prompt, cached PhoWhisper Large main/history transcripts, thinking enabled, temperature 0, 768-token cap, and official BTC ToolEnv scoring.", "",
        "| Branch | BTC after 30s gate | Before gate | Hash matches | Runtime errors | Mean latency | Exact gold asks |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key in predictions:
        r = all_group[key]
        lines.append(f"| {names[key]} | {r['btc_0_100']:.2f} | {r['btc_before_runtime_gate_0_100']:.2f} | {r['hash_matches']}/309 | {r['runtime_errors']} | {r['mean_latency_seconds']:.2f}s | {r['gold_ask_exact']}/47 |")
    for key in ("step120", "adapter_final1"):
        d = all_group[f"paired_{key}_minus_best_baseline"]
        ci = d["bootstrap_95pct_CI_points"]
        lines += ["", f"{names[key]} vs best baseline: {d['btc_delta_points']:+.2f} points (paired 95% CI {ci[0]:+.2f} to {ci[1]:+.2f}); wins/ties/losses {d['checkpoint_wins']}/{d['ties']}/{d['baseline_wins']}."]
    lines += ["", "The public submission will use the highest after-gate BTC score among these three standard-configuration branches.", "", summary["caveat"], ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true"); p.add_argument("--run", action="store_true"); p.add_argument("--analyze", action="store_true")
    a = p.parse_args()
    if sum((a.prepare, a.run, a.analyze)) != 1: p.error("choose one of --prepare, --run, --analyze")
    if a.prepare: prepare()
    elif a.run: run()
    else: analyze()
if __name__ == "__main__": main()
