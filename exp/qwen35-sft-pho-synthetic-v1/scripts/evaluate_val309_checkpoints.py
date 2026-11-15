from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, random, statistics, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "exp/btc-scoring/scripts"), str(ROOT / "exp/quan-continuation-cot-v1/scripts")]
from score_btc import score_sample, call_key
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/val309-checkpoint-eval"
ARCHIVE = EXP / "output/checkpoint-60-120-steps.zip"
BASELINE = ROOT / "exp/ask-lookup-phowhisper-asr-val309/output/phowhisper-large/predictions-COT.jsonl"
BASE_MODEL = "lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
BASE_DIGEST = "fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
MODEL_TAGS = {
    "step60": "lifemate-qwen35-phosynth-val309-step60-q4km:latest",
    "step120": "lifemate-qwen35-phosynth-val309-step120-q4km:latest",
}
SEED = 17
BOOTSTRAP_SEED = 20261114
BLOCK_SIZE = 15


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def context():
    compare = load_module("phowhisper_val309_compare_reused", ROOT / "exp/ask-lookup-phowhisper-asr-val309/scripts/compare.py")
    prompt, runner, ids, items, main, history, data, old_lock = compare.build()
    if len(ids) != 309 or [row["id"] for row in read_jsonl(BASELINE)] != ids:
        raise RuntimeError("The locked PhoWhisper baseline does not cover canonical val309")
    if old_lock["prompt"]["sha256"] != "88b0264bd512a90368d0a64aa2241ff59da2660790c0725781358986bb140785":
        raise RuntimeError("Focused prompt hash differs from the best baseline")
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=30) as response:
        tags = {row["name"]: row["digest"] for row in json.load(response)["models"]}
    missing = [tag for tag in MODEL_TAGS.values() if tag not in tags]
    if missing:
        raise RuntimeError(f"Checkpoint Ollama models are missing: {missing}")
    states = {}
    with zipfile.ZipFile(ARCHIVE) as zf:
        for step in (60, 120):
            raw = zf.read(f"checkpoints/checkpoint-{step}/trainer_state.json")
            state = json.loads(raw)
            states[str(step)] = {"global_step": state["global_step"], "epoch": state["epoch"],
                                 "max_steps": state["max_steps"]}
    asr = compare.ASR
    inputs = [ROOT / "data/splits/val_ids.json", data / "train/train.jsonl",
              asr / "predictions.jsonl", asr / "history_predictions.jsonl", BASELINE,
              ROOT / "src/lifemate/agent.py", ROOT / "src/lifemate/pipeline.py",
              ROOT / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
              ROOT / "exp/ask-lookup-disambiguation-followup-val309/scripts/benchmark.py",
              ROOT / "exp/qwen35-model-size-colab-val309/scripts/run_compare.py",
              Path(__file__).resolve()]
    adapters = {}
    for step in (60, 120):
        prefix = f"checkpoints/checkpoint-{step}/"
        with zipfile.ZipFile(ARCHIVE) as zf:
            adapters[str(step)] = {name: hashlib.sha256(zf.read(prefix + name)).hexdigest()
                                   for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "chat_template.jinja")}
    payload = {
        "experiment": EXP.name,
        "cohort": {"count": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "baseline": {"model": BASE_MODEL, "digest": BASE_DIGEST,
                     "predictions": str(BASELINE.relative_to(ROOT)).replace("\\", "/"),
                     "predictions_sha256": sha(BASELINE), "btc_val309": 42.7067251339096},
        "checkpoints": {key: {"model": MODEL_TAGS[key], "digest": tags[MODEL_TAGS[key]],
                              "training_state": states[key.replace("step", "")], "adapter_sha256": adapters[key.replace("step", "")]}
                        for key in MODEL_TAGS},
        "base_and_quantization": {"base_model": "models/kaggle-qwen3-5-4b-v1/Qwen3.5-4B",
            "base_manifest_sha256": sha(ROOT / "exp/qwen35-sft-epoch3-en-cot-val60/output/model-provenance/kaggle-version1-sha256.json"),
            "base_gguf_sha256": "5f74bc0f26801a814e874bf088079ff5006c0b24dc528e0ca32a88171b9d30f7",
            "conversion": "checkpoint adapter -> BF16 LoRA GGUF -> llama-export-lora onto verified BF16 base GGUF -> Q4_K_M",
            "quantizer_commit": "c0ab6e5c0bab3fb21e6cb9d2371f3c41d85ea775"},
        "prompt": {"variant": "focused post-lookup disambiguation", "sha256": old_lock["prompt"]["sha256"]},
        "settings": {"temperature": 0, "thinking": True, "protocol": "JSON", "num_predict": 768,
                      "num_ctx": 8192, "num_batch": 512, "seed": SEED, "max_action_turns": 5,
                      "asr": "PhoWhisper Large v2 cached for main val309 and val692 history"},
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in inputs},
        "method": "reused exact archived best-baseline predictions; new checkpoint inferences use identical PhoWhisper transcripts and prompt",
        "caveat": "Adapter config identifies Qwen/Qwen3.5-4B; the checkpoint archive does not include a weight hash for its downloaded base snapshot. Tokenizer vocabulary and chat template match the verified Kaggle v1 base, but the original training base weights cannot be independently hash-verified from this archive."
    }
    return compare, prompt, runner, ids, items, main, history, data, payload


def prepare():
    *_, payload = context()
    lock_path = OUT / "comparison_lock.json"
    if lock_path.exists() and read_json(lock_path) != payload:
        raise RuntimeError("Existing comparison lock differs from current inputs/models")
    if not lock_path.exists():
        write_json(lock_path, payload)
    write_json(OUT / "cohort_manifest.json", {"count": 309, "ids_sha256": payload["cohort"]["ids_sha256"]})
    print(json.dumps({"cohort": 309, "prompt_sha256": payload["prompt"]["sha256"],
                      "models": payload["checkpoints"]}, ensure_ascii=False, indent=2), flush=True)


def run_one(key: str, limit: int | None):
    compare, prompt, runner, ids, items, main, history, data, payload = context()
    locked_payload = read_json(OUT / "comparison_lock.json")
    script_key = Path(__file__).resolve().relative_to(ROOT).as_posix()
    inference_script_sha256 = locked_payload["input_sha256"][script_key]
    current_payload = json.loads(json.dumps(payload))
    current_payload["input_sha256"][script_key] = inference_script_sha256
    if locked_payload != current_payload:
        raise RuntimeError("Locked evaluation inputs, models, or prompt changed")
    analysis_script_sha256 = sha(Path(__file__).resolve())
    model = MODEL_TAGS[key]
    runner.MODEL = model
    runner.EXPECTED_DIGEST = payload["checkpoints"][key]["digest"]
    runner.OUT = OUT / "inference" / key
    runner.inject_cot_original = runner.inject_cot
    runner.inject_cot = prompt.focused_inject
    runner.PhoASRModel = compare.CachedPhoWhisper
    runner.PhoASRModel.lookup = history
    runner.seed_everything(SEED)
    runner.OUT.mkdir(parents=True, exist_ok=True)
    pred_path = runner.OUT / "predictions-COT.jsonl"
    prior = read_jsonl(pred_path)
    if [row["id"] for row in prior] != ids[:len(prior)]:
        raise RuntimeError(f"Existing predictions for {key} are not a canonical val309 prefix")
    target = min(len(ids), limit) if limit is not None else len(ids)
    if target < len(prior):
        raise RuntimeError("Requested limit is below the number of saved predictions")
    for round_no, start in enumerate(range(0, target, BLOCK_SIZE), 1):
        block = ids[start:min(start + BLOCK_SIZE, target)]
        done = {row["id"] for row in read_jsonl(pred_path)}
        todo = [sid for sid in block if sid not in done]
        if not todo:
            continue
        runner.warmup("COT", round_no)
        while todo:
            before = len(read_jsonl(pred_path))
            try:
                runner.run_block("COT", round_no, todo, main, items, cot=True)
            except Exception:
                after = read_jsonl(pred_path)
                if len(after) <= before or not after[-1].get("error"):
                    raise
            done = {row["id"] for row in read_jsonl(pred_path)}
            todo = [sid for sid in block if sid not in done]
        print(f"{key}: block {round_no} complete; {len(read_jsonl(pred_path))}/{len(ids)}", flush=True)
    print(f"{key}: saved {len(read_jsonl(pred_path))}/{len(ids)} predictions", flush=True)


def pct(values):
    return 100 * statistics.mean(values) if values else None


def boot_ci(deltas, seed, draws=10000):
    rng = random.Random(seed)
    vals = sorted(100 * statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(draws))
    return [vals[int(.025 * draws)], vals[int(.975 * draws) - 1]]


def parse_label(item):
    label = item.get("label", {})
    return json.loads(label) if isinstance(label, str) else label


def analyze(selected):
    compare, prompt, runner, ids, items, main, history, data, payload = context()
    locked_payload = read_json(OUT / "comparison_lock.json")
    script_key = Path(__file__).resolve().relative_to(ROOT).as_posix()
    inference_script_sha256 = locked_payload["input_sha256"][script_key]
    current_payload = json.loads(json.dumps(payload))
    current_payload["input_sha256"][script_key] = inference_script_sha256
    if locked_payload != current_payload:
        raise RuntimeError("Locked evaluation inputs, models, or prompt changed")
    analysis_script_sha256 = sha(Path(__file__).resolve())
    models = {"best_baseline": read_jsonl(BASELINE)}
    for key in selected:
        models[key] = read_jsonl(OUT / "inference" / key / "predictions-COT.jsonl")
    for key, rows in models.items():
        if [row["id"] for row in rows] != ids:
            raise RuntimeError(f"{key} does not have all 309 predictions in canonical order")
    gold_ask = {sid for sid in ids if any(call.get("name") == "sys.ask_missing_info"
                 for call in parse_label(items[sid]).get("tool_calls", []))}
    fresh = [sid for sid in ids if sid not in gold_ask]
    diagnostic = [sid for sid in ids if sid in gold_ask]
    rows_by_model = {key: {row["id"]: row for row in rows} for key, rows in models.items()}
    scored = {}
    for key, predictions in rows_by_model.items():
        scored[key] = {sid: score_sample(sid, items[sid], predictions[sid], data) for sid in ids}
    details = []
    for sid in ids:
        gold_calls = parse_label(items[sid]).get("tool_calls", [])
        gold_asks = [call for call in gold_calls if call.get("name") == "sys.ask_missing_info"]
        record = {"id": sid, "group": "audited_47_diagnostic" if sid in gold_ask else "fresh_262_primary",
                  "gold_ask": sid in gold_ask}
        for key in models:
            pred = rows_by_model[key][sid]
            score = scored[key][sid]
            asks = [call for call in (pred.get("tool_calls") or []) if call.get("name") == "sys.ask_missing_info"]
            exact = any(call_key(a) == call_key(g) for a in asks for g in gold_asks)
            record[key] = {"btc": score["official_sample_score"],
                "btc_before_runtime_gate": score["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match": score["hash_match"], "runtime_error": bool(pred.get("error")),
                "error": pred.get("error"), "latency_seconds": pred.get("latency_seconds"),
                "ask": bool(asks), "ask_exact": exact}
        details.append(record)
    summaries = {}
    for group_name, subset in (("all_val309", ids), ("fresh_262_primary", fresh), ("audited_47_diagnostic", diagnostic)):
        group = [row for row in details if row["id"] in set(subset)]
        group_summary = {}
        for key in models:
            latencies = [row[key]["latency_seconds"] for row in group if row[key]["latency_seconds"] is not None]
            group_summary[key] = {"btc_0_100": pct([row[key]["btc"] for row in group]),
                "btc_before_runtime_gate_0_100": pct([row[key]["btc_before_runtime_gate"] for row in group]),
                "hash_matches": sum(row[key]["hash_match"] for row in group),
                "runtime_errors": sum(row[key]["runtime_error"] for row in group),
                "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
                "gold_ask_exact": sum(row[key]["ask_exact"] for row in group),
                "ask_count": sum(row[key]["ask"] for row in group)}
        for key in selected:
            ds = [row[key]["btc"] - row["best_baseline"]["btc"] for row in group]
            ungated = [row[key]["btc_before_runtime_gate"] - row["best_baseline"]["btc_before_runtime_gate"] for row in group]
            seed_offset = 60 if key == "step60" else 120
            group_summary[f"paired_{key}_minus_best_baseline"] = {"btc_delta_points": pct(ds),
                "bootstrap_95pct_CI_points": boot_ci(ds, BOOTSTRAP_SEED + seed_offset),
                "checkpoint_wins": sum(x > 0 for x in ds), "baseline_wins": sum(x < 0 for x in ds),
                "ties": sum(x == 0 for x in ds), "ungated_delta_points": pct(ungated),
                "ungated_bootstrap_95pct_CI_points": boot_ci(ungated, BOOTSTRAP_SEED + seed_offset + 1)}
        summaries[group_name] = group_summary
    branches = {"best_baseline": {"model": BASE_MODEL, "predictions_reused": True,
                "asr": "PhoWhisper Large v2 main + history", "btc_val309_archived": payload["baseline"]["btc_val309"]}}
    for key in selected:
        step = 60 if key == "step60" else 120
        branches[key] = {"model": MODEL_TAGS[key], "global_step": step, "predictions_reinferred": True,
                         "samples_completed": 309, "status": "complete"}
    paused = [key for key in MODEL_TAGS if key not in selected]
    for key in paused:
        partial_path = OUT / "inference" / key / "predictions-COT.jsonl"
        partial_count = len(read_jsonl(partial_path))
        branches[key] = {"model": MODEL_TAGS[key], "global_step": 60 if key == "step60" else 120,
                         "predictions_reinferred": False, "samples_completed": partial_count,
                         "status": "paused; not scored"}
    summary = {"experiment": EXP.name, "cohort": {"count": 309, "fresh_primary": 262,
                "audited_gold_ask_diagnostic": 47, "ids_sha256": payload["cohort"]["ids_sha256"]},
        "branches": branches, "paused_branches": paused,
        "code_sha256": {"inference_script": inference_script_sha256, "analysis_script": analysis_script_sha256},
        "prompt_and_settings": {**payload["prompt"], **payload["settings"]}, "groups": summaries,
        "method": {"scoring": "BTC official ToolEnv scorer; local runtime-gated score and pre-gate diagnostic reported separately",
            "paired_by_id": True, "baseline": "reused exact archived best PhoWhisper focused-prompt predictions",
            "bootstrap_draws": 10000, "bootstrap_seed": BOOTSTRAP_SEED}, "caveat": payload["caveat"]}
    if "step60" in selected and "step120" in selected:
        step_delta = [row["step120"]["btc"] - row["step60"]["btc"] for row in details]
        summary["paired_step120_minus_step60_all_val309"] = {"btc_delta_points": pct(step_delta),
            "bootstrap_95pct_CI_points": boot_ci(step_delta, BOOTSTRAP_SEED + 1200),
            "step120_wins": sum(x > 0 for x in step_delta), "step60_wins": sum(x < 0 for x in step_delta),
            "ties": sum(x == 0 for x in step_delta)}
    write_json(OUT / "summary.json", summary)
    write_jsonl(OUT / "paired_details.jsonl", details)
    all_group = summaries["all_val309"]
    report = ["# Qwen3.5-4B synthetic SFT checkpoint on PhoWhisper val309", "",
        "The evaluated checkpoint uses the archived best baseline's PhoWhisper Large main/history transcripts, focused post-lookup prompt, decoding settings, and BTC ToolEnv scorer. Baseline predictions are reused exactly.", "",
        "## All val309", "", "| Branch | BTC after 30s gate | Before gate | Hash matches | Runtime errors | Mean seconds | Exact gold asks |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"best_baseline": "Best baseline (epoch3 focused)", "step60": "Synthetic SFT step 60", "step120": "Synthetic SFT step 120"}
    for key in models:
        row = all_group[key]
        report.append(f"| {labels[key]} | {row['btc_0_100']:.2f} | {row['btc_before_runtime_gate_0_100']:.2f} | {row['hash_matches']}/309 | {row['runtime_errors']} | {row['mean_latency_seconds']:.2f} | {row['gold_ask_exact']}/47 |")
    for key in selected:
        result = all_group[f"paired_{key}_minus_best_baseline"]
        ci = result["bootstrap_95pct_CI_points"]
        step = 60 if key == "step60" else 120
        report.extend(["", f"Step {step} vs baseline: **{result['btc_delta_points']:+.2f} points** (paired bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]); wins/ties/losses {result['checkpoint_wins']}/{result['ties']}/{result['baseline_wins']}."])
    if paused:
        for key in paused:
            report.extend(["", f"Branch {key} paused after {branches[key]['samples_completed']}/309 predictions; it is not included in the score table."])
    if "paired_step120_minus_step60_all_val309" in summary:
        report.extend(["", f"Step 120 vs step 60: **{summary['paired_step120_minus_step60_all_val309']['btc_delta_points']:+.2f} points**."])
    report.extend(["", "## Interpretation caveat", "", payload["caveat"],
        "The checkpoint model was merged onto the same verified BF16 base GGUF and quantized to Q4_K_M with the pinned llama.cpp converter/quantizer. Runtime excludes ASR, as in the val309 model-only evaluation.", "",
        "Per-ID scores and all fingerprints are in paired_details.jsonl, summary.json, and comparison_lock.json.", ""])
    (OUT / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--checkpoint", choices=["step60", "step120", "all"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if sum((args.prepare, args.run, args.analyze)) != 1:
        parser.error("choose one of --prepare, --run, or --analyze")
    if args.prepare:
        prepare()
    elif args.run:
        keys = list(MODEL_TAGS) if args.checkpoint == "all" else [args.checkpoint]
        for key in keys:
            run_one(key, args.limit)
    else:
        selected = list(MODEL_TAGS) if args.checkpoint == "all" else [args.checkpoint]
        analyze(selected)


if __name__ == "__main__":
    main()







