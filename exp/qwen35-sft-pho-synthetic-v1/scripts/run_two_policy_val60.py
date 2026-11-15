from __future__ import annotations
import hashlib, importlib.util, json, math, random, statistics, sys, time
from pathlib import Path

ROOT = Path(r"D:\Projects\zalo-ai-challenge-2026\lifemate")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "exp/btc-scoring/scripts"),
                str(ROOT / "exp/quan-continuation-cot-v1/scripts")]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/two-policy-val60"
MODEL = "lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
THRESHOLD = 20.0
SEED = 17
CAP = 768
CONTEXT = 8192
BATCH = 512
MAX_STEPS = 5

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def read_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()] if p.exists() else []

def write_json(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def build():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run with native system Python so local Ollama is reachable.")
    bench = load_module("final4_benchmark_for_two_policy_val60", EXP / "scripts/benchmark_adapter_final4.py")
    ev, compare, prompt, runner, ids, items, main, history, data, bench_lock = bench.build()
    cohort_manifest = read_json(ROOT / "exp/qwen35-thinking-budget-val60/output/cohort_manifest.json")
    cohort = cohort_manifest["ids"]
    if len(cohort) != 60 or len(set(cohort)) != 60 or not set(cohort).issubset(set(ids)):
        raise RuntimeError("The existing fixed pilot60 does not match val309 IDs.")
    final4 = read_jsonl(EXP / "output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl")
    if [x["id"] for x in final4] != ids:
        raise RuntimeError("Locked final4 predictions are incomplete or out of canonical order.")
    model_record = bench_lock["adapter_final4"]
    runner.MODEL = MODEL
    runner.EXPECTED_DIGEST = model_record["digest"]
    runner.CAP = CAP; runner.CONTEXT = CONTEXT; runner.BATCH = BATCH
    runner.SEED = SEED; runner.MAX_STEPS = MAX_STEPS
    runner.inject_cot_original = runner.inject_cot
    runner.inject_cot = prompt.focused_inject
    import lifemate.asr as asr_mod
    compare.CachedPhoWhisper.lookup = history
    asr_mod.PhoASRModel = compare.CachedPhoWhisper
    return ev, compare, prompt, runner, ids, cohort, items, main, history, data, model_record, final4, cohort_manifest

def boot_ci(diffs, seed, draws=10000):
    rng = random.Random(seed)
    vals = sorted(100 * statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return [vals[int(.025 * draws)], vals[int(.975 * draws) - 1]]

def adaptive_factory(base_agent):
    def make(*args, **kwargs):
        agent = base_agent(*args, **kwargs)
        original_next = agent.next_action
        original_request = agent._ollama_request
        state = {"sample": None, "start": 0, "switched": False}
        def next_action(*a, **k):
            sid = agent._telemetry_sample_id
            if sid != state["sample"]:
                state.update({"sample": sid, "start": len(agent.request_telemetry), "switched": False})
                agent.ollama_thinking = True
            elapsed = sum(float(x.get("latency_seconds") or 0) for x in agent.request_telemetry[state["start"]:])
            if not state["switched"] and elapsed >= THRESHOLD:
                agent.ollama_thinking = False
                state["switched"] = True
            return original_next(*a, **k)
        def request(*a, **k):
            start = len(agent.request_telemetry)
            thinking = bool(agent.ollama_thinking)
            response = original_request(*a, **k)
            for row in agent.request_telemetry[start:]:
                row["adaptive_thinking_enabled"] = thinking
                row["adaptive_switch_threshold_seconds"] = THRESHOLD
            return response
        agent.next_action = next_action
        agent._ollama_request = request
        return agent
    return make

def patch_pipeline(module, source, enabled):
    old = "            self.agent.record_tool_result(action, observation)\n        calls = env.get_tool_calls()"
    new = ("            self.agent.record_tool_result(action, observation)\n"
           "            if action.get(\"tool\") == \"sys.ask_missing_info\":\n"
           "                break\n"
           "        calls = env.get_tool_calls()")
    if enabled:
        if old not in source:
            raise RuntimeError("Could not place terminal ask guard in the loaded pipeline source.")
        exec(compile(source.replace(old, new, 1), str(module.__file__), "exec"), module.__dict__)
    else:
        exec(compile(source, str(module.__file__), "exec"), module.__dict__)

def run():
    ev, compare, prompt, runner, ids, cohort, items, main, history, data, model_record, final4, cohort_manifest = build()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "comparison_lock.json"
    import lifemate.pipeline as pipeline_mod
    import lifemate.agent as agent_mod
    original_pipeline_source = Path(pipeline_mod.__file__).read_text(encoding="utf-8")
    base_agent = agent_mod.Agent
    input_paths = [
        Path(__file__).resolve(),
        EXP / "scripts/benchmark_adapter_final4.py",
        ROOT / "exp/qwen35-thinking-budget-val60/output/cohort_manifest.json",
        EXP / "output/adapter-final-4epoch-benchmark/val309/comparison_lock.json",
        EXP / "output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl",
        ROOT / "src/lifemate/pipeline.py", ROOT / "src/lifemate/agent.py",
        ROOT / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
    ]
    lock = {
        "experiment": "qwen35-sft-pho-synthetic-v1-two-policy-pilot60",
        "cohort": {"n": 60, "ids": cohort, "source": "existing enriched fixed pilot60",
                   "source_manifest_sha256": sha(cohort_manifest and ROOT / "exp/qwen35-thinking-budget-val60/output/cohort_manifest.json"),
                   "ids_sha256": hashlib.sha256("\n".join(cohort).encode()).hexdigest()},
        "model": {"tag": MODEL, "digest": model_record["digest"]},
        "prompt": {"variant": "focused post-lookup disambiguation", "sha256": prompt.prompt_hash(prompt.focused_inject)},
        "settings": {"temperature": 0, "protocol": "JSON", "num_predict": CAP, "num_ctx": CONTEXT,
                     "num_batch": BATCH, "seed": SEED, "max_action_turns": MAX_STEPS,
                     "asr": "PhoWhisper Large v2 cached main/history"},
        "factors": {"stop_after_successful_ask": [False, True],
                    "adaptive_thinking": [False, True],
                    "adaptive_threshold_seconds": THRESHOLD},
        "fixed_final4_prediction_sha256": sha(EXP / "output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl"),
        "inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in input_paths},
        "caveat": "Enriched diagnostic pilot60; not an unbiased estimate of full val309 or public score."
    }
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Existing experiment lock differs from current inputs; refusing to overwrite.")
    write_json(lock_path, lock)
    branch_defs = {
        "fixed_full": (False, False),
        "fixed_ask_stop": (False, True),
        "adaptive20_full": (True, False),
        "adaptive20_ask_stop": (True, True),
    }
    branch_order = [
        ["fixed_full", "adaptive20_full", "fixed_ask_stop", "adaptive20_ask_stop"],
        ["adaptive20_ask_stop", "fixed_ask_stop", "adaptive20_full", "fixed_full"],
        ["fixed_ask_stop", "fixed_full", "adaptive20_ask_stop", "adaptive20_full"],
        ["adaptive20_full", "adaptive20_ask_stop", "fixed_full", "fixed_ask_stop"],
    ]
    try:
        for round_no, start in enumerate(range(0, 60, 15), 1):
            block = cohort[start:start + 15]
            for key in branch_order[round_no - 1]:
                adaptive, ask_stop = branch_defs[key]
                branch_out = OUT / "inference" / key
                branch_out.mkdir(parents=True, exist_ok=True)
                runner.OUT = branch_out
                done = {x["id"] for x in read_jsonl(branch_out / "predictions-COT.jsonl")}
                todo = [sid for sid in block if sid not in done]
                if not todo:
                    continue
                patch_pipeline(pipeline_mod, original_pipeline_source, ask_stop)
                agent_mod.Agent = adaptive_factory(base_agent) if adaptive else base_agent
                runner.warmup("COT", round_no)
                runner.run_block("COT", round_no, todo, main, items, cot=True)
                print(f"{key}: block {round_no}/4 complete ({start + len(block)}/60)", flush=True)
        patch_pipeline(pipeline_mod, original_pipeline_source, False)
        agent_mod.Agent = base_agent
    finally:
        patch_pipeline(pipeline_mod, original_pipeline_source, False)
        agent_mod.Agent = base_agent
    analyze(ev, ids, cohort, items, data, final4, model_record, lock)

def analyze(ev, ids, cohort, items, data, final4, model_record, lock):
    branches = ["fixed_full", "fixed_ask_stop", "adaptive20_full", "adaptive20_ask_stop"]
    preds = {}
    reqs = {}
    for key in branches:
        folder = OUT / "inference" / key
        rows = read_jsonl(folder / "predictions-COT.jsonl")
        if [x["id"] for x in rows] != cohort:
            raise RuntimeError(f"{key} has {len(rows)} rows or differs from locked cohort order.")
        preds[key] = {x["id"]: x for x in rows}
        reqs[key] = read_jsonl(folder / "requests-COT.jsonl")
    from score_btc import score_sample, call_key
    records = []
    for sid in cohort:
        gold = items[sid].get("label", {}).get("tool_calls", [])
        gold_asks = [x for x in gold if x.get("name") == "sys.ask_missing_info"]
        row = {"id": sid, "gold_ask": bool(gold_asks)}
        for key in branches:
            p = preds[key][sid]
            sc = score_sample(sid, items[sid], p, data)
            calls = p.get("tool_calls") or []
            asks = [x for x in calls if x.get("name") == "sys.ask_missing_info"]
            row[key] = {
                "btc_after_runtime_gate": sc["official_sample_score"],
                "btc_before_runtime_gate": sc["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match": bool(sc["hash_match"]), "latency_seconds": p.get("latency_seconds"),
                "runtime_error": bool(p.get("error")), "over_30s": bool(sc["runtime_over_30s"]),
                "matched_calls": sc["matched_calls"], "unmatched_calls": sc["unmatched_submitted_calls"],
                "gold_calls": sc["gold_calls"], "ask_count": len(asks),
                "ask_exact": any(call_key(a) == call_key(g) for a in asks for g in gold_asks),
            }
        records.append(row)
    summaries = {}
    for key in branches:
        rows = [r[key] for r in records]
        lat = [float(r["latency_seconds"]) for r in rows if r["latency_seconds"] is not None]
        summary = {
            "btc_after_30s_gate": 100 * statistics.mean(x["btc_after_runtime_gate"] for x in rows),
            "btc_before_gate": 100 * statistics.mean(x["btc_before_runtime_gate"] for x in rows),
            "over_30s": sum(x["over_30s"] for x in rows),
            "runtime_errors": sum(x["runtime_error"] for x in rows),
            "hash_matches": sum(x["hash_match"] for x in rows),
            "mean_latency_seconds": statistics.mean(lat) if lat else None,
            "p50_latency_seconds": statistics.median(lat) if lat else None,
            "p90_latency_seconds": sorted(lat)[math.ceil(.90 * len(lat)) - 1] if lat else None,
            "matched_calls": sum(x["matched_calls"] for x in rows),
            "unmatched_calls": sum(x["unmatched_calls"] for x in rows),
            "gold_calls": sum(x["gold_calls"] for x in rows),
            "ask_total": sum(x["ask_count"] for x in rows),
            "exact_gold_ask_samples": sum(x["ask_exact"] for x in rows),
        }
        active_req = [x for x in reqs[key] if "adaptive_thinking_enabled" in x]
        summary["adaptive_requests"] = {
            "total": len(active_req),
            "thinking_off": sum(not x["adaptive_thinking_enabled"] for x in active_req),
            "samples_switched": len({x.get("sample_id") for x in active_req if not x["adaptive_thinking_enabled"]}),
        }
        baseline_map = {x["id"]: x for x in final4}
        diff_calls = sum(preds["fixed_full"][sid].get("tool_calls") == baseline_map[sid].get("tool_calls") for sid in cohort)
        diff_hash = sum(preds["fixed_full"][sid].get("state_hash") == baseline_map[sid].get("state_hash") for sid in cohort)
        summary["fixed_full_reproducibility_vs_archived_final4"] = {
            "identical_ordered_calls": diff_calls, "identical_state_hashes": diff_hash,
            "n": len(cohort)
        } if key == "fixed_full" else None
        summaries[key] = summary
    contrasts = [
        ("ask_stop_effect_fixed", "fixed_ask_stop", "fixed_full"),
        ("adaptive_effect_without_stop", "adaptive20_full", "fixed_full"),
        ("ask_stop_effect_adaptive", "adaptive20_ask_stop", "adaptive20_full"),
        ("adaptive_effect_with_stop", "adaptive20_ask_stop", "fixed_ask_stop"),
        ("combined_vs_fixed_full", "adaptive20_ask_stop", "fixed_full"),
    ]
    paired = {}
    for name, left, right in contrasts:
        d = [records[i][left]["btc_after_runtime_gate"] - records[i][right]["btc_after_runtime_gate"] for i in range(len(records))]
        u = [records[i][left]["btc_before_runtime_gate"] - records[i][right]["btc_before_runtime_gate"] for i in range(len(records))]
        paired[name] = {
            "left": left, "right": right,
            "delta_after_gate_points": 100 * statistics.mean(d),
            "bootstrap_95pct_CI_after_gate": boot_ci(d, 20261114 + len(paired)),
            "delta_before_gate_points": 100 * statistics.mean(u),
            "bootstrap_95pct_CI_before_gate": boot_ci(u, 20261214 + len(paired)),
            "wins_ties_losses": [sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)],
        }
    result = {
        "experiment": lock["experiment"], "cohort": lock["cohort"], "model": lock["model"],
        "prompt": lock["prompt"], "settings": lock["settings"], "policy": lock["factors"],
        "branches": summaries, "paired_contrasts": paired,
        "method": {"scoring": "Official BTC ToolEnv formula; exact calls with multiplicity and state-hash gate; 30s local runtime gate and pre-gate reported separately.",
                   "paired_by_sample": True, "bootstrap_draws": 10000,
                   "fixed_baseline": "fresh same-session fixed-thinking control; archived final4 used for repeatability comparison"},
        "caveat": lock["caveat"]
    }
    write_json(OUT / "paired_details.json", records)
    write_json(OUT / "summary.json", result)
    lines = [
        "# Ask-stop and adaptive-thinking pilot60", "",
        "Same Qwen3.5 4-epoch adapter, PhoWhisper Large v2, focused prompt, JSON, temperature 0, num_predict 768, context 8192, batch 512, seed 17, and five action turns. Cohort is the existing enriched fixed pilot60, so scores are diagnostic rather than estimates of full val309/public performance.", "",
        "| Branch | BTC before 30s gate | BTC after gate | >30s | Mean sec | p90 sec | Hashes | Exact asks | Ask calls | Adaptive off reqs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for key in branches:
        x = summaries[key]; a = x["adaptive_requests"]
        lines.append(f"| {key} | {x['btc_before_gate']:.2f} | {x['btc_after_30s_gate']:.2f} | {x['over_30s']}/60 | {x['mean_latency_seconds']:.2f} | {x['p90_latency_seconds']:.2f} | {x['hash_matches']}/60 | {x['exact_gold_ask_samples']}/60 | {x['ask_total']} | {a['thinking_off']}/{a['total']} |")
    lines += ["", "## Paired contrasts", ""]
    for name, *_ in contrasts:
        x = paired[name]; ci = x["bootstrap_95pct_CI_after_gate"]
        lines.append(f"- {name}: {x['delta_after_gate_points']:+.2f} points after gate, 95% bootstrap CI [{ci[0]:+.2f}, {ci[1]:+.2f}]; before gate {x['delta_before_gate_points']:+.2f}. Wins/ties/losses {x['wins_ties_losses']}.")
    lines += ["", f"Fixed-thinking fresh run reproduced archived final4 on {summaries['fixed_full']['fixed_full_reproducibility_vs_archived_final4']['identical_ordered_calls']}/60 ordered call lists and {summaries['fixed_full']['fixed_full_reproducibility_vs_archived_final4']['identical_state_hashes']}/60 state hashes.", "",
              "This pilot does not establish an unbiased score gain. Confirm any promising branch on full val309 before promoting it.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

def main():
    run()

if __name__ == "__main__":
    main()
