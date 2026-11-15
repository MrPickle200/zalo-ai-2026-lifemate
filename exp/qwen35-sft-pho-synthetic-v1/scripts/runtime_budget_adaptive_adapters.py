from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, statistics, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/runtime-budget-adaptive-adapters"
SOURCE_OUT = EXP / "output/runtime-budget-ablation"
FINAL1_PRED = EXP / "output/adapter-final-1epoch-benchmark/val309/inference/adapter_final1/predictions-COT.jsonl"
FINAL3_PRED = EXP / "output/adapter-final-3epoch-benchmark/val309/inference/adapter_final3/predictions-COT.jsonl"
THRESHOLD_SEC = 20.0
CAP = 768
BLOCK = 15
SEED = 17
ADAPTER_KEYS = ("adapter_final1", "adapter_final3")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def context():
    bench = load_module("adapter_final3_benchmark_context_for_adaptive", EXP / "scripts/benchmark_adapter_final3.py")
    compare, prompt, runner, ids, items, main, history, data, bench_lock = bench.build()
    prior_lock_path = SOURCE_OUT / "ablation_lock.json"
    prior_lock = read_json(prior_lock_path)
    cohort_ids = prior_lock["cohort"]["ids"]
    if len(cohort_ids) != 120 or len(set(cohort_ids)) != 120 or not set(cohort_ids).issubset(set(ids)):
        raise RuntimeError("Prior adaptive cohort is invalid or differs from the canonical val309 IDs")
    fixed_paths = {"adapter_final1": FINAL1_PRED, "adapter_final3": FINAL3_PRED}
    fixed = {}
    for key, path in fixed_paths.items():
        rows = read_jsonl(path)
        if [r["id"] for r in rows] != ids:
            raise RuntimeError(f"Fixed thinking-on predictions are not complete/canonical for {key}")
        fixed[key] = rows
    expected = {key: bench_lock[key] for key in ADAPTER_KEYS}
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=30) as response:
        tags = {row["name"]: row["digest"] for row in json.load(response)["models"]}
    for key in ADAPTER_KEYS:
        if tags.get(expected[key]["model"]) != expected[key]["digest"]:
            raise RuntimeError(f"Ollama tag/digest drift for {key}")
    return bench, compare, prompt, runner, ids, items, main, history, data, bench_lock, prior_lock, fixed, expected


def make_lock():
    *_, bench_lock, prior_lock, fixed, expected = context()
    prior_lock_path = SOURCE_OUT / "ablation_lock.json"
    files = [Path(__file__).resolve(), EXP / "scripts/runtime_budget_adaptive.py",
             EXP / "scripts/runtime_budget_ablation.py", prior_lock_path,
             FINAL1_PRED, FINAL3_PRED,
             EXP / "output/adapter-final-1epoch-benchmark/val309/comparison_lock.json",
             EXP / "output/adapter-final-3epoch-benchmark/val309/comparison_lock.json"]
    return {
        "experiment": "qwen35-sft-pho-synthetic-v1-adaptive-thinking-adapters",
        "cohort": {"n": 120, "ids": prior_lock["cohort"]["ids"],
                   "slow_n": prior_lock["cohort"]["slow_n"], "slow_ids": prior_lock["cohort"]["slow_ids"],
                   "matched_fast_controls_n": prior_lock["cohort"]["fast_controls_n"],
                   "cohort_ids_sha256": prior_lock["cohort"]["cohort_ids_sha256"],
                   "source": "same step120 stress cohort as runtime-budget-ablation; 42 slow + 78 matched fast controls"},
        "policy": {"initial_thinking": True, "switch_to_thinking_false_when_cumulative_request_latency_seconds_exceeds": THRESHOLD_SEC,
                    "switch_applies": "all subsequent LLM requests for the current sample", "num_predict": CAP},
        "prompt": bench_lock["prompt"],
        "common_settings": {"temperature": 0, "protocol": "JSON", "num_ctx": 8192, "num_batch": 512,
                            "seed": SEED, "max_action_turns": 5, "asr": "PhoWhisper Large v2 cached main/history",
                            "toolenv": "official BTC ToolEnv per sample"},
        "adapters": {key: {"model": expected[key]["model"], "digest": expected[key]["digest"],
                            "fixed_thinking_predictions_sha256": sha(FINAL1_PRED if key == "adapter_final1" else FINAL3_PRED),
                            "fixed_val309_btc_after_30s_gate": bench_lock[key].get("btc_val309")}
                      for key in ADAPTER_KEYS},
        "inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files},
        "source_adaptive_cohort_lock_sha256": sha(prior_lock_path),
        "method_note": "Stress-cohort screening only; the 120 cases oversample step120 slow samples and are not a population estimate for full val309/public test."
    }


def boot_ci(values, seed, draws=10000):
    rng = random.Random(seed)
    draws_sorted = sorted(100 * statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [draws_sorted[int(0.025 * draws)], draws_sorted[int(0.975 * draws) - 1]] if draws_sorted else [None, None]


def run():
    bench, compare, prompt, runner, ids, items, main, history, data, bench_lock, prior_lock, fixed, expected = context()
    lock = make_lock()
    lock_path = OUT / "adaptive_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Adaptive lock changed; refusing to overwrite existing results")
    write_json(lock_path, lock)
    write_json(OUT / "cohort_manifest.json", lock["cohort"])
    cohort_ids = lock["cohort"]["ids"]
    runner.inject_cot_original = runner.inject_cot
    runner.inject_cot = prompt.focused_inject
    runner.CAP = CAP
    runner.MAX_STEPS = 5
    runner.seed_everything(SEED)
    compare.CachedPhoWhisper.lookup = history
    import lifemate.asr as asr_module
    asr_module.PhoASRModel = compare.CachedPhoWhisper
    asr_module.PhoASRModel.lookup = history
    import lifemate.agent as agent_module
    base_agent = agent_module.Agent

    def make_adaptive_agent(*args, **kwargs):
        agent = base_agent(*args, **kwargs)
        original_next = agent.next_action
        original_request = agent._ollama_request
        state = {"sample_id": None, "start": 0, "switched": False}
        def adaptive_next(*a, **k):
            sid = agent._telemetry_sample_id
            if sid != state["sample_id"]:
                state.update({"sample_id": sid, "start": len(agent.request_telemetry), "switched": False})
                agent.ollama_thinking = True
            elapsed = sum(float(row.get("latency_seconds") or 0) for row in agent.request_telemetry[state["start"]:])
            if not state["switched"] and elapsed >= THRESHOLD_SEC:
                agent.ollama_thinking = False
                state["switched"] = True
            return original_next(*a, **k)
        def request_with_mode(*a, **k):
            start = len(agent.request_telemetry)
            thinking = bool(agent.ollama_thinking)
            response = original_request(*a, **k)
            for row in agent.request_telemetry[start:]:
                row["adaptive_thinking_enabled"] = thinking
                row["adaptive_switch_threshold_seconds"] = THRESHOLD_SEC
            return response
        agent.next_action = adaptive_next
        agent._ollama_request = request_with_mode
        return agent

    agent_module.Agent = make_adaptive_agent
    try:
        for adapter_i, key in enumerate(ADAPTER_KEYS):
            tag = expected[key]["model"]
            branch_out = OUT / "inference" / f"{key}_adaptive20"
            branch_out.mkdir(parents=True, exist_ok=True)
            runner.MODEL = tag
            runner.EXPECTED_DIGEST = expected[key]["digest"]
            runner.OUT = branch_out
            pred_path = branch_out / "predictions-COT.jsonl"
            prior_rows = read_jsonl(pred_path)
            if [row["id"] for row in prior_rows] != cohort_ids[:len(prior_rows)]:
                raise RuntimeError(f"Saved predictions are not a canonical cohort prefix for {key}")
            for round_no, start in enumerate(range(0, len(cohort_ids), BLOCK), 1):
                block = cohort_ids[start:start + BLOCK]
                done = {row["id"] for row in read_jsonl(pred_path)}
                todo = [sid for sid in block if sid not in done]
                if not todo:
                    continue
                warm_started = time.perf_counter()
                warm = runner.api_json("/api/chat", {"model": tag, "messages": [{"role": "user", "content": "Return {} as JSON."}],
                    "stream": False, "think": True, "format": "json",
                    "options": {"temperature": 0, "num_predict": 8, "num_ctx": 8192, "num_batch": 512}})
                write_json(branch_out / f"warmup-COT-round{round_no}.json", {"branch": key, "round": round_no,
                    "excluded_from_measurements": True, "wall_seconds": time.perf_counter() - warm_started,
                    "eval_count": warm.get("eval_count"), "load_duration_ns": warm.get("load_duration")})
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
                meta_path = branch_out / "run-COT.json"
                meta = read_json(meta_path) if meta_path.exists() else {}
                meta.update({"branch": f"{key}_adaptive20", "status": "complete" if len(read_jsonl(pred_path)) == start + len(block) else "in_progress",
                    "thinking_policy": lock["policy"], "model_digest": expected[key]["digest"],
                    "prompt_variant": "focused post-lookup disambiguation", "cohort_ids": cohort_ids,
                    "executed_rounds": round_no})
                write_json(meta_path, meta)
                print(f"{key} adaptive block {round_no}/8 complete ({start + len(block)}/120)", flush=True)
            if [row["id"] for row in read_jsonl(pred_path)] != cohort_ids:
                raise RuntimeError(f"Adaptive inference incomplete for {key}")
    finally:
        agent_module.Agent = base_agent
    analyze()


def analyze():
    bench, compare, prompt, runner, ids, items, main, history, data, bench_lock, prior_lock, fixed, expected = context()
    lock = make_lock()
    if read_json(OUT / "adaptive_lock.json") != lock:
        raise RuntimeError("Adaptive lock no longer matches model digests, prompts, fixed predictions, or cohort")
    from score_btc import score_sample, call_key
    cohort_ids = lock["cohort"]["ids"]
    slow = set(lock["cohort"]["slow_ids"])
    groups = {"all_120": set(cohort_ids), "slow_gt30s": slow, "matched_fast_control": set(cohort_ids) - slow}
    details = []
    summaries = {}
    req_counts = {}
    for key in ADAPTER_KEYS:
        fixed_map = {row["id"]: row for row in fixed[key]}
        adaptive_path = OUT / "inference" / f"{key}_adaptive20" / "predictions-COT.jsonl"
        adaptive_rows = read_jsonl(adaptive_path)
        if [row["id"] for row in adaptive_rows] != cohort_ids:
            raise RuntimeError(f"Adaptive predictions incomplete or out of order for {key}")
        adaptive_map = {row["id"]: row for row in adaptive_rows}
        scored = {"fixed_think": {}, "adaptive20": {}}
        for sid in cohort_ids:
            label = items[sid].get("label", {})
            label = json.loads(label) if isinstance(label, str) else label
            gold_asks = [c for c in label.get("tool_calls", []) if c.get("name") == "sys.ask_missing_info"]
            record = {"id": sid, "cohort_group": "slow_gt30s" if sid in slow else "matched_fast_control",
                      "gold_ask": bool(gold_asks), "adapter": key}
            for mode, pred in (("fixed_think", fixed_map[sid]), ("adaptive20", adaptive_map[sid])):
                score = score_sample(sid, items[sid], pred, data)
                asks = [c for c in (pred.get("tool_calls") or []) if c.get("name") == "sys.ask_missing_info"]
                ask_exact = any(call_key(a) == call_key(g) for a in asks for g in gold_asks)
                scored[mode][sid] = {"score": score, "pred": pred, "ask_exact": ask_exact}
                record[mode] = {"btc": score["official_sample_score"],
                    "btc_before_runtime_gate": score["btc_formula_score_before_runtime_gate_diagnostic"],
                    "latency_seconds": pred.get("latency_seconds"), "over_30s": bool(score["runtime_over_30s"]),
                    "runtime_error": bool(pred.get("error")), "hash_match": bool(score["hash_match"]),
                    "gold_ask_exact": ask_exact, "ask_count": len(asks)}
            details.append(record)
        summaries[key] = {}
        for group_name, members in groups.items():
            summary = {}
            selected = [row for row in details if row["adapter"] == key and row["id"] in members]
            for mode in ("fixed_think", "adaptive20"):
                values = [row[mode] for row in selected]
                lat = [x["latency_seconds"] for x in values if x["latency_seconds"] is not None]
                summary[mode] = {"btc_after_30s_gate": 100 * statistics.mean(x["btc"] for x in values),
                    "btc_before_gate": 100 * statistics.mean(x["btc_before_runtime_gate"] for x in values),
                    "n_over_30s": sum(x["over_30s"] for x in values),
                    "runtime_errors": sum(x["runtime_error"] for x in values),
                    "hash_matches": sum(x["hash_match"] for x in values),
                    "mean_latency_seconds": statistics.mean(lat) if lat else None,
                    "gold_ask_exact": sum(x["gold_ask_exact"] for x in values),
                    "ask_count": sum(x["ask_count"] for x in values)}
            diffs = [row["adaptive20"]["btc"] - row["fixed_think"]["btc"] for row in selected]
            summary["paired_adaptive20_minus_fixed_think"] = {
                "delta_after_gate_points": 100 * statistics.mean(diffs),
                "bootstrap_95pct_CI_after_gate": boot_ci(diffs, 20261120 + (1 if key == "adapter_final1" else 3)),
                "wins": sum(x > 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "losses": sum(x < 0 for x in diffs)}
            summaries[key][group_name] = summary
        req_path = OUT / "inference" / f"{key}_adaptive20" / "requests-COT.jsonl"
        requests = read_jsonl(req_path)
        adaptive_req = [r for r in requests if "adaptive_thinking_enabled" in r]
        off = [r for r in adaptive_req if not r["adaptive_thinking_enabled"]]
        req_counts[key] = {"total_requests": len(adaptive_req), "thinking_enabled": sum(bool(r["adaptive_thinking_enabled"]) for r in adaptive_req),
                           "thinking_disabled": len(off), "samples_with_switch": len({r.get("sample_id") for r in off})}
    write_jsonl(OUT / "paired_details.jsonl", details)
    summary = {"experiment": lock["experiment"], "cohort": lock["cohort"], "policy": lock["policy"],
               "adapters": lock["adapters"], "groups": summaries, "adaptive_requests": req_counts,
               "method": {"scoring": "Official BTC ToolEnv per sample; 30-second local runtime gate and ungated diagnostic reported separately",
                          "paired": "same sample IDs and same model; adaptive predictions compared against existing fixed-thinking predictions",
                          "selection_bias": "stress cohort oversamples slow step120 examples; not a population estimate for val309 or public test"}}
    write_json(OUT / "summary.json", summary)
    lines = ["# Adaptive thinking on synthetic adapters", "",
             "Same 120-sample stress cohort used by the earlier step120 adaptive screen. Each adapter starts with thinking enabled and disables it for later requests once cumulative request latency for that sample exceeds 20 seconds. Fixed-thinking adapter predictions are reused from the complete val309 benchmark. The cohort oversamples slow step120 cases and is diagnostic, not a full-val estimate.", "",
             "| Adapter / group | Mode | BTC after 30s gate | Before gate | >30s | Hash matches | Mean sec | Exact asks |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key in ADAPTER_KEYS:
        for group_name in ("all_120", "slow_gt30s", "matched_fast_control"):
            for mode in ("fixed_think", "adaptive20"):
                x = summaries[key][group_name][mode]
                lines.append(f"| {key} / {group_name} | {mode} | {x['btc_after_30s_gate']:.2f} | {x['btc_before_gate']:.2f} | {x['n_over_30s']} | {x['hash_matches']}/{len(groups[group_name])} | {x['mean_latency_seconds']:.2f} | {x['gold_ask_exact']} |")
    lines += ["", "## Paired adaptive deltas", ""]
    for key in ADAPTER_KEYS:
        x = summaries[key]["all_120"]["paired_adaptive20_minus_fixed_think"]
        ci = x["bootstrap_95pct_CI_after_gate"]
        req = req_counts[key]
        lines.append(f"- {key}: {x['delta_after_gate_points']:+.2f} points after gate (paired 95% CI {ci[0]:+.2f} to {ci[1]:+.2f}); wins/ties/losses {x['wins']}/{x['ties']}/{x['losses']}. Thinking disabled on {req['samples_with_switch']}/120 samples and {req['thinking_disabled']}/{req['total_requests']} requests.")
    lines += ["", "Use this as a matched screening result only. If the adaptive result is promising, rerun the winning adapter on all 309 val IDs before making a submission choice.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        lock = make_lock()
        path = OUT / "adaptive_lock.json"
        if path.exists() and read_json(path) != lock:
            raise RuntimeError("Existing adaptive lock differs; refusing to replace it")
        write_json(path, lock)
        write_json(OUT / "cohort_manifest.json", lock["cohort"])
        print(json.dumps({"samples": 120, "policy": lock["policy"], "adapters": lock["adapters"], "cohort": lock["cohort"]}, ensure_ascii=False, indent=2), flush=True)
    elif args.run:
        run()
    else:
        analyze()

if __name__ == "__main__":
    main()
