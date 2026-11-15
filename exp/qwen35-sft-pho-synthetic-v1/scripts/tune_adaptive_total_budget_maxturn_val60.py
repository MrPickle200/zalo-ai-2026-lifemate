from __future__ import annotations
import copy, hashlib, importlib.util, json, os, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
OUT = EXP / "output/adaptive-totalbudget-maxturn-screen"
ARCHIVED = EXP / "output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl"
ASR_DIR = ROOT / "exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho"
ASR_MAIN = ASR_DIR / "predictions.jsonl"
ASR_HISTORY = ASR_DIR / "history_predictions.jsonl"
MODEL = "lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
THRESHOLDS = [15.0, 20.0, 25.0, 30.0]
CAPS = [4, 5, 6, 7]
RUN_CAP = max(CAPS)
RANDOM_N = 60
SEED = 20261115
BOOT_DRAWS = 10000
BLOCK = 15
CONTEXT = 16384
GEN_CAP = 768
BATCH = 512

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def build():
    bench = load_module("adaptive_total_budget_benchmark_final4", EXP / "scripts/benchmark_adapter_final4.py")
    ev, compare, prompt, runner, ids, items, main, history, data, model_lock = bench.build()
    if len(ids) != 309:
        raise RuntimeError(f"Expected canonical val309, got {len(ids)}")
    main_asr = read_jsonl(ASR_MAIN)
    history_asr = read_jsonl(ASR_HISTORY)
    main_seconds = {row["id"]: float(row["inference_seconds"]) for row in main_asr}
    history_seconds = {}
    for row in history_asr:
        history_seconds[row["id"]] = history_seconds.get(row["id"], 0.0) + float(row["inference_seconds"])
    if set(main_seconds) != set(ids):
        raise RuntimeError("PhoWhisper main ASR timing does not cover canonical val309.")
    if not set(history_seconds).issubset(set(ids)):
        raise RuntimeError("PhoWhisper history ASR timing has unexpected IDs.")
    asr_seconds = {sid: main_seconds[sid] + history_seconds.get(sid, 0.0) for sid in ids}
    return bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds

def make_cohorts(ids, archived):
    rng = random.Random(SEED)
    random_set = set(rng.sample(ids, RANDOM_N))
    random_ids = [sid for sid in ids if sid in random_set]
    eligible = []
    by_id = {row["id"]: row for row in archived}
    for sid in ids:
        row = by_id[sid]
        actions = row.get("model_output") or []
        first5 = actions[:5]
        if (len(first5) == 5 and all(x.get("action") == "tool_call" for x in first5)
                and not any(x.get("tool") == "sys.ask_missing_info" for x in first5)
                and not row.get("error")):
            eligible.append(sid)
    if len(eligible) != 25:
        raise RuntimeError(f"Expected 25 eligible continuation IDs, got {len(eligible)}")
    chosen = set(random_ids) | set(eligible)
    union = [sid for sid in ids if sid in chosen]
    return random_ids, eligible, union

def make_lock(ids, items, archived, asr_seconds, cohort, model_lock, prompt):
    script = Path(__file__).resolve()
    digest = model_lock["adapter_final4"]["digest"]
    input_paths = [
        script, EXP / "scripts/benchmark_adapter_final4.py",
        EXP / "scripts/run_two_policy_val60.py",
        EXP / "output/adapter-final-4epoch-benchmark/val309/comparison_lock.json",
        ARCHIVED, ASR_MAIN, ASR_HISTORY,
        ROOT / "src/lifemate/agent.py", ROOT / "src/lifemate/pipeline.py",
        ROOT / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        ROOT / "exp/btc-scoring/scripts/score_btc.py",
    ]
    return {
        "experiment": "qwen35-final4-adaptive-total-budget-max-turn-screen",
        "cohort": cohort,
        "random_seed": SEED,
        "model": {"tag": MODEL, "digest": digest},
        "policy": {
            "thresholds_are_end_to_end_seconds": THRESHOLDS,
            "threshold_includes_asr": True,
            "adaptive_rule": "first request starts with thinking on; before later requests, turn thinking off when measured ASR plus cumulative prior Qwen request latency reaches threshold",
            "successful_ask_stop": True,
            "run_cap_for_nested_turn_comparison": RUN_CAP,
            "turn_caps_scored": CAPS,
        },
        "settings": {
            "temperature": 0, "protocol": "JSON", "num_predict": GEN_CAP,
            "num_ctx": CONTEXT, "num_batch": BATCH, "seed": 17,
            "prompt": "focused post-lookup", "asr": "PhoWhisper Large v2 main/history",
            "asr_timing_scope": "per-clip inference_seconds from the local val309 PhoWhisper cache; model-load startup excluded",
        },
        "asr_seconds": {
            "mean_per_sample": statistics.mean(asr_seconds.values()),
            "median_per_sample": statistics.median(asr_seconds.values()),
            "p95_per_sample": sorted(asr_seconds.values())[int(.95 * (len(ids) - 1))],
            "main_clips": 309, "history_clips": 692,
            "main_sha256": sha(ASR_MAIN), "history_sha256": sha(ASR_HISTORY),
        },
        "focused_prompt_sha256": prompt.prompt_hash(prompt.focused_inject),
        "inputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in input_paths},
        "caveat": "60 random val IDs screen overall policy; all 25 eligible five-call continuations screen turn caps. Local RTX 4050 timing, not A30.",
    }

def adaptive_factory(base_agent, threshold_total, asr_seconds):
    def make(*args, **kwargs):
        agent = base_agent(*args, **kwargs)
        original_next = agent.next_action
        original_request = agent._ollama_request
        state = {"sample": None, "start": 0, "switched": False, "asr": 0.0}

        def next_action(*a, **k):
            sid = agent._telemetry_sample_id
            if sid != state["sample"]:
                state.update({"sample": sid, "start": len(agent.request_telemetry),
                              "switched": False, "asr": asr_seconds[sid]})
                agent.ollama_thinking = True
            prior_requests = agent.request_telemetry[state["start"]:]
            elapsed_total = state["asr"] + sum(float(row.get("latency_seconds") or 0) for row in prior_requests)
            if prior_requests and not state["switched"] and elapsed_total >= threshold_total:
                agent.ollama_thinking = False
                state["switched"] = True
            return original_next(*a, **k)

        def request(*a, **k):
            before = len(agent.request_telemetry)
            thinking = bool(agent.ollama_thinking)
            response = original_request(*a, **k)
            for row in agent.request_telemetry[before:]:
                row["adaptive_thinking_enabled"] = thinking
                row["adaptive_total_budget_threshold_seconds"] = threshold_total
                row["asr_latency_seconds_for_sample"] = state["asr"]
            return response

        agent.next_action = next_action
        agent._ollama_request = request
        return agent
    return make

def patch_ask_stop(pipeline, source):
    old = "            self.agent.record_tool_result(action, observation)\n        calls = env.get_tool_calls()"
    new = (
        "            self.agent.record_tool_result(action, observation)\n"
        "            if action.get(\"tool\") == \"sys.ask_missing_info\" and observation.get(\"ok\") is True:\n"
        "                break\n"
        "        calls = env.get_tool_calls()"
    )
    if old not in source:
        raise RuntimeError("Could not find action-loop location for successful ask-stop.")
    exec(compile(source.replace(old, new, 1), str(pipeline.__file__), "exec"), pipeline.__dict__)

def replay_hash(sample_id, calls, data):
    from lifemate.tool_executor import create_tool_env
    env = create_tool_env("train", sample_id, data)
    for call in calls:
        env.validate(call.get("name"), call.get("arguments") or {})
        env.call(call.get("name"), call.get("arguments") or {})
    replayed = env.get_tool_calls()
    if replayed != calls:
        left = json.dumps(replayed, sort_keys=True, ensure_ascii=False)
        right = json.dumps(calls, sort_keys=True, ensure_ascii=False)
        if left != right:
            raise RuntimeError(f"ToolEnv replay mismatch for {sample_id}")
    return env.get_state_hash(), env.get_state()

def project_cap(row, cap, request_rows, asr_seconds, data):
    result = copy.deepcopy(row)
    actions = list(result.get("model_output") or [])
    calls = list(result.get("tool_calls") or [])
    observations = list(result.get("observations") or [])
    if len(calls) != len(observations):
        raise RuntimeError(f"Calls/observations cannot be aligned for {result.get('id')}")
    tool_action_indices = [i for i, action in enumerate(actions) if action.get("action") == "tool_call"]
    if len(tool_action_indices) != len(calls):
        raise RuntimeError(f"Actions/calls cannot be aligned for {result.get('id')}")
    request_keep = None
    if len(actions) > cap:
        actions = actions[:cap]
        kept_calls = sum(action.get("action") == "tool_call" for action in actions)
        calls = calls[:kept_calls]
        observations = observations[:kept_calls]
        request_keep = cap
        result["state_hash"], result["final_state"] = replay_hash(result["id"], calls, data)
    result["model_output"] = actions
    result["tool_calls"] = calls
    result["observations"] = observations
    result["model_turns"] = len(actions)
    ordered_requests = sorted(request_rows, key=lambda x: int(x.get("action_turn") or 0))
    if request_keep is None:
        kept_requests = ordered_requests
    else:
        kept_requests = [x for x in ordered_requests if int(x.get("action_turn") or 0) <= request_keep]
    qwen_seconds = sum(float(x.get("latency_seconds") or 0) for x in kept_requests)
    if qwen_seconds == 0 and actions:
        qwen_seconds = sum(float((x.get("_diagnostics") or {}).get("latency_seconds") or 0) for x in actions)
    result["qwen_latency_seconds"] = qwen_seconds
    result["asr_latency_seconds"] = asr_seconds
    result["latency_seconds"] = qwen_seconds + asr_seconds
    result["runtime_over_30s"] = result["latency_seconds"] > 30
    return result

def ci_mean_delta(values, seed):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(100 * statistics.mean(rng.choices(values, k=n)) for _ in range(BOOT_DRAWS))
    return [draws[int(.025 * BOOT_DRAWS)], draws[int(.975 * BOOT_DRAWS) - 1]]

def summarize(predictions, ids, items, data, asr_seconds, request_groups, cap):
    from score_btc import score_sample
    scored = []
    for sid in ids:
        pred = project_cap(predictions[sid], cap, request_groups.get(sid, []), asr_seconds[sid], data)
        score = score_sample(sid, items[sid], pred, data)
        scored.append({
            "id": sid,
            "raw": score["btc_formula_score_before_runtime_gate_diagnostic"],
            "gated": score["official_sample_score"],
            "latency_total_seconds": pred["latency_seconds"],
            "qwen_seconds": pred["qwen_latency_seconds"],
            "asr_seconds": asr_seconds[sid],
            "over_30s": pred["latency_seconds"] > 30,
            "runtime_error": bool(pred.get("error")),
            "tool_calls": len(pred.get("tool_calls") or []),
            "model_turns": len(pred.get("model_output") or []),
            "used_cap": len(pred.get("model_output") or []) >= cap,
            "thinking_off": any(x.get("adaptive_thinking_enabled") is False for x in request_groups.get(sid, [])),
        })
    return scored

def stats(rows):
    return {
        "n": len(rows),
        "btc_raw": 100 * statistics.mean(x["raw"] for x in rows),
        "btc_asr_aware_gated": 100 * statistics.mean(x["gated"] for x in rows),
        "mean_asr_seconds": statistics.mean(x["asr_seconds"] for x in rows),
        "mean_qwen_seconds": statistics.mean(x["qwen_seconds"] for x in rows),
        "mean_total_seconds": statistics.mean(x["latency_total_seconds"] for x in rows),
        "over_30s": sum(x["over_30s"] for x in rows),
        "runtime_errors": sum(x["runtime_error"] for x in rows),
        "mean_tool_calls": statistics.mean(x["tool_calls"] for x in rows),
        "samples_reaching_cap": sum(x["used_cap"] for x in rows),
        "thinking_switched_samples": sum(x["thinking_off"] for x in rows),
    }

def analyze():
    bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds = build()
    lock = read_json(OUT / "comparison_lock.json")
    archive = read_jsonl(ARCHIVED)
    random_ids, eligible, union = make_cohorts(ids, archive)
    expected_cohort = {"random60": random_ids, "eligible25": eligible, "run_union": union}
    if lock["cohort"] != expected_cohort:
        raise RuntimeError("Cohort differs from prepared lock.")
    results = {}
    details = []
    for threshold in THRESHOLDS:
        key = str(int(threshold))
        branch = OUT / f"total-{key}s"
        pred_rows = read_jsonl(branch / "predictions-COT.jsonl")
        if [x["id"] for x in pred_rows] != union:
            raise RuntimeError(f"Threshold {threshold} is incomplete or noncanonical.")
        predictions = {x["id"]: x for x in pred_rows}
        request_rows = read_jsonl(branch / "requests-COT.jsonl")
        request_groups = {}
        for row in request_rows:
            request_groups.setdefault(row["sample_id"], []).append(row)
        threshold_results = {}
        for cap in CAPS:
            random_scored = summarize(predictions, random_ids, items, data, asr_seconds, request_groups, cap)
            eligible_scored = summarize(predictions, eligible, items, data, asr_seconds, request_groups, cap)
            threshold_results[str(cap)] = {
                "random60": stats(random_scored),
                "eligible25": stats(eligible_scored),
            }
            for split, rows in (("random60", random_scored), ("eligible25", eligible_scored)):
                details.extend({"threshold_total_seconds": threshold, "max_action_turns": cap,
                                "cohort": split, **row} for row in rows)
        results[key] = threshold_results
    write_jsonl(OUT / "paired_details.jsonl", details)
    summary = {
        "experiment": lock["experiment"],
        "cohort": lock["cohort"],
        "asr_timing": lock["asr_seconds"],
        "thresholds_seconds_include_asr": THRESHOLDS,
        "caps_scored": CAPS,
        "screen_results": results,
        "selection_rule": "Use random60 for overall threshold screening and eligible25 for max-turn screening; add ASR time per sample before applying the local 30-second gate.",
        "caveat": lock["caveat"],
    }
    write_json(OUT / "summary.json", summary)
    lines = [
        "# Adaptive threshold and max-turn screen with ASR-inclusive runtime",
        "",
        "Thresholds are end-to-end seconds: per-sample PhoWhisper main plus history inference_seconds, plus prior Qwen request latency. The first Qwen request always uses thinking. The runner generated up to seven turns once per threshold; caps 4, 5, 6, and 7 are scored from nested prefixes of the same trajectory.",
        "",
        f"ASR timing covers {lock['asr_seconds']['main_clips']} main clips and {lock['asr_seconds']['history_clips']} history clips. Mean per task {lock['asr_seconds']['mean_per_sample']:.2f}s, median {lock['asr_seconds']['median_per_sample']:.2f}s, p95 {lock['asr_seconds']['p95_per_sample']:.2f}s; ASR model-load startup is excluded.",
        "",
    ]
    for threshold in THRESHOLDS:
        lines.extend([f"## Total-time threshold {int(threshold)}s", "",
                      "| Max turns | Cohort | BTC raw | BTC with ASR+Qwen 30s gate | Mean total s | >30s | Mean tool calls |",
                      "|---:|---|---:|---:|---:|---:|---:|"])
        for cap in CAPS:
            for cohort_name in ("random60", "eligible25"):
                m = results[str(int(threshold))][str(cap)][cohort_name]
                lines.append(f"| {cap} | {cohort_name} | {m['btc_raw']:.2f} | {m['btc_asr_aware_gated']:.2f} | {m['mean_total_seconds']:.2f} | {m['over_30s']}/{m['n']} | {m['mean_tool_calls']:.2f} |")
        lines.append("")
    lines.extend([
        "The random60 cohort screens overall threshold behavior; eligible25 is deliberately selected for five-call continuations and only informs the marginal value and runtime of extra turns. Neither screen is a substitute for full val309 or A30 timing.",
        "",
    ])
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "analyzed", "summary": str(OUT / "summary.json"),
                      "asr_seconds": lock["asr_seconds"]}, ensure_ascii=False, indent=2), flush=True)

def run():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use native system Python so local Ollama is reachable.")
    bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds = build()
    archive = read_jsonl(ARCHIVED)
    random_ids, eligible, union = make_cohorts(ids, archive)
    cohort = {"random60": random_ids, "eligible25": eligible, "run_union": union}
    lock = make_lock(ids, items, archive, asr_seconds, cohort, model_lock, prompt)
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "comparison_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Existing lock differs; refusing to mix this run with another configuration.")
    if not lock_path.exists():
        write_json(lock_path, lock)
    write_json(OUT / "cohort_manifest.json", cohort)

    model_digest = model_lock["adapter_final4"]["digest"]
    runner.MODEL = MODEL
    runner.EXPECTED_DIGEST = model_digest
    runner.CAP = GEN_CAP
    runner.CONTEXT = CONTEXT
    runner.BATCH = BATCH
    runner.SEED = 17
    runner.MAX_STEPS = RUN_CAP
    runner.inject_cot_original = runner.inject_cot
    runner.inject_cot = prompt.focused_inject
    runner.seed_everything(17)
    runner.model_fingerprint()

    import lifemate.pipeline as pipeline
    import lifemate.agent as agent_module
    import lifemate.asr as asr_module
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    original_agent = agent_module.Agent
    asr_module.PhoASRModel = compare.CachedPhoWhisper
    asr_module.PhoASRModel.lookup = history
    try:
        patch_ask_stop(pipeline, source)
        for threshold in THRESHOLDS:
            branch = OUT / f"total-{int(threshold)}s"
            branch.mkdir(parents=True, exist_ok=True)
            runner.OUT = branch
            agent_module.Agent = adaptive_factory(original_agent, threshold, asr_seconds)
            pred_path = branch / "predictions-COT.jsonl"
            done_rows = read_jsonl(pred_path)
            done_ids = [row["id"] for row in done_rows]
            if done_ids != union[:len(done_ids)]:
                raise RuntimeError(f"Saved threshold {threshold} output is not a canonical union prefix.")
            for round_no, start in enumerate(range(0, len(union), BLOCK), 1):
                block = union[start:start + BLOCK]
                completed = {row["id"] for row in read_jsonl(pred_path)}
                todo = [sid for sid in block if sid not in completed]
                if not todo:
                    continue
                runner.seed_everything(17)
                runner.warmup("COT", round_no)
                while todo:
                    before = len(read_jsonl(pred_path))
                    try:
                        runner.run_block("COT", round_no, todo, main, items, cot=True)
                    except Exception:
                        after_rows = read_jsonl(pred_path)
                        if len(after_rows) <= before:
                            raise
                        completed = {row["id"] for row in after_rows}
                        todo = [sid for sid in block if sid not in completed]
                        continue
                    completed = {row["id"] for row in read_jsonl(pred_path)}
                    todo = [sid for sid in block if sid not in completed]
            final_rows = read_jsonl(pred_path)
            if [row["id"] for row in final_rows] != union:
                raise RuntimeError(f"Threshold {threshold} did not complete the locked cohort.")
            print(f"[threshold total={threshold:.0f}s] complete {len(final_rows)}/{len(union)}", flush=True)
    finally:
        agent_module.Agent = original_agent
        exec(compile(source, str(pipeline.__file__), "exec"), pipeline.__dict__)
    analyze()

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"
    if mode == "--run":
        run()
    elif mode == "--analyze":
        analyze()
    elif mode == "--prepare":
        bench, compare, prompt, runner, ids, items, main_asr, history, data, model_lock, asr_seconds = build()
        archive = read_jsonl(ARCHIVED)
        random_ids, eligible, union = make_cohorts(ids, archive)
        cohort = {"random60": random_ids, "eligible25": eligible, "run_union": union}
        lock = make_lock(ids, items, archive, asr_seconds, cohort, model_lock, prompt)
        OUT.mkdir(parents=True, exist_ok=True)
        lock_path = OUT / "comparison_lock.json"
        if lock_path.exists() and read_json(lock_path) != lock:
            raise RuntimeError("Existing lock differs; refusing to overwrite it.")
        write_json(lock_path, lock)
        write_json(OUT / "cohort_manifest.json", cohort)
        print(json.dumps({"status": "prepared", "cohort": cohort,
                          "asr_seconds": lock["asr_seconds"], "lock": str(lock_path)},
                         ensure_ascii=False, indent=2), flush=True)
    else:
        raise SystemExit("Use --prepare, --run, or --analyze")

if __name__ == "__main__":
    main()





