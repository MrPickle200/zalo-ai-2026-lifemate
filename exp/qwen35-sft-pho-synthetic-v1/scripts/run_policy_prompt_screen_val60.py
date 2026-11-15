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
OUT = EXP / "output/policy-prompt-screen-val60"
BASE_OUT = EXP / "output/two-policy-val60"
MODEL = "lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
SEED = 17
BOOTSTRAP_SEED = 20261115
CAP = 768
CONTEXT = 8192
BATCH = 512
MAX_STEPS = 5
ASK_STOP_PATCH_OLD = (
    "            self.agent.record_tool_result(action, observation)\n"
    "        calls = env.get_tool_calls()"
)
ASK_STOP_PATCH_NEW = (
    "            self.agent.record_tool_result(action, observation)\n"
    "            if action.get(\"tool\") == \"sys.ask_missing_info\":\n"
    "                break\n"
    "        calls = env.get_tool_calls()"
)

VARIANTS = {
    "ask_precision": (
        "Use sys.ask_missing_info only when a required field is still missing or ambiguous "
        "after checking the transcript, conversation history, profile, and lookup results. "
        "If those sources identify one clear candidate, proceed without asking. Ask only for "
        "unresolved required fields, and never ask again for information already available."
    ),
    "completion_audit": (
        "Keep track of every outcome the user explicitly requested. A read-only lookup does "
        "not complete a requested change: continue with the needed action using the latest "
        "observation. Stop only after all requested outcomes are complete, or when the next "
        "step cannot be done safely."
    ),
    "argument_grounding": (
        "Ground every argument in the user's words, context, history, or the latest tool "
        "observation. Preserve entity IDs, numbers, and exact values from those sources. "
        "Use only schema-supported arguments; omit optional values that are not supplied or "
        "safely inferable, and never invent or substitute an identifier."
    ),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if path.exists()
        else []
    )


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use system Python so the local Ollama service is reachable.")
    base = load_module(
        "two_policy_pilot60_for_prompt_screen",
        EXP / "scripts/run_two_policy_val60.py",
    )
    ev, compare, prompt, runner, ids, cohort, items, main, history, data, model_record, final4, cohort_manifest = base.build()
    if len(cohort) != 60:
        raise RuntimeError(f"Expected the existing pilot60, found {len(cohort)} IDs.")
    baseline_path = BASE_OUT / "inference/fixed_ask_stop/predictions-COT.jsonl"
    baseline_rows = read_jsonl(baseline_path)
    if [row["id"] for row in baseline_rows] != cohort:
        raise RuntimeError("The fixed ask-stop pilot60 baseline is incomplete or out of order.")
    baseline_details = read_json(BASE_OUT / "paired_details.json")
    if [row["id"] for row in baseline_details] != cohort:
        raise RuntimeError("The archived paired details do not match the pilot60 cohort.")
    runner.MODEL = MODEL
    runner.EXPECTED_DIGEST = model_record["digest"]
    runner.CAP = CAP
    runner.CONTEXT = CONTEXT
    runner.BATCH = BATCH
    runner.SEED = SEED
    runner.MAX_STEPS = MAX_STEPS
    return (
        base, ev, compare, prompt, runner, ids, cohort, items, main, history,
        data, model_record, final4, cohort_manifest, baseline_rows, baseline_details,
    )


def make_injector(prompt, instruction: str):
    def inject(messages):
        result = prompt.focused_inject(messages)
        systems = [message for message in result if message.get("role") == "system"]
        if len(systems) != 1:
            raise RuntimeError("Expected exactly one system message.")
        systems[0]["content"] += "\n\nEXPERIMENTAL POLICY:\n" + instruction
        return result
    return inject


def patch_ask_stop(pipeline_module, source: str, enabled: bool):
    if enabled:
        if ASK_STOP_PATCH_OLD not in source:
            raise RuntimeError("Could not place the successful-ask stop guard.")
        patched = source.replace(ASK_STOP_PATCH_OLD, ASK_STOP_PATCH_NEW, 1)
    else:
        patched = source
    exec(compile(patched, str(pipeline_module.__file__), "exec"), pipeline_module.__dict__)


def bootstrap_ci(differences, seed: int, draws: int = 10000):
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(
        100 * statistics.mean(rng.choices(differences, k=n))
        for _ in range(draws)
    )
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def run():
    (
        base, ev, compare, prompt, runner, ids, cohort, items, main, history,
        data, model_record, final4, cohort_manifest, baseline_rows, baseline_details,
    ) = build()
    OUT.mkdir(parents=True, exist_ok=True)
    import lifemate.pipeline as pipeline_module
    import lifemate.agent as agent_module

    original_pipeline_source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
    base_agent = agent_module.Agent
    input_paths = [
        Path(__file__).resolve(),
        EXP / "scripts/run_two_policy_val60.py",
        EXP / "scripts/benchmark_adapter_final4.py",
        ROOT / "exp/qwen35-thinking-budget-val60/output/cohort_manifest.json",
        EXP / "output/adapter-final-4epoch-benchmark/val309/comparison_lock.json",
        EXP / "output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl",
        BASE_OUT / "comparison_lock.json",
        BASE_OUT / "paired_details.json",
        BASE_OUT / "inference/fixed_ask_stop/predictions-COT.jsonl",
        ROOT / "src/lifemate/pipeline.py",
        ROOT / "src/lifemate/agent.py",
        ROOT / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
    ]
    prompt_hashes = {
        key: prompt.prompt_hash(make_injector(prompt, instruction))
        for key, instruction in VARIANTS.items()
    }
    lock = {
        "experiment": "qwen35-sft-pho-synthetic-v1-policy-prompt-screen-val60",
        "cohort": {
            "n": 60,
            "ids": cohort,
            "source": "existing enriched fixed pilot60",
            "source_manifest_sha256": sha(ROOT / "exp/qwen35-thinking-budget-val60/output/cohort_manifest.json"),
            "ids_sha256": hashlib.sha256("\n".join(cohort).encode()).hexdigest(),
        },
        "model": {"tag": MODEL, "digest": model_record["digest"]},
        "baseline": {
            "branch": "fixed_ask_stop",
            "prediction_sha256": sha(BASE_OUT / "inference/fixed_ask_stop/predictions-COT.jsonl"),
            "summary_sha256": sha(BASE_OUT / "summary.json"),
        },
        "prompts": {
            key: {"instruction": instruction, "sha256": prompt_hashes[key]}
            for key, instruction in VARIANTS.items()
        },
        "settings": {
            "temperature": 0,
            "thinking": True,
            "protocol": "JSON",
            "num_predict": CAP,
            "num_ctx": CONTEXT,
            "num_batch": BATCH,
            "seed": SEED,
            "max_action_turns": MAX_STEPS,
            "asr": "PhoWhisper Large v2 cached main/history",
        },
        "policy": {"stop_after_successful_sys_ask_missing_info": True},
        "execution_order": {
            "paired_block_size": 15,
            "rotating_variant_order": [
                ["ask_precision", "completion_audit", "argument_grounding"],
                ["completion_audit", "argument_grounding", "ask_precision"],
                ["argument_grounding", "ask_precision", "completion_audit"],
                ["ask_precision", "argument_grounding", "completion_audit"],
            ],
        },
        "caveat": (
            "Enriched diagnostic pilot60, not an unbiased full-val309 or Public LB estimate. "
            "Timing measured on local RTX 4050, not the competition A30."
        ),
        "inputs_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in input_paths
        },
    }
    lock_path = OUT / "comparison_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Existing screen lock differs from current inputs; refusing to overwrite.")
    write_json(lock_path, lock)

    orders = lock["execution_order"]["rotating_variant_order"]
    try:
        patch_ask_stop(pipeline_module, original_pipeline_source, True)
        for block_index, start in enumerate(range(0, 60, 15)):
            block = cohort[start : start + 15]
            for key in orders[block_index]:
                branch_out = OUT / "inference" / key
                branch_out.mkdir(parents=True, exist_ok=True)
                runner.OUT = branch_out
                runner.inject_cot = make_injector(prompt, VARIANTS[key])
                done = {row["id"] for row in read_jsonl(branch_out / "predictions-COT.jsonl")}
                todo = [sample_id for sample_id in block if sample_id not in done]
                if not todo:
                    continue
                runner.warmup("COT", block_index + 1)
                runner.run_block("COT", block_index + 1, todo, main, items, cot=True)
                print(
                    f"completed {key}: pilot block {block_index + 1}/4 "
                    f"({start + len(block)}/60 IDs)",
                    flush=True,
                )
    finally:
        patch_ask_stop(pipeline_module, original_pipeline_source, False)
        agent_module.Agent = base_agent

    analyze(prompt, cohort, items, data, model_record, baseline_rows, baseline_details, lock)


def analyze(prompt, cohort, items, data, model_record, baseline_rows, baseline_details, lock):
    from score_btc import call_key, score_sample

    baseline_by_id = {row["id"]: row for row in baseline_rows}
    baseline_detail_by_id = {row["id"]: row["fixed_ask_stop"] for row in baseline_details}
    records = []
    summaries = {}
    for key in VARIANTS:
        branch_out = OUT / "inference" / key
        predictions = read_jsonl(branch_out / "predictions-COT.jsonl")
        if [row["id"] for row in predictions] != cohort:
            raise RuntimeError(f"{key} predictions are incomplete or out of pilot cohort order.")
        pred_by_id = {row["id"]: row for row in predictions}
        req_rows = read_jsonl(branch_out / "requests-COT.jsonl")
        scores = []
        branch_records = []
        for sample_id in cohort:
            prediction = pred_by_id[sample_id]
            result = score_sample(sample_id, items[sample_id], prediction, data)
            calls = prediction.get("tool_calls") or []
            asks = [call for call in calls if call.get("name") == "sys.ask_missing_info"]
            gold = items[sample_id].get("label", {}).get("tool_calls", [])
            gold_asks = [call for call in gold if call.get("name") == "sys.ask_missing_info"]
            score_row = {
                "btc_after_runtime_gate": result["official_sample_score"],
                "btc_before_gate": result["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match": result["hash_match"],
                "runtime_error": result["runtime_failure"],
                "over_30s": result["runtime_over_30s"],
                "latency_seconds": prediction.get("latency_seconds"),
                "matched_calls": result["matched_calls"],
                "unmatched_calls": result["unmatched_submitted_calls"],
                "gold_calls": result["gold_calls"],
                "ask_count": len(asks),
                "ask_exact": any(call_key(a) == call_key(g) for a in asks for g in gold_asks),
                "repeated_ask": len(asks) > 1,
            }
            scores.append(score_row)
            baseline_score = baseline_detail_by_id[sample_id]
            branch_records.append({
                "id": sample_id,
                "baseline_fixed_ask_stop": baseline_score,
                key: score_row,
                "prediction_tool_calls": calls,
                "prediction_state_hash": prediction.get("state_hash"),
            })
        records.extend(branch_records)
        latencies = [float(row["latency_seconds"]) for row in scores if row["latency_seconds"] is not None]
        summaries[key] = {
            "btc_before_gate": 100 * statistics.mean(row["btc_before_gate"] for row in scores),
            "btc_after_30s_gate": 100 * statistics.mean(row["btc_after_runtime_gate"] for row in scores),
            "over_30s": sum(row["over_30s"] for row in scores),
            "runtime_errors": sum(row["runtime_error"] for row in scores),
            "hash_matches": sum(row["hash_match"] for row in scores),
            "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
            "p50_latency_seconds": statistics.median(latencies) if latencies else None,
            "p90_latency_seconds": sorted(latencies)[math.ceil(0.90 * len(latencies)) - 1] if latencies else None,
            "matched_calls": sum(row["matched_calls"] for row in scores),
            "unmatched_calls": sum(row["unmatched_calls"] for row in scores),
            "gold_calls": sum(row["gold_calls"] for row in scores),
            "ask_calls": sum(row["ask_count"] for row in scores),
            "exact_gold_ask_samples": sum(row["ask_exact"] for row in scores),
            "samples_with_repeated_ask": sum(row["repeated_ask"] for row in scores),
            "requests": len(req_rows),
            "prompt_sha256": lock["prompts"][key]["sha256"],
        }
        before_diffs = []
        after_diffs = []
        wins = ties = losses = 0
        for sample_id, current in zip(cohort, scores):
            baseline_score = baseline_detail_by_id[sample_id]
            after = current["btc_after_runtime_gate"] - baseline_score["btc_after_runtime_gate"]
            before = current["btc_before_gate"] - baseline_score["btc_before_runtime_gate"]
            after_diffs.append(after)
            before_diffs.append(before)
            wins += after > 0
            ties += after == 0
            losses += after < 0
        summaries[key]["paired_vs_fixed_ask_stop"] = {
            "delta_after_gate_points": 100 * statistics.mean(after_diffs),
            "ci95_after_gate_points": bootstrap_ci(after_diffs, BOOTSTRAP_SEED + len(summaries)),
            "delta_before_gate_points": 100 * statistics.mean(before_diffs),
            "ci95_before_gate_points": bootstrap_ci(before_diffs, BOOTSTRAP_SEED + 100 + len(summaries)),
            "wins_ties_losses": [wins, ties, losses],
        }

    result = {
        "experiment": lock["experiment"],
        "cohort": lock["cohort"],
        "model": lock["model"],
        "baseline": lock["baseline"],
        "prompts": lock["prompts"],
        "settings": lock["settings"],
        "policy": lock["policy"],
        "branches": summaries,
        "method": {
            "scoring": "BTC official ToolEnv replay and exact-call formula with duplicate multiplicity; runtime gate and ungated diagnostic reported separately.",
            "paired_by_sample": True,
            "bootstrap_draws": 10000,
            "baseline": "Existing same-cohort fixed-thinking final4 + ask-stop branch.",
        },
        "caveat": lock["caveat"],
    }
    write_json(OUT / "paired_details.json", records)
    write_json(OUT / "summary.json", result)
    lines = [
        "# Prompt-only policy screen — enriched pilot60",
        "",
        "Final4 adapter, PhoWhisper Large v2, fixed thinking on, JSON, temperature 0, "
        "num_predict 768, context 8192, batch 512, seed 17, five action turns, and the "
        "successful sys.ask_missing_info stop guard were held constant. Each arm adds one "
        "instruction to the focused system prompt. Baseline is the archived fixed_ask_stop "
        "branch on the same enriched pilot60. Scores and local RTX 4050 timing are diagnostic, "
        "not an unbiased estimate of full val309/Public LB or A30 runtime.",
        "",
        "| Prompt arm | BTC raw | BTC after 30s gate | Δ vs baseline (gated) | 95% CI | >30s | Mean sec | Hashes | Exact asks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, summary in summaries.items():
        paired = summary["paired_vs_fixed_ask_stop"]
        ci = paired["ci95_after_gate_points"]
        lines.append(
            f"| {key} | {summary['btc_before_gate']:.2f} | {summary['btc_after_30s_gate']:.2f} | "
            f"{paired['delta_after_gate_points']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{summary['over_30s']}/60 | {summary['mean_latency_seconds']:.2f} | "
            f"{summary['hash_matches']}/60 | {summary['exact_gold_ask_samples']}/60 |"
        )
    lines += [
        "",
        "## Decision rule",
        "",
        "Treat this as a low-cost screen. Do not promote a prompt arm based on a pilot60 win alone; "
        "confirm the strongest arm against full val309 with the 30-second gate before using it "
        "for another public candidate.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()


