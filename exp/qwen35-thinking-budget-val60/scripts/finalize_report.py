from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/qwen35-thinking-budget-val60/output"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp/btc-scoring/scripts"))
from lifemate.utils import find_data_root, load_jsonl  # noqa: E402
from score_btc import call_key, score_sample  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.fmean(values) if values else None,
            "p50": quantile(values, .5), "p90": quantile(values, .9),
            "p95": quantile(values, .95), "max": max(values) if values else None}


def comparisons(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> dict[str, Any]:
    ordered = multiset = state = 0
    per_sample = []
    for sid in ids:
        aa, bb = a[sid], b[sid]
        ca, cb = aa.get("tool_calls") or [], bb.get("tool_calls") or []
        ordered_diff = ca != cb
        multiset_diff = Counter(call_key(x) for x in ca) != Counter(call_key(x) for x in cb)
        state_diff = aa.get("state_hash") != bb.get("state_hash")
        ordered += ordered_diff
        multiset += multiset_diff
        state += state_diff
        per_sample.append({"id": sid, "ordered_calls_match": not ordered_diff,
                           "call_multiset_match": not multiset_diff, "state_hash_match": not state_diff})
    return {"samples": len(ids), "ordered_call_mismatches": ordered,
            "call_multiset_mismatches": multiset, "state_hash_mismatches": state,
            "identical_calls_and_hash": ordered == 0 and state == 0, "per_sample": per_sample}


def main() -> int:
    ids = read_json(OUT / "cohort_manifest.json")["ids"]
    primary = {r["id"]: r for r in read_rows(OUT / "predictions-512.jsonl")}
    repeat = {r["id"]: r for r in read_rows(OUT / "predictions-512-repeat.jsonl")}
    historical = {r["id"]: r for r in read_rows(ROOT / "exp/num-predict-512-vs-768-val/output/predictions-512.jsonl")}
    if set(primary) != set(ids) or set(repeat) != set(ids):
        raise RuntimeError("The current/repeated controls do not cover the exact pilot60 cohort")

    primary_transcripts = read_rows(OUT / "requests-512.jsonl")
    repeat_transcripts = read_rows(OUT / "requests-512-repeat.jsonl")
    def first_requests(rows: list[dict]) -> dict[str, dict]:
        return {r["sample_id"]: r for r in rows if r.get("sample_request_index") == 1}
    first_a, first_b = first_requests(primary_transcripts), first_requests(repeat_transcripts)
    same_first_payloads = sum(first_a[sid]["payload_sha256"] == first_b[sid]["payload_sha256"] for sid in ids)
    same_first_api_counts = sum(all(first_a[sid].get(k) == first_b[sid].get(k) for k in
        ("prompt_eval_count", "eval_count", "done_reason")) for sid in ids)
    same_first_parsed_action = 0
    for sid in ids:
        aa = primary[sid].get("model_output", [])
        bb = repeat[sid].get("model_output", [])
        if aa and bb:
            action = lambda x: {k: x.get(k) for k in ("action", "tool", "arguments")}
            same_first_parsed_action += action(aa[0]) == action(bb[0])

    control_repeat = comparisons(primary, repeat, ids)
    historical_replay = comparisons(primary, historical, ids)
    repeat_to_historical = comparisons(repeat, historical, ids)
    data_root = find_data_root("train+public_test/train+public_test")
    items = {r["id"]: r for r in load_jsonl(data_root / "train/train.jsonl")}
    repeat_scores = [score_sample(sid, items[sid], repeat[sid], data_root) for sid in ids]
    if any(row["gold_replay_error"] for row in repeat_scores):
        raise RuntimeError("Gold ToolEnv replay failed while scoring repeat control")
    (OUT / "btc-per-sample-512-repeat.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in repeat_scores), encoding="utf-8")
    repeat_score = 100 * statistics.fmean(x["btc_formula_score_before_runtime_gate_diagnostic"] for x in repeat_scores)

    with (OUT / "gpu_samples.csv").open(encoding="utf-8-sig", newline="") as f:
        gpu = list(csv.DictReader(f))
    gpu_stats = {"samples": len(gpu), "interval_seconds": 5,
        "coverage_note": "Sampling began during the first experiment session after its initial requests and continued through the 512 repeat control; not full coverage of the first sessions."}
    for field in ("utilization_gpu_pct", "memory_used_mib", "temperature_c", "power_w"):
        vals = [float(x[field]) for x in gpu if x.get(field) not in (None, "", "[N/A]")]
        gpu_stats[field] = describe(vals)

    paired = read_json(OUT / "paired_analysis.json")
    branches = paired["branches"]
    adjacent = paired["paired_adjacent_caps"]
    request1536 = read_rows(OUT / "requests-1536.jsonl")
    hit1536 = [r for r in request1536 if (r.get("eval_count") or 0) >= 1536]
    cap_gate = {
        "requests_reaching_num_predict": len(hit1536), "requests_total": len(request1536),
        "request_rate": len(hit1536) / len(request1536),
        "samples_with_num_predict_reached": len({r["sample_id"] for r in hit1536}),
        "context_limited_candidates": branches["1536"]["context_limited_candidates_inferred"],
        "paired_1536_vs_1024": adjacent["1024_vs_1536"],
        "decision": "Do not run 2048: true cap hits were rare and paired quality did not trend upward; the control repeatability check also failed."}

    src_current = sha256(ROOT / "src/lifemate/agent.py")
    prior_config = read_json(ROOT / "exp/num-predict-512-vs-768-val/output/config_manifest.json")
    older_config = read_json(ROOT / "exp/inference-latency-optimization-val60/output/config_manifest.json")
    repro = {
        "primary_512_vs_same_setting_512_repeat": control_repeat,
        "primary_512_vs_archived_512_pilot": historical_replay,
        "repeat_512_vs_archived_512_pilot": repeat_to_historical,
        "same_first_request_payload_sha256": {"matches": same_first_payloads, "samples": len(ids)},
        "same_first_request_api_counts": {"matches": same_first_api_counts, "samples": len(ids)},
        "same_first_parsed_action_ignoring_telemetry": {"matches": same_first_parsed_action, "samples": len(ids)},
        "repeat_control_btc_formula_score_before_runtime_gate_0_100": repeat_score,
        "primary_control_btc_formula_score_before_runtime_gate_0_100": branches["512"]["btc_formula_score_0_100"],
        "ollama_version_current_prior": ["0.40.14", prior_config.get("ollama_version"),
            older_config.get("locked_config", {}).get("ollama_version")],
        "model_digest_current_prior": ["d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04",
            prior_config.get("model_digest"), older_config.get("locked_config", {}).get("model_digest")],
        "prompt_hash_current_prior": [read_json(OUT / "config_manifest.json").get("source_prompt_sha256_utf8"),
            prior_config.get("prompt_sha256_utf8"), older_config.get("inputs_sha256", {}).get("prompt_utf8")],
        "agent_source_sha256_current": src_current,
        "agent_source_sha256_prior_pilot": prior_config.get("inputs_sha256", {}).get("src/lifemate/agent.py"),
        "prior_request_telemetry_had_num_batch": any("num_batch" in x for x in read_rows(ROOT / "exp/num-predict-512-vs-768-val/output/token_usage-512.jsonl")),
        "current_request_num_batch_values": sorted({x.get("num_batch") for x in primary_transcripts}),
        "ollama_seed_option_sent": False,
        "python_seed_everything": 17,
        "quality_selection_gate_passed": control_repeat["identical_calls_and_hash"],
        "interpretation": "Identical first request payload SHA-256 across the current control repeat, but model action/state outputs diverged. This indicates run-to-run response instability under the observed setup; do not select a cap from these quality point estimates. Historical pilot mismatch is also retained separately.",
    }
    write_json(OUT / "reproducibility.json", repro)
    paired["reproducibility"] = repro
    paired["gpu_observations"] = gpu_stats
    paired["cap2048_gate"] = cap_gate
    paired["quality_selection_gate_passed"] = control_repeat["identical_calls_and_hash"]
    paired["final_cap_recommendation"] = "No cap selected: same-setting 512 control failed exact output reproducibility. Resolve Ollama response variance and rerun controlled quality comparison."
    write_json(OUT / "paired_analysis.json", paired)

    cfg = read_json(OUT / "config_manifest.json")
    cfg["source_sha256_final"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (
        ROOT / "src/lifemate/agent.py", Path(__file__).resolve(), ROOT / "exp/qwen35-thinking-budget-val60/scripts/run_pilot.py",
        ROOT / "exp/qwen35-thinking-budget-val60/scripts/analyze_pilot.py",
        ROOT / "exp/qwen35-thinking-budget-val60/scripts/repeat_control_512.py",
        ROOT / "exp/qwen35-thinking-budget-val60/scripts/recover_prefix_duplicates.py",
        ROOT / "exp/qwen35-thinking-budget-val60/scripts/sample_gpu.py")}
    cfg["branch_run_manifest_sha256"] = {str(cap): sha256(OUT / f"run-{cap}.json") for cap in (512, 768, 1024, 1536)}
    cfg["repeat_control_run_manifest_sha256"] = sha256(OUT / "run-512-repeat.json")
    cfg["main_run_runner_sha256_at_preflight"] = cfg.get("input_sha256", {}).get("exp\\qwen35-thinking-budget-val60\\scripts\\run_pilot.py")
    cfg["repeat_control_runner_sha256"] = sha256(ROOT / "exp/qwen35-thinking-budget-val60/scripts/run_pilot.py")
    cfg["runner_recovery_note"] = "The main run used the preflight runner whose prefix flush duplicated old rows in files; inference was not repeated during recovery. The later runner changed output writing and telemetry labels only. Exact duplicate content was audited and archived before analysis."
    cfg["control_reproducibility_passed"] = control_repeat["identical_calls_and_hash"]
    cfg["quality_cap_selected"] = None
    write_json(OUT / "config_manifest.json", cfg)
    for cap in (512, 768, 1024, 1536):
        run = read_json(OUT / f"run-{cap}.json")
        write_json(OUT / f"run-config-{cap}.json", {"locked_common_config": cfg["locked_common_config"],
            "num_predict": cap, "effective_num_batch": 512, "num_ctx": 8192,
            "branch_run_manifest_sha256": sha256(OUT / f"run-{cap}.json"),
            "prediction_sha256": sha256(OUT / f"predictions-{cap}.jsonl"),
            "request_telemetry_sha256": sha256(OUT / f"requests-{cap}.jsonl"),
            "samples": run["samples_completed"]})

    summary = (OUT / "summary.md").read_text(encoding="utf-8").split(
        "\n## Reproducibility investigation and final decision", 1)[0]
    summary += "\n## Adjacent cap paired trend\n\n"
    summary += "| Lower → upper cap | BTC delta points (95% paired CI) | Improved / worsened / tied | Sample latency delta seconds (95% paired CI) |\n|---:|---:|---:|---:|\n"
    for lower, upper in ((512, 768), (768, 1024), (1024, 1536)):
        item = adjacent[f"{lower}_vs_{upper}"]
        score_ci = item["btc_delta_points_upper_minus_lower"]
        latency_ci = item["latency_delta_seconds_upper_minus_lower"]
        outcome = item["score_outcomes_upper_vs_lower"]
        summary += (f"| {lower} → {upper} | {score_ci['mean']:+.3f} [{score_ci['ci95_low']:+.3f}, {score_ci['ci95_high']:+.3f}] | "
            f"{outcome.get('improved',0)} / {outcome.get('worsened',0)} / {outcome.get('tied',0)} | "
            f"{latency_ci['mean']:+.2f} [{latency_ci['ci95_low']:+.2f}, {latency_ci['ci95_high']:+.2f}] |\n")
    summary += "\n"
    summary += "\n## Reproducibility investigation and final decision\n\n"
    summary += (f"The `num_predict=512` primary control and same-setting repeat differed on **{control_repeat['ordered_call_mismatches']}/60** ordered call sequences "
        f"(**{control_repeat['call_multiset_mismatches']}/60** call multisets) and **{control_repeat['state_hash_mismatches']}/60** state hashes. "
        f"All 60 first-request payload SHA-256 values matched, while first parsed actions matched {same_first_parsed_action}/60; so identical inputs still produced different first responses on 11 samples. "
        f"The primary control also differs from the archived 512 artifact on {historical_replay['ordered_call_mismatches']}/60 ordered calls and {historical_replay['state_hash_mismatches']}/60 state hashes. "
        f"The repeat BTC formula score was {repeat_score:.3f}/100 and it matched the archived calls/hashes on {repeat_to_historical['samples'] - repeat_to_historical['ordered_call_mismatches']}/60. The Ollama version/model digest and prompt/transcript fingerprints are reported in `reproducibility.json`; current source and request telemetry are fingerprinted in `config_manifest.json`. Python/torch seed 17 was set, but the Ollama request did not pass an explicit server-side `seed` option, which remains a detail to resolve separately. Quality selection is therefore **not cleared**: the cap scores below are screening measurements and no cap is recommended from them.\n\n")
    summary += (f"The 1536 branch had **{len(hit1536)}/{len(request1536)}** requests ({100*len(hit1536)/len(request1536):.1f}%) reach `num_predict`, across "
        f"{len({r['sample_id'] for r in hit1536})}/60 samples; {cap_gate['context_limited_candidates']} additional length cases were classified separately as context-limited candidates. "
        f"Paired 1536−1024 BTC delta was {adjacent['1024_vs_1536']['btc_delta_points_upper_minus_lower']['mean']:+.3f} points "
        f"(95% CI [{adjacent['1024_vs_1536']['btc_delta_points_upper_minus_lower']['ci95_low']:+.3f}, {adjacent['1024_vs_1536']['btc_delta_points_upper_minus_lower']['ci95_high']:+.3f}]); "
        f"paired outcomes were {adjacent['1024_vs_1536']['score_outcomes_upper_vs_lower']}. The cap-hit rate is low and the paired point trend does not improve, so the conditional 2048 branch was **not run**.\n\n")
    summary += (f"GPU sampling: {gpu_stats['samples']} samples at 5-second intervals, partial experiment coverage. Utilization mean/p95/max "
        f"{gpu_stats['utilization_gpu_pct']['mean']:.1f}/{gpu_stats['utilization_gpu_pct']['p95']:.0f}/{gpu_stats['utilization_gpu_pct']['max']:.0f}%; "
        f"VRAM mean/max {gpu_stats['memory_used_mib']['mean']:.0f}/{gpu_stats['memory_used_mib']['max']:.0f} MiB; "
        f"temperature mean/max {gpu_stats['temperature_c']['mean']:.1f}/{gpu_stats['temperature_c']['max']:.0f}°C; "
        f"power mean/max {gpu_stats['power_w']['mean']:.1f}/{gpu_stats['power_w']['max']:.1f} W. Detailed observations are in `gpu_samples.csv`.\n\n")
    summary += "## Latency and token telemetry\n\n"
    summary += "| Cap | Sample latency p50 / p90 / p95 / max (s) | Request latency p50 / p90 / p95 / max (s) | Prompt tokens/request p50 / p90 / p95 / max | Generated tokens/request p50 / p90 / p95 / max | Prompt+generated/request p50 / p90 / p95 / max | Request headroom p50 / p90 / p95 / max | Minimum context headroom/sample p50 / p90 / p95 / max | Prompt-eval / decode / load total (s) | `length` / cap-hit / context-candidate / unresolved | Repair attempts / failures |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for cap in (512, 768, 1024, 1536):
        m = branches[str(cap)]
        sl, rl = m["sample_latency_seconds"], m["request_latency_seconds"]
        pp, gg, tg, hh = (m["prompt_tokens_request"], m["generated_tokens_request_including_thinking"],
                          m["prompt_plus_generated_tokens_request"], m["sample_minimum_context_headroom"])
        fmt = lambda x: "/".join(f"{x[k]:.0f}" for k in ("p50", "p90", "p95", "max") if x.get(k) is not None)
        req_rows = read_rows(OUT / f"requests-{cap}.jsonl")
        req_headroom = describe([float(8192 - (x.get("prompt_eval_count") or 0) - (x.get("eval_count") or 0)) for x in req_rows])
        summary += (f"| {cap} | {fmt(sl)} | {fmt(rl)} | {fmt(pp)} | {fmt(gg)} | {fmt(tg)} | {fmt(req_headroom)} | {fmt(hh)} | "
            f"{m['prompt_eval_duration_seconds']:.1f} / {m['decode_duration_seconds']:.1f} / {m['load_duration_seconds']:.1f} | "
            f"{m['done_reason_length']} / {m['requests_reaching_num_predict']} / {m['context_limited_candidates_inferred']} / {m['length_reason_unresolved']} | "
            f"{m['repair_turns_attempted']} / {m['repair_turns_failed']} |\n")
    summary += "\n`eval_count` and generated-token totals include thinking plus answer; no exact thinking-token counter was available. `prompt_eval_count` is Ollama's counter; component estimates are approximate.\n\n"
    summary += "## Separate quality diagnostics\n\n"
    summary += "| Cap | Exact-call precision / recall / F1 diagnostic | Final state match /60 | Local proxy mean diagnostic | Schema rejections | Runtime errors | Empty predictions | Extra calls/duplicates U | Under-calls | Samples >30s local |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for cap in (512, 768, 1024, 1536):
        m = branches[str(cap)]
        summary += (f"| {cap} | {m['aggregate_exact_call_precision_diagnostic']:.3f} / {m['aggregate_exact_call_recall_diagnostic']:.3f} / {m['aggregate_exact_call_f1_diagnostic']:.3f} | "
            f"{m['final_state_matches']}/60 | {m['local_proxy_mean_diagnostic']:.3f} | {m['schema_rejections']} | {m['runtime_errors']} | "
            f"{m['empty_predictions']} | {m['unmatched_calls_U']} | {m['under_calls']} | {m['samples_over_30s_local']} |\n")
    summary += "\nThese are distinct diagnostics; neither aggregate call F1, final-state match, local proxy, nor the local timeout count substitutes for the BTC formula score.\n\n"
    summary += "Current server runner command-line settings were not collected in this run; the native Ollama log was unavailable. The API version/model digest, request options, effective `num_batch=512`, GPU telemetry, and load counters were recorded.\n\n"
    summary += "Final outcome: **no cap recommendation pending resolution of control reproducibility**. The pilot and telemetry artifacts are complete; cap2048 was not warranted by the observed rare cap hits and non-improving paired point trend. This is a local RTX 4050 diagnostic on an enriched train-only cohort, not val309 or A30 compliance evidence.\n"
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    cfg = read_json(OUT / "config_manifest.json")
    artifact_paths = [OUT / f"{prefix}-{cap}.jsonl" for cap in (512, 768, 1024, 1536)
                      for prefix in ("predictions", "step-traces", "requests", "btc-per-sample")]
    artifact_paths += [OUT / "predictions-512-repeat.jsonl", OUT / "requests-512-repeat.jsonl",
        OUT / "btc-per-sample-512-repeat.jsonl", OUT / "paired_analysis.json", OUT / "reproducibility.json",
        OUT / "recovery_audit.json", OUT / "execution_order.json", OUT / "summary.md", OUT / "gpu_samples.csv"]
    cfg["artifact_sha256"] = {p.name: sha256(p) for p in artifact_paths if p.exists()}
    write_json(OUT / "config_manifest.json", cfg)
    print(json.dumps({"control_repeat_mismatches": {"ordered_calls": control_repeat["ordered_call_mismatches"],
        "call_multisets": control_repeat["call_multiset_mismatches"], "state_hashes": control_repeat["state_hash_mismatches"]},
        "first_payload_hash_matches": same_first_payloads, "cap2048": cap_gate["decision"],
        "quality_recommendation": paired["final_cap_recommendation"], "gpu_samples": len(gpu)}, indent=2))
    return 0


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
