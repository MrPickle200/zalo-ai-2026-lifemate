from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.asr import PhoASRModel
from lifemate.pipeline import LifeMatePipeline
from lifemate.tool_executor import create_tool_env
from lifemate.utils import find_data_root, load_jsonl


ROOT = Path("exp/qwen35-thinking-tool-protocol-val309/output")
REFERENCE = Path("exp/session4-qwen3-vs-qwen35-val309/output/qwen35-en-cap5/predictions.jsonl")
VARIANTS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}


def smoke_ids(ids: list[str], items: dict[str, dict]) -> list[str]:
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for sid in ids:
        calls = items[sid].get("label", {}).get("tool_calls", [])
        bucket = "0" if len(calls) == 0 else "1" if len(calls) == 1 else "2" if len(calls) == 2 else "3+"
        missing = any(c.get("name") == "sys.ask_missing_info" for c in calls)
        strata[(bucket, missing)].append(sid)
    selected = []
    keys = [(b, m) for b in ("0", "1", "2", "3+") for m in (False, True)]
    # Fixed-order round robin gives length coverage and includes missing-info cases when present.
    while len(selected) < 12 and any(strata.values()):
        for key in keys:
            while strata[key] and strata[key][0] in selected:
                strata[key].pop(0)
            if strata[key] and len(selected) < 12:
                selected.append(strata[key].pop(0))
    return selected


def gpu_snapshot() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=[*VARIANTS, "all"], required=True)
    p.add_argument("--cohort", choices=("smoke", "val309"), required=True)
    p.add_argument("--data-root")
    p.add_argument("--resume", action="store_true", help="resume an existing output only when it is a strict prefix of the fixed cohort")
    a = p.parse_args()
    root = find_data_root(a.data_root)
    ids = json.loads(Path("data/splits/val_ids.json").read_text(encoding="utf-8"))
    items = {x["id"]: x for x in load_jsonl(root / "train" / "train.jsonl")}
    reference = {x["id"]: x for x in (json.loads(line) for line in REFERENCE.read_text(encoding="utf-8").splitlines() if line.strip())}
    if set(ids) - set(reference):
        raise RuntimeError("The existing Qwen3.5 transcript cache does not cover fixed val309")
    selected_ids = smoke_ids(ids, items) if a.cohort == "smoke" else ids
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "val309_ids.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    manifest = {"objective": "2x2 Qwen3.5 thinking by Ollama tool protocol experiment",
                "model": "qwen3.5:4b-q4_K_M", "parameters": "4.2B", "quantization": "Q4_K_M", "ollama_version": "0.40.13",
                "fixed_cohort": "fixed train-only val309", "cohort_ids_file": "val309_ids.json",
                "transcripts": str(REFERENCE), "prompt_policy": "system policy v2 English; observation-grounded-v2",
                "temperature": 0, "num_ctx": 8192, "num_predict": 512, "max_action_turns": 5,
                "toolenv_sessions": "exactly one per sample per branch", "variants": VARIANTS,
                "native_schema_conversion": "JSON Schema function parameters preserve names, descriptions, types, required, enum, and standard constraints; x-one-of becomes anyOf.required; unsupported x-ref/x-match are recorded per request and enforced by ToolEnv.validate",
                "production_default_changed": False, "submission_created": False, "public_lb_sent": False}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if a.cohort == "smoke":
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "smoke_ids.json").write_text(json.dumps(selected_ids, indent=2) + "\n", encoding="utf-8")
    if a.variant == "all":
        variants = list(VARIANTS)
    else:
        variants = [a.variant]

    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = "qwen3.5:4b-q4_K_M"
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "8192"
    os.environ["LIFEMATE_PROMPT_LANGUAGE"] = "en"
    for variant in variants:
        thinking, native = VARIANTS[variant]
        out = ROOT / f"{variant}-{ 'think' if thinking else 'nothink'}-{ 'native' if native else 'json'}" / a.cohort
        out.mkdir(parents=True, exist_ok=True)
        config = {"variant": variant, "model": "qwen3.5:4b-q4_K_M", "parameters": "4.2B", "quantization": "Q4_K_M",
                  "ollama_version": "0.40.13",
                  "thinking": thinking, "protocol": "native_tools_role_tool" if native else "json_observations_payload",
                  "cohort": "fixed train-only val309" if a.cohort == "val309" else "fixed stratified 12-ID smoke from val309",
                  "ids": selected_ids, "phoasr_transcript_source": str(REFERENCE), "prompt": "system policy v2 English",
                  "temperature": 0, "num_ctx": 8192, "num_predict": 512, "max_action_turns": 5,
                  "toolenv_per_sample": 1, "gpu_before": gpu_snapshot()}
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        agent = Agent(max_new_tokens=512, context_limit=8192, ollama_thinking=thinking, ollama_native_tools=native)
        pipeline = LifeMatePipeline(root, "train", asr=PhoASRModel(), agent=agent, max_steps=5)
        prediction_path = out / "predictions.jsonl"
        diagnostics_path = out / "diagnostics.jsonl"
        completed_ids = []
        if a.resume:
            if prediction_path.exists() != diagnostics_path.exists():
                raise RuntimeError("Cannot resume: prediction and diagnostics artifacts are not both present")
            if prediction_path.exists():
                prior_predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                prior_diagnostics = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                completed_ids = [row["id"] for row in prior_predictions]
                if len(prior_diagnostics) != len(completed_ids) or completed_ids != selected_ids[:len(completed_ids)]:
                    raise RuntimeError("Cannot resume: saved rows are not an aligned prefix of this fixed cohort")
        stream_mode = "a" if a.resume else "w"
        with prediction_path.open(stream_mode, encoding="utf-8") as predictions, diagnostics_path.open(stream_mode, encoding="utf-8") as diagnostics:
            for index, sid in enumerate(selected_ids[len(completed_ids):], start=len(completed_ids) + 1):
                item = items[sid]
                transcript = reference[sid].get("transcript_used") or reference[sid].get("asr_text") or ""
                started = time.perf_counter()
                try:
                    result = pipeline.run(item, transcript_override=transcript)
                    pred_calls = [{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in result["tool_calls"]]
                    row = {"id": sid, "asr_text": transcript, "transcript_used": transcript,
                           "model_output": result["model_output"], "tool_calls": pred_calls,
                           "state_hash": result["state_hash"], "observations": result["observations"],
                           "latency_seconds": result["latency_seconds"], "final_state": result["final_state"]}
                    turn_diagnostics = [step.get("_diagnostics", {}) for step in result["model_output"]]
                    diag = {"id": sid, "latency_seconds": result["latency_seconds"], "turns": turn_diagnostics,
                            "action_count": sum(step.get("action") == "tool_call" for step in result["model_output"]),
                            "stop_early": bool(result["model_output"] and result["model_output"][-1].get("action") == "stop"
                                               and len(pred_calls) < len(item.get("label", {}).get("tool_calls", []))),
                            "action_cap_hit": sum(step.get("action") == "tool_call" for step in result["model_output"]) >= 5,
                            "cap_exhausted_before_gold": (sum(step.get("action") == "tool_call" for step in result["model_output"]) >= 5
                                                           and len(pred_calls) < len(item.get("label", {}).get("tool_calls", []))),
                            "schema_repair_turns": sum(bool(d.get("repair_attempted")) for d in turn_diagnostics),
                            "schema_rejections": sum(not x.get("result", {}).get("ok", True) for x in result["observations"]),
                            "parse_schema_failures": sum(bool(d.get("repair_attempted") and not d.get("repair_succeeded", False)) for d in turn_diagnostics),
                            "native_overcalls": sum(max(0, d.get("native_call_count", 0) - 1) for d in turn_diagnostics),
                            "schema_extension_count": sum(len(d.get("schema_extensions", [])) for d in turn_diagnostics),
                            "error_turns": sum(bool(d.get("repair_attempted") and not d.get("repair_succeeded", False)) for d in turn_diagnostics),
                            "context_failures": 0, "runtime_error": None}
                except Exception as exc:
                    row = {"id": sid, "asr_text": transcript, "transcript_used": transcript,
                           "tool_calls": [], "state_hash": None, "observations": [],
                           "latency_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}
                    diag = {"id": sid, "latency_seconds": row["latency_seconds"], "runtime_error": row["error"],
                            "context_failures": int("context" in str(exc).lower()), "turns": []}
                predictions.write(json.dumps(row, ensure_ascii=False) + "\n"); predictions.flush()
                diagnostics.write(json.dumps(diag, ensure_ascii=False) + "\n"); diagnostics.flush()
                print(f"[{variant} {a.cohort} {index}/{len(selected_ids)}] {sid} {row.get('latency_seconds', 0):.2f}s calls={len(row['tool_calls'])} error={row.get('error')}", flush=True)
        all_rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_diagnostics = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        latencies = sorted(float(r.get("latency_seconds", 0)) for r in all_rows)
        turn_diags = [d for row in all_diagnostics for d in row.get("turns", [])]
        summary = {"variant": variant, "cohort": a.cohort, "samples": len(all_rows), "runtime_failures": sum("error" in r for r in all_rows),
                   "mean_latency_seconds": sum(latencies) / len(latencies), "p95_latency_seconds": latencies[min(len(latencies)-1, int(.95 * len(latencies)))],
                   "max_latency_seconds": max(latencies), "over_30s": sum(x > 30 for x in latencies),
                   "turns": len(turn_diags), "ollama_requests": sum(d.get("request_count", 1) for d in turn_diags),
                   "prompt_eval_tokens": sum((d.get("prompt_eval_count") or 0) + (d.get("initial_prompt_eval_count") or 0) for d in turn_diags),
                   "eval_tokens": sum((d.get("eval_count") or 0) + (d.get("initial_eval_count") or 0) for d in turn_diags),
                   "context_failures": sum("context" in str(r.get("error", "")).lower() for r in all_rows),
                   "schema_repair_turns": sum(bool(d.get("repair_attempted")) for d in turn_diags),
                   "schema_rejections": sum(x.get("schema_rejections", 0) for x in all_diagnostics),
                   "parse_schema_failures": sum(x.get("parse_schema_failures", 0) for x in all_diagnostics),
                   "stop_early_samples": sum(x.get("stop_early", 0) for x in all_diagnostics),
                   "action_cap_hit_samples": sum(x.get("action_cap_hit", 0) for x in all_diagnostics),
                   "cap_exhausted_before_gold": sum(x.get("cap_exhausted_before_gold", 0) for x in all_diagnostics),
                   "mean_action_calls_per_sample": sum(x.get("action_count", 0) for x in all_diagnostics) / len(all_rows),
                   "native_overcalls": sum(max(0, d.get("native_call_count", 0)-1) for d in turn_diags),
                   "schema_extension_count": sum(len(d.get("schema_extensions", [])) for d in turn_diags),
                   "gpu_after": gpu_snapshot(), "note": "BTC scorer is run separately after each full 309-ID branch."}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
