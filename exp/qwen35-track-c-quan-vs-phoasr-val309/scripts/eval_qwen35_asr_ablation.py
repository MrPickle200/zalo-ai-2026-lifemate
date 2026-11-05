from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from inference_common import edit_distance, normalize
from lifemate.agent import Agent
from lifemate.asr import PhoASRModel
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl


ROOT = Path("exp/qwen35-track-c-quan-vs-phoasr-val309/output")
PHO_TRANSCRIPTS = Path("exp/phoasr-lifemate-domain-v1-val309/output/predictions.jsonl")
QUAN_TRANSCRIPTS = Path("exp/quan-transcribe-lifemate-domain-v1-val309/output/predictions.jsonl")
PHO_CHECKPOINT = Path("exp/phoasr-lifemate-domain-v1/output/model/final-model")
QUAN_CHECKPOINT = Path("exp/quan-transcribe-lifemate-domain-v1/output/model")
BRANCHES = {"phoasr-ft": PHO_TRANSCRIPTS, "quan-ft-beam32": QUAN_TRANSCRIPTS}
SEED = 20261104
PILOT_N = 60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot_ids(ids: list[str], items: dict[str, dict]) -> list[str]:
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for sid in ids:
        gold = items[sid].get("label", {}).get("tool_calls", [])
        calls = "0" if not gold else "1" if len(gold) == 1 else "2" if len(gold) == 2 else "3+"
        missing = any(call.get("name") == "sys.ask_missing_info" for call in gold)
        strata[(calls, missing)].append(sid)
    chosen: list[str] = []
    keys = [(n, m) for n in ("0", "1", "2", "3+") for m in (False, True)]
    while len(chosen) < PILOT_N and any(strata.values()):
        for key in keys:
            if strata[key] and len(chosen) < PILOT_N:
                chosen.append(strata[key].pop(0))
    return chosen


def load_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap_ci(values: list[float], seed: int = SEED, iterations: int = 10_000) -> dict:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return {"mean": sum(values) / n, "ci95_percentile": [means[int(.025 * iterations)], means[min(iterations - 1, int(.975 * iterations))]],
            "resamples": iterations, "seed": seed}


def main() -> int:
    project = _bootstrap.ROOT
    root = find_data_root("train+public_test/train+public_test")
    ids = json.loads((project / "data/splits/val_ids.json").read_text(encoding="utf-8"))
    items = {row["id"]: row for row in load_jsonl(root / "train/train.jsonl")}
    selected = pilot_ids(ids, items)
    if len(selected) != PILOT_N:
        raise RuntimeError(f"Expected {PILOT_N} fixed pilot IDs, got {len(selected)}")

    pho_audit = json.loads((PHO_CHECKPOINT.parent / "dataset-audit.json").read_text(encoding="utf-8"))
    quan_audit = json.loads((QUAN_CHECKPOINT / "dataset-audit.json").read_text(encoding="utf-8"))
    expected_val_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    for name, audit in (("PhoASR", pho_audit), ("QUAN", quan_audit)):
        if audit["train_ids"] != 1236 or audit["train_validation_overlap"] != 0 or audit["validation_ids_sha256"] != expected_val_hash:
            raise RuntimeError(f"{name} fine-tune audit does not match fixed train-only split")
    transcript_rows = {name: load_rows(path) for name, path in BRANCHES.items()}
    for name, rows in transcript_rows.items():
        if set(rows) != set(ids):
            raise RuntimeError(f"{name} transcript cache does not exactly cover val309")

    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "val309_ids.json", ids)
    cohort_dir = ROOT / "pilot60"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    write_json(cohort_dir / "pilot_ids.json", selected)
    (cohort_dir / "val_ids.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    strata = Counter((len(items[sid].get("label", {}).get("tool_calls", [])),
                      any(c.get("name") == "sys.ask_missing_info" for c in items[sid].get("label", {}).get("tool_calls", []))) for sid in selected)
    manifest = {"objective": "Paired pilot for Track C ASR ablation", "cohort": "fixed 60-ID pilot selected before model scoring",
                "cohort_selection": "deterministic round robin by gold call count (0/1/2/3+) and missing-info label",
                "ids": selected, "strata_counts": {f"calls={k[0]},missing={k[1]}": v for k, v in strata.items()},
                "all_validation_ids_sha256": expected_val_hash, "seed": SEED,
                "model": "qwen3.5:4b-q4_K_M", "ollama": {"thinking": True, "protocol": "JSON", "temperature": 0,
                    "num_predict": 512, "num_ctx": 8192, "prompt": "observation-grounded-v2 English"},
                "max_action_turns": 5, "toolenv_per_sample_per_branch": 1,
                "branches": {}, "scoring": "exp/btc-scoring/scripts/score_btc.py; exact tool name+arguments, duplicate multiplicity, gold replay/hash gate; before runtime mask",
                "time_estimate_seconds": 2 * PILOT_N * 31.867839240776664,
                "full_val309_estimated_seconds": 2 * 309 * 31.867839240776664,
                "full_run_policy": "not started: estimated model runtime exceeds 4 hours; pilot only"}
    for name, path in BRANCHES.items():
        manifest["branches"][name] = {"transcript_source": str(path), "transcript_sha256": sha256(path),
            "checkpoint": str(PHO_CHECKPOINT if name == "phoasr-ft" else QUAN_CHECKPOINT),
            "checkpoint_selection": "fixed final checkpoint; no validation selection",
            "dataset_audit": pho_audit if name == "phoasr-ft" else quan_audit}
        transcript_dir = cohort_dir / name
        transcript_dir.mkdir(parents=True, exist_ok=True)
        # Save a self-contained transcript cache for all 309 fixed validation IDs.
        rows = transcript_rows[name]
        cache_rows = []
        for sid in ids:
            row = rows[sid]
            text = row.get("prediction") or row.get("transcript_used") or row.get("asr_text") or ""
            cache_rows.append({"id": sid, "reference": row.get("reference", items[sid].get("audio_text", "")), "transcript": text})
        cache_path = transcript_dir / "transcripts-val309.jsonl"
        cache_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cache_rows), encoding="utf-8")
        manifest["branches"][name]["experiment_transcript_cache"] = str(cache_path)
        manifest["branches"][name]["experiment_transcript_cache_sha256"] = sha256(cache_path)
    write_json(ROOT / "manifest.json", manifest)

    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = "qwen3.5:4b-q4_K_M"
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "8192"
    os.environ["LIFEMATE_OLLAMA_NUM_PREDICT"] = "512"
    os.environ["LIFEMATE_OLLAMA_THINK"] = "true"
    os.environ["LIFEMATE_OLLAMA_PROTOCOL"] = "json"
    os.environ["LIFEMATE_PROMPT_LANGUAGE"] = "en"

    for branch, source in BRANCHES.items():
        out = cohort_dir / branch
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "run-config.json", {"branch": branch, "transcript_source": str(source),
            "checkpoint": manifest["branches"][branch]["checkpoint"], "model": "qwen3.5:4b-q4_K_M",
            "backend": "Ollama", "thinking": True, "protocol": "JSON", "prompt": "observation-grounded-v2 English",
            "temperature": 0, "num_predict": 512, "num_ctx": 8192, "max_action_turns": 5,
            "toolenv_per_sample": 1, "cohort": "fixed stratified pilot60"})
        predictions_path = out / "predictions.jsonl"
        diagnostics_path = out / "runtime-diagnostics.jsonl"
        existing = load_rows(predictions_path) if predictions_path.exists() else {}
        done = [sid for sid in selected if sid in existing]
        if done != selected[:len(done)]:
            raise RuntimeError(f"Cannot resume {branch}: prediction rows are not a pilot prefix")
        diag_rows = load_rows(diagnostics_path) if diagnostics_path.exists() else {}
        asr = PhoASRModel(model_path="models/pho-asr-whisper-small")
        agent = Agent(max_new_tokens=512, context_limit=8192, ollama_thinking=True, ollama_native_tools=False)
        pipeline = LifeMatePipeline(root, "train", asr=asr, agent=agent, max_steps=5)
        with predictions_path.open("a" if done else "w", encoding="utf-8") as predictions, diagnostics_path.open("a" if done else "w", encoding="utf-8") as diagnostics:
            for index, sid in enumerate(selected[len(done):], start=len(done) + 1):
                item = items[sid]
                transcript_row = transcript_rows[branch][sid]
                transcript = transcript_row.get("prediction") or transcript_row.get("transcript_used") or transcript_row.get("asr_text") or ""
                started = time.perf_counter()
                try:
                    result = pipeline.run(item, transcript_override=transcript)
                    row = {"id": sid, "asr_text": transcript, "transcript_used": transcript,
                           "model_output": result["model_output"],
                           "tool_calls": [{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in result["tool_calls"]],
                           "state_hash": result["state_hash"], "observations": result["observations"],
                           "latency_seconds": result["latency_seconds"], "final_state": result["final_state"]}
                    turns = [s.get("_diagnostics", {}) for s in result["model_output"]]
                    diag = {"id": sid, "latency_seconds": row["latency_seconds"], "turns": turns,
                            "schema_rejections": sum(not o.get("result", {}).get("ok", True) for o in result["observations"]),
                            "parse_schema_failures": sum(bool(d.get("repair_attempted") and not d.get("repair_succeeded", False)) for d in turns),
                            "early_stop": bool(result["model_output"] and result["model_output"][-1].get("action") == "stop"
                                               and len(row["tool_calls"]) < len(item.get("label", {}).get("tool_calls", []))),
                            "action_calls": len(row["tool_calls"]), "runtime_error": None}
                except Exception as exc:
                    row = {"id": sid, "asr_text": transcript, "transcript_used": transcript, "tool_calls": [],
                           "state_hash": None, "observations": [], "latency_seconds": time.perf_counter() - started,
                           "error": f"{type(exc).__name__}: {exc}"}
                    diag = {"id": sid, "latency_seconds": row["latency_seconds"], "turns": [],
                            "schema_rejections": 0, "parse_schema_failures": 0, "runtime_error": row["error"]}
                predictions.write(json.dumps(row, ensure_ascii=False) + "\n"); predictions.flush()
                diagnostics.write(json.dumps(diag, ensure_ascii=False) + "\n"); diagnostics.flush()
                print(f"[{branch} {index}/{PILOT_N}] {sid} {row['latency_seconds']:.2f}s calls={len(row['tool_calls'])} error={row.get('error')}", flush=True)
        all_pred = load_rows(predictions_path)
        all_diag = load_rows(diagnostics_path)
        latencies = [float(all_pred[sid].get("latency_seconds", 0)) for sid in selected]
        summary = {"branch": branch, "samples": len(all_pred), "runtime_failures": sum("error" in all_pred[sid] for sid in selected),
                   "mean_latency_seconds": sum(latencies) / len(latencies),
                   "p95_latency_seconds": sorted(latencies)[min(len(latencies) - 1, math.ceil(.95 * len(latencies)) - 1)],
                   "max_latency_seconds": max(latencies), "samples_over_30s": sum(x > 30 for x in latencies),
                   "schema_rejections": sum(all_diag[sid].get("schema_rejections", 0) for sid in selected),
                   "parse_schema_failures": sum(all_diag[sid].get("parse_schema_failures", 0) for sid in selected),
                   "transcript_model": manifest["branches"][branch]["checkpoint"], "cohort": "pilot60"}
        write_json(out / "runtime-summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    # Official scoring replays every gold trajectory once using the supplied ToolEnv API.
    for branch in BRANCHES:
        out = cohort_dir / branch
        command = [sys.executable, "exp/btc-scoring/scripts/score_btc.py", "--predictions", str(out / "predictions.jsonl"),
                   "--split-dir", str(cohort_dir), "--details", str(out / "btc-scores.jsonl"),
                   "--summary", str(out / "btc-summary.json")]
        subprocess.run(command, check=True)

    pho_scores = load_rows(cohort_dir / "phoasr-ft/btc-scores.jsonl")
    quan_scores = load_rows(cohort_dir / "quan-ft-beam32/btc-scores.jsonl")
    deltas = [quan_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] -
              pho_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in selected]
    bootstrap = bootstrap_ci([delta * 100 for delta in deltas])
    improved = [sid for sid in selected if deltas[selected.index(sid)] > 1e-12]
    worsened = [sid for sid in selected if deltas[selected.index(sid)] < -1e-12]
    tied = [sid for sid in selected if abs(deltas[selected.index(sid)]) <= 1e-12]
    transcripts = {name: transcript_rows[name] for name in BRANCHES}
    comparisons = []
    corpus_metrics = {}
    pred_rows = {name: load_rows(cohort_dir / name / "predictions.jsonl") for name in BRANCHES}
    for sid in selected:
        reference = items[sid].get("audio_text", "")
        wer_cer = {}
        for branch in BRANCHES:
            text = transcripts[branch][sid].get("prediction") or ""
            ref_n, text_n = normalize(reference), normalize(text)
            ref_chars, text_chars = list("".join(ref_n)), list("".join(text_n))
            wer_cer[branch] = {"wer": edit_distance(ref_n, text_n) / max(1, len(ref_n)),
                               "cer": edit_distance(ref_chars, text_chars) / max(1, len(ref_chars))}
        comparisons.append({"id": sid, "btc_delta_quan_minus_phoasr": deltas[selected.index(sid)],
            "phoasr_score": pho_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "quan_score": quan_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"],
            "wer_cer": wer_cer, "reference": reference,
            "phoasr_transcript": transcripts["phoasr-ft"][sid].get("prediction", ""),
            "quan_transcript": transcripts["quan-ft-beam32"][sid].get("prediction", ""),
            "gold_calls": items[sid].get("label", {}).get("tool_calls", []),
            "phoasr_calls": pred_rows["phoasr-ft"][sid].get("tool_calls", []),
            "quan_calls": pred_rows["quan-ft-beam32"][sid].get("tool_calls", [])})
    for branch in BRANCHES:
        ref_words = hyp_words = ref_chars = hyp_chars = 0
        for sid in selected:
            ref_n = normalize(items[sid].get("audio_text", ""))
            hyp_n = normalize(transcripts[branch][sid].get("prediction") or "")
            ref_tokens, hyp_tokens = ref_n, hyp_n
            ref_letters, hyp_letters = list("".join(ref_n)), list("".join(hyp_n))
            ref_words += len(ref_tokens)
            hyp_words += edit_distance(ref_tokens, hyp_tokens)
            ref_chars += len(ref_letters)
            hyp_chars += edit_distance(ref_letters, hyp_letters)
        corpus_metrics[branch] = {"corpus_wer": hyp_words / max(1, ref_words),
                                  "corpus_cer": hyp_chars / max(1, ref_chars)}
    summary = {"metric": "paired BTC sample score before runtime mask", "split": "fixed train-only val309 stratified pilot60",
               "matching": "exact tool name and arguments; duplicates by multiplicity; gold replay/hash gate",
               "phoasr_score_0_100": 100 * sum(pho_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in selected) / len(selected),
               "quan_score_0_100": 100 * sum(quan_scores[sid]["btc_formula_score_before_runtime_gate_diagnostic"] for sid in selected) / len(selected),
               "paired_difference_quan_minus_phoasr_0_100": bootstrap,
               "corpus_wer_cer": corpus_metrics,
               "bootstrap_resamples": 10_000, "improved_samples": len(improved), "worsened_samples": len(worsened), "tied_samples": len(tied),
               "improved_ids": improved, "worsened_ids": worsened, "hash_match_rate": {
                   "phoasr": sum(pho_scores[sid]["hash_match"] for sid in selected) / len(selected),
                   "quan": sum(quan_scores[sid]["hash_match"] for sid in selected) / len(selected)},
               "wer_cer_method": "normalized Vietnamese whitespace WER; CER after whitespace removal",
               "runtime_warning": "Pilot only. Track C full-val reference averages 31.87s/sample; no full paired run started.",
               "analysis_samples": comparisons}
    write_json(cohort_dir / "paired-analysis.json", summary)
    write_json(ROOT / "summary.json", {"status": "pilot_complete", "cohort": "pilot60", "paired_report": "pilot60/paired-analysis.json",
        "full_val309_run": "not started; estimated paired model runtime 19,694 seconds (>4 hours)"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
