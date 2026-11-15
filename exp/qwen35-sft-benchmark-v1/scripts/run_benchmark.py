from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp" / "btc-scoring" / "scripts"))

EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OUT = EXP / "output" / "inference"
PROVENANCE = EXP / "output" / "model-provenance"
DOWNSTREAM_RUNNER_PATH = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "run_downstream.py"
ANALYZER_PATH = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "analyze_downstream.py"
MODEL_MANIFEST = PROVENANCE / "ollama-models.json"
SMOKE_MANIFEST = EXP / "output" / "smoke" / "smoke_manifest.json"
COHORT_SOURCE = ROOT / "exp" / "inference-latency-optimization-val60" / "output" / "cohort_manifest.json"
EXPECTED_IDS_SHA256 = "87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9"
BRANCH_ORDER = [["A", "B", "COT"], ["B", "COT", "A"],
                ["COT", "A", "B"], ["A", "COT", "B"]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def branch_models() -> dict[str, dict[str, str]]:
    manifest = read_json(MODEL_MANIFEST)
    return {
        "A": {"branch": "base", "model": manifest["branches"]["base"]["tag"],
              "digest": manifest["branches"]["base"]["ollama_digest"],
              "gguf_sha256": manifest["branches"]["base"]["gguf"]["sha256"]},
        "B": {"branch": "sft", "model": manifest["branches"]["sft"]["tag"],
              "digest": manifest["branches"]["sft"]["ollama_digest"],
              "gguf_sha256": manifest["branches"]["sft"]["gguf"]["sha256"]},
        "COT": {"branch": "sft", "model": manifest["branches"]["sft"]["tag"],
                "digest": manifest["branches"]["sft"]["ollama_digest"],
                "gguf_sha256": manifest["branches"]["sft"]["gguf"]["sha256"]},
    }


def lock_config(runner) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    if any((OUT / name).exists() for name in ("config_manifest.json", "cohort_manifest.json",
                                                "benchmark_lock.json")):
        raise FileExistsError(f"Benchmark lock already exists under {OUT}")
    if not SMOKE_MANIFEST.is_file() or read_json(SMOKE_MANIFEST).get("status") != "passed":
        raise RuntimeError("Both model tags must pass the real ToolEnv smoke before cohort inference")

    model_map = branch_models()
    runner.MODEL = model_map["A"]["model"]
    runner.EXPECTED_DIGEST = model_map["A"]["digest"]
    runner.OUT = OUT
    runner.prepare()
    config_path = OUT / "config_manifest.json"
    config = read_json(config_path)
    cohort = read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if len(ids) != 60 or len(set(ids)) != 60 or ids_hash != EXPECTED_IDS_SHA256:
        raise RuntimeError("The inherited enriched validation cohort does not match the locked 60-ID screen")

    anchors = ids[:10] + ids[-10:]
    config.update({"experiment": "qwen35-sft-benchmark-v1", "status": "benchmark_config_locked",
        "branches": {
            "A": {"model": model_map["A"]["model"], "model_digest": model_map["A"]["digest"],
                  "base": "Exact local Kaggle model limamateus/qwen3-5-4b/PyTorch/default/version1",
                  "asr": "selected QUAN checkpoint", "prompt": "observation-grounded-v2 English"},
            "B": {"model": model_map["B"]["model"], "model_digest": model_map["B"]["digest"],
                  "base": "Kaggle version1 + final PEFT adapter merged in BF16", "asr": "same selected QUAN checkpoint",
                  "prompt": "observation-grounded-v2 English"},
            "C": {"model": model_map["COT"]["model"], "model_digest": model_map["COT"]["digest"],
                  "base": "Kaggle version1 + final PEFT adapter merged in BF16", "asr": "same selected QUAN checkpoint",
                  "prompt": "observation-grounded-v2 English + exact TASK CoT checklist",
                  "internal_artifact_prefix": "COT"},
        },
        "branch_model_mapping": model_map,
        "branch_order": BRANCH_ORDER,
        "repeat_control": {"branch": "A-repeat", "ids": anchors,
                           "selection": "first 10 and last 10 IDs from the fixed cohort"},
        "ollama": {**config["ollama"], "model_mapping": model_map,
                   "version": read_json(MODEL_MANIFEST)["ollama_version"],
                   "thinking": True, "temperature": 0, "protocol": "JSON",
                   "num_predict": 768, "num_ctx": 8192, "num_batch": 512},
        "model_provenance": {
            "kaggle_model_ref": "limamateus/qwen3-5-4b/pytorch/default/version1",
            "kaggle_model_manifest_sha256": sha256(PROVENANCE / "kaggle-version1-sha256.json"),
            "base_tree_sha256": read_json(PROVENANCE / "kaggle-version1-sha256.json")["base_directory_tree_sha256"],
            "adapter_sha256": read_json(PROVENANCE / "local-audit.json")["adapter"]["final"]["sha256"],
            "merged_hf_tree_sha256": read_json(PROVENANCE / "merge-manifest.json")["merged_tree_sha256"],
            "conversion_manifest_sha256": sha256(PROVENANCE / "conversion-manifest.json"),
            "llama_cpp_commit": read_json(PROVENANCE / "conversion-manifest.json")["llama_cpp_commit"],
            "quantizer_sha256": read_json(PROVENANCE / "llama-toolchain-manifest.json")["quantizer"]["sha256"],
            "base_gguf_sha256": model_map["A"]["gguf_sha256"],
            "sft_gguf_sha256": model_map["B"]["gguf_sha256"],
            "adapter_training_report_has_input_hash": False,
            "base_identity_note": "Adapter config stores Kaggle model path/version but no revision or weight hash. User attested identity; exact local Kaggle v1 bytes are independently hashed. This does not prove the remote training run consumed those exact bytes.",
        },
        "public_private_test_used": False,
        "execution_environment": "system Python with Ollama localhost; not .venv",
    })

    extra_inputs = [
        EXP / "scripts" / "run_benchmark.py",
        EXP / "scripts" / "register_ollama_models.py",
        EXP / "scripts" / "smoke_ollama_models.py",
        ANALYZER_PATH,
        ROOT / "TASK_benchmark.md", ROOT / "COMPETITION.md",
        PROVENANCE / "local-audit.json", PROVENANCE / "kaggle-version1-sha256.json",
        PROVENANCE / "merge-manifest.json", PROVENANCE / "conversion-manifest.json",
        PROVENANCE / "llama-toolchain-manifest.json", MODEL_MANIFEST, SMOKE_MANIFEST,
    ]
    for path in extra_inputs:
        config["input_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    write_json(config_path, config)

    order_manifest = {"cohort_ids_sha256_newline_joined": ids_hash,
        "rounds": [{"round": index, "ids": ids[(index - 1) * 15:index * 15],
                    "branch_order_internal": order,
                    "branch_order_user_facing": ["C" if name == "COT" else name for name in order]}
                   for index, order in enumerate(BRANCH_ORDER, 1)],
        "A_repeat_anchor_ids": anchors,
        "paired_by_id": True,
        "warmup_before_each_branch_block": True,
        "settings": {"temperature": 0, "thinking": True, "format": "JSON",
                     "num_predict": 768, "num_ctx": 8192, "num_batch": 512,
                     "max_action_turns": 5},
        "main_branch_map": {"A": "base", "B": "SFT", "C": "SFT + CoT (stored as COT)"},
    }
    write_json(OUT / "step3-execution-order.json", order_manifest)
    lock = {"status": "locked", "cohort_ids_sha256_newline_joined": ids_hash,
            "samples": len(ids), "model_map": model_map, "execution_order": order_manifest,
            "config_manifest_sha256": sha256(config_path), "cohort_manifest_sha256": sha256(OUT / "cohort_manifest.json"),
            "smoke_manifest_sha256": sha256(SMOKE_MANIFEST), "created_epoch": time.time()}
    write_json(OUT / "benchmark_lock.json", lock)
    print(json.dumps({"status": lock["status"], "cohort_samples": len(ids), "ids_sha256": ids_hash,
                      "models": {key: value["model"] for key, value in model_map.items()},
                      "repeat_anchors": len(anchors), "execution_order": BRANCH_ORDER}, indent=2))
    return config


def run_inference() -> None:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run benchmark with system Python, outside .venv, to reach Ollama")
    config_path = OUT / "config_manifest.json"
    config = read_json(config_path)
    if config.get("status") != "benchmark_config_locked":
        raise RuntimeError("The benchmark config is not locked")
    lock = read_json(OUT / "benchmark_lock.json")
    if lock.get("status") != "locked" or lock["config_manifest_sha256"] != sha256(config_path):
        raise RuntimeError("Benchmark lock/config hash mismatch")
    runner = load_module("qwen35_sft_benchmark_downstream_runner", DOWNSTREAM_RUNNER_PATH)
    runner.OUT = OUT
    model_map = branch_models()
    runner.MODEL = model_map["A"]["model"]
    runner.EXPECTED_DIGEST = model_map["A"]["digest"]
    runner.verify_locked_inputs(config)
    if any((OUT / f"{name}-{branch}.jsonl").exists()
           for branch in ("A", "B", "COT", "A-repeat")
           for name in ("predictions", "step-traces", "requests")):
        raise FileExistsError("Prediction artifacts exist; this runner will not overwrite or silently resume")

    ids = config["cohort"]["ids"]
    transcript_rows = runner.read_jsonl(runner.ASR_PREDICTIONS)
    val_ids = json.loads((ROOT / "data" / "splits" / "val_ids.json").read_text(encoding="utf-8-sig"))
    transcript_map = {row["id"]: row["prediction"] for row in transcript_rows}
    data_root = runner.find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    if [row["id"] for row in transcript_rows] != val_ids or set(ids) - transcript_map.keys() or set(ids) - items.keys():
        raise RuntimeError("Fixed held-out transcript or ToolEnv item coverage changed after preflight")

    def do_block(internal_branch: str, round_index: int, block: list[str], cot: bool) -> None:
        model_key = "A" if internal_branch == "A-repeat" else (
            "COT" if internal_branch == "COT" else internal_branch)
        model = model_map[model_key]
        runner.MODEL = model["model"]
        runner.EXPECTED_DIGEST = model["digest"]
        version, digest = runner.model_fingerprint()
        if version != config["ollama"]["version"] or digest != model["digest"]:
            raise RuntimeError(f"Ollama tag changed before {internal_branch} round {round_index}")
        runner.warmup(internal_branch, round_index)
        runner.run_block(internal_branch, round_index, block, transcript_map, items, cot=cot)

    started = time.time()
    blocks = [ids[index * 15:(index + 1) * 15] for index in range(4)]
    for round_index, block in enumerate(blocks, 1):
        for internal_branch in BRANCH_ORDER[round_index - 1]:
            do_block(internal_branch, round_index, block, cot=(internal_branch == "COT"))
    for internal_branch in ("A", "B", "COT"):
        pred_rows = runner.read_jsonl(OUT / f"predictions-{internal_branch}.jsonl")
        trace_rows = runner.read_jsonl(OUT / f"step-traces-{internal_branch}.jsonl")
        if [row["id"] for row in pred_rows] != ids or [row["id"] for row in trace_rows] != ids:
            raise RuntimeError(f"Branch {internal_branch} has incomplete or misordered cohort artifacts")
        meta_path = OUT / f"run-{internal_branch}.json"
        meta = runner.read_json(meta_path)
        meta.update({"status": "complete", "user_facing_branch": "C" if internal_branch == "COT" else internal_branch,
                     "ended_epoch": time.time(), "samples_completed": len(ids)})
        runner.write_json(meta_path, meta)

    anchors = config["repeat_control"]["ids"]
    do_block("A-repeat", 5, anchors, cot=False)
    repeat_rows = runner.read_jsonl(OUT / "predictions-A-repeat.jsonl")
    if [row["id"] for row in repeat_rows] != anchors:
        raise RuntimeError("A-repeat did not cover its exact ordered anchor IDs")
    meta_path = OUT / "run-A-repeat.json"
    meta = runner.read_json(meta_path)
    meta.update({"status": "complete", "user_facing_branch": "A-repeat",
                 "ended_epoch": time.time(), "samples_completed": len(anchors),
                 "overall_run_started_epoch": started})
    runner.write_json(meta_path, meta)
    run_complete = {"status": "complete", "started_epoch": started, "ended_epoch": time.time(),
                    "elapsed_seconds": time.time() - started, "main_samples_per_branch": len(ids),
                    "repeat_anchor_samples": len(anchors), "branches": ["A", "B", "C"],
                    "internal_branch_prefix_for_C": "COT", "public_private_test_used": False}
    write_json(OUT / "inference-run.json", run_complete)
    print(json.dumps(run_complete, indent=2))


def run_repeat_only() -> None:
    """Finish the repeat control after all primary branches completed."""
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run benchmark with system Python, outside .venv, to reach Ollama")
    config_path = OUT / "config_manifest.json"
    config = read_json(config_path)
    lock = read_json(OUT / "benchmark_lock.json")
    if (config.get("status") != "benchmark_config_locked" or lock.get("status") != "locked"
            or lock["config_manifest_sha256"] != sha256(config_path)):
        raise RuntimeError("Benchmark lock/config hash mismatch")
    if (OUT / "inference-run.json").exists():
        raise FileExistsError("Inference is already marked complete; refusing to rerun A-repeat")
    if any((OUT / f"{name}-A-repeat.jsonl").exists()
           for name in ("predictions", "step-traces", "requests")) or (OUT / "run-A-repeat.json").exists():
        raise FileExistsError("A-repeat artifacts exist; refusing to overwrite or silently resume")

    runner = load_module("qwen35_sft_benchmark_downstream_runner", DOWNSTREAM_RUNNER_PATH)
    runner.OUT = OUT
    model_map = branch_models()
    ids = config["cohort"]["ids"]
    for branch in ("A", "B", "COT"):
        meta = read_json(OUT / f"run-{branch}.json")
        predictions = runner.read_jsonl(OUT / f"predictions-{branch}.jsonl")
        traces = runner.read_jsonl(OUT / f"step-traces-{branch}.jsonl")
        if (meta.get("status") != "complete" or meta.get("model_digest") != model_map[branch]["digest"]
                or [row["id"] for row in predictions] != ids
                or [row["id"] for row in traces] != ids):
            raise RuntimeError(f"Primary branch {branch} is not complete against the locked cohort/model")

    runner.MODEL = model_map["A"]["model"]
    runner.EXPECTED_DIGEST = model_map["A"]["digest"]
    runner.verify_locked_inputs(config)
    version, digest = runner.model_fingerprint()
    if version != config["ollama"]["version"] or digest != model_map["A"]["digest"]:
        raise RuntimeError("Ollama fingerprint changed after primary branch inference")

    transcript_rows = runner.read_jsonl(runner.ASR_PREDICTIONS)
    val_ids = json.loads((ROOT / "data" / "splits" / "val_ids.json").read_text(encoding="utf-8-sig"))
    transcript_map = {row["id"]: row["prediction"] for row in transcript_rows}
    data_root = runner.find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    anchors = config["repeat_control"]["ids"]
    if ([row["id"] for row in transcript_rows] != val_ids
            or set(anchors) - transcript_map.keys() or set(anchors) - items.keys()):
        raise RuntimeError("Fixed held-out transcripts or ToolEnv item coverage changed before A-repeat")

    started = time.time()
    runner.warmup("A-repeat", 5)
    runner.run_block("A-repeat", 5, anchors, transcript_map, items, cot=False)
    for artifact in ("predictions", "step-traces"):
        rows = runner.read_jsonl(OUT / f"{artifact}-A-repeat.jsonl")
        if [row["id"] for row in rows] != anchors:
            raise RuntimeError(f"A-repeat {artifact} did not cover the exact ordered anchors")
    meta_path = OUT / "run-A-repeat.json"
    meta = runner.read_json(meta_path)
    meta.update({"status": "complete", "user_facing_branch": "A-repeat",
                 "ended_epoch": time.time(), "samples_completed": len(anchors)})
    runner.write_json(meta_path, meta)
    overall_started = min(
        block["started_epoch"]
        for branch in ("A", "B", "COT")
        for block in read_json(OUT / f"run-{branch}.json")["blocks"]
    )
    run_complete = {"status": "complete", "started_epoch": overall_started,
                    "ended_epoch": time.time(), "elapsed_seconds": time.time() - overall_started,
                    "main_samples_per_branch": len(ids), "repeat_anchor_samples": len(anchors),
                    "branches": ["A", "B", "C"], "internal_branch_prefix_for_C": "COT",
                    "public_private_test_used": False,
                    "recovery_note": "Primary A/B/C artifacts were retained; only the previously missing A-repeat block was run after fixing its model-map key."}
    write_json(OUT / "inference-run.json", run_complete)
    print(json.dumps(run_complete, indent=2))


def analyze() -> None:
    analysis_path = OUT / "analysis.json"
    if analysis_path.exists():
        raise FileExistsError(f"Refusing to overwrite {analysis_path}")
    analyzer = load_module("qwen35_sft_benchmark_downstream_analyzer", ANALYZER_PATH)
    analyzer.OUT = OUT
    cohort = read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    anchors = read_json(OUT / "benchmark_lock.json")["execution_order"]["A_repeat_anchor_ids"]
    data_root = analyzer.find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in analyzer.load_jsonl(data_root / "train" / "train.jsonl")}
    branch_results = {}
    for label, internal in (("A", "A"), ("B", "B"), ("C", "COT")):
        branch_results[label] = analyzer.compute_branch(internal, ids, items, data_root)
    repeat_results = analyzer.compute_branch("A-repeat", anchors, items, data_root)

    a_pred, a_score, a_metrics, a_requests = branch_results["A"]
    b_pred, b_score, b_metrics, b_requests = branch_results["B"]
    c_pred, c_score, c_metrics, c_requests = branch_results["C"]
    repeat_pred, repeat_score, repeat_metrics, repeat_requests = repeat_results
    rng = __import__("random").Random(analyzer.BOOTSTRAP_SEED)
    comparisons = {
        "A_vs_B": analyzer.pair_metrics("A-vs-B", ids, a_pred, b_pred, a_score, b_score,
                                         a_requests, b_requests, rng),
        "B_vs_C": analyzer.pair_metrics("B-vs-C", ids, b_pred, c_pred, b_score, c_score,
                                         b_requests, c_requests, rng),
        "A_vs_C": analyzer.pair_metrics("A-vs-C", ids, a_pred, c_pred, a_score, c_score,
                                         a_requests, c_requests, rng),
        "A_vs_A_repeat_anchors": analyzer.pair_metrics(
            "A-vs-A-repeat", anchors, {sid: a_pred[sid] for sid in anchors}, repeat_pred,
            {sid: a_score[sid] for sid in anchors}, repeat_score, a_requests, repeat_requests, rng),
    }
    if any(metric["gold_replay_failures"] for metric in (a_metrics, b_metrics, c_metrics, repeat_metrics)):
        raise RuntimeError("Official gold ToolEnv replay failed for at least one sample")

    analysis = {
        "cohort": "fixed enriched subset of held-out val309; 60-ID screening only",
        "official_scoring": {
            "formula": "For each sample: score=0 when ToolEnv state hash differs or M=0; otherwise exact-call P=M/(M+U), R=M/G, harmonic mean. Final score=100 times mean over N.",
            "matching": "exact tool name and arguments; duplicate calls matched by multiplicity",
            "gold_replay": "official ToolEnv replay via exp/btc-scoring/scripts/score_btc.py",
            "local_runtime_gate_is_official": False,
        },
        "branches": {"A": a_metrics, "B": b_metrics, "C": c_metrics,
                     "A_repeat_anchors": repeat_metrics},
        "paired_comparisons": comparisons,
        "raw_chain_of_thought_saved": False,
        "public_private_test_used": False,
        "interpretation_note": "Enriched cohort is screening only. Treat paired differences as preliminary; the repeat-A anchor comparison estimates local output variance. The adapter training report does not include an executed-input hash, so the user attestation and local Kaggle version1 identity are recorded as provenance, not proof of the remote run bytes.",
    }
    write_json(analysis_path, analysis)
    write_summary(analysis)
    print(json.dumps({"status": "analyzed", "official_btc_0_100": {
        "A": a_metrics["btc_formula_score_before_runtime_gate_0_100"],
        "B": b_metrics["btc_formula_score_before_runtime_gate_0_100"],
        "C": c_metrics["btc_formula_score_before_runtime_gate_0_100"]},
        "paired_delta_points": {key: value["paired_formula_delta_points_right_minus_left"]["mean"]
                                 for key, value in comparisons.items()}}, indent=2))


def write_summary(analysis: dict[str, Any]) -> None:
    branches = analysis["branches"]
    comparisons = analysis["paired_comparisons"]
    lines = [
        "# Qwen3.5 SFT benchmark summary", "",
        "## Scope and method", "",
        "Fixed enriched validation cohort60, screening only; not full val309 or Public LB. Each sample used one prediction ToolEnv per branch. Gold hashes were obtained by official ToolEnv replay. Calls were matched by exact tool name and arguments with duplicate multiplicity.", "",
        "Official score is 0 when the predicted ToolEnv hash differs from gold or when M=0; otherwise it is the harmonic mean of exact-call precision M/(M+U) and recall M/G. The final score is the mean sample score on a 0–100 scale. The 30-second local runtime gate is reported only as a diagnostic.", "",
        "| Branch | Model/prompt | Official BTC | Hash matches | M / U / G | >30s samples | Mean latency | Mean generated tokens/request |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    model_map = branch_models()
    labels = {
        "A": f"{model_map['A']['model']} / English prompt",
        "B": f"{model_map['B']['model']} / English prompt",
        "C": f"{model_map['COT']['model']} / English + CoT",
    }
    for key in ("A", "B", "C"):
        metric = branches[key]
        lines.append(f"| {key} | {labels[key]} | {metric['btc_formula_score_before_runtime_gate_0_100']:.2f} | {metric['hash_matches']}/{metric['samples']} | {metric['M_matched_calls']} / {metric['U_unmatched_submitted_calls']} / {metric['G_gold_calls']} | {metric['samples_over_30s_local']} | {metric['sample_latency_seconds']['mean']:.2f}s | {metric['generated_tokens_per_request_including_thinking']['mean']:.1f} |")
    lines += ["", "## Paired official-score differences", "",
              "Positive deltas mean the right-hand branch scored higher. CIs use paired bootstrap over the fixed cohort; A-repeat uses the 20 anchor IDs.", "",
              "| Contrast (right − left) | Mean delta points | 95% CI | Improved / worsened / tied |",
              "|---|---:|---:|---:|"]
    for key, title in (("A_vs_B", "B − A (SFT effect)"), ("B_vs_C", "C − B (CoT effect on SFT)"),
                       ("A_vs_C", "C − A (combined change)"),
                       ("A_vs_A_repeat_anchors", "A-repeat − A on anchors (run variance)")):
        paired = comparisons[key]
        delta = paired["paired_formula_delta_points_right_minus_left"]
        outcomes = paired["paired_outcomes_formula_score"]
        lines.append(f"| {title} | {delta['mean']:+.2f} | [{delta['ci95_low']:+.2f}, {delta['ci95_high']:+.2f}] | {outcomes.get('improved', 0)} / {outcomes.get('worsened', 0)} / {outcomes.get('tied', 0)} |")
    lines += ["", "## Runtime and diagnostics", "",
              "The local 30-second gated score, aggregate exact-call F1, and final-state match are diagnostics, not official BTC. The local RTX 4050 host is not an A30 timing benchmark.", ""]
    for key in ("A", "B", "C"):
        metric = branches[key]
        lines.append(f"- **{key}:** local 30s-gated score {metric['local_30s_gated_score_diagnostic_0_100']:.2f}; mean/p50/p95/max latency {metric['sample_latency_seconds']['mean']:.2f}/{metric['sample_latency_seconds']['p50']:.2f}/{metric['sample_latency_seconds']['p95']:.2f}/{metric['sample_latency_seconds']['max']:.2f}s; {metric['requests_reaching_num_predict']} requests reached the 768-token cap; {metric['runtime_errors']} runtime/parser errors; {metric['schema_rejections']} schema rejections.")
    lines += ["", "## Provenance and limits", "",
              "Base A uses the downloaded and hashed Kaggle model version1. B/C use the final adapter merged into that base, converted with the pinned llama.cpp revision to BF16 GGUF and Q4_K_M. Base and SFT tags are separate and retain the GGUF tokenizer/chat template. The adapter archive records the Kaggle model path/version and the user attested the identity, but the training report has no input-weight hash; local provenance therefore cannot prove that the remote training job used identical weight bytes.", "",
              "The SFT training data used a Vietnamese system prompt, while all benchmark branches use the English production prompt; this language mismatch limits interpretation. The enriched 60-ID cohort is a screen. If an effect is positive and stable against A-repeat, run a separately locked full-val309 evaluation before considering any submission.", "",
              "Smoke passed for both tags on fixed official train sample `train_0050`: each emitted a JSON action accepted by parser/schema and executed one call in ToolEnv. The earlier `train_0001` smoke returned valid `stop` JSON with no tool call; that failed attempt is retained in `output/smoke/base.json` and was excluded from the gate.", "",
              "Raw chain-of-thought was never saved.", ""]
    (EXP / "output" / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--repeat", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        runner = load_module("qwen35_sft_benchmark_downstream_prepare", DOWNSTREAM_RUNNER_PATH)
        lock_config(runner)
    elif args.run:
        run_inference()
    elif args.repeat:
        run_repeat_only()
    else:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
