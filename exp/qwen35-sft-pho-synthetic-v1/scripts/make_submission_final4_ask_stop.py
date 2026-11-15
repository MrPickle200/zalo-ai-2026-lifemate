from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
PILOT = EXP / "output/two-policy-val60"
OUT = EXP / "output/submission-final4-ask-stop-pilot60"
FOCUSED_WRAPPER = ROOT / "exp/phowhisper-large-qwen-focused-public-v1/scripts/make_submission.py"
STEM = "phowhisper-large-v2-qwen35-phosynth-final4-ask-stop-pilot60-public-test"
SUBMISSION_CONTEXT = 16384


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load submission workflow: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure():
    pilot_summary_path = PILOT / "summary.json"
    pilot_lock_path = PILOT / "comparison_lock.json"
    summary = read_json(pilot_summary_path)
    lock = read_json(pilot_lock_path)
    branches = summary["branches"]
    winner = max(branches, key=lambda key: branches[key]["btc_after_30s_gate"])
    if winner != "fixed_ask_stop":
        raise RuntimeError(f"Pilot winner changed; refusing to generate this candidate: {winner}")
    if lock["model"]["tag"] != "lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest":
        raise RuntimeError("Pilot did not use the expected final4 adapter.")
    if summary["policy"]["stop_after_successful_ask"] != [False, True]:
        raise RuntimeError("Pilot policy lock does not contain the ask-stop intervention.")

    module = load_module("focused_ask_stop_public_submission", FOCUSED_WRAPPER)
    module.__file__ = str(Path(__file__).resolve())
    module.MODEL = lock["model"]["tag"]
    module.DIGEST = lock["model"]["digest"]
    module.STEM = STEM
    module.EXP = EXP
    module.OUT = OUT
    module.base.MODEL = module.MODEL
    module.base.DIGEST = module.DIGEST
    module.base.STEM = STEM
    module.base.EXP = EXP
    module.base.OUT = OUT
    module.base.MODEL_PARAMS = 4_205_751_296
    module.base.CONTEXT = SUBMISSION_CONTEXT

    original_input_lock = module.input_lock

    def input_lock_with_policy():
        submission_lock = original_input_lock()
        submission_lock["settings"]["num_ctx"] = SUBMISSION_CONTEXT
        submission_lock["ask_stop_selection"] = {
            "selected_branch": winner,
            "selection_rule": "Highest official BTC score after the local 30-second gate among the four paired pilot60 branches.",
            "pilot_branch_score_after_gate": branches[winner]["btc_after_30s_gate"],
            "pilot_branch_score_before_gate": branches[winner]["btc_before_gate"],
            "pilot_cohort_n": summary["cohort"]["n"],
            "pilot_cohort_source": summary["cohort"]["source"],
            "pilot_summary": str(pilot_summary_path.relative_to(ROOT)).replace("\\", "/"),
            "pilot_comparison_lock_sha256": file_sha256(pilot_lock_path),
            "pilot_caveat": summary["caveat"],
        }
        submission_lock["policy"] = {
            "name": "stop after a successful sys.ask_missing_info call",
            "thinking": "fixed on for every request",
            "implementation": "In-memory LifeMatePipeline loop patch; source files remain unchanged.",
        }
        submission_lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/output/two-policy-val60/summary.json"] = file_sha256(pilot_summary_path)
        submission_lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/output/two-policy-val60/comparison_lock.json"] = file_sha256(pilot_lock_path)
        submission_lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/scripts/make_submission_final4_ask_stop.py"] = file_sha256(Path(__file__).resolve())
        submission_lock["context_adjustment"] = {
            "num_ctx": SUBMISSION_CONTEXT,
            "reason": "Use the competition maximum to cover the known 8,265-token val309 request that exceeded the 8,192-token benchmark context.",
            "benchmark_context": lock["settings"]["num_ctx"],
            "competition_context_limit": 16_384,
        }
        return submission_lock

    module.input_lock = input_lock_with_policy
    module.SELECTED_BRANCH = winner
    module.PILOT_SCORE = branches[winner]["btc_after_30s_gate"]
    return module, summary, lock, winner


def install_ask_stop(module):
    import lifemate.pipeline as pipeline_module

    source_path = Path(pipeline_module.__file__).resolve()
    original_source = source_path.read_text(encoding="utf-8")
    old = "            self.agent.record_tool_result(action, observation)\n        calls = env.get_tool_calls()"
    new = (
        "            self.agent.record_tool_result(action, observation)\n"
        "            if action.get(\"tool\") == \"sys.ask_missing_info\":\n"
        "                break\n"
        "        calls = env.get_tool_calls()"
    )
    if old not in original_source:
        raise RuntimeError("Could not locate the action loop insertion point.")
    exec(compile(original_source.replace(old, new, 1), str(source_path), "exec"), pipeline_module.__dict__)
    module.base.LifeMatePipeline = pipeline_module.LifeMatePipeline

    original_create = pipeline_module.create_tool_env
    env_counts: dict[str, int] = {}

    def one_env(split, sample_id, data_root, *args, **kwargs):
        env_counts[sample_id] = env_counts.get(sample_id, 0) + 1
        if env_counts[sample_id] != 1:
            raise RuntimeError(f"More than one ToolEnv created for sample {sample_id}")
        return original_create(split, sample_id, data_root, *args, **kwargs)

    pipeline_module.create_tool_env = one_env
    return pipeline_module, source_path, original_source, original_create, env_counts


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--prepare", "--run"):
        raise SystemExit("Use --prepare or --run")
    module, pilot_summary, pilot_lock, winner = configure()
    OUT.mkdir(parents=True, exist_ok=True)
    selection = {
        "status": "selected_from_pilot60",
        "winner": winner,
        "model": module.MODEL,
        "model_digest": module.DIGEST,
        "pilot_btc_after_local_30s_gate": module.PILOT_SCORE,
        "pilot_summary": str((PILOT / "summary.md").relative_to(ROOT)).replace("\\", "/"),
        "pilot_comparison_lock_sha256": file_sha256(PILOT / "comparison_lock.json"),
        "public_submission_config": {
            "prompt": "focused post-lookup disambiguation",
            "asr": "PhoWhisper Large v2 main and history",
            "policy": "stop after successful sys.ask_missing_info",
            "thinking": True,
            "temperature": 0,
            "protocol": "JSON",
            "num_predict": 768,
            "num_ctx": SUBMISSION_CONTEXT,
            "num_batch": 512,
            "seed": 17,
            "max_action_turns": 5,
        },
        "public_lb_submitted": False,
    }
    selection_path = OUT / "winner_selection.json"
    if sys.argv[1] == "--prepare":
        write_json(selection_path, selection)
        module.prepare()
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
        return

    pipeline_module, source_path, original_source, original_create, env_counts = install_ask_stop(module)
    try:
        module.run()
    finally:
        exec(compile(original_source, str(source_path), "exec"), pipeline_module.__dict__)
        pipeline_module.create_tool_env = original_create
        module.base.LifeMatePipeline = pipeline_module.LifeMatePipeline

    public_ids = [row["id"] for row in module.base.load_jsonl(
        module.base.find_data_root("train+public_test/train+public_test") / "public_test/public_test.jsonl")]
    if len(env_counts) != 634 or set(env_counts) != set(public_ids) or any(count != 1 for count in env_counts.values()):
        raise RuntimeError("ToolEnv count audit failed; expected exactly one environment per public-test sample.")

    manifest_path = ROOT / "outputs/submissions" / (STEM + "_manifest.json")
    manifest = read_json(manifest_path)
    manifest["validation_winner"] = winner
    manifest["pilot60_btc_after_local_30s_gate"] = module.PILOT_SCORE
    manifest["pilot60_btc_before_gate"] = pilot_summary["branches"][winner]["btc_before_gate"]
    manifest["pilot60_summary"] = str((PILOT / "summary.md").relative_to(ROOT)).replace("\\", "/")
    manifest["pilot60_comparison_lock_sha256"] = file_sha256(PILOT / "comparison_lock.json")
    manifest["policy"] = "stop after successful sys.ask_missing_info; thinking fixed on"
    manifest["policy_implementation"] = "in-memory pipeline patch; source pipeline unchanged"
    manifest["toolenv_count_audit"] = {"samples": len(env_counts), "exactly_one_per_sample": True}
    manifest["public_lb_submitted"] = False
    write_json(manifest_path, manifest)
    write_json(OUT / "submission_summary.json", manifest)
    selection["status"] = "public_submission_generated_and_schema_validated"
    selection["submission_sha256"] = manifest["submission_sha256"]
    selection["rows"] = manifest["rows"]
    selection["schema_validation"] = manifest["schema_validation"]
    selection["runtime_errors"] = manifest["runtime_errors"]
    write_json(selection_path, selection)
    print(json.dumps({
        "submission": manifest["submission"],
        "winner": winner,
        "pilot60_btc_after_gate": module.PILOT_SCORE,
        "rows": manifest["rows"],
        "schema_validation": manifest["schema_validation"],
        "runtime_errors": manifest["runtime_errors"],
        "local_latency_mean_seconds": manifest["local_latency_mean_seconds"],
        "local_over_30_seconds": manifest["local_over_30_seconds"],
        "toolenv_count_audit": manifest["toolenv_count_audit"],
        "public_lb_submitted": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

