from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
EVAL = EXP / "output/adapter-final-4epoch-benchmark/val309"
OUT = EXP / "output/submission-final3-vs-final4"
FOCUSED_WRAPPER = ROOT / "exp/phowhisper-large-qwen-focused-public-v1/scripts/make_submission.py"
STEM = "phowhisper-large-v2-qwen35-phosynth-final4-val309-winner-focused-public-test"
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
    summary = read_json(EVAL / "summary.json")
    lock = read_json(EVAL / "comparison_lock.json")
    scores = summary["groups"]["all_val309"]
    candidates = ("adapter_final3", "adapter_final4")
    for key in candidates:
        branch = summary["branches"].get(key, {})
        if branch.get("status") != "complete" or branch.get("samples_completed") != 309:
            raise RuntimeError(f"Cannot select incomplete val309 branch: {key}")
    selected = summary["public_candidate_winner_by_raw_val309"]
    if selected not in candidates:
        raise RuntimeError(f"Unexpected val309 selection: {selected}")
    model = lock[selected]["model"]
    digest = lock[selected]["digest"]

    module = load_module("focused_submission_from_final3_final4_comparison", FOCUSED_WRAPPER)
    module.__file__ = str(Path(__file__).resolve())
    module.MODEL = model
    module.DIGEST = digest
    module.STEM = STEM
    module.EXP = EXP
    module.OUT = OUT
    module.base.MODEL = model
    module.base.DIGEST = digest
    module.base.STEM = STEM
    module.base.EXP = EXP
    module.base.OUT = OUT
    module.base.MODEL_PARAMS = 4_205_751_296
    module.base.CONTEXT = SUBMISSION_CONTEXT

    original_input_lock = module.input_lock

    def input_lock_with_selection():
        submission_lock = original_input_lock()
        submission_lock["settings"]["num_ctx"] = SUBMISSION_CONTEXT
        submission_lock["val309_selection"] = {
            "selected_checkpoint": selected,
            "selected_raw_btc_before_runtime_gate": scores[selected]["btc_before_runtime_gate"],
            "selected_btc_after_local_30s_gate": scores[selected]["btc_after_30s_gate"],
            "compared_candidates": {
                key: {
                    "raw_btc_before_runtime_gate": scores[key]["btc_before_runtime_gate"],
                    "btc_after_local_30s_gate": scores[key]["btc_after_30s_gate"],
                    "model": lock[key]["model"],
                    "digest": lock[key]["digest"],
                }
                for key in candidates
            },
            "benchmark_summary": str((EVAL / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            "benchmark_lock_sha256": file_sha256(EVAL / "comparison_lock.json"),
            "selection_rule": "Select by full-thinking BTC before the local 30-second gate, matching the fixed-thinking public-test policy.",
        }
        submission_lock["context_adjustment"] = {
            "num_ctx": SUBMISSION_CONTEXT,
            "reason": "The shared val309 failure train_1725 produced an 8,265-token request, exceeding the 8,192-token benchmark context. The submission context is raised to the competition maximum of 16,384 so the candidate can process that prompt.",
            "benchmark_context": lock["settings"]["num_ctx"],
            "competition_context_limit": 16_384,
        }
        submission_lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/scripts/make_submission_final3_vs_final4.py"] = file_sha256(Path(__file__).resolve())
        submission_lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/output/adapter-final-4epoch-benchmark/val309/summary.json"] = file_sha256(EVAL / "summary.json")
        submission_lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/output/adapter-final-4epoch-benchmark/val309/comparison_lock.json"] = file_sha256(EVAL / "comparison_lock.json")
        return submission_lock

    module.input_lock = input_lock_with_selection
    module.SELECTED_CHECKPOINT = selected
    module.SELECTED_MODEL = model
    module.SELECTED_DIGEST = digest
    module.VAL309_SCORES = {
        key: {
            "raw_btc_before_runtime_gate": scores[key]["btc_before_runtime_gate"],
            "btc_after_local_30s_gate": scores[key]["btc_after_30s_gate"],
        }
        for key in candidates
    }
    module.VAL309_LOCK_SHA256 = file_sha256(EVAL / "comparison_lock.json")
    return module


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--prepare", "--run"):
        raise SystemExit("Use --prepare or --run")
    module = configure()
    OUT.mkdir(parents=True, exist_ok=True)
    selection = {
        "status": "selected_from_complete_val309_benchmark",
        "winner": module.SELECTED_CHECKPOINT,
        "model": module.SELECTED_MODEL,
        "model_digest": module.SELECTED_DIGEST,
        "val309_scores": module.VAL309_SCORES,
        "benchmark_summary": str((EVAL / "summary.md").relative_to(ROOT)).replace("\\", "/"),
        "benchmark_lock_sha256": module.VAL309_LOCK_SHA256,
        "public_submission_config": {
            "prompt": "focused post-lookup disambiguation",
            "asr": "PhoWhisper Large v2 main and history",
            "thinking": True,
            "temperature": 0,
            "protocol": "JSON",
            "num_predict": 768,
            "num_ctx": SUBMISSION_CONTEXT,
            "num_batch": 512,
            "seed": 17,
            "max_action_turns": 5,
            "context_note": "Raised from 8192 to 16384 to handle the shared 8265-token val309 overflow; 16384 is the competition maximum.",
        },
        "public_lb_submitted": False,
    }
    selection_path = OUT / "winner_selection.json"
    if sys.argv[1] == "--prepare":
        write_json(selection_path, selection)
        module.prepare()
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
        return

    module.run()
    selection["status"] = "public_submission_generated_and_schema_validated"
    write_json(selection_path, selection)
    manifest_path = ROOT / "outputs/submissions" / (STEM + "_manifest.json")
    manifest = read_json(manifest_path)
    manifest["validation_winner"] = module.SELECTED_CHECKPOINT
    manifest["val309_scores"] = module.VAL309_SCORES
    manifest["val309_benchmark_summary"] = selection["benchmark_summary"]
    manifest["val309_comparison_lock_sha256"] = module.VAL309_LOCK_SHA256
    manifest["context_adjustment"] = selection["public_submission_config"]["context_note"]
    manifest["public_lb_submitted"] = False
    write_json(manifest_path, manifest)
    write_json(OUT / "submission_summary.json", manifest)
    print(json.dumps({
        "submission": manifest["submission"],
        "winner": module.SELECTED_CHECKPOINT,
        "val309_scores": module.VAL309_SCORES,
        "rows": manifest["rows"],
        "runtime_errors": manifest["runtime_errors"],
        "schema_validation": manifest["schema_validation"],
        "num_ctx": SUBMISSION_CONTEXT,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

