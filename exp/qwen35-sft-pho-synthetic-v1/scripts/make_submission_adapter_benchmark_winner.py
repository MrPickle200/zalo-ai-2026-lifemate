from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
EVAL = EXP / "output/adapter-final-1epoch-benchmark/val309"
OUT = EXP / "output/submission-adapter-benchmark-winner"
FOCUSED_WRAPPER = ROOT / "exp/phowhisper-large-qwen-focused-public-v1/scripts/make_submission.py"
STEM = "phowhisper-large-v2-qwen35-val309-benchmark-winner-focused-public-test"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load submission workflow: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure():
    summary = read_json(EVAL / "summary.json")
    lock = read_json(EVAL / "comparison_lock.json")
    scores = summary["groups"]["all_val309"]
    candidates = ("best_baseline", "step120", "adapter_final1")
    for key in candidates:
        branch = summary["branches"].get(key, {})
        if branch.get("status") != "complete" or branch.get("samples_completed") != 309:
            raise RuntimeError(f"Cannot select incomplete val309 branch: {key}")
    selected = max(candidates, key=lambda key: scores[key]["btc_0_100"])
    if selected == "best_baseline":
        model = lock["baseline"]["model"]
        digest = lock["baseline"].get("digest") or "fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
    elif selected == "step120":
        model = lock["step120"]["model"]
        digest = lock["step120"]["digest"]
    else:
        model = lock["adapter_final1"]["model"]
        digest = lock["adapter_final1"]["digest"]
    module = load_module("focused_submission_from_val309_benchmark_winner", FOCUSED_WRAPPER)
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
    module.base.MODEL_PARAMS = 4205751296
    module.SELECTED_CHECKPOINT = selected
    module.SELECTED_VAL309_SCORE = scores[selected]["btc_0_100"]
    module.VAL309_SCORES = {key: scores[key]["btc_0_100"] for key in candidates}
    module.VAL309_LOCK_SHA256 = __import__("hashlib").sha256((EVAL / "comparison_lock.json").read_bytes()).hexdigest()
    return module


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--prepare", "--run"):
        raise SystemExit("Use --prepare or --run")
    module = configure()
    selection = {"status": "selected_from_complete_val309_benchmark", "winner": module.SELECTED_CHECKPOINT,
        "model": module.MODEL, "model_digest": module.DIGEST, "val309_btc_after_30s_gate": module.SELECTED_VAL309_SCORE,
        "all_val309_scores": module.VAL309_SCORES, "benchmark_summary": str((EVAL / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "benchmark_comparison_lock_sha256": module.VAL309_LOCK_SHA256,
        "public_submission_config": {"prompt": "focused post-lookup", "asr": "PhoWhisper Large v2 main and history",
            "thinking": True, "temperature": 0, "protocol": "JSON", "num_predict": 768,
            "num_ctx": 8192, "num_batch": 512, "seed": 17, "max_action_turns": 5}}
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "--prepare":
        write_json(OUT / "winner_selection.json", selection)
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
        module.prepare()
        return
    module.run()
    selection["status"] = "public_submission_generated"
    write_json(OUT / "winner_selection.json", selection)
    manifest_path = ROOT / "outputs/submissions" / (STEM + "_manifest.json")
    manifest = read_json(manifest_path)
    manifest["validation_winner"] = module.SELECTED_CHECKPOINT
    manifest["val309_btc_after_30s_gate"] = module.SELECTED_VAL309_SCORE
    manifest["val309_scores"] = module.VAL309_SCORES
    manifest["val309_benchmark_summary"] = selection["benchmark_summary"]
    manifest["val309_comparison_lock_sha256"] = module.VAL309_LOCK_SHA256
    write_json(manifest_path, manifest)
    write_json(OUT / "submission_summary.json", manifest)
    print(json.dumps({"submission": manifest["submission"], "winner": module.SELECTED_CHECKPOINT,
        "val309_btc": module.SELECTED_VAL309_SCORE, "rows": manifest["rows"],
        "runtime_errors": manifest["runtime_errors"], "schema_validation": manifest["schema_validation"]}, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
