from __future__ import annotations
import json, sys, time
from pathlib import Path
import prepare_benchmark as prep

ROOT, OUT = prep.ROOT, prep.OUT
ORDER = [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]]

def main():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run native system Python outside the project .venv to reach Ollama")
    runner_module = prep.module("val60_downstream_runner", prep.NORMALIZATION_RUNNER)
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    runner.OUT = OUT
    lock = prep.read_json(OUT / "benchmark_lock.json")
    if lock.get("status") != "prepared": raise RuntimeError("Run prepare_benchmark.py first")
    version, digest = runner.model_fingerprint()
    if version != lock["qwen"]["ollama_version"] or digest != lock["qwen"]["digest"]:
        raise RuntimeError("Locked Qwen/Ollama fingerprint changed")
    if runner.canonical_prompt_hash(True) != lock["qwen"]["prompt_sha256_utf8"]:
        raise RuntimeError("Locked CoT prompt hash changed")
    cohort = prep.read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    rows = prep.read_jsonl(OUT / "branch_transcripts.jsonl")
    if [row["id"] for row in rows] != ids: raise RuntimeError("Prepared transcript rows do not match cohort order")
    transcripts = {
        branch: {row["id"]: row[key] for row in rows}
        for branch, key in (("A", "A_quan_raw"), ("B", "B_quan_number_normalized_then_map"),
                            ("C", "C_btc_original_script"))
    }
    data_root = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    if set(ids) - items.keys(): raise RuntimeError("Missing task inputs for locked validation IDs")

    scratch_root = (OUT / "_scratch").resolve()
    if not scratch_root.is_relative_to(OUT.resolve()) or scratch_root == OUT.resolve():
        raise RuntimeError(f"Scratch root escaped benchmark output: {scratch_root}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    lock.update({"status": "running", "inference_started": True, "inference_start_epoch": time.time()})
    prep.write_json(OUT / "benchmark_lock.json", lock)
    started = time.time()
    for round_index, order in enumerate(ORDER, 1):
        block = ids[(round_index - 1) * 20:round_index * 20]
        for branch in order:
            scratch = (scratch_root / f"{branch}-round{round_index}").resolve()
            if not scratch.is_relative_to(OUT.resolve()) or scratch == OUT.resolve():
                raise RuntimeError(f"Refusing unexpected scratch path: {scratch}")
            if scratch.exists(): raise FileExistsError(f"Refusing to reuse scratch {scratch}")
            runner_module.run_branch_block(runner, branch, round_index, block,
                                           transcripts[branch], items, cot=True)
            prep.write_json(OUT / "run-progress.json", {
                "status": "in_progress", "completed_blocks": [
                    f"{b}-round{r}" for r in range(1, round_index + 1)
                    for b in (ORDER[r - 1] if r < round_index else order[:order.index(branch) + 1])
                ],
                "last_completed": f"{branch}-round{round_index}",
                "elapsed_seconds": time.time() - started,
            })
    for branch in ("A", "B", "C"):
        predictions = prep.read_jsonl(OUT / f"predictions-{branch}.jsonl")
        traces = prep.read_jsonl(OUT / f"step-traces-{branch}.jsonl")
        if [row["id"] for row in predictions] != ids or [row["id"] for row in traces] != ids:
            raise RuntimeError(f"Branch {branch} does not cover all 60 IDs in fixed order")
    manifest = {"status": "complete", "started_epoch": started, "ended_epoch": time.time(),
                "elapsed_seconds": time.time() - started, "samples_per_branch": {"A": 60, "B": 60, "C": 60},
                "branch_order": ORDER, "raw_chain_of_thought_saved": False}
    prep.write_json(OUT / "inference_run.json", manifest)
    lock.update({"status": "inference_complete", "inference_end_epoch": manifest["ended_epoch"],
                 "inference_elapsed_seconds": manifest["elapsed_seconds"]})
    prep.write_json(OUT / "benchmark_lock.json", lock)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

