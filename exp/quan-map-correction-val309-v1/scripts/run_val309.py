from __future__ import annotations
import hashlib, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp/quan-map-correction-val60-v1/scripts"))
import prepare_benchmark as prep

EXP = ROOT / "exp/quan-map-correction-val309-v1"
OUT = EXP / "output" / "benchmark"
NORMALIZATION_RUNNER = ROOT / "exp/quan-transcript-normalization-val60/scripts/run_benchmark.py"
BRANCHES = ("A", "B", "C", "D")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)

def scratch_dir(branch, round_index):
    path = (OUT / "_scratch" / f"{branch}-round{round_index}").resolve()
    if path == OUT.resolve() or not path.is_relative_to(OUT.resolve()):
        raise RuntimeError(f"Scratch path escaped benchmark output: {path}")
    return path

def promote_block(runner, branch, round_index, block_ids):
    scratch = scratch_dir(branch, round_index)
    scratch.mkdir(parents=True, exist_ok=True)
    runner.OUT = scratch

    warm_path = scratch / f"warmup-COT-round{round_index}.json"
    if not warm_path.exists():
        runner.warmup("COT", round_index)
    pred_path = scratch / "predictions-COT.jsonl"
    trace_path = scratch / "step-traces-COT.jsonl"
    req_path = scratch / "requests-COT.jsonl"

    while True:
        preds = read_jsonl(pred_path)
        done_ids = [row["id"] for row in preds]
        if len(done_ids) != len(set(done_ids)) or set(done_ids) - set(block_ids):
            raise RuntimeError(f"Invalid staged predictions for {branch}/round{round_index}")
        if done_ids != [sid for sid in block_ids if sid in set(done_ids)]:
            raise RuntimeError(f"Staged predictions are out of block order for {branch}/round{round_index}")
        if len(done_ids) == len(block_ids):
            break
        remaining = [sid for sid in block_ids if sid not in set(done_ids)]
        before = len(done_ids)
        try:
            runner.run_block("COT", round_index, remaining, transcript_map[branch], items, cot=True)
        except Exception as exc:
            after_rows = read_jsonl(pred_path)
            if len(after_rows) <= before:
                raise
            new_rows = after_rows[before:]
            if not any(row.get("error") for row in new_rows):
                raise
            print(f"recorded runtime error in {branch}/round{round_index}; continuing after {new_rows[0]['id']}: {new_rows[0].get('error')}", flush=True)
        after_ids = [row["id"] for row in read_jsonl(pred_path)]
        if len(after_ids) <= before:
            raise RuntimeError(f"Runner made no progress in {branch}/round{round_index}")

    preds = read_jsonl(pred_path)
    traces = read_jsonl(trace_path)
    requests = read_jsonl(req_path)
    if [row["id"] for row in preds] != block_ids or [row["id"] for row in traces] != block_ids:
        raise RuntimeError(f"Staged rows are incomplete or misordered for {branch}/round{round_index}")
    for row in traces:
        row["branch"] = branch
    for row in requests:
        row["request_id"] = row["request_id"].replace("COT:", branch + ":", 1)
        row["branch"] = branch

    block_set = set(block_ids)
    for suffix, staged, id_key in (
        ("predictions", preds, "id"),
        ("step-traces", traces, "id"),
        ("requests", requests, "sample_id"),
    ):
        target = OUT / f"{suffix}-{branch}.jsonl"
        prior = read_jsonl(target)
        remaining_prior = [row for row in prior if row[id_key] not in block_set]
        write_jsonl_atomic(target, remaining_prior + staged)

    warm = read_json(warm_path)
    warm["branch"] = branch
    write_json(OUT / f"warmup-{branch}-round{round_index}.json", warm)
    if scratch.exists():
        shutil.rmtree(scratch)

def main():
    global transcript_map, items
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use native system Python outside the project .venv to reach Ollama")
    lock_path = OUT / "benchmark_lock.json"
    lock = read_json(lock_path)
    if lock.get("status") == "inference_complete":
        print("Inference is already complete; refusing to duplicate Qwen calls.")
        return 0
    if lock.get("status") not in {"prepared", "inference_running"}:
        raise RuntimeError(f"Unexpected benchmark lock status: {lock.get('status')}")

    for rel, expected in lock["source_hashes"].items():
        path = ROOT / rel
        if not path.is_file() or prep.sha(path) != expected:
            raise RuntimeError(f"Locked source changed: {rel}")

    runner_module = prep.module("quan_val309_downstream_runner", NORMALIZATION_RUNNER)
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    runner.OUT = OUT
    version, model_digest = runner.model_fingerprint()
    if version != lock["qwen"]["ollama_version"] or model_digest != lock["qwen"]["digest"]:
        raise RuntimeError("Qwen/Ollama fingerprint differs from the val309 preflight lock")
    if runner.canonical_prompt_hash(True) != lock["qwen"]["prompt_sha256_utf8"]:
        raise RuntimeError("Qwen prompt hash differs from the val309 preflight lock")

    cohort = read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    if len(ids) != 309 or cohort.get("is_entire_split") is not True:
        raise RuntimeError("Locked cohort is not the complete val309 split")
    rows = read_jsonl(OUT / "branch_transcripts.jsonl")
    if [row["id"] for row in rows] != ids:
        raise RuntimeError("Transcript rows do not match the locked val309 order")
    transcript_map = {
        branch: {row["id"]: row[key] for row in rows}
        for branch, key in (
            ("A", "A_quan_epoch06_raw"),
            ("B", "B_epoch06_number_normalized_then_train_map"),
            ("C", "C_btc_original_script_oracle"),
            ("D", "D_quan_epoch02_raw"),
        )
    }
    runner_module_data = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(runner_module_data / "train" / "train.jsonl")}
    if set(ids) - items.keys():
        raise RuntimeError("Missing validation task inputs")

    lock.update({"status": "inference_running", "inference_started": True,
                 "inference_start_epoch": lock.get("inference_start_epoch", time.time())})
    write_json(lock_path, lock)
    started = lock["inference_start_epoch"]
    blocks = read_json(OUT / "execution_order.json")["blocks"]
    completed = []
    for block in blocks:
        round_index = block["round"]
        block_ids = block["ids"]
        order = block["branch_order"]
        for branch in order:
            expected_set = set(block_ids)
            artifact_states = []
            for suffix, id_key in (("predictions", "id"), ("step-traces", "id"), ("requests", "sample_id")):
                existing = read_jsonl(OUT / f"{suffix}-{branch}.jsonl")
                found = {row[id_key] for row in existing} & expected_set
                artifact_states.append(found)
            if all(found == expected_set for found in artifact_states):
                stage = scratch_dir(branch, round_index)
                if stage.exists():
                    shutil.rmtree(stage)
                completed.append(f"{branch}-round{round_index}")
                continue
            stage = scratch_dir(branch, round_index)
            if not stage.exists() and any(artifact_states):
                raise RuntimeError(f"Global artifacts are partial and staging is absent: {branch}/round{round_index}")
            promote_block(runner, branch, round_index, block_ids)
            completed.append(f"{branch}-round{round_index}")
            counts = {}
            for name in BRANCHES:
                rows_done = read_jsonl(OUT / f"predictions-{name}.jsonl")
                counts[name] = len(rows_done)
            progress = {"status": "in_progress", "completed_blocks": list(completed),
                        "completed_samples_per_branch": counts, "last_completed": f"{branch}-round{round_index}",
                        "elapsed_seconds": time.time() - started}
            write_json(OUT / "run-progress.json", progress)
            print(json.dumps(progress, ensure_ascii=True), flush=True)

    errors = {}
    for branch in BRANCHES:
        predictions = read_jsonl(OUT / f"predictions-{branch}.jsonl")
        traces = read_jsonl(OUT / f"step-traces-{branch}.jsonl")
        requests = read_jsonl(OUT / f"requests-{branch}.jsonl")
        if [row["id"] for row in predictions] != ids or [row["id"] for row in traces] != ids:
            raise RuntimeError(f"{branch} output does not cover all val IDs in order")
        if any(pred.get("transcript_used") != transcript_map[branch][pred["id"]] for pred in predictions):
            raise RuntimeError(f"{branch} did not use its locked transcript input")
        errors[branch] = sum(bool(row.get("runtime_error")) for row in traces)
        run_meta = {"branch": branch, "status": "complete", "samples": len(ids),
                    "requests": len(requests), "runtime_errors": errors[branch],
                    "execution_blocks": sum(branch in block["branch_order"] for block in blocks)}
        write_json(OUT / f"run-{branch}.json", run_meta)

    ended = time.time()
    manifest = {"status": "complete", "started_epoch": started, "ended_epoch": ended,
        "elapsed_seconds": ended - started, "samples_per_branch": {b: len(ids) for b in BRANCHES},
        "total_branch_sample_runs": len(ids) * len(BRANCHES), "runtime_errors": errors,
        "branch_order": [{"round": row["round"], "branch_order": row["branch_order"]} for row in blocks],
        "raw_chain_of_thought_saved": False, "public_private_test_used": False}
    write_json(OUT / "inference_run.json", manifest)
    lock.update({"status": "inference_complete", "inference_end_epoch": ended,
                 "inference_elapsed_seconds": ended - started, "runtime_errors": errors})
    write_json(lock_path, lock)
    print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    transcript_map = {}
    items = {}
    raise SystemExit(main())

