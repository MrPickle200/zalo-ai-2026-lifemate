from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import prepare_benchmark as prep

ROOT, OUT = prep.ROOT, prep.OUT
ASR_DIR = ROOT / "exp/quan-continuation-cot-v1/output/asr/epoch-02-val309"
ASR_PREDICTIONS = ASR_DIR / "predictions.jsonl"
ASR_SUMMARY = ASR_DIR / "summary.json"
CHECKPOINT = ROOT / "exp/quan-continuation-cot-v1/output/checkpoints/epoch-02/model.safetensors"

def main():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use native system Python outside the project .venv")
    runner_module = prep.module("val60_epoch02_runner", prep.NORMALIZATION_RUNNER)
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    runner.OUT = OUT
    lock = prep.read_json(OUT / "benchmark_lock.json")
    if lock.get("status") != "complete": raise RuntimeError("The A/B/C benchmark is not complete")
    if runner.model_fingerprint() != (lock["qwen"]["ollama_version"], lock["qwen"]["digest"]):
        raise RuntimeError("Locked Qwen/Ollama fingerprint changed")
    if runner.canonical_prompt_hash(True) != lock["qwen"]["prompt_sha256_utf8"]:
        raise RuntimeError("Locked Qwen prompt changed")
    if any((OUT / f"{name}-D.jsonl").exists() for name in ("predictions", "step-traces", "requests")):
        raise FileExistsError("Branch D output exists; refusing to rerun or overwrite it")

    cohort = prep.read_json(OUT / "cohort_manifest.json")
    ids = cohort["ids"]
    val_ids = json.loads((ROOT / "data/splits/val_ids.json").read_text(encoding="utf-8-sig"))
    train_ids = json.loads((ROOT / "data/splits/train_ids.json").read_text(encoding="utf-8-sig"))
    if set(ids) - set(val_ids) or set(ids) & set(train_ids) or len(ids) != 60:
        raise RuntimeError("Locked cohort is not the expected held-out val60 subset")
    rows = prep.read_jsonl(ASR_PREDICTIONS)
    if [row["id"] for row in rows] != val_ids:
        raise RuntimeError("The epoch-02 transcript cache does not exactly cover val309 in order")
    transcripts = {row["id"]: row["prediction"] for row in rows}
    if any(not transcripts.get(sid) for sid in ids):
        raise RuntimeError("Empty epoch-02 QUAN prediction in the cohort")
    summary = prep.read_json(ASR_SUMMARY)
    checkpoint_sha = prep.sha(CHECKPOINT)
    if summary.get("model_sha256") != checkpoint_sha:
        raise RuntimeError("Epoch-02 prediction summary does not match the requested checkpoint")
    if abs(summary["metrics"]["corpus_wer"] - 0.16334955034645438) > 1e-10:
        raise RuntimeError("Unexpected epoch-02 validation metrics/checkpoint")

    gold_rows = prep.read_jsonl(OUT / "gold_references.jsonl")
    if [row["id"] for row in gold_rows] != ids:
        raise RuntimeError("Gold reference order differs from the locked val60 cohort")
    gold = {row["id"]: row["reference"] for row in gold_rows}
    normalizer = prep.module("val60_epoch02_number_normalizer", prep.NORMALIZER)
    project_wer, number_wer = [], []
    source_rows = []
    for sid in ids:
        hyp = transcripts[sid]
        ref = gold[sid]
        hyp_number, _ = prep.normalize(hyp, normalizer)
        ref_number, _ = prep.normalize(ref, normalizer)
        project_wer.append((sid, prep.tokens(ref), prep.tokens(hyp)))
        number_wer.append((sid, ref_number, hyp_number))
        source_rows.append({"id": sid, "prediction": hyp, "source_checkpoint": "epoch-02"})
    prep.write_jsonl(OUT / "transcripts-D-epoch02.jsonl", source_rows)
    wer = {"branch": "D", "input": "raw QUAN two-epoch continuation checkpoint",
           "checkpoint": "exp/quan-continuation-cot-v1/output/checkpoints/epoch-02",
           "checkpoint_sha256": checkpoint_sha, "validation_transcript_file": str(ASR_PREDICTIONS.relative_to(ROOT)),
           "validation_transcript_sha256": prep.sha(ASR_PREDICTIONS),
           "WER_project_normalization": prep.corpus_wer(project_wer),
           "WER_number_value_normalization": prep.corpus_wer(number_wer),
           "number_parser_applied_for_metric_only": True,
           "prediction_input_uses_raw_epoch02_transcript": True}
    prep.write_json(OUT / "transcript-analysis-D.json", wer)

    data_root = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    if set(ids) - items.keys(): raise RuntimeError("Missing validation task inputs")
    scratch_root = (OUT / "_scratch").resolve()
    if not scratch_root.is_relative_to(OUT.resolve()) or scratch_root == OUT.resolve():
        raise RuntimeError(f"Scratch directory escaped benchmark output: {scratch_root}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for round_index in range(1, 4):
        block = ids[(round_index - 1) * 20:round_index * 20]
        scratch = (scratch_root / f"D-round{round_index}").resolve()
        if not scratch.is_relative_to(OUT.resolve()) or scratch == OUT.resolve():
            raise RuntimeError(f"Unexpected scratch path: {scratch}")
        if scratch.exists(): raise FileExistsError(f"Refusing to reuse scratch path: {scratch}")
        runner_module.run_branch_block(runner, "D", round_index, block, transcripts, items, cot=True)
        prep.write_json(OUT / "run-progress-D.json", {"status": "in_progress",
            "completed_blocks": round_index, "completed_samples": round_index * 20,
            "elapsed_seconds": time.time() - started})
    predictions = prep.read_jsonl(OUT / "predictions-D.jsonl")
    traces = prep.read_jsonl(OUT / "step-traces-D.jsonl")
    if [row["id"] for row in predictions] != ids or [row["id"] for row in traces] != ids:
        raise RuntimeError("Branch D does not cover all fixed val60 IDs in order")
    if any(row.get("transcript_used") != transcripts[row["id"]] for row in predictions):
        raise RuntimeError("Branch D inference did not use the epoch-02 transcript inputs")
    manifest = {"status": "complete", "branch": "D", "samples": len(ids),
                "started_epoch": started, "ended_epoch": time.time(),
                "elapsed_seconds": time.time() - started, "checkpoint_sha256": checkpoint_sha,
                "transcript_file_sha256": prep.sha(ASR_PREDICTIONS),
                "qwen_model": runner.MODEL, "qwen_digest": lock["qwen"]["digest"],
                "prompt_sha256_utf8": lock["qwen"]["prompt_sha256_utf8"],
                "branch_input": "raw epoch-02 QUAN transcript, no map; fixed 60-ID cohort"}
    prep.write_json(OUT / "inference-D.json", manifest)
    print(json.dumps({"status": "complete", "branch": "D", "samples": 60,
        "elapsed_seconds": manifest["elapsed_seconds"], "checkpoint_sha256": checkpoint_sha,
        "WER_project": wer["WER_project_normalization"]["corpus_wer"],
        "WER_number_value": wer["WER_number_value_normalization"]["corpus_wer"]},
        ensure_ascii=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

