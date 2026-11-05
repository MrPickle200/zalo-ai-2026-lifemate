from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.asr import PhoASRModel
from lifemate.pipeline import MAX_STEPS, LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pairwise_wer(a: str, b: str) -> float:
    a_words, b_words = a.split(), b.split()
    return edit_distance(a_words, b_words) / max(1, len(a_words))


def consensus_choice(candidates: list[dict]) -> dict:
    # Deterministic medoid: minimize mean token-WER to the other cached candidates.
    return min(candidates, key=lambda c: (sum(pairwise_wer(c["text"], other["text"])
                                             for other in candidates),
                                          c["decode_config"], c["rank"]))


def main() -> int:
    p = argparse.ArgumentParser(description="Create a public-test submission using cached PhoASR N-best consensus")
    p.add_argument("--transcripts", default="exp/phoasr-nbest-v1/output/public_test/transcripts.jsonl")
    p.add_argument("--data-root")
    p.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter")
    p.add_argument("--output", default="outputs/submissions/phoasr-nbest-consensus-v1.jsonl")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.resume and args.overwrite:
        p.error("--resume and --overwrite are mutually exclusive")
    seed_everything()
    root = find_data_root(args.data_root)
    split = "public_test"
    items = list(load_jsonl(root / split / f"{split}.jsonl"))
    cache_path = Path(args.transcripts)
    cache = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [item["id"] for item in items]
    cache_ids = [row["id"] for row in cache]
    if cache_ids != ids:
        raise SystemExit("N-best cache IDs/order do not exactly match the public-test manifest")
    if any(not row.get("candidates") for row in cache):
        raise SystemExit("N-best cache contains a sample without candidates")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("exp/phoasr-nbest-v1/output/submission-artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = artifact_dir / (output.stem + "_predictions.jsonl")
    manifest_path = artifact_dir / (output.stem + "_manifest.json")
    if output.exists() and not (args.resume or args.overwrite):
        raise SystemExit(f"Refusing to overwrite {output}; pass --overwrite or --resume")
    count = 0
    if args.resume and output.exists():
        existing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(existing) > len(items):
            raise SystemExit("Existing submission has more rows than the manifest")
        for i, row in enumerate(existing):
            if row.get("id") != ids[i] or set(row) != {"id", "tool_calls", "state_hash"}:
                raise SystemExit(f"Existing output is not a valid manifest prefix at row {i + 1}")
        if not diagnostic_path.exists():
            raise SystemExit("Cannot resume: diagnostic prediction sidecar is missing")
        diag = [line for line in diagnostic_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(diag) != len(existing):
            raise SystemExit("Cannot resume: submission and diagnostic sidecar row counts differ")
        count = len(existing)

    asr = PhoASRModel(cache_dir="cache/transcripts")
    agent = Agent(model_path=args.model, adapter_path=args.adapter)
    pipeline = LifeMatePipeline(root, split, asr=asr, agent=agent, max_steps=args.max_steps)
    mode = "a" if args.resume and output.exists() else "w"
    total_latency = 0.0
    started = time.perf_counter()
    with output.open(mode, encoding="utf-8", newline="\n") as submission, \
            diagnostic_path.open(mode, encoding="utf-8", newline="\n") as diagnostic:
        for index in range(count, len(items)):
            item = items[index]
            record = cache[index]
            selected = consensus_choice(record["candidates"])
            result = pipeline.run(item, transcript_override=selected["text"])
            row = {"id": result["id"], "tool_calls": result["tool_calls"], "state_hash": result["state_hash"]}
            detail = {"id": result["id"], "selected_candidate": selected,
                      "tool_calls": result["tool_calls"], "state_hash": result["state_hash"],
                      "observations": result["observations"], "latency_seconds": result["latency_seconds"]}
            submission.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            diagnostic.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
            submission.flush()
            diagnostic.flush()
            total_latency += result["latency_seconds"]
            print(f"[{index + 1}/{len(items)}] {item['id']} calls={len(row['tool_calls'])} "
                  f"latency={result['latency_seconds']:.2f}s", flush=True)

    manifest = {"split": split, "rows": len(items), "submission": str(output),
                "diagnostics": str(diagnostic_path), "transcripts": str(cache_path),
                "transcript_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "selector": "consensus medoid minimizing average pairwise token WER",
                "gold_used_for_selection": False, "asr": "PhoASR public N-best cache",
                "llm_model": str(args.model), "adapter": args.adapter,
                "llm_backend": os.environ.get("LIFEMATE_LLM_BACKEND", "transformers"),
                "ollama_context": int(os.environ.get("LIFEMATE_OLLAMA_CONTEXT", "16384"))
                if os.environ.get("LIFEMATE_LLM_BACKEND", "transformers").lower() == "ollama" else None,
                "max_steps": args.max_steps,
                "elapsed_seconds_this_run": time.perf_counter() - started,
                "note": "Full public-test submission; each sample used one ToolEnv."}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
