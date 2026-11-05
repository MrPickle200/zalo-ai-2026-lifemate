from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.asr import PhoASRModel
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl, seed_everything


VARIANTS = {
    "B": (False, True, "nothink-native"),
    "C": (True, False, "think-json"),
    "D": (True, True, "think-native"),
}
DEFAULT_TRANSCRIPTS = Path("exp/phoasr-nbest-v1/output/public_test/transcripts.jsonl")


def ollama_version() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    request = urllib.request.Request(host + "/api/version")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))["version"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one Qwen3.5 B/C/D public-test submission")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS))
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seed_everything()
    version = ollama_version()
    root = find_data_root(args.data_root)
    split = "public_test"
    items = list(load_jsonl(root / split / f"{split}.jsonl"))
    ids = [item["id"] for item in items]
    transcript_path = Path(args.transcripts)
    transcript_rows = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
    transcript_ids = [row["id"] for row in transcript_rows]
    if transcript_ids != ids:
        raise SystemExit("PhoASR transcript cache IDs/order do not exactly match the public-test manifest")
    transcripts = {}
    for row in transcript_rows:
        candidates = row.get("candidates") or []
        production = next((candidate["text"] for candidate in candidates
                           if candidate.get("decode_config") == "production_greedy"), None)
        production = production or row.get("production_1best")
        if not isinstance(production, str):
            raise SystemExit(f"No production-greedy PhoASR transcript for {row.get('id')}")
        transcripts[row["id"]] = production

    thinking, native, suffix = VARIANTS[args.variant]
    output = Path(args.output or f"outputs/submissions/qwen35-4b-promptv2-en-phoasr-{args.variant}-{suffix}-v1.jsonl")
    artifact_dir = Path("exp/qwen35-promptv2-submissions/output/submission-artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = artifact_dir / (output.stem + "_predictions.jsonl")
    manifest_path = artifact_dir / (output.stem + "_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output}; pass --resume to continue a valid prefix")
    completed = 0
    if args.resume and output.exists():
        prior = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        prior_diag = ([json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                      if diagnostics_path.exists() else [])
        if len(prior) != len(prior_diag) or len(prior) > len(items):
            raise SystemExit("Cannot resume: submission and diagnostic rows are not aligned")
        for index, row in enumerate(prior):
            if row.get("id") != ids[index] or set(row) != {"id", "tool_calls", "state_hash"}:
                raise SystemExit(f"Cannot resume: invalid manifest prefix at row {index + 1}")
        completed = len(prior)

    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = "qwen3.5:4b-q4_K_M"
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "8192"
    os.environ["LIFEMATE_PROMPT_LANGUAGE"] = "en"
    agent = Agent(max_new_tokens=512, context_limit=8192,
                  ollama_thinking=thinking, ollama_native_tools=native)
    pipeline = LifeMatePipeline(root, split, asr=PhoASRModel(), agent=agent, max_steps=5)

    total_latency = 0.0
    runtime_failures = 0
    started = time.perf_counter()
    mode = "a" if args.resume and output.exists() else "w"
    with output.open(mode, encoding="utf-8", newline="\n") as submission, \
            diagnostics_path.open(mode, encoding="utf-8", newline="\n") as diagnostics:
        for index in range(completed, len(items)):
            item = items[index]
            sample_started = time.perf_counter()
            try:
                result = pipeline.run(item, transcript_override=transcripts[item["id"]])
                calls = [{"name": call.get("name"), "arguments": call.get("arguments", {})}
                         for call in result["tool_calls"]]
                row = {"id": item["id"], "tool_calls": calls, "state_hash": result["state_hash"]}
                detail = {"id": item["id"], "transcript_used": transcripts[item["id"]],
                          "tool_calls": calls, "state_hash": result["state_hash"],
                          "observations": result["observations"], "model_output": result["model_output"],
                          "latency_seconds": result["latency_seconds"], "runtime_error": None}
                total_latency += result["latency_seconds"]
            except Exception as exc:
                runtime_failures += 1
                latency = time.perf_counter() - sample_started
                row = {"id": item["id"], "tool_calls": [], "state_hash": "0" * 64}
                detail = {"id": item["id"], "transcript_used": transcripts[item["id"]],
                          "tool_calls": [], "state_hash": row["state_hash"], "observations": [],
                          "model_output": [], "latency_seconds": latency,
                          "runtime_error": f"{type(exc).__name__}: {exc}"}
                total_latency += latency
            submission.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            diagnostics.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
            submission.flush()
            diagnostics.flush()
            print(f"[{args.variant} {index + 1}/{len(items)}] {item['id']} "
                  f"calls={len(row['tool_calls'])} latency={detail['latency_seconds']:.2f}s "
                  f"error={detail['runtime_error']}", flush=True)

    manifest = {
        "variant": args.variant, "split": split, "rows": len(items),
        "submission": str(output), "diagnostics": str(diagnostics_path),
        "transcripts": str(transcript_path),
        "transcript_cache_sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b-q4_K_M", "parameters": "4.2B", "quantization": "Q4_K_M",
        "ollama_version": version, "asr": "PhoASR production-greedy; audio in history transcribed locally",
        "prompt_version": "observation-grounded-v2", "prompt_language": "en",
        "thinking": thinking, "protocol": "native Ollama tools with role=tool observations" if native else
                    "JSON actions with observations in payload",
        "temperature": 0, "num_predict": 512, "num_ctx": 8192, "max_action_turns": 5,
        "average_latency_seconds_this_run": total_latency / max(1, len(items) - completed),
        "runtime_failures": runtime_failures,
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "toolenv_sessions": "exactly one per sample", "public_lb_submitted": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 1 if runtime_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
