from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lifemate.agent import Agent  # noqa: E402
from lifemate.pipeline import LifeMatePipeline  # noqa: E402
from lifemate.schemas import parse_action, validate_action  # noqa: E402
from lifemate.utils import find_data_root, load_jsonl, seed_everything  # noqa: E402

EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OUT = EXP / "output"
PROVENANCE = OUT / "model-provenance"
OLLAMA_MANIFEST = PROVENANCE / "ollama-models.json"
SFT_TRAIN = ROOT / "exp" / "qwen35-sft-next-action-v1" / "kaggle-upload" / "sft_train.jsonl"
SAMPLE_ID = "train_0050"
HOST = "http://127.0.0.1:11434"
CAP = 768
CONTEXT = 8192
BATCH = 512


def api_json(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(HOST + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_hashable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_smoke_input(data_root: Path) -> tuple[dict[str, Any], str, str]:
    if not SFT_TRAIN.is_file():
        raise FileNotFoundError(SFT_TRAIN)
    row = None
    with SFT_TRAIN.open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate.get("sample_id") == SAMPLE_ID:
                row = candidate
                break
    if row is None:
        raise RuntimeError(f"Fixed smoke sample {SAMPLE_ID} is absent from the SFT training artifact")
    user_messages = [m for m in row.get("messages", []) if m.get("role") == "user"]
    if len(user_messages) != 1:
        raise RuntimeError("The fixed smoke row must have exactly one serialized user message")
    payload = json.loads(user_messages[0]["content"])
    transcript = payload.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise RuntimeError("The fixed smoke row has no transcript")
    item = next((item for item in load_jsonl(data_root / "train" / "train.jsonl")
                 if item.get("id") == SAMPLE_ID), None)
    if item is None:
        raise RuntimeError(f"Fixed smoke item {SAMPLE_ID} is absent from official training data")
    if item.get("history"):
        raise RuntimeError("Fixed smoke item unexpectedly has audio history; keep smoke ASR independent")
    return item, transcript, row["sample_id"]


def main() -> int:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run smoke with system Python, outside .venv, to reach local Ollama")
    out_dir = OUT / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "smoke_manifest.json"
    sample_slug = SAMPLE_ID.replace("/", "_").replace("\\", "_")
    if summary_path.exists() or any((out_dir / f"{branch}-{sample_slug}.json").exists()
                                    for branch in ("base", "sft")):
        raise FileExistsError("Smoke artifacts already exist; refusing to overwrite")

    ollama_manifest = json.loads(OLLAMA_MANIFEST.read_text(encoding="utf-8"))
    tags = api_json("/api/tags").get("models", [])
    by_name = {model.get("name"): model for model in tags}
    data_root = find_data_root("train+public_test/train+public_test")
    item, transcript, source_id = read_smoke_input(data_root)
    inputs = {"sample_id": source_id, "source": str(SFT_TRAIN.relative_to(ROOT)),
              "transcript_sha256": sha256_text(transcript), "transcript_characters": len(transcript),
              "target_used_for_prompt": False, "split": "official train"}

    records: dict[str, dict[str, Any]] = {}
    for branch in ("base", "sft"):
        model = ollama_manifest["branches"][branch]
        tag = model["tag"]
        found = by_name.get(tag)
        if not found or found.get("digest") != model.get("ollama_digest"):
            raise RuntimeError(f"Ollama model fingerprint mismatch for {tag}: {found}")

        os.environ.update({
            "OLLAMA_HOST": HOST,
            "LIFEMATE_LLM_BACKEND": "ollama",
            "LIFEMATE_OLLAMA_MODEL": tag,
            "LIFEMATE_OLLAMA_CONTEXT": str(CONTEXT),
            "LIFEMATE_OLLAMA_NUM_PREDICT": str(CAP),
            "LIFEMATE_OLLAMA_NUM_BATCH": str(BATCH),
            "LIFEMATE_OLLAMA_THINK": "true",
            "LIFEMATE_OLLAMA_PROTOCOL": "json",
            "LIFEMATE_PROMPT_LANGUAGE": "en",
            "LIFEMATE_OLLAMA_TELEMETRY": "1",
        })
        seed_everything(17)
        agent = Agent(max_new_tokens=CAP, context_limit=CONTEXT,
                      ollama_thinking=True, ollama_native_tools=False)
        raw_actions: list[dict[str, Any]] = []
        original_generate = agent._generate

        def capture_json(messages: list[dict[str, str]]) -> str:
            raw = original_generate(messages)
            record: dict[str, Any] = {"content_sha256": sha256_text(raw), "content_characters": len(raw)}
            try:
                parsed = parse_action(raw)
                valid, error = validate_action(parsed, agent.catalog, set(agent._last_allowed_tools))
                record.update({"parsed_action": parsed, "schema_valid": valid, "schema_error": error})
            except Exception as exc:
                record.update({"parse_valid": False, "parse_error": f"{type(exc).__name__}: {exc}"})
            else:
                record["parse_valid"] = True
            raw_actions.append(record)
            return raw

        agent._generate = capture_json
        pipeline = LifeMatePipeline(data_root, "train", agent=agent, max_steps=5)
        agent._last_allowed_tools = set()
        # Capture the exact tool set presented by the real pipeline without changing the prompt.
        original_next = agent.next_action

        def track_tools(transcript_arg, context, history, available_tools, observations, **kwargs):
            agent._last_allowed_tools = set(available_tools)
            return original_next(transcript_arg, context, history, available_tools, observations, **kwargs)

        agent.next_action = track_tools
        started = time.time()
        error = None
        result = None
        try:
            result = pipeline.run(item, transcript_override=transcript)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        executed = [] if result is None else result.get("tool_calls", [])
        valid_generated = [r.get("parsed_action") for r in raw_actions
                           if r.get("parse_valid") and r.get("schema_valid")
                           and r.get("parsed_action", {}).get("action") == "tool_call"]
        executed_keys = {json_hashable({"name": c.get("name"), "arguments": c.get("arguments", {})})
                         for c in executed}
        matched_execution = next((a for a in valid_generated
                                  if json_hashable({"name": a["tool"], "arguments": a["arguments"]}) in executed_keys), None)
        record: dict[str, Any] = {
            "branch": branch, "model_tag": tag, "model_digest": model["ollama_digest"],
            "settings": {"think": True, "temperature": 0, "format": "json",
                         "num_ctx": CONTEXT, "num_predict": CAP, "num_batch": BATCH,
                         "max_action_turns": 5},
            "input": inputs, "generation_json": raw_actions,
            "request_telemetry": agent.request_telemetry,
            "executed_tool_calls": executed,
            "toolenv_state_hash": None if result is None else result.get("state_hash"),
            "toolenv_observations": [] if result is None else result.get("observations", []),
            "model_output_actions": [] if result is None else result.get("model_output", []),
            "valid_json_schema_action_executed_in_toolenv": matched_execution,
            "elapsed_seconds": time.time() - started,
            "error": error,
            "raw_thinking_saved": False,
        }
        branch_path = out_dir / f"{branch}-{sample_slug}.json"
        branch_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records[branch] = {"artifact": str(branch_path.relative_to(ROOT)),
                           "model_tag": tag, "model_digest": model["ollama_digest"],
                           "error": error,
                           "valid_json_schema_action_executed_in_toolenv": matched_execution,
                           "tool_call_count": len(executed),
                           "state_hash": record["toolenv_state_hash"],
                           "elapsed_seconds": record["elapsed_seconds"]}
        if error or matched_execution is None:
            raise RuntimeError(f"Smoke failed for {branch}; details saved to {branch_path}")

    manifest = {"status": "passed", "sample_id": source_id,
                "input_source_sha256": hashlib.sha256(SFT_TRAIN.read_bytes()).hexdigest(),
                "sample_split": "train", "raw_chain_of_thought_saved": False,
                "branches": records}
    summary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
