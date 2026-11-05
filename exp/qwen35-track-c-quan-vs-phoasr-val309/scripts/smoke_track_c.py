from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.agent import Agent
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root, load_jsonl
from lifemate.schemas import parse_action


def ollama_tags() -> set[str]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    with urllib.request.urlopen(host + "/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {model.get("name", "") for model in payload.get("models", [])}


def main() -> int:
    project = _bootstrap.ROOT
    output = project / "exp/qwen35-track-c-quan-vs-phoasr-val309/output/smoke"
    output.mkdir(parents=True, exist_ok=True)
    os.environ["LIFEMATE_LLM_BACKEND"] = "ollama"
    os.environ["LIFEMATE_OLLAMA_MODEL"] = os.environ.get("LIFEMATE_OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
    os.environ["LIFEMATE_OLLAMA_CONTEXT"] = "8192"
    os.environ["LIFEMATE_OLLAMA_NUM_PREDICT"] = "512"
    os.environ["LIFEMATE_OLLAMA_THINK"] = "true"
    os.environ["LIFEMATE_OLLAMA_PROTOCOL"] = "json"
    os.environ["LIFEMATE_PROMPT_LANGUAGE"] = "en"
    model_tag = os.environ["LIFEMATE_OLLAMA_MODEL"]
    tags = ollama_tags()
    if model_tag not in tags:
        raise RuntimeError(f"Ollama model tag {model_tag!r} is unavailable; installed tags: {sorted(tags)}")

    agent = Agent(max_new_tokens=512, context_limit=8192)
    prompt_output = agent._generate([{"role": "user", "content": "Return exactly this JSON object: {\"smoke\": true}"}])
    prompt_parsed = json.loads(prompt_output)
    if prompt_parsed != {"smoke": True}:
        raise RuntimeError(f"JSON prompt smoke returned unexpected object: {prompt_parsed}")
    parsed_action = parse_action('{"action":"stop"}')
    if parsed_action.get("action") != "stop":
        raise RuntimeError("LifeMate action parser smoke failed")

    root = find_data_root("train+public_test/train+public_test")
    ids = json.loads((project / "data/splits/val_ids.json").read_text(encoding="utf-8"))
    reference_path = project / "exp/session4-qwen3-vs-qwen35-val309/output/qwen35-en-cap5/predictions.jsonl"
    reference = {row["id"]: row for row in (json.loads(line) for line in reference_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    sample_id = ids[0]
    item = next(row for row in load_jsonl(root / "train/train.jsonl") if row["id"] == sample_id)
    transcript = reference[sample_id].get("transcript_used") or reference[sample_id].get("asr_text") or ""
    agent = Agent(max_new_tokens=512, context_limit=8192)
    pipeline = LifeMatePipeline(root, "train", agent=agent, max_steps=5)
    result = pipeline.run(item, transcript_override=transcript)
    validation = (result["observations"][0]["result"] if result["observations"] else
                  {"ok": None, "detail": "sample produced no tool calls; ToolEnv was still exercised once by pipeline"})
    artifact = {"model_tag": model_tag, "installed_tags": sorted(tags), "backend": "ollama",
                "thinking": agent.ollama_thinking, "protocol": "native" if agent.ollama_native_tools else "json",
                "num_ctx": 8192, "num_predict": 512, "temperature": 0, "sample_id": sample_id,
                "parser_smoke": {"raw": prompt_output, "parsed_json": prompt_parsed, "action_parser": parsed_action},
                "prediction": {"tool_calls": result["tool_calls"], "state_hash": result["state_hash"],
                               "latency_seconds": result["latency_seconds"], "observations": result["observations"],
                               "first_call_schema_validation": validation}}
    (output / "track-c-smoke.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in artifact.items() if k != "prediction"}, ensure_ascii=False, indent=2))
    print(json.dumps({"sample_id": sample_id, "latency_seconds": result["latency_seconds"],
                      "tool_calls": result["tool_calls"], "first_call_schema_validation": validation}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
