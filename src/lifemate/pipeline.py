from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .asr import ASRModel
from .agent import Agent
from .tool_executor import available_tool_names, create_tool_env
from .utils import resolve_audio

MAX_STEPS = 5


class LifeMatePipeline:
    def __init__(self, data_root: Path, split: str, asr: ASRModel | None = None,
                 agent: Agent | None = None, max_steps: int = MAX_STEPS):
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        self.asr = asr or ASRModel()
        self.agent = agent or Agent()
        self.max_steps = max_steps
        catalog_path = self.split_dir / f"{split}_tools_catalog.json"
        self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.agent.catalog = self.catalog

    def run(self, item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        env = create_tool_env(self.split, item["id"], self.data_root)
        transcript = self.asr.transcribe(str(resolve_audio(self.data_root, self.split_dir, item, item["audio"])))
        history = []
        for turn in item.get("history", []):
            converted = dict(turn)
            if turn.get("audio"):
                converted["text"] = self.asr.transcribe(str(resolve_audio(self.data_root, self.split_dir, item, turn["audio"])))
                converted.pop("audio", None)
            history.append(converted)
        allowed = available_tool_names(item)
        observations = []
        for _ in range(self.max_steps):
            action = self.agent.next_action(transcript, item.get("context", {}), history, allowed, observations)
            if action.get("action") == "stop":
                break
            tool = action["tool"]
            args = action.get("arguments", {})
            # Validate before execution using ToolEnv's authoritative schema.
            result = env.validate(tool, args)
            if not result.get("ok"):
                observations.append({"tool": tool, "arguments": args, "result": result})
                # One schema repair is handled by the agent; if it nevertheless emits an invalid call,
                # record the attempted call through ToolEnv and stop instead of looping.
                env.call(tool, args)
                break
            observation = env.call(tool, args)
            observations.append({"tool": tool, "arguments": args, "result": observation})
        calls = env.get_tool_calls()
        return {"id": item["id"], "tool_calls": calls, "state_hash": env.get_state_hash(),
                "final_state": env.get_state(), "transcript": transcript,
                "latency_seconds": time.perf_counter() - started}

