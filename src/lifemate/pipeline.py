from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .asr import ASRModel
from .agent import Agent
from .retrieval import TrainRetriever
from .retrieval import DEFAULT_TOP_K, MAX_CANDIDATE_TOOLS
from .tool_executor import available_tool_names, create_tool_env
from .utils import resolve_audio

MAX_STEPS = 5


class LifeMatePipeline:
    def __init__(self, data_root: Path, split: str, asr: ASRModel | None = None,
                 agent: Agent | None = None, max_steps: int = MAX_STEPS,
                 retrieval_assisted: bool = False, retrieval_top_k: int = DEFAULT_TOP_K,
                 retrieval_max_candidate_tools: int = MAX_CANDIDATE_TOOLS,
                 soft_retrieval: bool = False, curated_fewshot: bool = False):
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        self.asr = asr or ASRModel()
        self.agent = agent or Agent()
        self.max_steps = max_steps
        self.retriever = TrainRetriever(self.data_root) if retrieval_assisted else None
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_max_candidate_tools = retrieval_max_candidate_tools
        self.soft_retrieval = soft_retrieval
        self.curated_fewshot = curated_fewshot
        catalog_path = self.split_dir / f"{split}_tools_catalog.json"
        self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.agent.catalog = self.catalog

    def run(self, item: dict[str, Any], transcript_override: str | None = None,
            forced_first_tool_name: str | None = None,
            injected_first_call: dict[str, Any] | None = None,
            injected_first_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if injected_first_call is not None and injected_first_calls is not None:
            raise ValueError("Pass injected_first_call or injected_first_calls, not both")
        calls_to_inject = injected_first_calls if injected_first_calls is not None else (
            [injected_first_call] if injected_first_call is not None else [])
        if forced_first_tool_name is not None and calls_to_inject:
            raise ValueError("Only one first-action diagnostic intervention may be used")
        started = time.perf_counter()
        if hasattr(self.agent, "native_history"):
            self.agent.native_history = []
        env = create_tool_env(self.split, item["id"], self.data_root)
        if transcript_override is None:
            asr_text = self.asr.transcribe(str(resolve_audio(self.data_root, self.split_dir, item, item["audio"])))
            transcript = asr_text
        else:
            asr_text = ""
            transcript = transcript_override
        history = []
        for turn in item.get("history", []):
            converted = dict(turn)
            if turn.get("audio"):
                converted["text"] = self.asr.transcribe(str(resolve_audio(self.data_root, self.split_dir, item, turn["audio"])))
                converted.pop("audio", None)
            history.append(converted)
        allowed = available_tool_names(item)
        observations = []
        actions = []
        retrieved_examples = self.retriever.search(transcript, top_k=self.retrieval_top_k) if self.retriever else []
        candidate_tools, retrieval_confidence = (self.retriever.shortlist(
            retrieved_examples, allowed, top_tools=self.retrieval_max_candidate_tools)
                                                  if self.retriever and not self.soft_retrieval and not self.curated_fewshot else (None, 0.0))
        missing_info_demos = self.retriever.missing_info_demos if self.retriever else []
        few_shot_examples = (self.retriever.curated_demos if self.curated_fewshot else
                             retrieved_examples[:DEFAULT_TOP_K] if self.retriever and self.soft_retrieval else
                             retrieved_examples[:DEFAULT_TOP_K] if candidate_tools else [])
        first_turn = 0
        for injected_index, injected_call in enumerate(calls_to_inject):
            if injected_index >= self.max_steps:
                raise ValueError("Injected calls exceed the pipeline turn cap")
            tool = injected_call.get("name")
            args = injected_call.get("arguments", {})
            actions.append({"action": "tool_call", "tool": tool, "arguments": args,
                            "diagnostic_injection": f"gold_first_call_{injected_index + 1}"})
            result = env.validate(tool, args)
            if not result.get("ok"):
                observation = env.call(tool, args)
                observations.append({"tool": tool, "arguments": args, "result": observation})
                first_turn = self.max_steps
                break
            else:
                observation = env.call(tool, args)
                observations.append({"tool": tool, "arguments": args, "result": observation})
                first_turn = injected_index + 1
        for step in range(first_turn, self.max_steps):
            action = self.agent.next_action(transcript, item.get("context", {}), history, allowed, observations,
                                            candidate_tools=candidate_tools,
                                            few_shot_examples=few_shot_examples,
                                            missing_info_demos=missing_info_demos,
                                            missing_info_rule=self.retriever is not None or self.curated_fewshot,
                                            forced_tool_name=forced_first_tool_name if step == 0 else None)
            # Keep internal Ollama assistant messages in the agent history only; prediction
            # artifacts retain the chosen action and bounded diagnostics, not chain-of-thought.
            actions.append({key: value for key, value in action.items() if key != "_native_message"})
            if action.get("action") == "stop":
                break
            tool = action["tool"]
            args = action.get("arguments", {})
            # Validate before execution using ToolEnv's authoritative schema.
            result = env.validate(tool, args)
            if not result.get("ok") and getattr(self.agent, "ollama_native_tools", False):
                repair = self.agent.repair_native_toolenv_rejection(action, result)
                actions.append({key: value for key, value in repair.items() if key != "_native_message"})
                if repair.get("action") == "tool_call":
                    action = repair
                    tool = action["tool"]
                    args = action.get("arguments", {})
                    result = env.validate(tool, args)
                else:
                    break
            if not result.get("ok"):
                observations.append({"tool": tool, "arguments": args, "result": result})
                # One schema repair is handled by the agent; if it nevertheless emits an invalid call,
                # record the attempted call through ToolEnv and stop instead of looping.
                env.call(tool, args)
                self.agent.record_tool_result(action, result)
                break
            observation = env.call(tool, args)
            observations.append({"tool": tool, "arguments": args, "result": observation})
            self.agent.record_tool_result(action, observation)
        calls = env.get_tool_calls()
        return {"id": item["id"], "asr_text": asr_text, "input_transcript": transcript,
                "model_output": actions, "observations": observations,
                "retrieved_examples": retrieved_examples, "candidate_tools": candidate_tools,
                "retrieval_confidence": retrieval_confidence,
                "tool_calls": calls, "state_hash": env.get_state_hash(),
                "final_state": env.get_state(), "transcript": transcript,
                "latency_seconds": time.perf_counter() - started}

