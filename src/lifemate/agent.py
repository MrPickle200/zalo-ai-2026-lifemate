from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import parse_action, validate_action


class Agent:
    """Small, greedy local causal LLM for structured tool actions."""

    def __init__(self, model_path: str | Path | None = None, max_new_tokens: int = 128, context_limit: int = 6144):
        self.model_path = Path(model_path or os.environ.get("LIFEMATE_LLM_MODEL", "models/Qwen3-4B-Instruct-2507")).expanduser()
        self.max_new_tokens = max_new_tokens
        self.context_limit = min(context_limit, 16384)
        self.tokenizer = None
        self.model = None
        self.catalog: dict = {}

    def load(self):
        if self.model is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"LLM weights must be available locally at {self.model_path}. "
                    "Set LIFEMATE_LLM_MODEL to a local model directory."
                )
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Missing torch/transformers; install requirements.txt into .venv") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False)
            if torch.cuda.is_available():
                total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # Leave room for context/KV cache and display usage on small GPUs;
                # let larger cards keep the whole model on-device for competition runtime.
                gpu_budget = max(2, int(total_gib - (3 if total_gib <= 8 else 4)))
                max_memory = {0: f"{gpu_budget}GiB", "cpu": "24GiB"}
            else:
                max_memory = None
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
                torch_dtype="auto", low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                max_memory=max_memory,
            )
            n_params = sum(p.numel() for p in self.model.parameters())
            if n_params >= 5_000_000_000:
                self.model = None
                raise ValueError(f"LLM has {n_params:,} parameters; competition limit is below 5B")
            self.model.eval()
        return self.model

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if os.environ.get("LIFEMATE_LLM_BACKEND", "transformers").lower() == "ollama":
            model_name = os.environ.get("LIFEMATE_OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
            ollama_context = min(int(os.environ.get("LIFEMATE_OLLAMA_CONTEXT", "16384")), 16384)
            payload = json.dumps({
                "model": model_name,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": self.max_new_tokens,
                    "num_ctx": ollama_context,
                },
            }).encode("utf-8")
            request = urllib.request.Request(
                os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                detail = ""
                if hasattr(exc, "read"):
                    try:
                        detail = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"Ollama request failed for {model_name}: {exc}{suffix}") from exc
            return result.get("message", {}).get("content", "").strip()
        model = self.load()
        import torch
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        encoded = self.tokenizer(rendered, return_tensors="pt", truncation=True,
                                 max_length=self.context_limit - self.max_new_tokens)
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                                    num_beams=1, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _tool_specs(available_tools: list[str], catalog: dict) -> list[dict]:
        selected = set(available_tools)
        compact = []
        for tool in catalog.get("apps", []):
            for spec in tool.get("tools", []):
                if spec.get("name") not in selected:
                    continue
                params = spec.get("parameters", {})
                properties = {}
                for name, prop in params.get("properties", {}).items():
                    properties[name] = {key: prop[key] for key in
                                         ("type", "description", "enum", "format", "minimum", "maximum",
                                          "exclusiveMinimum", "exclusiveMaximum", "x-ref", "x-match", "items")
                                         if key in prop}
                compact.append({"name": spec["name"], "description": spec.get("description", "")[:160],
                                "required": params.get("required", []), "one_of": params.get("x-one-of", []),
                                "args": properties})
        return compact

    def next_action(self, transcript: str, context: dict, history: list, available_tools: list[str], observations: list):
        allowed = set(available_tools)
        tool_specs = self._tool_specs(available_tools, self.catalog)
        system = (
            "Bạn là agent LifeMate tiếng Việt. Dùng transcript cuối cùng, context, history và tool schema để thực hiện yêu cầu. "
            "Chỉ gọi tool đã cài và có trong schema. Không bịa dữ liệu; thiếu thông tin thiết yếu thì gọi sys.ask_missing_info. "
            "Sau khi hoàn tất, trả action stop. Mỗi lần chỉ trả đúng một JSON object, không giải thích. "
            'Ví dụ: {"action":"tool_call","tool":"app.tool","arguments":{}} hoặc {"action":"stop"}. /no_think'
        )
        # installed_apps only repeats tool names; the separate schema list above is authoritative.
        compact_context = {key: value for key, value in context.items() if key != "installed_apps"}
        payload = {"transcript": transcript, "context": compact_context, "history": history,
                   "available_tools": tool_specs, "observations": observations}
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
        raw = self._generate(messages)
        try:
            action = parse_action(raw)
            valid, error = validate_action(action, self.catalog, allowed)
            if valid:
                return action
        except (ValueError, TypeError) as exc:
            error = str(exc)
        # A single bounded repair attempt; never repair semantic argument values in code.
        messages.extend([{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Output không hợp lệ ({error}). Hãy sửa format/schema một lần; không tự đoán giá trị. Chỉ trả JSON action."}])
        try:
            action = parse_action(self._generate(messages))
            valid, _ = validate_action(action, self.catalog, allowed)
            return action if valid else {"action": "stop"}
        except (ValueError, TypeError):
            return {"action": "stop"}

