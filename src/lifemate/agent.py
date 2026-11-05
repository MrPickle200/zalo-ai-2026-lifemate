from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import parse_action, validate_action


class Agent:
    """Small, greedy local causal LLM for structured tool actions."""

    def __init__(self, model_path: str | Path | None = None, adapter_path: str | Path | None = None,
                 max_new_tokens: int | None = None, context_limit: int | None = None,
                 ollama_thinking: bool | None = None, ollama_native_tools: bool | None = None):
        self.model_path = Path(model_path or os.environ.get("LIFEMATE_LLM_MODEL", "models/Qwen3-4B-Instruct-2507")).expanduser()
        adapter = adapter_path or os.environ.get("LIFEMATE_LLM_ADAPTER")
        self.adapter_path = Path(adapter).expanduser() if adapter else None
        default_tokens = 512 if os.environ.get("LIFEMATE_LLM_BACKEND", "transformers").lower() == "ollama" else 128
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else default_tokens
        context_limit = context_limit if context_limit is not None else int(os.environ.get("LIFEMATE_LLM_CONTEXT_LIMIT", "6144"))
        self.context_limit = min(context_limit, 16384)
        self.tokenizer = None
        self.model = None
        self.catalog: dict = {}
        self.ollama_thinking = (os.environ.get("LIFEMATE_OLLAMA_THINK", "true").lower() in {"1", "true", "yes", "on"}
                                if ollama_thinking is None else ollama_thinking)
        protocol = os.environ.get("LIFEMATE_OLLAMA_PROTOCOL", "json").strip().lower()
        self.ollama_native_tools = (protocol in {"native", "tools", "tool_calling"}
                                    if ollama_native_tools is None else ollama_native_tools)
        self.last_request_diagnostics: dict[str, Any] = {}
        self.native_history: list[dict[str, Any]] = []
        self._last_native_messages: list[dict[str, Any]] = []
        self._last_native_tools: list[dict[str, Any]] = []

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
            if self.adapter_path is not None and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise RuntimeError("Loading a QLoRA adapter on CUDA requires bitsandbytes") from exc
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                # The adapter was trained against a 4-bit NF4 base. Keeping the
                # complete quantized base on CUDA also avoids PEFT hooks crossing
                # CPU/GPU offload boundaries on small local cards.
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), local_files_only=True, trust_remote_code=False,
                    quantization_config=quantization, torch_dtype="auto",
                    low_cpu_mem_usage=True, device_map={"": 0},
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), local_files_only=True, trust_remote_code=False,
                    torch_dtype="auto", low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    max_memory=max_memory,
                )
            if self.adapter_path is not None:
                if not self.adapter_path.is_dir():
                    raise FileNotFoundError(f"LoRA adapter directory not found: {self.adapter_path}")
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise RuntimeError("Loading a LoRA adapter requires peft") from exc
                self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path),
                                                       is_trainable=False, local_files_only=True)
            n_params = sum(p.numel() for p in self.model.parameters())
            if n_params >= 5_000_000_000:
                self.model = None
                raise ValueError(f"LLM has {n_params:,} parameters; competition limit is below 5B")
            self.model.eval()
        return self.model

    def _ollama_request(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> dict:
        model_name = os.environ.get("LIFEMATE_OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
        ollama_context = min(int(os.environ.get("LIFEMATE_OLLAMA_CONTEXT", "8192")), 16384)
        body = {"model": model_name, "messages": messages, "stream": False,
                "think": self.ollama_thinking,
                "options": {"temperature": 0, "num_predict": int(os.environ.get("LIFEMATE_OLLAMA_NUM_PREDICT", self.max_new_tokens)),
                            "num_ctx": ollama_context}}
        if tools:
            body["tools"] = tools
        else:
            body["format"] = "json"
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat",
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            detail = ""
            if hasattr(exc, "read"):
                try: detail = exc.read().decode("utf-8", errors="replace")
                except Exception: pass
            raise RuntimeError(f"Ollama request failed for {model_name}: {exc}{': ' + detail if detail else ''}") from exc
        self.last_request_diagnostics = {"model": model_name, "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"), "total_duration_ns": result.get("total_duration"),
            "load_duration_ns": result.get("load_duration"), "latency_seconds": time.perf_counter() - started,
            "thinking_chars": len(result.get("message", {}).get("thinking") or ""),
            "request_roles": [m.get("role") for m in messages],
            "request_content_chars": sum(len(str(m.get("content", ""))) for m in messages),
            "request_tool_schema_count": len(tools or []),
            "response_content_chars": len(result.get("message", {}).get("content") or ""),
            "response_tool_names": [c.get("function", {}).get("name") for c in result.get("message", {}).get("tool_calls", [])],
            "done_reason": result.get("done_reason")}
        return result

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if os.environ.get("LIFEMATE_LLM_BACKEND", "transformers").lower() == "ollama":
            result = self._ollama_request(messages)
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

    def next_action(self, transcript: str, context: dict, history: list, available_tools: list[str], observations: list,
                    candidate_tools: list[str] | None = None, few_shot_examples: list[dict] | None = None,
                    missing_info_demos: list[dict] | None = None, missing_info_rule: bool = False,
                    forced_tool_name: str | None = None):
        allowed = set(available_tools)
        selected_tools = [name for name in candidate_tools if name in allowed] if candidate_tools is not None else available_tools
        if forced_tool_name is not None:
            if forced_tool_name not in allowed:
                raise ValueError(f"forced diagnostic tool {forced_tool_name!r} is not installed for this sample")
            selected_tools = [forced_tool_name]
        model_allowed = set(selected_tools)
        tool_specs = self._tool_specs(selected_tools, self.catalog)
        native_tools = []
        schema_extensions = []
        if self.ollama_native_tools:
            catalog_by_name = {spec["name"]: spec for app in self.catalog.get("apps", []) for spec in app.get("tools", [])}
            for tool_name in selected_tools:
                spec = catalog_by_name.get(tool_name, {})
                params = spec.get("parameters", {})
                properties = params.get("properties", {})
                clean_properties = {}
                for key, value in properties.items():
                    kept = {k: v for k, v in value.items() if k in {
                        "type", "description", "enum", "format", "minimum", "maximum",
                        "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                        "minItems", "maxItems", "uniqueItems", "pattern", "items", "properties", "required"}}
                    unsupported = sorted(set(value) - set(kept))
                    if unsupported: schema_extensions.append({"tool": tool_name, "field": key, "extensions": unsupported})
                    clean_properties[key] = kept
                unsupported_top = sorted(set(params) - {"type", "properties", "required", "additionalProperties"})
                if "x-one-of" in params:
                    clean_properties_schema = {"type": "object", "properties": clean_properties,
                                               "required": params.get("required", []),
                                               "additionalProperties": params.get("additionalProperties", False),
                                               "anyOf": [{"required": group} for group in params["x-one-of"]]}
                    unsupported_top = [x for x in unsupported_top if x != "x-one-of"]
                else:
                    clean_properties_schema = {"type": "object", "properties": clean_properties,
                                               "required": params.get("required", []),
                                               "additionalProperties": params.get("additionalProperties", False)}
                if unsupported_top: schema_extensions.append({"tool": tool_name, "schema_extensions": unsupported_top})
                native_tools.append({"type": "function", "function": {"name": tool_name,
                    "description": spec.get("description", ""), "parameters": clean_properties_schema}})
        dual_transcript = transcript.startswith("<DUAL_TRANSCRIPT>\n")
        prompt_language = os.environ.get("LIFEMATE_PROMPT_LANGUAGE", "vi").strip().lower()
        if prompt_language == "vi":
            system = (
                "Bạn là agent LifeMate, xử lý yêu cầu bằng tiếng Việt thông qua các tool được cung cấp. "
                "Mục tiêu là hoàn thành đúng toàn bộ yêu cầu hiện tại, không chỉ thực hiện bước đầu tiên. "
                "Căn cứ vào transcript hiện tại, context/profile, history, available tools và mọi observations từ các lượt trước. "
                "Chỉ dùng tool có trong schema của mẫu; đọc kỹ mô tả, required fields, kiểu dữ liệu và enum. "
                "Không gọi tool gần giống nếu không đáp ứng yêu cầu; nếu không có tool phù hợp hoặc yêu cầu không cần hành động thì trả stop. "
                "Trước mỗi lượt, xác định yêu cầu nào còn chưa hoàn thành và chọn đúng bước tiếp theo cần thiết. "
                "Sau mỗi tool call, đọc observation mới nhất: kiểm tra thành công/thất bại, lấy chính xác ID, giá trị và dữ liệu trả về để quyết định bước sau hoặc điền arguments. "
                "Không đoán, tự tạo hoặc thay đổi ID/giá trị mà tool đã trả về; không lặp lại action đã hoàn thành. "
                "Không dừng sau một bước tra cứu nếu yêu cầu còn cần hành động tiếp theo. Chỉ stop khi mọi phần yêu cầu đã hoàn tất, hoặc không thể tiến hành an toàn bằng các tool hiện có. "
                "Nếu thiếu required argument, trước hết kiểm tra transcript, context/profile, history và observations xem dữ kiện đã có chưa. "
                "Chỉ gọi sys.ask_missing_info khi dữ kiện thực sự còn thiếu và không thể suy ra an toàn; hỏi đúng các trường còn thiếu, không hỏi lại dữ kiện đã có. "
                "Nếu transcript có vẻ nhận dạng sai, dùng ngữ cảnh để hiểu ý định nhưng không tự bịa chi tiết quan trọng; hỏi thêm nếu cách hiểu làm thay đổi tool hoặc arguments. "
                "Tránh tool call thừa, trùng lặp hoặc gây thay đổi trạng thái ngoài yêu cầu. "
                "Mỗi lượt chỉ trả đúng một JSON object, không markdown, không giải thích: "
                '{"action":"tool_call","tool":"app.tool","arguments":{}} hoặc {"action":"stop"}. /no_think'
            )
        elif prompt_language == "en":
            system = (
                "You are LifeMate, a Vietnamese-language agent that fulfills the user's request with the available tools. "
                "Complete the entire current request, not just its first step. Use the current transcript, context/profile, history, available tools, and all observations from previous turns. "
                "Call only tools available in this sample and described by their schemas; check descriptions, required fields, types, and enums. "
                "Do not call a similar tool that does not satisfy the request. If no available tool applies or no action is needed, return stop. "
                "Before each turn, identify what remains unfinished and choose the next necessary action. "
                "After every tool call, inspect the newest observation for success or failure. Use returned IDs, values, and data exactly to choose the next action and fill its arguments. "
                "Never guess, invent, or alter an ID/value returned by a tool, and do not repeat a completed action. "
                "Do not stop after a lookup when the request requires a follow-up action. Return stop only when every part of the request is complete or cannot be carried out safely with the available tools. "
                "If a required argument is missing, first check whether it is already available in the transcript, context/profile, history, or observations. "
                "Call sys.ask_missing_info only when required information is genuinely missing and cannot be safely inferred; ask only for the missing fields and never ask again for information already available. "
                "If the transcript appears misrecognized, use context to understand the intent without inventing important details; ask when ambiguity would change the tool or its arguments. "
                "Avoid unnecessary or duplicate calls and state changes outside the request. "
                "Return exactly one JSON object per turn, with no Markdown or explanation: "
                '{"action":"tool_call","tool":"app.tool","arguments":{}} or {"action":"stop"}. /no_think'
            )
        else:
            raise ValueError("LIFEMATE_PROMPT_LANGUAGE must be 'vi' or 'en'")
        if self.ollama_thinking:
            system = system.replace(" /no_think", "")
        if self.ollama_native_tools:
            system = system.replace(
                'Return exactly one JSON object per turn, with no Markdown or explanation: '
                '{"action":"tool_call","tool":"app.tool","arguments":{}} or {"action":"stop"}.',
                "Use the native function calling interface to call exactly one appropriate tool. If no action is needed or no tool applies, return no tool call.")
            system = system.replace(
                'Mỗi lượt chỉ trả đúng một JSON object, không markdown, không giải thích: '
                '{"action":"tool_call","tool":"app.tool","arguments":{}} hoặc {"action":"stop"}.',
                "Dùng giao diện function calling gốc để gọi đúng một tool phù hợp. Nếu không cần hành động hoặc không có tool phù hợp, không gọi tool nào.")
        if forced_tool_name is not None:
            system += (f" Lượt diagnostic đầu tiên đã cố định tool là {forced_tool_name}. "
                       "Bắt buộc trả action tool_call với đúng tool này, tự tạo arguments chỉ từ transcript, context, "
                       "history và schema; không dùng arguments từ nhãn và không trả stop.")
        if dual_transcript:
            system += (
                " Transcript A và B là hai nhận dạng của cùng một câu nói. Hãy suy ra yêu cầu từ thông tin nhất quán giữa hai bản. "
                "Nếu khác nhau ở chi tiết quan trọng, không tự bịa hoặc tự chọn một giá trị thiếu căn cứ; "
                "dựa trên yêu cầu hiện tại và tool schema để quyết định tool call."
            )
        if missing_info_rule:
            system += (
                " Nếu đã xác định được tool phù hợp nhưng thiếu required argument và không thể suy ra an toàn từ context/history, "
                "không đoán; gọi sys.ask_missing_info với tool_name và params cần hỏi. Nếu request đã đủ thông tin thì không hỏi lại. "
                + ("Toàn bộ tool schema đã cho đều có thể xem xét." if candidate_tools is None else "Chỉ gọi các tool có schema trong danh sách rút gọn.")
            )
        examples = []
        seen_examples = set()
        for source, docs in (("retrieved", few_shot_examples or []), ("missing-info", missing_info_demos or [])):
            for doc in docs:
                sample_id = doc.get("sample_id")
                if sample_id in seen_examples:
                    continue
                seen_examples.add(sample_id)
                examples.append({"source": source, "category": doc.get("category"),
                                 "transcript": doc.get("transcript", ""),
                                 "expected_tool_calls": doc.get("expected_tool_calls", [])})
        if examples:
            lines = ["Retrieved examples are references only. Do not copy their arguments unless they are explicitly present in the current request. Always decide the correct tool from the current request and available tool schemas.",
                     "Ví dụ tham khảo từ train:"]
            for example in examples:
                lines.append("Example:\n\nUser request:\n" + example["transcript"])
                lines.append("\nCorrect tool calls:\n" + json.dumps(example["expected_tool_calls"], ensure_ascii=False, separators=(",", ":")))
            system += "\n" + "\n".join(lines)
        # installed_apps only repeats tool names; the separate schema list above is authoritative.
        compact_context = {key: value for key, value in context.items() if key != "installed_apps"}
        payload = {"transcript": transcript, "context": compact_context, "history": history,
                   "available_tools": tool_specs, "observations": [] if self.ollama_native_tools else observations}
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
        if self.ollama_native_tools:
            messages.extend(self.native_history)
            self._last_native_messages = list(messages)
            self._last_native_tools = native_tools
            response = self._ollama_request(messages, native_tools)
            model_message = response.get("message", {})
            tool_calls = model_message.get("tool_calls") or []
            self.last_request_diagnostics.update({"native_call_count": len(tool_calls), "schema_extensions": schema_extensions})
            if not tool_calls:
                return {"action": "stop", "_diagnostics": dict(self.last_request_diagnostics)}
            first = tool_calls[0].get("function", {})
            history_message = {**model_message, "tool_calls": [tool_calls[0]]}
            args = first.get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except json.JSONDecodeError: args = args
            action = {"action": "tool_call", "tool": first.get("name"), "arguments": args,
                      "_native_message": history_message, "_diagnostics": dict(self.last_request_diagnostics)}
            valid, error = validate_action(action, self.catalog, model_allowed) if isinstance(args, dict) else (False, "native arguments are not an object")
            if valid:
                return action
            repair_messages = messages + [model_message, {"role": "user", "content": f"Invalid tool call: {error}. Make one corrected tool call. Do not guess or change semantic values."}]
            initial_response = response
            repaired = self._ollama_request(repair_messages, native_tools)
            repaired_message = repaired.get("message", {})
            calls = repaired_message.get("tool_calls") or []
            self.last_request_diagnostics.update({"repair_attempted": True, "initial_schema_error": error,
                "repair_call_count": len(calls), "request_count": 2,
                "initial_prompt_eval_count": initial_response.get("prompt_eval_count"),
                "initial_eval_count": initial_response.get("eval_count"), "schema_extensions": schema_extensions})
            if not calls:
                return {"action": "stop", "_diagnostics": {**self.last_request_diagnostics,
                        "repair_attempted": True, "repair_succeeded": False, "initial_schema_error": error}}
            fn = calls[0].get("function", {}); repaired_args = fn.get("arguments", {})
            repaired_history_message = {**repaired_message, "tool_calls": [calls[0]]}
            if isinstance(repaired_args, str):
                try: repaired_args = json.loads(repaired_args)
                except json.JSONDecodeError: repaired_args = repaired_args
            repaired_action = {"action": "tool_call", "tool": fn.get("name"), "arguments": repaired_args,
                               "_native_message": repaired_history_message, "_diagnostics": dict(self.last_request_diagnostics)}
            ok, _ = validate_action(repaired_action, self.catalog, model_allowed) if isinstance(repaired_args, dict) else (False, None)
            if ok:
                repaired_action["_diagnostics"].update({"repair_succeeded": True})
                return repaired_action
            return {"action": "stop", "_diagnostics": {**self.last_request_diagnostics,
                    "repair_attempted": True, "repair_succeeded": False, "initial_schema_error": error}}
        raw = self._generate(messages)
        generation_diagnostics = dict(self.last_request_diagnostics)
        try:
            action = parse_action(raw)
            valid, error = validate_action(action, self.catalog, model_allowed)
            if valid and not (forced_tool_name is not None and action.get("action") == "stop"):
                if generation_diagnostics:
                    action["_diagnostics"] = generation_diagnostics
                return action
            if valid:
                error = f"diagnostic requires a tool_call to {forced_tool_name}"
        except (ValueError, TypeError) as exc:
            error = str(exc)
        # A single bounded repair attempt; never repair semantic argument values in code.
        messages.extend([{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Output không hợp lệ ({error}). Hãy sửa format/schema một lần; không tự đoán giá trị. Chỉ trả JSON action."}])
        try:
            action = parse_action(self._generate(messages))
            repair_diagnostics = dict(self.last_request_diagnostics)
            valid, _ = validate_action(action, self.catalog, model_allowed)
            valid_forced = forced_tool_name is None or action.get("action") == "tool_call"
            if valid and valid_forced:
                if self.last_request_diagnostics:
                    action["_diagnostics"] = {**repair_diagnostics, "request_count": 2,
                        "initial_prompt_eval_count": generation_diagnostics.get("prompt_eval_count"),
                        "initial_eval_count": generation_diagnostics.get("eval_count"),
                        "repair_attempted": True, "repair_succeeded": True,
                        "initial_schema_error": error}
                return action
            return {"action": "stop", "_diagnostics": {**repair_diagnostics, "request_count": 2,
                    "initial_prompt_eval_count": generation_diagnostics.get("prompt_eval_count"),
                    "initial_eval_count": generation_diagnostics.get("eval_count"), "repair_attempted": True,
                    "repair_succeeded": False, "initial_schema_error": error}}
        except (ValueError, TypeError):
            return {"action": "stop", "_diagnostics": {**generation_diagnostics, "request_count": 2,
                    "initial_prompt_eval_count": generation_diagnostics.get("prompt_eval_count"),
                    "initial_eval_count": generation_diagnostics.get("eval_count"), "repair_attempted": True,
                    "repair_succeeded": False, "initial_schema_error": error}}

    def record_tool_result(self, action: dict[str, Any], result: Any) -> None:
        if not self.ollama_native_tools:
            return
        message = action.get("_native_message")
        if message:
            self.native_history.append({"role": "assistant", **message})
            self.native_history.append({"role": "tool", "tool_name": action.get("tool"),
                                        "content": json.dumps(result, ensure_ascii=False, default=str)})

    def repair_native_toolenv_rejection(self, action: dict[str, Any], error: Any) -> dict[str, Any]:
        """Allow one model repair when ToolEnv rejects a native call's full catalog schema."""
        message = action.get("_native_message")
        if not self.ollama_native_tools or not message:
            return {"action": "stop", "_diagnostics": {"repair_attempted": False}}
        error_text = json.dumps(error, ensure_ascii=False, default=str)
        self.native_history.extend([{"role": "assistant", **message},
                                    {"role": "tool", "tool_name": action.get("tool"), "content": error_text}])
        repair_messages = self._last_native_messages + [
            {"role": "assistant", **message},
            {"role": "tool", "tool_name": action.get("tool"), "content": error_text},
            {"role": "user", "content": "ToolEnv rejected that call against the full tool schema. Correct the schema only; do not guess or alter semantic values."}]
        response = self._ollama_request(repair_messages, self._last_native_tools)
        calls = response.get("message", {}).get("tool_calls") or []
        initial_diag = action.get("_diagnostics", {})
        diag = {**self.last_request_diagnostics, "repair_attempted": True, "request_count": 2,
                "initial_prompt_eval_count": initial_diag.get("prompt_eval_count"),
                "initial_eval_count": initial_diag.get("eval_count"),
                "schema_extensions": initial_diag.get("schema_extensions", []),
                "repair_source": "ToolEnv full-schema validation", "initial_schema_error": error,
                "repair_call_count": len(calls), "native_call_count": len(calls),
                "native_overcalls": max(0, len(calls) - 1)}
        if not calls:
            return {"action": "stop", "_diagnostics": {**diag, "repair_succeeded": False}}
        repaired_message = response["message"]
        history_message = {**repaired_message, "tool_calls": [calls[0]]}
        fn = calls[0].get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except json.JSONDecodeError: args = args
        repaired = {"action": "tool_call", "tool": fn.get("name"), "arguments": args,
                    "_native_message": history_message, "_diagnostics": diag}
        allowed = {t["function"]["name"] for t in self._last_native_tools}
        valid, _ = validate_action(repaired, self.catalog, allowed) if isinstance(args, dict) else (False, None)
        if valid:
            repaired["_diagnostics"]["repair_succeeded"] = True
            return repaired
        return {"action": "stop", "_diagnostics": {**diag, "repair_succeeded": False,
                "repair_error": "corrected call failed local name/argument validation"}}

