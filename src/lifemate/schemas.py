from __future__ import annotations

import json
import re
from typing import Any


def parse_action(text: str) -> dict[str, Any]:
    """Parse one JSON object, allowing a surrounding Markdown code fence."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Model output did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or value.get("action") not in {"tool_call", "stop"}:
        raise ValueError("Expected action=tool_call or action=stop")
    if value["action"] == "stop":
        return {"action": "stop"}
    if not isinstance(value.get("tool"), str) or not isinstance(value.get("arguments", {}), dict):
        raise ValueError("tool_call requires a string tool and object arguments")
    return {"action": "tool_call", "tool": value["tool"], "arguments": value.get("arguments", {})}


def validate_action(action: dict[str, Any], catalog: dict, allowed: set[str]) -> tuple[bool, str | None]:
    if action.get("action") == "stop":
        return True, None
    tool = action.get("tool")
    if tool not in allowed:
        return False, f"tool {tool!r} is not installed for this sample"
    specs = {t["name"]: t for app in catalog.get("apps", []) for t in app.get("tools", [])}
    spec = specs.get(tool)
    if not spec:
        return False, f"tool {tool!r} is absent from catalog"
    params = spec.get("parameters", {})
    args = action.get("arguments", {})
    for field in params.get("required", []):
        if field not in args or args[field] in (None, "", [], {}):
            return False, f"missing required argument: {field}"
    groups = params.get("x-one-of", [])
    if groups and not any(all(args.get(k) not in (None, "", [], {}) for k in group) for group in groups):
        return False, "missing one-of argument group"
    for field, value in args.items():
        p = params.get("properties", {}).get(field)
        if not p or value in (None, "", [], {}):
            continue
        expected = p.get("type")
        good = {"string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), "boolean": lambda v: isinstance(v, bool),
                "array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict)}.get(expected, lambda _: True)(value)
        if not good:
            return False, f"argument {field} must have type {expected}"
        if p.get("enum") and not any(value == choice and type(value) is type(choice) for choice in p["enum"]):
            return False, f"argument {field} must be one of {p['enum']}"
    return True, None

