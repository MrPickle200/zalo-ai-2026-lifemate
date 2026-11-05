from __future__ import annotations

import json


def serialize_router_input(payload: dict, step: int) -> str:
    """Compact production-like routing context; current/future gold args are excluded."""
    fields = [f"TRANSCRIPT:\n{payload.get('transcript', '')}",
              f"CONTEXT:\n{json.dumps(payload.get('context', {}), ensure_ascii=False, default=str)}",
              f"HISTORY:\n{json.dumps(payload.get('history', []), ensure_ascii=False, default=str)}",
              f"AVAILABLE TOOLS:\n{', '.join(t.get('name', '') for t in payload.get('available_tools', []))}",
              f"PREVIOUS ACTIONS AND OBSERVATIONS:\n{json.dumps(payload.get('observations', []), ensure_ascii=False, default=str)}",
              f"STEP: {step}", "QUESTION:\nWhat is the next tool to call?"]
    return "\n\n".join(fields)
