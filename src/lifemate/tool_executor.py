from __future__ import annotations

import sys
from pathlib import Path


def create_tool_env(split: str, sample_id: str, data_root: Path):
    """Construct exactly one official ToolEnv session for this sample."""
    package_root = data_root / split
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from zaic_tools import ToolEnv
    return ToolEnv(split, sample_id, root=data_root)


def available_tool_names(item: dict) -> list[str]:
    names = {"sys.ask_missing_info"}
    for app in item.get("context", {}).get("installed_apps", {}).values():
        names.update(app.get("tools", []))
    return sorted(names)

