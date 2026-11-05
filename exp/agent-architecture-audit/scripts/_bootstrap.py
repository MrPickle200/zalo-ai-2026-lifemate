from __future__ import annotations
import sys
from pathlib import Path
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "lifemate").is_dir())
sys.path.insert(0, str(ROOT / "src"))
