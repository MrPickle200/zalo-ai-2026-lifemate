"""Development-only downloader for competition-eligible public model revisions.

The HF token is read from .env, kept in memory, and passed directly to Hugging
Face Hub. Inference code never downloads models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import _bootstrap  # noqa: F401


def env_token(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Token file not found: {path}")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            token = value.strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError("HF_TOKEN is missing or empty in .env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned, competition-eligible public model revisions")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    args = parser.parse_args()
    if args.asr_only and args.llm_only:
        parser.error("choose at most one of --asr-only and --llm-only")
    token = env_token(Path(args.env_file))
    root = Path(args.models_dir)
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    specs = []
    if not args.llm_only:
        specs.append(("Systran/faster-whisper-small", "536b066", root / "faster-whisper-small"))
    if not args.asr_only:
        specs.append(("Qwen/Qwen3-4B-Instruct-2507", "0c396dc021fde326c257328889b5c7e2e2d20c0a",
                      root / "Qwen3-4B-Instruct-2507"))
    for repo, revision, local_dir in specs:
        print(f"Downloading {repo} at pinned revision {revision} to {local_dir}")
        try:
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(local_dir), token=token)
        except Exception as exc:
            # Do not display request headers, environment values, or credential material.
            print(f"Could not download {repo}: {type(exc).__name__}: {str(exc).splitlines()[0]}", file=sys.stderr)
            return 1
        print(f"Completed {repo}")
    print("Download complete. Keep models/ out of git; inference remains local-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
