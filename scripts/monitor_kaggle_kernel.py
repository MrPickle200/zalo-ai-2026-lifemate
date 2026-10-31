from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}


def load_kaggle_token() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    env_path = Path(".env")
    if not env_path.is_file():
        raise RuntimeError("KAGGLE_API_TOKEN is not set and .env was not found")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("KAGGLE_API_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                os.environ["KAGGLE_API_TOKEN"] = value
                return
    raise RuntimeError("KAGGLE_API_TOKEN is not set in .env")


def run_cli(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, check=False)


def kernel_status(kernel_ref: str) -> tuple[str, str]:
    result = run_cli(["kaggle", "kernels", "status", kernel_ref], timeout=120)
    output = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if result.returncode or not match:
        raise RuntimeError(f"Could not parse Kaggle kernel status (exit={result.returncode}):\n{output[-3000:]}")
    return match.group(1), output


def start_live_logs(kernel_ref: str, output_dir: Path):
    log_path = output_dir / "execution.log"
    process = subprocess.Popen(["kaggle", "kernels", "logs", "-f", kernel_ref],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)

    def copy_stream():
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            if process.stdout is not None:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()

    thread = threading.Thread(target=copy_stream, name="kaggle-live-log-copy", daemon=True)
    thread.start()
    return process, thread, log_path


def stop_live_logs(process, thread) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    thread.join(timeout=5)


def collect_artifacts(kernel_ref: str, output_dir: Path) -> str:
    result = run_cli(["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir.resolve())], timeout=600)
    return (result.stdout + "\n" + result.stderr).strip()


def log_tail(path: Path | None, limit: int = 100) -> list[str]:
    if path is None:
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, dict):
                    data = item.get("data", "")
                    if data:
                        lines.extend(str(data).splitlines())
            return lines[-limit:]
    except json.JSONDecodeError:
        pass
    return raw.splitlines()[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll a Kaggle kernel and collect its run log")
    parser.add_argument("--kernel-ref", default="daoviet/lifemate-qwen-next-action-qlora-v1")
    parser.add_argument("--output-dir", default="outputs/training/qwen-next-action-v1/kaggle-monitor")
    parser.add_argument("--interval-seconds", type=int, default=300,
                        help="poll interval; default is five minutes")
    parser.add_argument("--once", action="store_true", help="query status and collect a log one time")
    parser.add_argument("--max-polls", type=int, default=0, help="optional cap; zero means continue until terminal")
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.max_polls < 0:
        parser.error("interval-seconds must be positive and max-polls cannot be negative")
    load_kaggle_token()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    snapshots = outdir / "status.jsonl"
    previous = None
    polls = 0
    live_process = live_thread = None
    log_path = outdir / "execution.log"
    if not args.once:
        log_path.write_text("", encoding="utf-8")
        live_process, live_thread, log_path = start_live_logs(args.kernel_ref, outdir)
    else:
        result = run_cli(["kaggle", "kernels", "logs", args.kernel_ref], timeout=300)
        if result.stdout.strip():
            log_path.write_text(result.stdout, encoding="utf-8")
    try:
        while True:
            polls += 1
            status, status_output = kernel_status(args.kernel_ref)
            artifact_output = collect_artifacts(args.kernel_ref, outdir) if status == "COMPLETE" else ""
            snapshot = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "kernel_ref": args.kernel_ref,
                        "status": status, "log_path": str(log_path) if log_path.is_file() else None,
                        "status_output": status_output[-1500:], "artifact_output": artifact_output[-1500:]}
            with snapshots.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            if status != previous:
                print(json.dumps({"event": "status_change", "status": status,
                                  "timestamp_utc": snapshot["timestamp_utc"],
                                  "log_path": snapshot["log_path"]}, ensure_ascii=False), flush=True)
                previous = status
            elif args.once:
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            if status not in TERMINAL and live_process is not None and live_process.poll() is not None:
                print(json.dumps({"event": "log_follow_restart", "exit_code": live_process.returncode,
                                  "timestamp_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      flush=True)
                live_process, live_thread, log_path = start_live_logs(args.kernel_ref, outdir)
            if status in TERMINAL:
                if live_process is not None:
                    stop_live_logs(live_process, live_thread)
                if status == "ERROR":
                    print("CODEX_ACTION_REQUIRED: Kaggle kernel failed. Latest run log follows:", flush=True)
                    for line in log_tail(log_path if log_path.is_file() else None):
                        print(line, flush=True)
                    return 1
                print(json.dumps({"event": "terminal", "status": status, "log_path": snapshot["log_path"]},
                                 ensure_ascii=False), flush=True)
                return 0
            if args.once or (args.max_polls and polls >= args.max_polls):
                return 0
            time.sleep(args.interval_seconds)
    finally:
        if live_process is not None:
            stop_live_logs(live_process, live_thread)


if __name__ == "__main__":
    raise SystemExit(main())
