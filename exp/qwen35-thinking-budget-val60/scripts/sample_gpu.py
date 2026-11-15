from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output" / "gpu_samples.csv"
FIELDS = ("utilization.gpu", "memory.used", "temperature.gpu", "power.draw")
OUT.parent.mkdir(parents=True, exist_ok=True)
new_file = not OUT.exists()
with OUT.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    if new_file:
        writer.writerow(("timestamp_local", "utilization_gpu_pct", "memory_used_mib", "temperature_c", "power_w"))
    handle.flush()
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
            writer.writerow((datetime.now().astimezone().isoformat(timespec="seconds"), *values))
            handle.flush()
        time.sleep(5)
