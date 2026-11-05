from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path


def normalize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score_rows(rows: list[dict]) -> dict:
    word_edits = word_count = char_edits = char_count = exact = 0
    utterance_wer = []
    latencies = []
    rtfs = []
    for row in rows:
        ref, hyp = normalize(row.get("reference") or ""), normalize(row.get("prediction") or "")
        word_edits += edit_distance(ref, hyp)
        word_count += len(ref)
        rc, hc = list("".join(ref)), list("".join(hyp))
        char_edits += edit_distance(rc, hc)
        char_count += len(rc)
        exact += ref == hyp
        utterance_wer.append(edit_distance(ref, hyp) / max(1, len(ref)))
        if row.get("inference_seconds") is not None:
            latencies.append(row["inference_seconds"])
            duration = row.get("audio_seconds")
            if duration and duration > 0:
                rtfs.append(row["inference_seconds"] / duration)
    sorted_latency = sorted(latencies)
    p95_index = min(len(sorted_latency) - 1, math.ceil(0.95 * len(sorted_latency)) - 1) if sorted_latency else 0
    return {
        "samples": len(rows),
        "corpus_wer": word_edits / max(1, word_count),
        "corpus_cer": char_edits / max(1, char_count),
        "macro_utterance_wer": statistics.mean(utterance_wer) if utterance_wer else None,
        "normalized_exact_match_rate": exact / len(rows) if rows else None,
        "mean_inference_seconds": statistics.mean(latencies) if latencies else None,
        "median_inference_seconds": statistics.median(latencies) if latencies else None,
        "p95_inference_seconds": sorted_latency[p95_index] if sorted_latency else None,
        "mean_real_time_factor": statistics.mean(rtfs) if rtfs else None,
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_validation(data_root: Path, ids_path: Path) -> list[dict]:
    ids = json.loads(ids_path.read_text(encoding="utf-8-sig"))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {ids_path}")
    manifest = data_root / "train" / "train.jsonl"
    samples = {row["id"]: row for row in read_jsonl(manifest)}
    missing = [sample_id for sample_id in ids if sample_id not in samples]
    if missing:
        raise ValueError(f"{len(missing)} validation IDs are missing from {manifest}: {missing[:5]}")
    return [samples[sample_id] for sample_id in ids]


def audio_duration(path: Path) -> float | None:
    try:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            if container.duration is not None:
                return float(container.duration) / av.time_base
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return None


def read_audio_16k(path: Path):
    import av
    import numpy as np
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(frame.to_ndarray() for frame in resampler.resample(frame))
        chunks.extend(frame.to_ndarray() for frame in resampler.resample(None))
    if not chunks:
        raise ValueError(f"No audio frames decoded: {path}")
    return np.concatenate(chunks, axis=-1).reshape(-1)
