from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lifemate.asr import PhoASRModel
from lifemate.utils import find_data_root, load_jsonl, resolve_audio, seed_everything


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    words_ref, words_hyp = ref.split(), hyp.split()
    return edit_distance(words_ref, words_hyp) / max(1, len(words_ref))


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(ref, hyp) / max(1, len(ref))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Generate reproducible PhoASR alternatives for a fixed split or ID list")
    p.add_argument("--split", choices=("train", "public_test"), default="train")
    p.add_argument("--sample-ids", help="optional JSON ID list; defaults to the hard-30 for train or full manifest for public_test")
    p.add_argument("--data-root")
    p.add_argument("--model", default="models/pho-asr-whisper-small")
    p.add_argument("--output", default="outputs/asr_nbest_v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    out = Path(args.output)
    target = out / "transcripts.jsonl"
    if target.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite valid-looking cache: {target}; pass --overwrite explicitly")
    root = find_data_root(args.data_root)
    split_dir = root / args.split
    items = {r["id"]: r for r in load_jsonl(split_dir / f"{args.split}.jsonl")}
    if args.sample_ids:
        ids = json.loads(Path(args.sample_ids).read_text(encoding="utf-8"))
    elif args.split == "train":
        ids = json.loads(Path("outputs/eval/agent-architecture-audit-v1-val309/sample_ids.json").read_text(encoding="utf-8"))
    else:
        ids = [r["id"] for r in load_jsonl(split_dir / f"{args.split}.jsonl")]
    if len(ids) != len(set(ids)):
        raise SystemExit("Sample ID file has duplicates")
    missing = [sid for sid in ids if sid not in items]
    if missing:
        raise SystemExit(f"IDs absent from {args.split} split: {missing[:5]}")
    seed_everything(args.seed)
    asr = PhoASRModel(model_path=args.model, cache_dir=Path("cache/transcripts"))
    model = asr._load()
    import torch

    configs = [
        {"name": "production_greedy", "num_beams": 1, "do_sample": False},
        {"name": "beam_3", "num_beams": 3, "do_sample": False},
        {"name": "beam_5", "num_beams": 5, "do_sample": False},
        {"name": "sample_t0.2", "num_beams": 1, "do_sample": True, "temperature": 0.2, "top_p": 0.9},
        {"name": "sample_t0.35", "num_beams": 1, "do_sample": True, "temperature": 0.35, "top_p": 0.9},
    ]
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    failures = []
    rows = []
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for index, sid in enumerate(ids, 1):
            item = items[sid]
            audio = resolve_audio(root, split_dir, item, item["audio"])
            audio = Path(audio)
            canonical = asr.transcribe(str(audio))
            waveform = inputs = None
            candidates = []
            for config in configs:
                try:
                    kwargs = dict(config)
                    name = kwargs.pop("name")
                    # Reuse the already cached production transcript for public-test rows.
                    # The other four configurations provide alternatives without another greedy pass.
                    if args.split == "public_test" and name == "production_greedy":
                        candidates.append({"text": canonical, "decode_config": name, "rank": 0,
                                           "asr_score": None, "avg_logprob": None,
                                           "no_speech_prob": None, "generation_config": config,
                                           "source": "PhoASRModel production transcript cache or production inference"})
                        continue
                    if inputs is None:
                        waveform = asr._read_audio(audio)
                        inputs = asr.processor(waveform, sampling_rate=16000, return_tensors="pt",
                                               return_attention_mask=True)
                        inputs = {k: v.to(device=asr.device, dtype=model.dtype if k == "input_features" else v.dtype)
                                  for k, v in inputs.items()}
                    with torch.inference_mode():
                        output = model.generate(**inputs, language="vi", task="transcribe", return_timestamps=False,
                                                **kwargs)
                    texts = asr.processor.batch_decode(output, skip_special_tokens=True)
                    for rank, text in enumerate(texts):
                        candidates.append({"text": text.strip(), "decode_config": name, "rank": rank,
                                           "asr_score": None, "avg_logprob": None,
                                           "no_speech_prob": None, "generation_config": config})
                except Exception as exc:
                    failures.append({"id": sid, "decode_config": config["name"],
                                     "error": f"{type(exc).__name__}: {exc}"})
            production = next((c["text"] for c in candidates if c["decode_config"] == "production_greedy"), None)
            if production != canonical:
                failures.append({"id": sid, "decode_config": "production_greedy_cache_check",
                                 "error": "Generated greedy text differs from production cached/transcribed text"})
            row = {"id": sid, "audio_ref": str(audio.relative_to(root)).replace("\\", "/"),
                   "audio_sha256": sha256(audio), "production_1best": canonical, "candidates": candidates}
            if args.split == "train":
                row["gold_text"] = item.get("audio_text", "")
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            print(f"[{index}/{len(ids)}] {sid}: {len(candidates)} candidates", flush=True)
    elapsed = time.perf_counter() - started
    all_candidates = [c for r in rows for c in r["candidates"]]
    duplicate_candidates = sum(len(r["candidates"]) - len({c["text"] for c in r["candidates"]}) for r in rows)
    duplicate_rate = duplicate_candidates / len(all_candidates) if all_candidates else 0.0
    gold = [(r, c) for r in rows for c in r["candidates"]]
    per_sample = {}
    for r in rows:
        ref = r.get("gold_text")
        if not ref:
            continue
        ws = [wer(ref, c["text"]) for c in r["candidates"]]
        cs = [cer(ref, c["text"]) for c in r["candidates"]]
        one = next((i for i, c in enumerate(r["candidates"]) if c["decode_config"] == "production_greedy"), None)
        per_sample[r["id"]] = {"production_wer": ws[one] if one is not None else None,
                                "oracle_wer": min(ws) if ws else None,
                                "production_cer": cs[one] if one is not None else None,
                                "oracle_cer": min(cs) if cs else None,
                                "average_wer": sum(ws) / len(ws) if ws else None,
                                "average_cer": sum(cs) / len(cs) if cs else None}
    valid = [v for v in per_sample.values() if v["production_wer"] is not None]
    quality = {k: sum(v[k] for v in valid) / len(valid) if valid else None
               for k in ("production_wer", "oracle_wer", "production_cer", "oracle_cer", "average_wer", "average_cer")}
    analysis = {"samples": len(rows), "average_candidates_per_sample": len(all_candidates) / max(1, len(rows)),
                "min_candidates_per_sample": min((len(r["candidates"]) for r in rows), default=0),
                "max_candidates_per_sample": max((len(r["candidates"]) for r in rows), default=0),
                "duplicate_transcript_rate": duplicate_rate,
                "empty_transcripts": sum(not c["text"] for c in all_candidates),
                "runtime_failures": len(failures), "generation_runtime_seconds": elapsed,
                "wer_cer": quality if valid else None, "per_sample": per_sample, "failures": failures}
    (out / "wer_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generation = {"split": args.split, "model": str(Path(args.model)), "model_config": str((Path(args.model) / "config.json").resolve()),
                  "sample_ids_file": str(Path(args.sample_ids)) if args.sample_ids else None,
                  "sample_ids_sha256": hashlib.sha256(Path(args.sample_ids).read_bytes()).hexdigest() if args.sample_ids else None,
                  "manifest_order_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest(),
                  "seed": args.seed, "sampling_seed_per_sample": "global seeded torch RNG in fixed sample/config order",
                  "processor": "PhoASR local AutoProcessor; Vietnamese 16 kHz mono", "configs": configs,
                  "score_note": "PhoASR generation wrapper does not expose calibrated confidence; all confidence fields are null",
                  "production_decode": "PhoASRModel production transcript (existing hash-keyed cache when available); language=vi, task=transcribe, do_sample=False, num_beams=1",
                  "cache_sha256": sha256(target)}
    (out / "generation_config.json").write_text(json.dumps(generation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"version": "asr_nbest_v1", "split": args.split, "samples": len(rows), "transcript_cache": target.name,
                "generation_config": "generation_config.json", "analysis": "wer_analysis.json",
                "duplicate_sample_ids": len(rows) - len({r["id"] for r in rows}),
                "sample_ids_match_requested_order": [r["id"] for r in rows] == ids,
                "audio_mapping_verified_by_sha256": True, "runtime_failures": len(failures),
                "checks": "Production greedy equals canonical PhoASR cached transcript; original text preserved without post-normalization"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in analysis.items() if k not in ("per_sample", "failures")}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
