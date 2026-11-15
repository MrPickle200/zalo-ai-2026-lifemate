from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import sys
sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "lifemate").is_dir()) / "exp" / "asr-modeling" / "scripts"))
import _bootstrap
from inference_common import audio_duration, read_audio_16k, read_jsonl, write_jsonl
from lifemate.utils import find_data_root, resolve_audio

def main() -> int:
    project = _bootstrap.ROOT
    model_dir = project / "models" / "quan-transcribe"
    sys.path.insert(0, str(model_dir / "inference"))
    from quan_transcribe import QuanTranscribe
    parser = argparse.ArgumentParser(description="Transcribe all examples in one LifeMate split.")
    parser.add_argument("--split", choices=("train", "public_test"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="train+public_test/train+public_test")
    parser.add_argument("--lm", default=str(model_dir / "lm"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=32)
    args = parser.parse_args()
    root = find_data_root(args.data_root)
    split_dir = root / args.split
    manifest = split_dir / ("train.jsonl" if args.split == "train" else "public_test.jsonl")
    samples = read_jsonl(manifest)
    if not samples or len({x["id"] for x in samples}) != len(samples):
        raise ValueError(f"Empty manifest or duplicate IDs: {manifest}")
    output = Path(args.output)
    rows_by_id = {r["id"]: r for r in read_jsonl(output)}
    ids = {x["id"] for x in samples}
    if set(rows_by_id) - ids:
        raise ValueError("Transcript file contains IDs outside requested split")
    model = QuanTranscribe.from_pretrained(args.model, device=args.device, lm=args.lm)
    todo = [x for x in samples if x["id"] not in rows_by_id]
    started = time.perf_counter()
    for index, item in enumerate(todo, 1):
        audio = resolve_audio(root, split_dir, item, item["audio"])
        duration = audio_duration(audio)
        t0 = time.perf_counter()
        waveform = read_audio_16k(audio)
        prediction = model.transcribe(waveform, sample_rate=16000, streaming=False,
                                      use_lm=True, beam_width=args.beam_width)
        rows_by_id[item["id"]] = {"id": item["id"], "prediction": prediction,
                                  "audio_seconds": duration, "inference_seconds": time.perf_counter()-t0}
        if index % 25 == 0 or index == len(todo):
            write_jsonl(output, [rows_by_id[x["id"]] for x in samples if x["id"] in rows_by_id])
            print(f"{args.split}: {index}/{len(todo)} new; {len(rows_by_id)}/{len(samples)} total", flush=True)
    rows = [rows_by_id[x["id"]] for x in samples]
    write_jsonl(output, rows)
    summary = {"split": args.split, "samples": len(rows), "model": str(Path(args.model).resolve()),
               "beam_width": args.beam_width, "decoder": "KenLM CTC beam search",
               "reference_text_saved": False, "new_inference_wall_seconds": time.perf_counter()-started,
               "transcript_file": str(output.resolve())}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
