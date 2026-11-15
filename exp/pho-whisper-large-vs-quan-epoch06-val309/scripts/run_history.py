from __future__ import annotations
import argparse, json, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "exp/asr-modeling/scripts"), str(ROOT / "models/quan-transcribe/inference")]
from inference_common import audio_duration, read_audio_16k, read_jsonl, score_rows, write_jsonl
from lifemate.utils import find_data_root, resolve_audio

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--branch", choices=["pho","quan"], required=True)
    a=p.parse_args()
    data=find_data_root("train+public_test/train+public_test")
    ids=json.loads((ROOT/"data/splits/val_ids.json").read_text(encoding="utf-8-sig"))
    all_items={r["id"]:r for r in read_jsonl(data/"train/train.jsonl")}
    samples=[]
    for sid in ids:
        item=all_items[sid]
        for i,h in enumerate(item.get("history", [])):
            if h.get("audio"):
                samples.append((sid,i,h["audio"],h.get("audio_text",""),item))
    assert len(samples)==692, len(samples)
    out=ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output"/a.branch
    path=out/"history_predictions.jsonl"
    saved={r["key"]:r for r in read_jsonl(path)}
    resumed_count=len(saved)
    keys=[f"{sid}:{i}" for sid,i,*_ in samples]
    assert len(saved)==len(set(saved)) and set(saved)<=set(keys)
    if a.branch=="pho":
        from lifemate.asr import PhoASRModel
        cache=tempfile.TemporaryDirectory(prefix="lifemate-history-pho-")
        model=PhoASRModel(ROOT/"models/pho-whisper-large-zaloai",cache.name,device="cuda")
        t=time.perf_counter(); model._load(); load=time.perf_counter()-t
    else:
        from quan_transcribe import QuanTranscribe
        t=time.perf_counter()
        model=QuanTranscribe.from_pretrained(ROOT/"exp/quan-10epoch-transcript-dictionary-v1/output/checkpoints/epoch-06",device="cuda",lm=ROOT/"models/quan-transcribe/lm")
        load=time.perf_counter()-t
    for n,(sid,i,rel,ref,item) in enumerate(samples,1):
        key=f"{sid}:{i}"
        if key in saved: continue
        audio=resolve_audio(data,data/"train",item,rel)
        duration=audio_duration(audio)
        t=time.perf_counter()
        if a.branch=="pho": pred=model.transcribe(str(audio))
        else: pred=model.transcribe(read_audio_16k(audio),sample_rate=16000,streaming=False,use_lm=True,beam_width=32)
        latency=time.perf_counter()-t
        saved[key]={"key":key,"id":sid,"history_index":i,"audio":rel,"reference":ref,"prediction":pred,"audio_seconds":duration,"inference_seconds":latency}
        if n%25==0 or n==len(samples):
            write_jsonl(path,[saved[k] for k in keys if k in saved])
            print(f"{a.branch} history {len(saved)}/692",flush=True)
    rows=[saved[k] for k in keys]
    summary={"branch":a.branch,"samples":len(rows),"metrics":score_rows(rows),"model_load_seconds":load,"resumed_predictions":resumed_count,"normalization":"inference_common.normalize"}
    (out/"history_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__": main()


