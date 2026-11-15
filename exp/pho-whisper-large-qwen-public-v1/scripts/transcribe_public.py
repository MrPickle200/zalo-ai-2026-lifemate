from __future__ import annotations
import hashlib,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from lifemate.asr import PhoASRModel
from lifemate.utils import find_data_root,load_jsonl,resolve_audio
EXP=ROOT/"exp/pho-whisper-large-qwen-public-v1"
OUT=EXP/"output"
MODEL=ROOT/"models/pho-whisper-large-zaloai"
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read_jsonl(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def save(path,rows):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    temp.replace(path)
def main():
    data=find_data_root("train+public_test/train+public_test")
    items=list(load_jsonl(data/"public_test/public_test.jsonl"))
    assert len(items)==634 and len({r["id"] for r in items})==634
    clips=[]
    for item in items:
        clips.append({"id":item["id"],"kind":"main","index":None,"audio":item["audio"]})
        for i,h in enumerate(item.get("history",[])):
            if h.get("audio"):clips.append({"id":item["id"],"kind":"history","index":i,"audio":h["audio"]})
    assert len(clips)==1933 and len({x["audio"] for x in clips})==1933
    by_id={x["id"]:x for x in items}
    OUT.mkdir(parents=True,exist_ok=True)
    lock={"split":"public_test","clips":len(clips),"tasks":len(items),"manifest_sha256":sha(data/"public_test/public_test.jsonl"),"release_sha256":sha(MODEL/"release.json"),"model_sha256":sha(MODEL/"model.safetensors"),"generation_config_sha256":sha(MODEL/"generation_config.json"),"decode":{"language":"vi","task":"transcribe","num_beams":1,"do_sample":False,"max_new_tokens":256,"sample_rate":16000}}
    lock_path=OUT/"asr_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text(encoding="utf-8"))==lock,"ASR input lock changed"
    else:lock_path.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    target=OUT/"transcripts_audio.jsonl"
    prior=read_jsonl(target)
    done={x["audio"]:x for x in prior}
    keys=[x["audio"] for x in clips]
    assert len(done)==len(prior) and set(done)<=set(keys)
    cache=tempfile.TemporaryDirectory(prefix="lifemate-phowhisper-public-")
    asr=PhoASRModel(MODEL,cache.name,device="cuda")
    t=time.perf_counter();asr._load();load_seconds=time.perf_counter()-t
    for n,clip in enumerate(clips,1):
        if clip["audio"] in done:continue
        item=by_id[clip["id"]]
        path=resolve_audio(data,data/"public_test",item,clip["audio"])
        t=time.perf_counter();text=asr.transcribe(str(path));seconds=time.perf_counter()-t
        done[clip["audio"]]={**clip,"transcript":text,"inference_seconds":seconds}
        if n%25==0 or n==len(clips):
            save(target,[done[k] for k in keys if k in done])
            print(f"PhoWhisper public {len(done)}/{len(clips)}",flush=True)
    rows=[done[k] for k in keys]
    assert all(isinstance(x["transcript"],str) and x["transcript"].strip() for x in rows)
    save(target,rows)
    summary={"status":"complete","clips":len(rows),"main":sum(x["kind"]=="main" for x in rows),"history":sum(x["kind"]=="history" for x in rows),"model_load_seconds":load_seconds,"transcripts_sha256":sha(target),"model_sha256":lock["model_sha256"],"resumed":len(prior)}
    (OUT/"asr_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    cache.cleanup()
if __name__=="__main__":main()

