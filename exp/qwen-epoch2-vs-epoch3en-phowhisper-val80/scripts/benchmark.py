from __future__ import annotations
import hashlib,importlib.util,json,os,random,statistics,sys,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"exp/btc-scoring/scripts"),
             str(ROOT/"exp/quan-continuation-cot-v1/scripts")]
from lifemate.utils import find_data_root,load_jsonl,resolve_audio,seed_everything
from lifemate import asr as asr_module
from score_btc import score_sample

EXP=ROOT/"exp/qwen-epoch2-vs-epoch3en-phowhisper-val80"
OUT=EXP/"output"
ASR_DIR=ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho"
BASE="lifemate-qwen35-kagglev1-sft-epoch2-q4km:latest"
BASE_DIGEST="973ae821f84016daad8db7ebf494fd2ddf98db7d1f14b5502ffb608cbfc8304d"
NEW="lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
NEW_DIGEST="fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
PROMPT_HASH="cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950"
SEED=20261111
N=80

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read_json(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def read_jsonl(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def load_runner():
    p=ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py"
    spec=importlib.util.spec_from_file_location("shared_qwen_runner",p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
def build():
    data=find_data_root("train+public_test/train+public_test")
    val_ids=json.loads((ROOT/"data/splits/val_ids.json").read_text(encoding="utf-8-sig"))
    rng=random.Random(SEED)
    ids=set(rng.sample(val_ids,N))
    ids=[sid for sid in val_ids if sid in ids]
    items={r["id"]:r for r in load_jsonl(data/"train/train.jsonl")}
    main={r["id"]:r for r in read_jsonl(ASR_DIR/"predictions.jsonl")}
    history={r["key"]:r for r in read_jsonl(ASR_DIR/"history_predictions.jsonl")}
    assert len(main)==309 and len(history)==692 and len(ids)==N
    transcript_map={}
    history_map={}
    for sid in ids:
        assert sid in main and sid in items
        transcript_map[sid]=main[sid]["prediction"]
        item=items[sid]
        for i,h in enumerate(item.get("history",[])):
            if not h.get("audio"):continue
            key=f"{sid}:{i}"
            row=history[key]
            assert row["audio"]==h["audio"]
            path=resolve_audio(data,data/"train",item,h["audio"])
            history_map[str(path.resolve())]=row["prediction"]
    assert len(history_map)==sum(bool(h.get("audio")) for sid in ids for h in items[sid].get("history",[]))
    id_hash=hashlib.sha256("\n".join(ids).encode()).hexdigest()
    source_paths=[ROOT/"data/splits/val_ids.json",data/"train/train.jsonl",
        ASR_DIR/"predictions.jsonl",ASR_DIR/"history_predictions.jsonl",
        ROOT/"src/lifemate/agent.py",ROOT/"src/lifemate/pipeline.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",EXP/"scripts/benchmark.py",EXP/"README.md"]
    cohort={"seed":SEED,"samples":N,"ids":ids,"ids_sha256_newline_joined":id_hash,
            "selection":"uniform random sample from val309, ordered by canonical val order"}
    return data,ids,items,transcript_map,history_map,cohort,{str(p.relative_to(ROOT)):sha(p) for p in source_paths}
def prep():
    OUT.mkdir(parents=True,exist_ok=True)
    data,ids,items,transcripts,hist,cohort,hashes=build()
    runner=load_runner()
    os.environ.update({"LIFEMATE_LLM_BACKEND":"ollama","LIFEMATE_OLLAMA_THINK":"true",
        "LIFEMATE_OLLAMA_PROTOCOL":"json","LIFEMATE_PROMPT_LANGUAGE":"en"})
    prompt=runner.canonical_prompt_hash(True)
    if prompt!=PROMPT_HASH:raise RuntimeError(f"English + CoT prompt hash mismatch: {prompt}")
    locks={}
    for tag,digest in ((BASE,BASE_DIGEST),(NEW,NEW_DIGEST)):
        runner.MODEL=tag;runner.EXPECTED_DIGEST=digest
        version,actual=runner.model_fingerprint()
        if actual!=digest:raise RuntimeError(f"Model fingerprint mismatch for {tag}")
        locks[tag]={"digest":actual,"ollama_version":version}
    lock={"experiment":"qwen-epoch2-vs-epoch3en-phowhisper-val80","cohort":cohort,
        "models":locks,"prompt_sha256":prompt,"prompt_variant":"English observation-grounded-v2 + existing CoT checklist",
        "settings":{"thinking":True,"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,"num_batch":512,"seed":17,"max_action_turns":5},
        "asr":{"model":"PhoWhisper large v2 on main and history","main":len(transcripts),"history":len(hist),
            "main_file_sha256":sha(ASR_DIR/"predictions.jsonl"),"history_file_sha256":sha(ASR_DIR/"history_predictions.jsonl")},
        "input_sha256":hashes}
    path=OUT/"benchmark_lock.json"
    if path.exists():
        if read_json(path)!=lock:raise RuntimeError("Benchmark lock differs from the existing run")
    else:write_json(path,lock)
    write_json(OUT/"cohort_manifest.json",cohort)
    print(json.dumps({"ids":N,"history_audio":len(hist),"prompt_sha256":prompt,"models":locks},ensure_ascii=False),flush=True)
def run():
    lock=read_json(OUT/"benchmark_lock.json")
    data,ids,items,transcripts,hist,cohort,hashes=build()
    assert lock["cohort"]==cohort and lock["input_sha256"]==hashes
    runner=load_runner()
    import lifemate
    class CachedPhoWhisper:
        def __init__(self,*args,**kwargs):pass
        def transcribe(self,path):
            key=str(Path(path).resolve())
            if key not in hist:raise KeyError(f"History transcript missing for {key}")
            return hist[key]
    asr_module.PhoASRModel=CachedPhoWhisper
    # run_downstream.run_block imports PhoASRModel lazily from lifemate.asr.
    seed_everything(17)
    blocks=[ids[i:i+20] for i in range(0,len(ids),20)]
    order=[["epoch2","epoch3en"],["epoch3en","epoch2"],["epoch2","epoch3en"],["epoch3en","epoch2"]]
    write_json(OUT/"execution_order.json",{"blocks":[{"round":i+1,"ids":b,"branch_order":order[i]} for i,b in enumerate(blocks)],"paired":True})
    for ri,block in enumerate(blocks,1):
        for branch in order[ri-1]:
            tag,digest=(BASE,BASE_DIGEST) if branch=="epoch2" else (NEW,NEW_DIGEST)
            runner.MODEL=tag;runner.EXPECTED_DIGEST=digest
            version,actual=runner.model_fingerprint()
            if actual!=digest or version!=lock["models"][tag]["ollama_version"]:raise RuntimeError("Ollama fingerprint changed")
            branch_out=OUT/branch;branch_out.mkdir(parents=True,exist_ok=True)
            runner.OUT=branch_out
            done={r["id"] for r in read_jsonl(branch_out/"predictions-COT.jsonl")}
            todo=[sid for sid in block if sid not in done]
            if not todo:continue
            runner.warmup("COT",ri)
            runner.run_block("COT",ri,todo,transcripts,items,cot=True)
            print(f"completed {branch} {ri}/4 total={len(read_jsonl(branch_out/'predictions-COT.jsonl'))}/80",flush=True)
    for branch in ("epoch2","epoch3en"):
        rows=read_jsonl(OUT/branch/"predictions-COT.jsonl")
        if [r["id"] for r in rows]!=ids:raise RuntimeError(f"{branch} missing or out-of-order IDs")
    analyze()
def analyze():
    data=find_data_root("train+public_test/train+public_test")
    items={r["id"]:r for r in load_jsonl(data/"train/train.jsonl")}
    ids=read_json(OUT/"cohort_manifest.json")["ids"]
    rows={b:{r["id"]:r for r in read_jsonl(OUT/b/"predictions-COT.jsonl")} for b in ("epoch2","epoch3en")}
    scores={b:[] for b in rows};details=[]
    for sid in ids:
        row={"id":sid}
        for b in rows:
            score=score_sample(sid,items[sid],rows[b][sid],data)
            scores[b].append(score["official_sample_score"])
            row[b]={"btc":score["official_sample_score"],"btc_before_runtime_gate":score["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match":score["hash_match"],"matched_calls":score["matched_calls"],"gold_calls":score["gold_calls"],
                "extra_calls":score["unmatched_submitted_calls"],"over30":score["runtime_over_30s"],"runtime_error":score["runtime_failure"]}
        details.append(row)
    delta=[x["epoch3en"]["btc"]-x["epoch2"]["btc"] for x in details]
    rng=random.Random(SEED);boots=sorted(100*statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000))
    result={}
    for b in rows:
        result[b]={"btc_0_100":100*statistics.mean(scores[b]),
            "btc_before_runtime_gate_0_100":100*statistics.mean(x[b]["btc_before_runtime_gate"] for x in details),
            "hash_matches":sum(x[b]["hash_match"] for x in details),
            "over30":sum(x[b]["over30"] for x in details),"runtime_errors":sum(x[b]["runtime_error"] for x in details),
            "mean_latency_seconds":statistics.mean(rows[b][sid]["latency_seconds"] for sid in ids)}
    result["paired_epoch3_minus_epoch2"]={"btc_delta_points":100*statistics.mean(delta),
        "bootstrap_95pct_CI":[boots[249],boots[9749]],"epoch3_wins":sum(x>0 for x in delta),
        "epoch2_wins":sum(x<0 for x in delta),"ties":sum(x==0 for x in delta)}
    (OUT/"paired_details.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in details),encoding="utf-8")
    write_json(OUT/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("--prepare","--run"):raise SystemExit("use --prepare or --run")
    prep() if sys.argv[1]=="--prepare" else run()


