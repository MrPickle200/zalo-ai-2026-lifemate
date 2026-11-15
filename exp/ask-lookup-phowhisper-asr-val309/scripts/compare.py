from __future__ import annotations
import hashlib, importlib.util, json, math, os, random, statistics, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/ask-lookup-phowhisper-asr-val309"
OUT=EXP/"output"
PREV=ROOT/"exp/ask-lookup-disambiguation-followup-val309"
PREV_OUT=PREV/"output"
ASR=ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho"
MODEL="lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
DIGEST="fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
SEED=17
BOOTSTRAP_SEED=20261114
BLOCK=15
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"exp/btc-scoring/scripts"),str(ROOT/"exp/quan-continuation-cot-v1/scripts")]
from lifemate.utils import find_data_root,load_jsonl,resolve_audio
from score_btc import score_sample,call_key

def sha(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read_json(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def read_jsonl(p):return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if p.exists() else []
def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def load_prompt_benchmark():
    p=PREV/"scripts/benchmark.py"
    spec=importlib.util.spec_from_file_location("focused_prompt_benchmark_for_asr_comparison",p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def build():
    prompt=load_prompt_benchmark()
    runner=prompt.runner
    old_lock=read_json(PREV_OUT/"benchmark_lock.json")
    ids=old_lock["cohort"]["all_val_ids"]
    if len(ids)!=309:raise RuntimeError("Expected locked val309 cohort")
    pho_main_rows=read_jsonl(ASR/"predictions.jsonl")
    pho_hist_rows=read_jsonl(ASR/"history_predictions.jsonl")
    main={r["id"]:r["prediction"] for r in pho_main_rows}
    hist_rows={r["key"]:r for r in pho_hist_rows}
    if [r["id"] for r in pho_main_rows]!=ids or len(main)!=309 or len(hist_rows)!=692:
        raise RuntimeError("PhoWhisper ASR artifacts do not exactly cover val309 main/history")
    data=find_data_root("train+public_test/train+public_test")
    items={r["id"]:r for r in load_jsonl(data/"train/train.jsonl")}
    expected_keys=[];history={}
    for sid in ids:
        item=items[sid]
        for i,h in enumerate(item.get("history",[])):
            if not h.get("audio"):continue
            key=f"{sid}:{i}"
            row=hist_rows.get(key)
            if row is None or row["audio"]!=h["audio"]:raise RuntimeError(f"Missing/mismatched history transcript {key}")
            path=resolve_audio(data,data/"train",item,h["audio"])
            history[str(path.resolve())]=row["prediction"]
            expected_keys.append(key)
    if set(expected_keys)!=set(hist_rows):raise RuntimeError("PhoWhisper history includes unexpected/missing IDs")
    baseline=read_jsonl(PREV_OUT/"focused/predictions-COT.jsonl")
    if [r["id"] for r in baseline]!=ids:raise RuntimeError("Archived focused QUAN/PhoASR predictions are incomplete")
    prompt.runner.inject_cot_original=prompt.runner.inject_cot
    os.environ.update({"OLLAMA_HOST":runner.HOST,"LIFEMATE_LLM_BACKEND":"ollama",
        "LIFEMATE_OLLAMA_MODEL":MODEL,"LIFEMATE_OLLAMA_CONTEXT":"8192",
        "LIFEMATE_OLLAMA_NUM_PREDICT":"768","LIFEMATE_OLLAMA_NUM_BATCH":"512",
        "LIFEMATE_OLLAMA_THINK":"true","LIFEMATE_OLLAMA_PROTOCOL":"json",
        "LIFEMATE_PROMPT_LANGUAGE":"en"})
    phash=prompt.prompt_hash(prompt.focused_inject)
    if phash!=old_lock["prompt"]["focused_sha256"]:raise RuntimeError("Focused prompt hash changed")
    runner.MODEL=MODEL;runner.EXPECTED_DIGEST=DIGEST
    version,digest=runner.model_fingerprint()
    if digest!=DIGEST or digest!=old_lock["model"]["digest"]:raise RuntimeError("Qwen model digest changed")
    if version!=old_lock["model"]["ollama_version"]:raise RuntimeError("Ollama version differs from the locked prompt run")
    inputs=[PREV_OUT/"benchmark_lock.json",PREV_OUT/"focused/predictions-COT.jsonl",
        PREV_OUT/"paired_details.jsonl",ASR/"predictions.jsonl",ASR/"history_predictions.jsonl",
        ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output/provenance.json",
        ROOT/"data/splits/val_ids.json",data/"train/train.jsonl",ROOT/"src/lifemate/agent.py",
        ROOT/"src/lifemate/pipeline.py",ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        PREV/"scripts/benchmark.py",Path(__file__).resolve()]
    cohort_hash=hashlib.sha256("\n".join(ids).encode()).hexdigest()
    lock={"experiment":EXP.name,"cohort":{"count":309,"ids":ids,"ids_sha256":cohort_hash},
        "model":{"tag":MODEL,"digest":digest,"ollama_version":version},
        "prompt":{"variant":"focused post-lookup disambiguation","sha256":phash},
        "settings":{"temperature":0,"thinking":True,"protocol":"JSON","num_predict":768,"num_ctx":8192,"num_batch":512,"seed":SEED,"max_action_turns":5},
        "baseline":{"path":str((PREV_OUT/"focused/predictions-COT.jsonl").relative_to(ROOT)),
            "sha256":sha(PREV_OUT/"focused/predictions-COT.jsonl"),
            "primary_asr":old_lock["transcript"]["primary_asr"],"history_asr":old_lock["transcript"]["history_asr"]},
        "treatment":{"primary_asr":"PhoWhisper large v2 val309","history_asr":"PhoWhisper large v2 val692",
            "main_count":len(main),"history_count":len(history),
            "main_sha256":sha(ASR/"predictions.jsonl"),"history_sha256":sha(ASR/"history_predictions.jsonl")},
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in inputs}}
    return prompt,runner,ids,items,main,history,data,lock

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    *_,lock=build()
    p=OUT/"comparison_lock.json"
    if p.exists() and read_json(p)!=lock:raise RuntimeError("Comparison lock changed")
    if not p.exists():write_json(p,lock)
    write_json(OUT/"cohort_manifest.json",{"ids":lock["cohort"]["ids"],"ids_sha256":lock["cohort"]["ids_sha256"]})
    print(json.dumps({"tasks":309,"PhoWhisper_main":634,"PhoWhisper_val_history":692,
        "prompt_sha256":lock["prompt"]["sha256"],"model_digest":lock["model"]["digest"]},ensure_ascii=False),flush=True)

class CachedPhoWhisper:
    def __init__(self,*args,**kwargs):pass
    def transcribe(self,path):
        key=str(Path(path).resolve())
        if key not in self.lookup:raise KeyError(f"Uncached PhoWhisper history path: {key}")
        return self.lookup[key]

def run():
    if sys.prefix!=sys.base_prefix:raise RuntimeError("Use native system Python so local Ollama is reachable")
    prompt,runner,ids,items,main,history,data,current=build()
    lock=read_json(OUT/"comparison_lock.json")
    if current!=lock:raise RuntimeError("Comparison inputs/model/prompt no longer match the lock")
    branch=OUT/"phowhisper-large"
    branch.mkdir(parents=True,exist_ok=True)
    predpath=branch/"predictions-COT.jsonl"
    existing=read_jsonl(predpath)
    existing_ids=[r["id"] for r in existing]
    if existing_ids!=ids[:len(existing_ids)]:raise RuntimeError("Existing PhoWhisper output is not a canonical ID prefix")
    runner.MODEL=MODEL;runner.EXPECTED_DIGEST=DIGEST;runner.OUT=branch
    runner.inject_cot_original=runner.inject_cot
    runner.inject_cot=prompt.focused_inject
    runner.PhoASRModel=CachedPhoWhisper
    runner.PhoASRModel.lookup=history
    runner.seed_everything(SEED)
    for ri,start in enumerate(range(0,len(ids),BLOCK),1):
        block=ids[start:start+BLOCK]
        done={r["id"] for r in read_jsonl(predpath)}
        todo=[sid for sid in block if sid not in done]
        if not todo:continue
        runner.warmup("COT",ri)
        while todo:
            before=len(read_jsonl(predpath))
            try:runner.run_block("COT",ri,todo,main,items,cot=True)
            except Exception:
                rows=read_jsonl(predpath)
                if len(rows)<=before or not rows[-1].get("error"):raise
            done={r["id"] for r in read_jsonl(predpath)}
            todo=[sid for sid in block if sid not in done]
        print(f"PhoWhisper block {ri}/{math.ceil(len(ids)/BLOCK)} complete; total={len(read_jsonl(predpath))}/309",flush=True)
    rows=read_jsonl(predpath)
    if [r["id"] for r in rows]!=ids:raise RuntimeError("PhoWhisper predictions do not cover canonical val309")
    meta=read_json(branch/"run-COT.json") if (branch/"run-COT.json").exists() else {}
    meta.update({"branch":"PhoWhisper-large-focused-COT","status":"complete","samples_completed":len(rows),
        "asr_main":"PhoWhisper large v2","asr_history":"PhoWhisper large v2","prompt_variant":"focused post-lookup disambiguation"})
    write_json(branch/"run-COT.json",meta)
    print("PhoWhisper downstream inference complete: 309/309",flush=True)

def pct(xs):return 100*statistics.mean(xs) if xs else None
def boot_ci(deltas,seed,n=10000):
    rng=random.Random(seed);v=sorted(100*statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(n))
    return [v[int(.025*n)],v[int(.975*n)-1]]
def label_of(item):
    x=item.get("label",{});return json.loads(x) if isinstance(x,str) else x
def analyze():
    prompt,runner,ids,items,main,history,data,current=build()
    if current!=read_json(OUT/"comparison_lock.json"):raise RuntimeError("Inputs changed since lock")
    baseline={r["id"]:r for r in read_jsonl(PREV_OUT/"focused/predictions-COT.jsonl")}
    pho={r["id"]:r for r in read_jsonl(OUT/"phowhisper-large/predictions-COT.jsonl")}
    if set(baseline)!=set(ids) or set(pho)!=set(ids):raise RuntimeError("One branch misses val IDs")
    gold_ask=set()
    records=[]
    for sid in ids:
        labels=label_of(items[sid]);gold=labels.get("tool_calls",[])
        goldask=[c for c in gold if c.get("name")=="sys.ask_missing_info"]
        if goldask:gold_ask.add(sid)
        row={"id":sid,"group":"audited_47_diagnostic" if goldask else "fresh_262_primary","gold_ask":bool(goldask)}
        for key,pred in (("quan_phoasr",baseline[sid]),("phowhisper_large",pho[sid])):
            s=score_sample(sid,items[sid],pred,data)
            asks=[c for c in pred.get("tool_calls",[]) if c.get("name")=="sys.ask_missing_info"]
            exact=any(call_key(a)==call_key(g) for a in asks for g in goldask)
            row[key]={"btc":s["official_sample_score"],"btc_before_runtime_gate":s["btc_formula_score_before_runtime_gate_diagnostic"],
                "hash_match":s["hash_match"],"runtime_error":bool(pred.get("error")),"error":pred.get("error"),
                "latency_seconds":pred.get("latency_seconds"),"ask":bool(asks),"ask_exact":exact}
        records.append(row)
    fresh=[sid for sid in ids if sid not in gold_ask];diag=[sid for sid in ids if sid in gold_ask]
    summaries={}
    for name,subset in (("all_val309",ids),("fresh_262_primary",fresh),("audited_47_diagnostic",diag)):
        group=[r for r in records if r["id"] in set(subset)]
        part={}
        for b in ("quan_phoasr","phowhisper_large"):
            lat=[r[b]["latency_seconds"] for r in group if r[b]["latency_seconds"] is not None]
            part[b]={"btc_0_100":pct([r[b]["btc"] for r in group]),
                "btc_before_runtime_gate_0_100":pct([r[b]["btc_before_runtime_gate"] for r in group]),
                "hash_matches":sum(r[b]["hash_match"] for r in group),
                "runtime_errors":sum(r[b]["runtime_error"] for r in group),
                "mean_latency_seconds":statistics.mean(lat) if lat else None,
                "gold_ask_exact":sum(r[b]["ask_exact"] for r in group),"ask_count":sum(r[b]["ask"] for r in group)}
        ds=[r["phowhisper_large"]["btc"]-r["quan_phoasr"]["btc"] for r in group]
        du=[r["phowhisper_large"]["btc_before_runtime_gate"]-r["quan_phoasr"]["btc_before_runtime_gate"] for r in group]
        part["paired_phowhisper_minus_quan_phoasr"]={"btc_delta_points":pct(ds),"bootstrap_95pct_CI_points":boot_ci(ds,BOOTSTRAP_SEED),
            "phowhisper_wins":sum(x>0 for x in ds),"quan_phoasr_wins":sum(x<0 for x in ds),"ties":sum(x==0 for x in ds),
            "ungated_delta_points":pct(du),"ungated_bootstrap_95pct_CI_points":boot_ci(du,BOOTSTRAP_SEED+1)}
        summaries[name]=part
    summary={"branches":{"quan_phoasr":{"main":"QUAN checkpoint-verification-rerun val309","history":"PhoASRModel","predictions_reused":True},
        "phowhisper_large":{"main":"PhoWhisper large v2 val309","history":"PhoWhisper large v2 val692","predictions_reinferred":True}},
        "same_model_prompt_settings":{"model":MODEL,"model_digest":DIGEST,"prompt_sha256":current["prompt"]["sha256"],
            "prompt_variant":"focused post-lookup disambiguation","thinking":True,"protocol":"JSON","temperature":0,
            "num_predict":768,"num_ctx":8192,"num_batch":512,"seed":SEED,"action_cap":5},
        "groups":summaries,"method":{"scoring":"official BTC ToolEnv replay; local 30-second runtime gate and pre-gate diagnostic reported separately",
            "primary_group":"all val309; 262 fresh non-gold-ask tasks and 47 audited gold-ask tasks also split out",
            "baseline":"reused exact archived focused-prompt predictions; PhoWhisper branch run once",
            "bootstrap_draws":10000,"bootstrap_seed":BOOTSTRAP_SEED}}
    write_json(OUT/"summary.json",summary)
    write_jsonl=OUT/"paired_details.jsonl"
    write_jsonl.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    a=summaries["all_val309"];p=a["paired_phowhisper_minus_quan_phoasr"]
    freshs=summaries["fresh_262_primary"];fp=freshs["paired_phowhisper_minus_quan_phoasr"]
    diags=summaries["audited_47_diagnostic"]
    report=("# PhoWhisper Large versus QUAN/PhoASR with focused prompt\n\n"
        "Both branches use Qwen epoch 3, the same focused post-lookup prompt, decoding settings and ToolEnv scoring. "
        "The archived QUAN/PhoASR branch is reused; PhoWhisper Large v2 replaces both main and history ASR in the new branch.\n\n"
        "## All val309\n\n"
        f"| Branch | BTC after local 30s gate | BTC before gate | Hash matches | Runtime errors | Mean seconds | Exact gold asks |\n|---|---:|---:|---:|---:|---:|---:|\n"
        f"| QUAN main + PhoASR history | {a['quan_phoasr']['btc_0_100']:.2f} | {a['quan_phoasr']['btc_before_runtime_gate_0_100']:.2f} | {a['quan_phoasr']['hash_matches']}/309 | {a['quan_phoasr']['runtime_errors']} | {a['quan_phoasr']['mean_latency_seconds']:.2f} | {a['quan_phoasr']['gold_ask_exact']}/47 |\n"
        f"| PhoWhisper Large main + history | {a['phowhisper_large']['btc_0_100']:.2f} | {a['phowhisper_large']['btc_before_runtime_gate_0_100']:.2f} | {a['phowhisper_large']['hash_matches']}/309 | {a['phowhisper_large']['runtime_errors']} | {a['phowhisper_large']['mean_latency_seconds']:.2f} | {a['phowhisper_large']['gold_ask_exact']}/47 |\n\n"
        f"Paired PhoWhisper−QUAN/PhoASR: **{p['btc_delta_points']:+.2f} points** (bootstrap 95% CI **[{p['bootstrap_95pct_CI_points'][0]:+.2f}, {p['bootstrap_95pct_CI_points'][1]:+.2f}]**); wins/ties/losses {p['phowhisper_wins']}/{p['ties']}/{p['quan_phoasr_wins']}. Pre-gate delta {p['ungated_delta_points']:+.2f} [{p['ungated_bootstrap_95pct_CI_points'][0]:+.2f}, {p['ungated_bootstrap_95pct_CI_points'][1]:+.2f}].\n\n"
        "## Cohort split\n\n"
        f"On the 262 non-gold-ask tasks, paired delta was {fp['btc_delta_points']:+.2f} [{fp['bootstrap_95pct_CI_points'][0]:+.2f}, {fp['bootstrap_95pct_CI_points'][1]:+.2f}]. "
        f"On the 47 audited ask tasks, exact asks were {diags['quan_phoasr']['gold_ask_exact']}/47 for QUAN/PhoASR and {diags['phowhisper_large']['gold_ask_exact']}/47 for PhoWhisper Large.\n\n"
        "## Caveats\n\n"
        "BTC runtime gating uses local elapsed times, not A30 timing. The prior branch predictions were reused and the PhoWhisper branch was inferred once, so run-to-run model variance may affect the difference. "
        "PhoWhisper validation transcripts may overlap its training corpus. No public-test submission was created.\n\n"
        "Per-ID outcomes, bootstrap inputs, model/prompt fingerprints and transcript hashes are recorded in paired_details.jsonl, summary.json and comparison_lock.json.\n")
    (OUT/"summary.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ("--prepare","--run","--analyze"):raise SystemExit("use --prepare, --run, or --analyze")
    {"--prepare":prepare,"--run":run,"--analyze":analyze}[sys.argv[1]]()
if __name__=="__main__":main()
