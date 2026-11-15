from __future__ import annotations
import hashlib,importlib.util,json,math,os,re,statistics,subprocess,sys,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from lifemate import pipeline as pipeline_module
from lifemate.agent import Agent
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import find_data_root,load_jsonl,resolve_audio,seed_everything

EXP=ROOT/"exp/pho-whisper-large-qwen-public-v1"
OUT=EXP/"output"
ASR_PATH=OUT/"transcripts_audio.jsonl"
PROMPT_EXP=ROOT/"exp/ask-ambiguity-prompt-val309"
MODEL="lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
DIGEST="fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
MODEL_PARAMS=4205751296
ASR_PARAMS=1543304960
HOST="http://127.0.0.1:11434"
CAP=768;CONTEXT=8192;BATCH=512;SEED=17;MAX_STEPS=5

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def read_json(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def read_jsonl(p):
    if not p.exists():return []
    return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def api(path):
    with urllib.request.urlopen(HOST+path,timeout=30) as r:return json.loads(r.read().decode())
def prompt_module():
    p=PROMPT_EXP/"scripts/run_prompt.py"
    s=importlib.util.spec_from_file_location("locked_prompt_experiment",p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m
def choose_prompt():
    summary=read_json(PROMPT_EXP/"output/summary.json")
    a,b=summary["baseline"],summary["variant"]
    paired=summary["paired"]
    # Treat local 30-second gate differences within paired uncertainty as unstable.
    if (paired["bootstrap_95pct_CI_points"][0]>0
            and b["ask_exact_gold_call"]>=a["ask_exact_gold_call"]):
        return "variant",summary
    return "baseline",summary
def settings():
    os.environ.update({"OLLAMA_HOST":HOST,"LIFEMATE_LLM_BACKEND":"ollama","LIFEMATE_OLLAMA_MODEL":MODEL,
        "LIFEMATE_OLLAMA_CONTEXT":str(CONTEXT),"LIFEMATE_OLLAMA_NUM_PREDICT":str(CAP),
        "LIFEMATE_OLLAMA_NUM_BATCH":str(BATCH),"LIFEMATE_OLLAMA_THINK":"true",
        "LIFEMATE_OLLAMA_PROTOCOL":"json","LIFEMATE_OLLAMA_TELEMETRY":"1",
        "LIFEMATE_PROMPT_LANGUAGE":"en"})
def input_lock():
    if sys.prefix!=sys.base_prefix:raise RuntimeError("Use native Python to reach Ollama")
    assert MODEL_PARAMS+ASR_PARAMS<6_000_000_000 and MODEL_PARAMS<5_000_000_000
    assert CONTEXT<=16384
    data=find_data_root("train+public_test/train+public_test")
    items=list(load_jsonl(data/"public_test/public_test.jsonl"))
    assert len(items)==634 and len({x["id"] for x in items})==634
    clips=read_jsonl(ASR_PATH)
    assert len(clips)==1933 and len({x["audio"] for x in clips})==1933
    main={x["id"]:x["transcript"] for x in clips if x["kind"]=="main"}
    hist={x["audio"]:x["transcript"] for x in clips if x["kind"]=="history"}
    assert len(main)==634 and len(hist)==1299 and set(main)=={x["id"] for x in items}
    for item in items:
        assert all(h["audio"] in hist for h in item.get("history",[]) if h.get("audio"))
    selected,summary=choose_prompt()
    p=prompt_module()
    settings()
    expected_hash=read_json(PROMPT_EXP/"output/manifest.json")[f"{selected}_prompt_sha256"]
    actual_hash=p.prompt_hash(p.original_inject if selected=="baseline" else p.variant_inject)
    assert expected_hash==actual_hash
    version=api("/api/version")["version"]
    model=next(x for x in api("/api/tags")["models"] if x["name"]==MODEL)
    assert model["digest"]==DIGEST
    asr_lock=read_json(OUT/"asr_lock.json")
    assert asr_lock["model_sha256"]==sha(ROOT/"models/pho-whisper-large-zaloai/model.safetensors")
    paths=[data/"public_test/public_test.jsonl",data/"public_test/public_test_tools_catalog.json",
           ASR_PATH,OUT/"asr_lock.json",PROMPT_EXP/"output/summary.json",PROMPT_EXP/"output/manifest.json",
           ROOT/"src/lifemate/agent.py",ROOT/"src/lifemate/pipeline.py",ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
           PROMPT_EXP/"scripts/run_prompt.py",Path(__file__).resolve()]
    hashes={str(x.relative_to(ROOT)).replace("\\","/"):sha(x) for x in paths}
    stem=f"phowhisper-large-v2-qwen35-epoch3-en-cot-{selected}-public-test"
    return {"status":"locked","split":"public_test","samples":634,"audio_clips":1933,
        "main_transcripts":634,"history_transcripts":1299,"model":MODEL,"model_digest":DIGEST,
        "ollama_version":version,"llm_parameters":MODEL_PARAMS,"asr_parameters":ASR_PARAMS,
        "total_parameters":MODEL_PARAMS+ASR_PARAMS,"prompt_branch":selected,"prompt_sha256":actual_hash,
        "prompt_val_btc":{k:summary[k]["btc_0_100"] for k in ("baseline","variant")},
        "settings":{"temperature":0,"thinking":True,"protocol":"JSON","num_predict":CAP,"num_ctx":CONTEXT,"num_batch":BATCH,"seed":SEED,"max_action_turns":MAX_STEPS},
        "inputs_sha256":hashes,"stem":stem}
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=input_lock();path=OUT/"submission_lock.json"
    if path.exists():assert read_json(path)==lock,"Submission lock changed"
    else:write_json(path,lock)
    print(json.dumps({k:lock[k] for k in ("stem","prompt_branch","prompt_sha256","samples","audio_clips","total_parameters")},ensure_ascii=False),flush=True)
class CachedHistoryASR:
    def __init__(self,lookup):self.lookup=lookup
    def transcribe(self,path):
        p=str(Path(path).resolve())
        if p not in self.lookup:raise KeyError(f"Uncached history audio: {p}")
        return self.lookup[p]
def run():
    lock=read_json(OUT/"submission_lock.json")
    assert input_lock()==lock,"Locked input changed"
    data=find_data_root("train+public_test/train+public_test")
    items=list(load_jsonl(data/"public_test/public_test.jsonl"))
    clips=read_jsonl(ASR_PATH)
    main={x["id"]:x["transcript"] for x in clips if x["kind"]=="main"}
    by_audio={x["audio"]:x["transcript"] for x in clips if x["kind"]=="history"}
    by_id={x["id"]:x for x in items}
    hist={}
    for x in clips:
        if x["kind"]=="history":
            path=resolve_audio(data,data/"public_test",by_id[x["id"]],x["audio"])
            hist[str(path.resolve())]=by_audio[x["audio"]]
    assert len(hist)==1299
    stem=lock["stem"]
    final_dir=ROOT/"outputs/submissions"
    final=final_dir/(stem+".jsonl")
    pred_final=final_dir/(stem+"_predictions.jsonl")
    manifest_final=final_dir/(stem+"_manifest.json")
    if any(x.exists() for x in (final,pred_final,manifest_final)):
        raise FileExistsError("Submission candidate already exists")
    working=OUT/"submission_working.jsonl";details=OUT/"predictions_working.jsonl";requests=OUT/"requests.jsonl"
    prior=read_jsonl(working);prior_details=read_jsonl(details)
    assert len(prior)==len(prior_details)
    assert [x["id"] for x in prior]==[x["id"] for x in items[:len(prior)]]
    assert [x["id"] for x in prior_details]==[x["id"] for x in items[:len(prior)]]
    seed_everything(SEED);settings()
    prompt=prompt_module()
    inject=prompt.original_inject if lock["prompt_branch"]=="baseline" else prompt.variant_inject
    agent=Agent(max_new_tokens=CAP,context_limit=CONTEXT,ollama_thinking=True,ollama_native_tools=False)
    original_generate=agent._generate
    agent._generate=lambda messages:original_generate(inject(messages))
    pipe=LifeMatePipeline(data,"public_test",asr=CachedHistoryASR(hist),agent=agent,max_steps=MAX_STEPS)
    original_create=pipeline_module.create_tool_env
    active={"count":0,"env":None}
    def one_env(*args,**kwargs):
        active["count"]+=1
        if active["count"]!=1:raise RuntimeError("Multiple ToolEnv in sample")
        active["env"]=original_create(*args,**kwargs)
        return active["env"]
    pipeline_module.create_tool_env=one_env
    try:
        with working.open("a",encoding="utf-8",newline="\n") as sf,details.open("a",encoding="utf-8",newline="\n") as df,requests.open("a",encoding="utf-8",newline="\n") as rf:
            for i,item in enumerate(items[len(prior):],len(prior)+1):
                sid=item["id"];active={"count":0,"env":None}
                agent._telemetry_sample_id=sid
                telemetry_start=len(agent.request_telemetry)
                t=time.perf_counter();result=None;error=None
                try:result=pipe.run(item,transcript_override=main[sid])
                except Exception as e:error=f"{type(e).__name__}: {e}"
                env=active["env"]
                if active["count"]!=1 or env is None:raise RuntimeError(f"Missing ToolEnv for {sid}: {error}")
                calls=env.get_tool_calls();state=env.get_state_hash()
                assert isinstance(state,str) and re.fullmatch("[0-9a-f]{64}",state)
                if result is not None:assert calls==result["tool_calls"]
                latency=result["latency_seconds"] if result else time.perf_counter()-t
                row={"id":sid,"tool_calls":calls,"state_hash":state}
                detail={"id":sid,"transcript_used":main[sid],"history_asr":"PhoWhisper-large-v2",
                    "tool_calls":calls,"state_hash":state,"observations":result["observations"] if result else [],
                    "model_output":result["model_output"] if result else [],"latency_seconds":latency,
                    "runtime_error":error,"toolenv_count":active["count"]}
                sf.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n");sf.flush()
                df.write(json.dumps(detail,ensure_ascii=False,separators=(",",":"))+"\n");df.flush()
                for j,x in enumerate(agent.request_telemetry[telemetry_start:],1):
                    x["request_id"]=f"P:{sid}:r{j}"
                    x["prompt_branch"]=lock["prompt_branch"]
                    x["prompt_sha256"]=lock["prompt_sha256"]
                    rf.write(json.dumps(x,ensure_ascii=False)+"\n")
                rf.flush()
                print(f"[{i}/{len(items)}] {sid} calls={len(calls)} latency={latency:.2f}s error={error}",flush=True)
    finally:pipeline_module.create_tool_env=original_create
    assert [x["id"] for x in read_jsonl(working)]==[x["id"] for x in items]
    validator=ROOT/"exp/lifemate-pipeline/scripts/validate_submission.py"
    check=subprocess.run([sys.executable,str(validator),"--split","public_test","--data-root",str(data),"--submission",str(working)],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace")
    (OUT/"schema_validation.log").write_text(check.stdout+"\n"+check.stderr,encoding="utf-8")
    if check.returncode:raise RuntimeError("Official submission schema validation failed")
    validation=json.loads(check.stdout)
    assert validation["submission_rows"]==634 and not validation["errors"]
    final_dir.mkdir(parents=True,exist_ok=True)
    working.replace(final);details.replace(pred_final)
    detail_rows=read_jsonl(pred_final)
    lat=[x["latency_seconds"] for x in detail_rows]
    asr_time={}
    for clip in clips:
        asr_time[clip["id"]]=asr_time.get(clip["id"],0.0)+clip["inference_seconds"]
    combined=[x["latency_seconds"]+asr_time[x["id"]] for x in detail_rows]
    manifest={"status":"generated_and_schema_validated","submission":str(final.relative_to(ROOT)),"predictions":str(pred_final.relative_to(ROOT)),
        "submission_sha256":sha(final),"predictions_sha256":sha(pred_final),"source_lock":str((OUT/"submission_lock.json").relative_to(ROOT)),
        "rows":634,"schema_validation":validation,"model":MODEL,"model_digest":DIGEST,"prompt_branch":lock["prompt_branch"],
        "prompt_sha256":lock["prompt_sha256"],"asr":"PhoWhisper-large-v2 for main and history","audio_clips":1933,
        "model_parameters_total":MODEL_PARAMS+ASR_PARAMS,"runtime_errors":sum(bool(x["runtime_error"]) for x in detail_rows),
        "local_latency_mean_seconds":statistics.mean(lat),"local_over_30_seconds":sum(x>30 for x in lat),
        "local_latency_scope":"Qwen and ToolEnv on precomputed ASR transcripts; excludes ASR inference",
        "estimated_combined_asr_qwen_mean_seconds":statistics.mean(combined),
        "estimated_combined_asr_qwen_over_30_seconds":sum(x>30 for x in combined),
        "toolenv_sessions":634,"public_lb_submitted":False}
    write_json(manifest_final,manifest)
    write_json(OUT/"submission_summary.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else None
    if mode=="--prepare":prepare()
    elif mode=="--run":run()
    else:raise SystemExit("Use --prepare or --run")



