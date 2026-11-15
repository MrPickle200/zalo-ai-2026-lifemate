from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/phowhisper-large-qwen-focused-public-v1"
OUT=EXP/"output"
SOURCE=ROOT/"exp/pho-whisper-large-qwen-public-v1"
PROMPT=ROOT/"exp/ask-lookup-disambiguation-followup-val309"
COMPARISON=ROOT/"exp/ask-lookup-phowhisper-asr-val309/output/summary.json"
MODEL="lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
DIGEST="fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
STEM="phowhisper-large-v2-qwen35-epoch3-en-cot-focused-public-test"
BASE_PATH=SOURCE/"scripts/make_submission.py"
ASR_PATH=SOURCE/"output/transcripts_audio.jsonl"

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
base=load_module("phowhisper_public_submission_base",BASE_PATH)
prompt=load_module("focused_lookup_prompt_for_public_submission",PROMPT/"scripts/benchmark.py")
runner=prompt.runner
runner.inject_cot_original=runner.inject_cot

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def read_json(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def focused_prompt_module():
    return SimpleNamespace(original_inject=prompt.original_inject,variant_inject=prompt.focused_inject)

def input_lock():
    if sys.prefix!=sys.base_prefix:raise RuntimeError("Use native system Python so local Ollama is reachable")
    data=base.find_data_root("train+public_test/train+public_test")
    items=list(base.load_jsonl(data/"public_test/public_test.jsonl"))
    clips=base.read_jsonl(ASR_PATH)
    if len(items)!=634 or len({x["id"] for x in items})!=634:raise RuntimeError("Expected 634 unique public-test tasks")
    if len(clips)!=1933 or len({x["audio"] for x in clips})!=1933:raise RuntimeError("Expected 1,933 cached PhoWhisper audio transcripts")
    main={x["id"]:x for x in clips if x.get("kind")=="main"}
    hist={x["audio"]:x for x in clips if x.get("kind")=="history"}
    if len(main)!=634 or len(hist)!=1299 or set(main)!={x["id"] for x in items}:raise RuntimeError("PhoWhisper ASR coverage mismatch")
    for item in items:
        if any(h.get("audio") and h["audio"] not in hist for h in item.get("history",[])):
            raise RuntimeError(f"Missing PhoWhisper history transcript for {item['id']}")
    prompt_lock=read_json(PROMPT/"output/benchmark_lock.json")
    base.settings()
    actual_prompt_hash=prompt.prompt_hash(prompt.focused_inject)
    if actual_prompt_hash!=prompt_lock["prompt"]["focused_sha256"]:raise RuntimeError("Focused prompt hash changed")
    version=base.api("/api/version")["version"]
    model=next(x for x in base.api("/api/tags")["models"] if x["name"]==MODEL)
    if model["digest"]!=DIGEST:raise RuntimeError("Qwen model digest changed")
    asr_lock=read_json(SOURCE/"output/asr_lock.json")
    model_file=ROOT/"models/pho-whisper-large-zaloai/model.safetensors"
    if asr_lock["model_sha256"]!=sha(model_file):raise RuntimeError("PhoWhisper model hash changed")
    val_summary=read_json(COMPARISON)
    paths=[data/"public_test/public_test.jsonl",data/"public_test/public_test_tools_catalog.json",
        ASR_PATH,SOURCE/"output/asr_lock.json",PROMPT/"output/benchmark_lock.json",COMPARISON,
        ROOT/"src/lifemate/agent.py",ROOT/"src/lifemate/pipeline.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        PROMPT/"scripts/benchmark.py",BASE_PATH,Path(__file__).resolve()]
    return {"status":"locked","split":"public_test","samples":634,"audio_clips":1933,
        "main_transcripts":634,"history_transcripts":1299,"model":MODEL,"model_digest":DIGEST,
        "ollama_version":version,"prompt_branch":"focused","prompt_sha256":actual_prompt_hash,
        "prompt_text":prompt.FOCUSED,"prompt_val_btc":{
            "QUAN_main_PhoASR_history":val_summary["groups"]["all_val309"]["quan_phoasr"]["btc_0_100"],
            "PhoWhisper_Large_main_and_history":val_summary["groups"]["all_val309"]["phowhisper_large"]["btc_0_100"]},
        "settings":{"temperature":0,"thinking":True,"protocol":"JSON","num_predict":768,
            "num_ctx":8192,"num_batch":512,"seed":17,"max_action_turns":5},
        "asr":"PhoWhisper large v2 for main and history",
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in paths},
        "stem":STEM}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=input_lock();p=OUT/"submission_lock.json"
    if p.exists() and read_json(p)!=lock:raise RuntimeError("Submission lock differs from existing artifact")
    if not p.exists():write_json(p,lock)
    print(json.dumps({k:lock[k] for k in ("stem","prompt_branch","prompt_sha256","samples","audio_clips","model_digest")},ensure_ascii=False),flush=True)

def run():
    base.EXP=EXP;base.OUT=OUT;base.ASR_PATH=ASR_PATH
    base.input_lock=input_lock
    base.prompt_module=focused_prompt_module
    base.run()
    lock=read_json(OUT/"submission_lock.json")
    manifest_path=ROOT/"outputs/submissions"/(STEM+"_manifest.json")
    manifest=read_json(manifest_path)
    manifest["prompt_branch"]="focused-post-lookup"
    manifest["prompt_sha256"]=lock["prompt_sha256"]
    manifest["prompt_val_btc"]=lock["prompt_val_btc"]
    manifest["asr"]="PhoWhisper-large-v2 for main and history"
    manifest["public_lb_submitted"]=False
    write_json(manifest_path,manifest)
    write_json(OUT/"submission_summary.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ("--prepare","--run"):raise SystemExit("use --prepare or --run")
    prepare() if sys.argv[1]=="--prepare" else run()
if __name__=="__main__":main()
