from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/qwen35-sft-pho-synthetic-v1"
OUT=EXP/"output/submission-final4-ask-stop-max6-full-thinking"
STEM="phowhisper-large-v2-qwen35-phosynth-final4-ask-stop-max6-full-thinking-public-test"
SUBMISSION_CONTEXT=16384


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def readj(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def writej(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def configure():
    maker=load("existing_final4_askstop_maker_for_max6",EXP/"scripts/make_submission_final4_ask_stop.py")
    module,pilot_summary,pilot_lock,winner=maker.configure()
    module.STEM=STEM;module.OUT=OUT;module.SUBMISSION_CONTEXT=SUBMISSION_CONTEXT
    module.base.STEM=STEM;module.base.OUT=OUT;module.base.CONTEXT=SUBMISSION_CONTEXT;module.base.MAX_STEPS=6
    original_lock=module.input_lock
    def input_lock_max6():
        lock=original_lock()
        lock["settings"]["max_action_turns"]=6
        lock["policy"]={"name":"stop after a successful sys.ask_missing_info call",
            "thinking":"fixed on for every request","max_action_turns":6,
            "implementation":"in-memory pipeline patch checks observation.ok; source pipeline unchanged"}
        lock["candidate_policy_selection"]={"basis":"explicit user request for fixed-thinking max6 public candidate",
            "full_val_raw_winner":"fixed-thinking max5 + successful ask-stop, BTC 63.8022/100",
            "full_val_adaptive_max6_raw_btc":63.1093,
            "fixed_thinking_max6_full_val":"not run; this is a requested public-test candidate, not a full-val winner claim"}
        lock["inputs_sha256"][str(Path(__file__).resolve().relative_to(ROOT)).replace("\\","/")]=sha(Path(__file__).resolve())
        lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/output/ask-stop-max6-adaptive20-val309/summary.json"]=sha(EXP/"output/ask-stop-max6-adaptive20-val309/summary.json")
        lock["inputs_sha256"]["exp/qwen35-sft-pho-synthetic-v1/output/ask-stop-max6-adaptive20-val309/comparison_lock.json"]=sha(EXP/"output/ask-stop-max6-adaptive20-val309/comparison_lock.json")
        lock["settings"]["num_ctx"]=SUBMISSION_CONTEXT
        lock["context_adjustment"]={"num_ctx":SUBMISSION_CONTEXT,
            "reason":"Use competition context maximum, consistent with the existing public candidate and known 8,265-token val309 request.",
            "benchmark_context":8192,"competition_context_limit":16384}
        return lock
    module.input_lock=input_lock_max6
    module.SELECTED_BRANCH="fixed_thinking_max6_ask_stop"
    module.PILOT_SCORE=None
    return module,pilot_summary,pilot_lock,winner

def install_successful_ask_stop(module):
    import lifemate.pipeline as pipeline_module
    source_path=Path(pipeline_module.__file__).resolve()
    original_source=source_path.read_text(encoding="utf-8")
    old="            self.agent.record_tool_result(action, observation)\n        calls = env.get_tool_calls()"
    new=("            self.agent.record_tool_result(action, observation)\n"
         "            if action.get(\"tool\") == \"sys.ask_missing_info\" and observation.get(\"ok\") is True:\n"
         "                break\n"
         "        calls = env.get_tool_calls()")
    if old not in original_source:raise RuntimeError("Could not find action-loop location for successful ask-stop")
    exec(compile(original_source.replace(old,new,1),str(source_path),"exec"),pipeline_module.__dict__)
    module.base.LifeMatePipeline=pipeline_module.LifeMatePipeline
    original_create=pipeline_module.create_tool_env
    counts={}
    def one_env(split,sample_id,data_root,*args,**kwargs):
        counts[sample_id]=counts.get(sample_id,0)+1
        if counts[sample_id]!=1:raise RuntimeError(f"Multiple ToolEnv sessions for {sample_id}")
        return original_create(split,sample_id,data_root,*args,**kwargs)
    pipeline_module.create_tool_env=one_env
    return pipeline_module,source_path,original_source,original_create,counts

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ("--prepare","--run"):raise SystemExit("Use --prepare or --run")
    module,pilot_summary,pilot_lock,pilot_winner=configure()
    OUT.mkdir(parents=True,exist_ok=True)
    selection={"status":"locked_for_generation","branch":"fixed_thinking_max6_ask_stop",
        "model":module.MODEL,"model_digest":module.DIGEST,"max_action_turns":6,"thinking":"fixed on",
        "ask_stop":"after a successful sys.ask_missing_info","context":SUBMISSION_CONTEXT,
        "full_val_reference":{"raw_winner":"fixed-thinking max5 ask-stop","btc_raw":63.80216870508133,
            "adaptive20_max6_btc_raw":63.109317866599426,
            "fixed_thinking_max6":"not full-val benchmarked"},
        "public_lb_submitted":False}
    selection_path=OUT/"winner_selection.json"
    if sys.argv[1]=="--prepare":
        module.prepare();writej(selection_path,selection)
        print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True);return
    pipeline_module,source_path,original_source,original_create,counts=install_successful_ask_stop(module)
    try:module.run()
    finally:
        exec(compile(original_source,str(source_path),"exec"),pipeline_module.__dict__)
        pipeline_module.create_tool_env=original_create
        module.base.LifeMatePipeline=pipeline_module.LifeMatePipeline
    data=module.base.find_data_root("train+public_test/train+public_test")
    public_ids=[r["id"] for r in module.base.load_jsonl(data/"public_test/public_test.jsonl")]
    if len(counts)!=634 or set(counts)!=set(public_ids) or any(x!=1 for x in counts.values()):
        raise RuntimeError("Expected exactly one ToolEnv session for each of 634 public samples")
    manifest_path=ROOT/"outputs/submissions"/(STEM+"_manifest.json")
    manifest=readj(manifest_path)
    manifest["policy"]={"thinking":"fixed on for every request","max_action_turns":6,
        "ask_stop":"break after sys.ask_missing_info returns observation.ok=true",
        "implementation":"in-memory LifeMatePipeline patch; source file unchanged"}
    manifest["candidate_policy_selection"]=selection["full_val_reference"]
    manifest["toolenv_count_audit"]={"samples":len(counts),"exactly_one_per_sample":True}
    manifest["public_lb_submitted"]=False
    manifest["public_lb_score"]=None
    manifest["note"]="Generated as the user-requested fixed-thinking max6 candidate. Max5 fixed-thinking ask-stop remains the full-val raw BTC winner; fixed-thinking max6 was not evaluated on full val309."
    writej(manifest_path,manifest);writej(OUT/"submission_summary.json",manifest)
    selection.update({"status":"public_submission_generated_and_schema_validated",
        "submission":manifest["submission"],"submission_sha256":manifest["submission_sha256"],
        "predictions":manifest["predictions"],"predictions_sha256":manifest["predictions_sha256"],
        "rows":manifest["rows"],"schema_validation":manifest["schema_validation"],
        "runtime_errors":manifest["runtime_errors"],"toolenv_count_audit":manifest["toolenv_count_audit"]})
    writej(selection_path,selection)
    print(json.dumps({"submission":manifest["submission"],"sha256":manifest["submission_sha256"],
        "rows":manifest["rows"],"schema_validation":manifest["schema_validation"],
        "runtime_errors":manifest["runtime_errors"],"toolenv_count_audit":manifest["toolenv_count_audit"],
        "public_lb_submitted":False},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
