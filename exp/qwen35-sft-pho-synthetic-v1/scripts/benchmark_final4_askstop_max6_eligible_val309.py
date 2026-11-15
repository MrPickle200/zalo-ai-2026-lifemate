from __future__ import annotations
import hashlib, importlib.util, json, random, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/qwen35-sft-pho-synthetic-v1"
OUT=EXP/"output/ask-stop-max6-eligible-val309"
MODEL="lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
BASELINE=EXP/"output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl"
SEED=20261119
DRAWS=10000

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def readj(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def readjl(p):return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if p.exists() else []
def writej(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def writejl(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in x),encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def ci(xs):
    rng=random.Random(SEED);n=len(xs)
    v=sorted(100*statistics.mean(rng.choices(xs,k=n)) for _ in range(DRAWS))
    return [v[int(.025*DRAWS)],v[int(.975*DRAWS)-1]]

def run():
    if sys.prefix!=sys.base_prefix:raise RuntimeError("Use system Python for local Ollama.")
    bench=load("benchmark_final4_for_eligible6",EXP/"scripts/benchmark_adapter_final4.py")
    ev,compare,prompt,runner,ids,items,main,history,data,lock=bench.build()
    archived=readjl(BASELINE)
    if [x["id"] for x in archived]!=ids:raise RuntimeError("Archived val309 baseline differs from canonical order.")
    old={x["id"]:x for x in archived}
    eligible=[]
    for sid in ids:
        actions=old[sid].get("model_output") or []
        if (len(actions)>=5 and all(x.get("action")=="tool_call" for x in actions[:5])
            and not any(x.get("tool")=="sys.ask_missing_info" for x in actions[:5])
            and not old[sid].get("error")):
            eligible.append(sid)
    if not eligible:raise RuntimeError("No eligible 5-action samples in locked val309.")
    OUT.mkdir(parents=True,exist_ok=True)
    lock_paths=[Path(__file__).resolve(),EXP/"scripts/benchmark_adapter_final4.py",
        EXP/"scripts/run_two_policy_val60.py",BASELINE,
        EXP/"output/adapter-final-4epoch-benchmark/val309/comparison_lock.json",
        ROOT/"src/lifemate/pipeline.py",ROOT/"src/lifemate/agent.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py"]
    payload={"experiment":"final4-q4-ask-stop-max6-eligible-val309",
        "cohort":{"n":len(eligible),"ids":eligible,"source":"val309 archived final4 max5; first five actions are tool calls and contain no ask"},
        "model":{"tag":MODEL,"digest":lock["adapter_final4"]["digest"]},
        "settings":{"thinking":True,"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,
            "num_batch":512,"seed":17,"max_steps":6,"focused_prompt":True,"stop_after_successful_ask":True},
        "baseline_predictions_sha256":sha(BASELINE),
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in lock_paths},
        "caveat":"Targeted cohort of 25 eligible val309 samples; timings local RTX 4050, not A30."}
    lp=OUT/"comparison_lock.json"; pred_path=OUT/"inference/max6/predictions-COT.jsonl"
    if lp.exists() and readj(lp)!=payload and pred_path.exists():raise RuntimeError("Existing inference output does not match this lock.")
    writej(lp,payload)
    branch=pred_path.parent;branch.mkdir(parents=True,exist_ok=True)
    done={x["id"] for x in readjl(pred_path)}
    if done-set(eligible):raise RuntimeError("Unexpected inference IDs.")
    runner.MODEL=MODEL;runner.EXPECTED_DIGEST=lock["adapter_final4"]["digest"]
    runner.CAP,runner.CONTEXT,runner.BATCH=768,8192,512
    runner.SEED,runner.MAX_STEPS=17,6
    runner.inject_cot_original=runner.inject_cot;runner.inject_cot=prompt.focused_inject
    runner.OUT=branch
    runner.model_fingerprint()
    import lifemate.asr as asr_module
    asr_module.PhoASRModel=compare.CachedPhoWhisper;asr_module.PhoASRModel.lookup=history
    patcher=load("two_policy_patch_for_eligible6",EXP/"scripts/run_two_policy_val60.py")
    import lifemate.pipeline as pipeline
    source=Path(pipeline.__file__).read_text(encoding="utf-8")
    todo=[sid for sid in eligible if sid not in done]
    try:
        patcher.patch_pipeline(pipeline,source,True)
        if todo:
            runner.warmup("COT",1)
            runner.run_block("COT",1,todo,main,items,cot=True)
    finally:patcher.patch_pipeline(pipeline,source,False)
    treatment=readjl(pred_path)
    if [x["id"] for x in treatment]!=eligible:raise RuntimeError("Eligible max6 predictions incomplete or out of order.")
    treat={x["id"]:x for x in treatment}
    from score_btc import score_sample
    details=[]
    for sid in eligible:
        b=score_sample(sid,items[sid],old[sid],data)
        t=score_sample(sid,items[sid],treat[sid],data)
        out=treat[sid].get("model_output") or []
        details.append({"id":sid,
            "baseline":{"raw":b["btc_formula_score_before_runtime_gate_diagnostic"],"gated":b["official_sample_score"],"latency":old[sid].get("latency_seconds"),"calls":old[sid].get("tool_calls")},
            "max6":{"raw":t["btc_formula_score_before_runtime_gate_diagnostic"],"gated":t["official_sample_score"],"latency":treat[sid].get("latency_seconds"),"error":bool(treat[sid].get("error")),"turns":len(out),"sixth_turn_used":len(out)>=6,"calls":treat[sid].get("tool_calls")}})
    raw=[x["max6"]["raw"]-x["baseline"]["raw"] for x in details]
    gate=[x["max6"]["gated"]-x["baseline"]["gated"] for x in details]
    summary={"experiment":payload["experiment"],"cohort":payload["cohort"],
        "baseline":{"raw":100*statistics.mean(x["baseline"]["raw"] for x in details),"gated":100*statistics.mean(x["baseline"]["gated"] for x in details),
            "mean_latency":statistics.mean(x["baseline"]["latency"] for x in details)},
        "max6":{"raw":100*statistics.mean(x["max6"]["raw"] for x in details),"gated":100*statistics.mean(x["max6"]["gated"] for x in details),
            "mean_latency":statistics.mean(x["max6"]["latency"] for x in details),"over_30s":sum(x["max6"]["latency"]>30 for x in details),
            "runtime_errors":sum(x["max6"]["error"] for x in details),"sixth_turn_used":sum(x["max6"]["sixth_turn_used"] for x in details)},
        "paired":{"raw_delta":100*statistics.mean(raw),"raw_ci95":ci(raw),"gated_delta":100*statistics.mean(gate),"gated_ci95":ci(gate),
            "wins_ties_losses":[sum(x>0 for x in raw),sum(x==0 for x in raw),sum(x<0 for x in raw)]},
        "reproducibility":{"same_calls":sum(x["max6"]["calls"]==x["baseline"]["calls"] for x in details),"n":len(details)},
        "caveat":payload["caveat"]}
    writej(OUT/"paired_details.json",details);writej(OUT/"summary.json",summary)
    b,m,p=summary["baseline"],summary["max6"],summary["paired"]
    report=["# Max-six-turn follow-up on eligible val309 samples","",
        "Selected archived final4 val309 samples whose first five outputs were tool calls, with no sys.ask_missing_info and no runtime error. For these IDs, applying ask-stop does not change the max-five prefix. Treatment reruns only the selected IDs with ask-stop retained and max_steps=6.",
        "",f"Cohort: {len(eligible)} eligible samples out of 309.","",
        "| Branch | BTC before gate | BTC after local 30s gate | >30s | Mean seconds |",
        "|---|---:|---:|---:|---:|",
        f"| Archived max-five baseline | {b['raw']:.2f} | {b['gated']:.2f} | - | {b['mean_latency']:.2f} |",
        f"| Fresh max-six treatment | {m['raw']:.2f} | {m['gated']:.2f} | {m['over_30s']}/{len(eligible)} | {m['mean_latency']:.2f} |","",
        f"Paired delta: raw {p['raw_delta']:+.2f} (95% CI {p['raw_ci95']}); after gate {p['gated_delta']:+.2f} (95% CI {p['gated_ci95']}); wins/ties/losses {p['wins_ties_losses']}.",
        f"Actually used sixth turn: {m['sixth_turn_used']}/{len(eligible)}; runtime errors {m['runtime_errors']}; identical call lists {summary['reproducibility']['same_calls']}/{len(eligible)}.",
        "","Targeted held-out diagnostic; small eligible cohort and local RTX 4050 timings are not a Public LB or A30 estimate.",""]
    (OUT/"summary.md").write_text("\n".join(report),encoding="utf-8")
    meta=readj(branch/"run-COT.json");meta["status"]="complete";meta["samples_completed"]=len(eligible);meta["max_action_turns"]=6;writej(branch/"run-COT.json",meta)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":run()
