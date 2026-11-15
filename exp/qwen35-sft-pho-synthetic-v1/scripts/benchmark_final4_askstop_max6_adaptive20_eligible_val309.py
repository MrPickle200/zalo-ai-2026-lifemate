from __future__ import annotations
import hashlib, importlib.util, json, random, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/qwen35-sft-pho-synthetic-v1"
OUT=EXP/"output/ask-stop-max6-adaptive20-eligible-val309"
BASELINE=EXP/"output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl"
FIXED6=EXP/"output/ask-stop-max6-eligible-val309/inference/max6/predictions-COT.jsonl"
FIXED6_LOCK=EXP/"output/ask-stop-max6-eligible-val309/comparison_lock.json"
MODEL="lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
THRESHOLD=20.0
SEED=20261120
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
    bench=load("benchmark_final4_for_adaptive6",EXP/"scripts/benchmark_adapter_final4.py")
    ev,compare,prompt,runner,ids,items,main,history,data,bench_lock=bench.build()
    archived=readjl(BASELINE); fixed=readjl(FIXED6); fixed_lock=readj(FIXED6_LOCK)
    if [x["id"] for x in archived]!=ids:raise RuntimeError("Archived val309 baseline differs from canonical order.")
    old={x["id"]:x for x in archived}; fixed_map={x["id"]:x for x in fixed}
    eligible=[sid for sid in ids if len(old[sid].get("model_output") or [])>=5
        and all(a.get("action")=="tool_call" for a in (old[sid].get("model_output") or [])[:5])
        and not any(a.get("tool")=="sys.ask_missing_info" for a in (old[sid].get("model_output") or [])[:5])
        and not old[sid].get("error")]
    if len(eligible)!=25 or [x["id"] for x in fixed]!=eligible:
        raise RuntimeError("Eligible cohort differs from completed fixed max6 run.")
    OUT.mkdir(parents=True,exist_ok=True)
    patcher=load("two_policy_for_adaptive6",EXP/"scripts/run_two_policy_val60.py")
    model_digest=bench_lock["adapter_final4"]["digest"]
    inputs=[Path(__file__).resolve(),EXP/"scripts/benchmark_adapter_final4.py",EXP/"scripts/run_two_policy_val60.py",
        BASELINE,FIXED6,FIXED6_LOCK,EXP/"output/adapter-final-4epoch-benchmark/val309/comparison_lock.json",
        ROOT/"src/lifemate/agent.py",ROOT/"src/lifemate/pipeline.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py"]
    payload={"experiment":"final4-q4-askstop-max6-adaptive20-eligible-val309",
        "cohort":{"n":len(eligible),"ids":eligible,"source":"all 25 val309 archived max5 rows with five tool calls and no ask"},
        "model":{"tag":MODEL,"digest":model_digest},
        "policy":{"thinking":"on initially; off before next request after cumulative per-sample request latency reaches 20s",
            "threshold_seconds":THRESHOLD,"action_cap":6,"stop_after_successful_ask":True},
        "settings":{"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,"num_batch":512,"seed":17,"focused_prompt":True},
        "fixed_max6_lock_sha256":sha(FIXED6_LOCK),
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in inputs},
        "caveat":"Targeted 25-sample held-out val309 cohort; local RTX 4050 timing is not A30."}
    lp=OUT/"comparison_lock.json";pred_path=OUT/"inference/adaptive20/predictions-COT.jsonl"
    if lp.exists() and readj(lp)!=payload and pred_path.exists():raise RuntimeError("Existing inference differs from this lock.")
    writej(lp,payload)
    branch=pred_path.parent;branch.mkdir(parents=True,exist_ok=True)
    done={x["id"] for x in readjl(pred_path)}
    if done-set(eligible):raise RuntimeError("Unexpected IDs in adaptive output.")
    runner.MODEL=MODEL;runner.EXPECTED_DIGEST=model_digest;runner.CAP=768;runner.CONTEXT=8192;runner.BATCH=512
    runner.SEED=17;runner.MAX_STEPS=6;runner.OUT=branch
    runner.inject_cot_original=runner.inject_cot;runner.inject_cot=prompt.focused_inject
    runner.model_fingerprint()
    import lifemate.pipeline as pipeline
    import lifemate.agent as agent_module
    base_agent=agent_module.Agent
    patcher.THRESHOLD=THRESHOLD
    source=Path(pipeline.__file__).read_text(encoding="utf-8")
    try:
        patcher.patch_pipeline(pipeline,source,True)
        agent_module.Agent=patcher.adaptive_factory(base_agent)
        todo=[sid for sid in eligible if sid not in done]
        if todo:
            runner.warmup("COT",1);runner.run_block("COT",1,todo,main,items,cot=True)
    finally:
        patcher.patch_pipeline(pipeline,source,False)
        agent_module.Agent=base_agent
    treatment=readjl(pred_path)
    if [x["id"] for x in treatment]!=eligible:raise RuntimeError("Adaptive max6 inference incomplete.")
    adaptive={x["id"]:x for x in treatment}
    reqs=readjl(branch/"requests-COT.jsonl")
    request_groups={}
    for r in reqs:request_groups.setdefault(r.get("sample_id"),[]).append(r)
    from score_btc import score_sample
    details=[]
    for sid in eligible:
        b=score_sample(sid,items[sid],old[sid],data)
        f=score_sample(sid,items[sid],fixed_map[sid],data)
        a=score_sample(sid,items[sid],adaptive[sid],data)
        outputs=adaptive[sid].get("model_output") or []
        sr=request_groups.get(sid,[])
        details.append({"id":sid,
            "max5":{"raw":b["btc_formula_score_before_runtime_gate_diagnostic"],"gated":b["official_sample_score"],"latency":old[sid].get("latency_seconds")},
            "fixed6":{"raw":f["btc_formula_score_before_runtime_gate_diagnostic"],"gated":f["official_sample_score"],"latency":fixed_map[sid].get("latency_seconds")},
            "adaptive6":{"raw":a["btc_formula_score_before_runtime_gate_diagnostic"],"gated":a["official_sample_score"],"latency":adaptive[sid].get("latency_seconds"),
                "runtime_error":bool(adaptive[sid].get("error")),"turns":len(outputs),"sixth_turn_used":len(outputs)>=6,
                "switched":any(x.get("adaptive_thinking_enabled") is False for x in sr),
                "thinking_off_requests":sum(x.get("adaptive_thinking_enabled") is False for x in sr),
                "calls":adaptive[sid].get("tool_calls")}})
    def stats(key,ref):
        d=[x[key]["raw"]-x[ref]["raw"] for x in details]
        g=[x[key]["gated"]-x[ref]["gated"] for x in details]
        return {"raw_delta":100*statistics.mean(d),"raw_ci95":ci(d),
            "gated_delta":100*statistics.mean(g),"gated_ci95":ci(g),
            "wins_ties_losses":[sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)]}
    def branch_summary(key):
        xs=[x[key] for x in details]
        return {"raw":100*statistics.mean(x["raw"] for x in xs),"gated":100*statistics.mean(x["gated"] for x in xs),
            "mean_latency":statistics.mean(x["latency"] for x in xs),"over_30s":sum(x["latency"]>30 for x in xs)}
    switched=[x["adaptive6"] for x in details]
    summary={"experiment":payload["experiment"],"cohort":payload["cohort"],
        "max5":branch_summary("max5"),"fixed6":branch_summary("fixed6"),"adaptive6":branch_summary("adaptive6"),
        "adaptive_telemetry":{"switched_samples":sum(x["switched"] for x in switched),
            "thinking_off_requests":sum(x["thinking_off_requests"] for x in switched),
            "sixth_turn_used":sum(x["sixth_turn_used"] for x in switched),
            "runtime_errors":sum(x["runtime_error"] for x in switched)},
        "paired":{"adaptive6_vs_max5":stats("adaptive6","max5"),
            "adaptive6_vs_fixed6":stats("adaptive6","fixed6"),
            "fixed6_vs_max5":stats("fixed6","max5")},
        "caveat":payload["caveat"]}
    writej(OUT/"paired_details.json",details);writej(OUT/"summary.json",summary)
    m5,f6,a6=summary["max5"],summary["fixed6"],summary["adaptive6"]
    p1,p2=summary["paired"]["adaptive6_vs_max5"],summary["paired"]["adaptive6_vs_fixed6"]
    report=["# Adaptive thinking with six-turn ask-stop — eligible val309","",
        "Same 25 eligible held-out val309 samples as the fixed max-six screen. Final4 Q4, focused prompt, PhoWhisper cache, ask-stop retained. Adaptive thinking starts on and turns off for later requests after cumulative request latency reaches 20 seconds.",
        "","| Branch | BTC before gate | BTC after local 30s gate | >30s | Mean seconds |",
        "|---|---:|---:|---:|---:|",
        f"| Max5 fixed-thinking baseline | {m5['raw']:.2f} | {m5['gated']:.2f} | - | {m5['mean_latency']:.2f} |",
        f"| Max6 fixed thinking | {f6['raw']:.2f} | {f6['gated']:.2f} | {f6['over_30s']}/25 | {f6['mean_latency']:.2f} |",
        f"| Max6 adaptive 20s | {a6['raw']:.2f} | {a6['gated']:.2f} | {a6['over_30s']}/25 | {a6['mean_latency']:.2f} |","",
        f"Adaptive−max5 paired delta: raw {p1['raw_delta']:+.2f} (95% CI {p1['raw_ci95']}); gated {p1['gated_delta']:+.2f} (95% CI {p1['gated_ci95']}).",
        f"Adaptive−fixed6 paired delta: raw {p2['raw_delta']:+.2f} (95% CI {p2['raw_ci95']}); gated {p2['gated_delta']:+.2f} (95% CI {p2['gated_ci95']}).",
        f"Adaptive switched {summary['adaptive_telemetry']['switched_samples']}/25 samples and disabled thinking on {summary['adaptive_telemetry']['thinking_off_requests']} requests; turn six was used on {summary['adaptive_telemetry']['sixth_turn_used']}/25; runtime errors {summary['adaptive_telemetry']['runtime_errors']}.",
        "","Targeted held-out diagnostic; local RTX 4050 timing is not A30 performance or a Public LB estimate.",""]
    (OUT/"summary.md").write_text("\n".join(report),encoding="utf-8")
    meta=readj(branch/"run-COT.json");meta["status"]="complete";meta["samples_completed"]=len(eligible);meta["max_action_turns"]=6;meta["adaptive_threshold_seconds"]=THRESHOLD;writej(branch/"run-COT.json",meta)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":run()
