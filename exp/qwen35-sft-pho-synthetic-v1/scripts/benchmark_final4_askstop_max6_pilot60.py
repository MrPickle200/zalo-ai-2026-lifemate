from __future__ import annotations
import hashlib, importlib.util, json, random, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/qwen35-sft-pho-synthetic-v1"
OUT=EXP/"output/ask-stop-max6-pilot60"
BASE=EXP/"output/two-policy-val60"
TAG="lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
DRAWS=10000
SEED=20261118

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def read_json(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def read_jsonl(path): return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if path.exists() else []
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def boot_ci(diffs):
    rng=random.Random(SEED);n=len(diffs)
    xs=sorted(100*statistics.mean(rng.choices(diffs,k=n)) for _ in range(DRAWS))
    return [xs[int(.025*DRAWS)],xs[int(.975*DRAWS)-1]]

def build():
    base=load_module("two_policy_for_askstop_max6",EXP/"scripts/run_two_policy_val60.py")
    ev,compare,prompt,runner,ids,cohort,items,main,history,data,record,final4,manifest=base.build()
    return base,ev,prompt,runner,cohort,items,main,history,data,record

def run():
    if sys.prefix!=sys.base_prefix:raise RuntimeError("Use system Python so local Ollama is reachable.")
    base,ev,prompt,runner,cohort,items,main,history,data,record=build()
    OUT.mkdir(parents=True,exist_ok=True)
    baseline_path=BASE/"inference/fixed_ask_stop/predictions-COT.jsonl"
    baseline=read_jsonl(baseline_path)
    if [x["id"] for x in baseline]!=cohort:raise RuntimeError("Archived ask-stop control differs from pilot60.")
    lock_paths=[Path(__file__).resolve(),EXP/"scripts/run_two_policy_val60.py",
        baseline_path,BASE/"comparison_lock.json",ROOT/"src/lifemate/pipeline.py",ROOT/"src/lifemate/agent.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        ROOT/"exp/qwen35-thinking-budget-val60/output/cohort_manifest.json",
        EXP/"output/adapter-final-4epoch-benchmark/val309/comparison_lock.json"]
    lock={"experiment":"final4-q4-ask-stop-max6-pilot60","cohort":{"n":len(cohort),"ids":cohort,"source":"existing enriched fixed pilot60"},
        "model":{"tag":TAG,"digest":record["digest"]},
        "settings":{"thinking":True,"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,
            "num_batch":512,"seed":17,"max_steps":6,"focused_prompt":True,"stop_after_successful_sys_ask_missing_info":True},
        "baseline":{"max_steps":5,"path":str(baseline_path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(baseline_path)},
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in lock_paths},
        "caveat":"Enriched diagnostic pilot60; local RTX 4050 timing is not A30 performance."}
    lp=OUT/"comparison_lock.json"
    pred_path=OUT/"inference/max6/predictions-COT.jsonl"
    if lp.exists() and read_json(lp)!=lock and pred_path.exists():raise RuntimeError("Existing run output differs from current inputs.")
    write_json(lp,lock)
    branch=pred_path.parent;branch.mkdir(parents=True,exist_ok=True)
    done={x["id"] for x in read_jsonl(pred_path)}
    if done-set(cohort):raise RuntimeError("Unexpected IDs in max6 output.")
    runner.MODEL=TAG;runner.EXPECTED_DIGEST=record["digest"];runner.OUT=branch
    runner.CAP,runner.CONTEXT,runner.BATCH=768,8192,512
    runner.SEED,runner.MAX_STEPS=17,6
    version,digest=runner.model_fingerprint()
    if digest!=record["digest"]:raise RuntimeError("Q4 model digest mismatch.")
    import lifemate.pipeline as pipeline
    source=Path(pipeline.__file__).read_text(encoding="utf-8")
    try:
        base.patch_pipeline(pipeline,source,True)
        for rn,start in enumerate(range(0,len(cohort),15),1):
            todo=[sid for sid in cohort[start:start+15] if sid not in done]
            if not todo:continue
            runner.warmup("COT",rn);runner.run_block("COT",rn,todo,main,items,cot=True)
            done.update(todo);print(f"max6 block {rn}/4: {len(done)}/60",flush=True)
    finally:base.patch_pipeline(pipeline,source,False)
    rows=read_jsonl(pred_path)
    if [x["id"] for x in rows]!=cohort:raise RuntimeError("Max6 predictions incomplete.")
    baseline_map={x["id"]:x for x in baseline};treat_map={x["id"]:x for x in rows}
    from score_btc import score_sample
    details=[]
    for sid in cohort:
        b=score_sample(sid,items[sid],baseline_map[sid],data)
        t=score_sample(sid,items[sid],treat_map[sid],data)
        model_output=treat_map[sid].get("model_output") or []
        details.append({"id":sid,
            "baseline":{"raw":b["btc_formula_score_before_runtime_gate_diagnostic"],"gated":b["official_sample_score"],"latency":baseline_map[sid].get("latency_seconds")},
            "max6":{"raw":t["btc_formula_score_before_runtime_gate_diagnostic"],"gated":t["official_sample_score"],"latency":treat_map[sid].get("latency_seconds"),
                "runtime_error":bool(treat_map[sid].get("error")),"turns":len(model_output),
                "sixth_turn_used":len(model_output)>=6,
                "last_action":model_output[-1].get("action") if model_output else None,
                "ask_called":any(x.get("tool")=="sys.ask_missing_info" for x in model_output)}})
    raw=[x["max6"]["raw"]-x["baseline"]["raw"] for x in details]
    gate=[x["max6"]["gated"]-x["baseline"]["gated"] for x in details]
    current=[x for x in rows]
    summary={"experiment":lock["experiment"],"cohort":lock["cohort"],
        "baseline":{"raw":100*statistics.mean(x["baseline"]["raw"] for x in details),
            "gated":100*statistics.mean(x["baseline"]["gated"] for x in details),
            "mean_latency":statistics.mean(x["baseline"]["latency"] for x in details)},
        "max6":{"raw":100*statistics.mean(x["max6"]["raw"] for x in details),
            "gated":100*statistics.mean(x["max6"]["gated"] for x in details),
            "mean_latency":statistics.mean(x["max6"]["latency"] for x in details),
            "over_30s":sum(bool(x["max6"]["latency"] and x["max6"]["latency"]>30) for x in details),
            "runtime_errors":sum(x["max6"]["runtime_error"] for x in details),
            "sixth_turn_used":sum(x["max6"]["sixth_turn_used"] for x in details),
            "max_turn_samples":sum(x["max6"]["turns"]>=6 for x in details)},
        "paired":{"raw_delta":100*statistics.mean(raw),"raw_ci95":boot_ci(raw),
            "gated_delta":100*statistics.mean(gate),"gated_ci95":boot_ci(gate),
            "wins_ties_losses":[sum(x>0 for x in raw),sum(x==0 for x in raw),sum(x<0 for x in raw)]},
        "reproducibility":{"same_ordered_calls":sum(treat_map[s]["tool_calls"]==baseline_map[s]["tool_calls"] for s in cohort),
            "same_hashes":sum(treat_map[s]["state_hash"]==baseline_map[s]["state_hash"] for s in cohort),"n":60},
        "caveat":lock["caveat"]}
    write_json(OUT/"paired_details.json",details);write_json(OUT/"summary.json",summary)
    b,m,p=summary["baseline"],summary["max6"],summary["paired"]
    report=["# Final4 Q4 ask-stop with max 6 turns — enriched pilot60","",
        "Same final4 Q4 adapter, focused prompt, fixed thinking, cached PhoWhisper, JSON/temp0/cap768/context8192/batch512/seed17. Successful sys.ask_missing_info still ends the loop; treatment raises only the action-turn cap from five to six.",
        "","| Branch | BTC before gate | BTC after local 30s gate | >30s | Mean seconds |",
        "|---|---:|---:|---:|---:|",
        f"| Q4 ask-stop max5 baseline | {b['raw']:.2f} | {b['gated']:.2f} | - | {b['mean_latency']:.2f} |",
        f"| Q4 ask-stop max6 | {m['raw']:.2f} | {m['gated']:.2f} | {m['over_30s']}/60 | {m['mean_latency']:.2f} |","",
        f"Paired max6−max5: raw {p['raw_delta']:+.2f} (95% CI {p['raw_ci95']}); after gate {p['gated_delta']:+.2f} (95% CI {p['gated_ci95']}); wins/ties/losses {p['wins_ties_losses']}.",
        f"Sixth turn used on {m['sixth_turn_used']}/60 samples; runtime errors {m['runtime_errors']}. Same calls on {summary['reproducibility']['same_ordered_calls']}/60 and same hashes on {summary['reproducibility']['same_hashes']}/60.",
        "","Enriched diagnostic pilot60; this does not estimate full-val or Public LB. Local RTX 4050 timing is not A30 performance.",""]
    (OUT/"summary.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":run()
