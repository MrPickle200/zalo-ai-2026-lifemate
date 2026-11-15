from __future__ import annotations
import hashlib, importlib.util, json, random, statistics, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/qwen35-sft-pho-synthetic-v1"
OUT=EXP/"output/ask-stop-max6-adaptive20-val309"
ARCHIVED=EXP/"output/adapter-final-4epoch-benchmark/val309/inference/adapter_final4/predictions-COT.jsonl"
MODEL="lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
THRESHOLD=20.0
SEED=20261121
DRAWS=10000
BLOCK=15

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
def call_key(call):return json.dumps({"name":call.get("name"),"arguments":call.get("arguments",{})},sort_keys=True,ensure_ascii=False,separators=(",",":"))
def ci(xs,seed=SEED):
    rng=random.Random(seed);n=len(xs)
    v=sorted(100*statistics.mean(rng.choices(xs,k=n)) for _ in range(DRAWS))
    return [v[int(.025*DRAWS)],v[int(.975*DRAWS)-1]]
def label(item):
    x=item.get("label",{})
    return json.loads(x) if isinstance(x,str) else x

def replay_hash(sample_id,calls,data):
    from lifemate.tool_executor import create_tool_env
    env=create_tool_env("train",sample_id,data)
    for call in calls:
        name=call.get("name");args=call.get("arguments",{})
        env.validate(name,args)
        env.call(name,args)
    observed=env.get_tool_calls()
    if [call_key(x) for x in observed]!=[call_key(x) for x in calls]:
        raise RuntimeError(f"ToolEnv replay call list changed for {sample_id}")
    return env.get_state_hash(),env.get_state()

def make_askstop_baseline(archived,ids,data):
    out=[]
    truncated=0
    for original in archived:
        row=json.loads(json.dumps(original))
        actions=row.get("model_output") or [];calls=row.get("tool_calls") or [];obs=row.get("observations") or []
        action_calls=[a for a in actions if a.get("action")=="tool_call"]
        if len(action_calls)!=len(calls) or len(obs)!=len(calls):
            raise RuntimeError(f"Cannot align actions, calls, and observations for {row['id']}")
        stop_action=None;stop_call_count=None
        call_index=0
        for action_index,action in enumerate(actions):
            if action.get("action")!="tool_call":continue
            observation=obs[call_index]
            if action.get("tool")!=observation.get("tool"):
                raise RuntimeError(f"Action/observation mismatch for {row['id']} at {call_index}")
            if action.get("tool")=="sys.ask_missing_info" and observation.get("result",{}).get("ok") is True:
                stop_action=action_index+1;stop_call_count=call_index+1;break
            call_index+=1
        if stop_action is not None:
            dropped=actions[stop_action:]
            dropped_latency=sum(float((a.get("_diagnostics") or {}).get("latency_seconds") or 0) for a in dropped)
            row["model_output"]=actions[:stop_action]
            row["tool_calls"]=calls[:stop_call_count]
            row["observations"]=obs[:stop_call_count]
            row["latency_seconds"]=max(0.0,float(row.get("latency_seconds") or 0)-dropped_latency)
            row["state_hash"],row["final_state"]=replay_hash(row["id"],row["tool_calls"],data)
            truncated+=1
        out.append(row)
    if [r["id"] for r in out]!=ids:raise RuntimeError("Reconstructed fixed max5 baseline is not canonical val309")
    return out,truncated

def summarize_branch(predictions,items,data,ids):
    from score_btc import score_sample
    detail=[]
    for sid in ids:
        pred=predictions[sid]
        sc=score_sample(sid,items[sid],pred,data)
        detail.append({"id":sid,"raw":sc["btc_formula_score_before_runtime_gate_diagnostic"],
            "gated":sc["official_sample_score"],"hash_match":sc["hash_match"],
            "runtime_error":bool(pred.get("error")),"over_30s":sc["runtime_over_30s"],
            "latency":pred.get("latency_seconds"),"ask_count":sum(c.get("name")=="sys.ask_missing_info" for c in pred.get("tool_calls") or []),
            "gold_ask":any(c.get("name")=="sys.ask_missing_info" for c in label(items[sid]).get("tool_calls",[])),
            "gold_ask_exact":any(call_key(a)==call_key(g) for a in (pred.get("tool_calls") or [])
                for g in label(items[sid]).get("tool_calls",[]) if g.get("name")=="sys.ask_missing_info")})
    return detail

def branch_stats(rows):
    lat=[x["latency"] for x in rows if x["latency"] is not None]
    n=len(rows)
    return {"n":n,"raw":100*statistics.mean(x["raw"] for x in rows),
        "gated":100*statistics.mean(x["gated"] for x in rows),
        "hash_matches":sum(x["hash_match"] for x in rows),"runtime_errors":sum(x["runtime_error"] for x in rows),
        "over_30s":sum(x["over_30s"] for x in rows),"mean_latency":statistics.mean(lat) if lat else None,
        "ask_count":sum(x["ask_count"] for x in rows),"exact_gold_ask_samples":sum(x["gold_ask_exact"] for x in rows)}
def paired(a,b,seed):
    raw=[x["raw"]-y["raw"] for x,y in zip(a,b)];gated=[x["gated"]-y["gated"] for x,y in zip(a,b)]
    return {"raw_delta":100*statistics.mean(raw),"raw_ci95":ci(raw,seed),
        "gated_delta":100*statistics.mean(gated),"gated_ci95":ci(gated,seed+1),
        "raw_wins_ties_losses":[sum(x>0 for x in raw),sum(x==0 for x in raw),sum(x<0 for x in raw)],
        "gated_wins_ties_losses":[sum(x>0 for x in gated),sum(x==0 for x in gated),sum(x<0 for x in gated)]}

def run():
    if sys.prefix!=sys.base_prefix:raise RuntimeError("Use system Python so local Ollama is reachable")
    bench=load("benchmark_final4_for_adaptive_val309",EXP/"scripts/benchmark_adapter_final4.py")
    ev,compare,prompt,runner,ids,items,main,history,data,model_lock=bench.build()
    archived=readjl(ARCHIVED)
    if [r["id"] for r in archived]!=ids or len(ids)!=309:raise RuntimeError("Archived max5 baseline differs from locked canonical val309")
    model_digest=model_lock["adapter_final4"]["digest"]
    baseline_path=OUT/"inference/fixed5_askstop_replayed/predictions-COT.jsonl"
    treatment_path=OUT/"inference/adaptive20_max6/predictions-COT.jsonl"
    inputs=[Path(__file__).resolve(),EXP/"scripts/benchmark_adapter_final4.py",EXP/"scripts/run_two_policy_val60.py",
        EXP/"scripts/evaluate_val309_checkpoints.py",EXP/"output/adapter-final-4epoch-benchmark/val309/comparison_lock.json",
        ARCHIVED,ROOT/"src/lifemate/agent.py",ROOT/"src/lifemate/pipeline.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",ROOT/"exp/btc-scoring/scripts/score_btc.py"]
    payload={"experiment":"final4-q4-askstop-max6-adaptive20-val309","cohort":{"count":309,"ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "model":{"tag":MODEL,"digest":model_digest},
        "control":{"method":"archived fixed-thinking max5 val309 predictions, truncated at first successful sys.ask_missing_info and replayed through official ToolEnv",
            "source_sha256":sha(ARCHIVED)},
        "treatment":{"method":"fresh adaptive-thinking inference with six-action cap and successful ask-stop guard",
            "threshold_seconds":THRESHOLD,"action_cap":6,"initial_thinking":True},
        "settings":{"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,"num_batch":512,"seed":17,
            "focused_prompt_sha256":"88b0264bd512a90368d0a64aa2241ff59da2660790c0725781358986bb140785",
            "asr":"PhoWhisper Large v2 cached main/history"},
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in inputs},
        "caveat":"Full held-out val309. Control uses archived predictions and removes post-ask model requests; treatment is freshly inferred. Local RTX 4050 runtime is not A30."}
    OUT.mkdir(parents=True,exist_ok=True);lock_path=OUT/"comparison_lock.json"
    if lock_path.exists() and readj(lock_path)!=payload:raise RuntimeError("Existing val309 experiment lock differs")
    if not lock_path.exists() and (baseline_path.exists() or treatment_path.exists()):raise RuntimeError("Predictions exist without their experiment lock")
    writej(lock_path,payload)
    if baseline_path.exists():
        baseline=readjl(baseline_path)
        if [r["id"] for r in baseline]!=ids:raise RuntimeError("Saved reconstructed baseline is incomplete/noncanonical")
        truncated=sum(1 for a,b in zip(archived,baseline) if len(b.get("tool_calls") or [])<len(a.get("tool_calls") or []))
    else:
        baseline,truncated=make_askstop_baseline(archived,ids,data)
        writejl(baseline_path,baseline)
    from score_btc import score_sample
    baseline_details=summarize_branch({r["id"]:r for r in baseline},items,data,ids)
    writej(OUT/"baseline_summary.json",{"branch":"fixed-thinking max5 with posthoc successful ask-stop","truncated_samples":truncated,
        "metrics":branch_stats(baseline_details)})
    done=readjl(treatment_path)
    if [r["id"] for r in done]!=ids[:len(done)]:raise RuntimeError("Adaptive predictions are not a canonical val309 prefix")
    runner.MODEL=MODEL;runner.EXPECTED_DIGEST=model_digest;runner.CAP=768;runner.CONTEXT=8192;runner.BATCH=512
    runner.SEED=17;runner.MAX_STEPS=6;runner.OUT=treatment_path.parent
    runner.inject_cot_original=runner.inject_cot;runner.inject_cot=prompt.focused_inject
    runner.seed_everything(17);runner.model_fingerprint()
    import lifemate.pipeline as pipeline
    import lifemate.agent as agent_module
    patcher=load("adaptive20_factory_for_fullval",EXP/"scripts/run_two_policy_val60.py")
    patcher.THRESHOLD=THRESHOLD
    original_pipeline_source=Path(pipeline.__file__).read_text(encoding="utf-8")
    old="            self.agent.record_tool_result(action, observation)\n        calls = env.get_tool_calls()"
    new=("            self.agent.record_tool_result(action, observation)\n"
         "            if action.get(\"tool\") == \"sys.ask_missing_info\" and observation.get(\"ok\") is True:\n"
         "                break\n"
         "        calls = env.get_tool_calls()")
    if old not in original_pipeline_source:raise RuntimeError("Could not locate pipeline point for successful ask-stop")
    base_agent=agent_module.Agent
    import lifemate.asr as asr_module
    asr_module.PhoASRModel=compare.CachedPhoWhisper;asr_module.PhoASRModel.lookup=history
    try:
        exec(compile(original_pipeline_source.replace(old,new,1),str(pipeline.__file__),"exec"),pipeline.__dict__)
        agent_module.Agent=patcher.adaptive_factory(base_agent)
        for round_no,start in enumerate(range(0,len(ids),BLOCK),1):
            block=ids[start:start+BLOCK];saved=readjl(treatment_path);completed={r["id"] for r in saved}
            todo=[sid for sid in block if sid not in completed]
            if not todo:continue
            runner.warmup("COT",round_no)
            while todo:
                before=len(readjl(treatment_path))
                try:runner.run_block("COT",round_no,todo,main,items,cot=True)
                except Exception:
                    after=readjl(treatment_path)
                    if len(after)<=before or not after[-1].get("error"):raise
                    error=after[-1].get("error","")
                    if "exceeds the available context size" not in error:
                        raise RuntimeError(f"Unexpected inference error at {after[-1]['id']}: {error}")
                completed={r["id"] for r in readjl(treatment_path)}
                todo=[sid for sid in block if sid not in completed]
            meta_path=treatment_path.parent/"run-COT.json";meta=readj(meta_path)
            meta.update({"status":"complete" if len(readjl(treatment_path))==start+len(block) else "in_progress",
                "thinking":"adaptive; starts true, turns off after cumulative request latency reaches 20 seconds",
                "adaptive_threshold_seconds":THRESHOLD,"ask_stop":"after successful sys.ask_missing_info",
                "max_action_turns":6,"cohort_ids":ids,"execution_order":"canonical blocks of 15"})
            writej(meta_path,meta)
            print(f"val309 adaptive block {round_no}/21 complete; {len(readjl(treatment_path))}/309",flush=True)
    finally:
        exec(compile(original_pipeline_source,str(pipeline.__file__),"exec"),pipeline.__dict__)
        agent_module.Agent=base_agent
    treatment=readjl(treatment_path)
    if [r["id"] for r in treatment]!=ids:raise RuntimeError("Adaptive max6 inference incomplete")
    meta=readj(treatment_path.parent/"run-COT.json");meta["status"]="complete";meta["samples_completed"]=309;writej(treatment_path.parent/"run-COT.json",meta)
    adaptive_details=summarize_branch({r["id"]:r for r in treatment},items,data,ids)
    writejl(OUT/"fixed5_details.jsonl",baseline_details);writejl(OUT/"adaptive6_details.jsonl",adaptive_details)
    reqs=readjl(treatment_path.parent/"requests-COT.jsonl")
    grouped={}
    for req in reqs:grouped.setdefault(req.get("sample_id"),[]).append(req)
    switched_samples=sum(any(r.get("adaptive_thinking_enabled") is False for r in grouped.get(sid,[])) for sid in ids)
    off_requests=sum(r.get("adaptive_thinking_enabled") is False for r in reqs)
    asks=sum(any(c.get("name")=="sys.ask_missing_info" for c in (p.get("tool_calls") or [])) for p in treatment)
    successful_asks=sum(any(o.get("tool")=="sys.ask_missing_info" and o.get("result",{}).get("ok") is True for o in (p.get("observations") or [])) for p in treatment)
    sixth=sum(sum(a.get("action")=="tool_call" for a in (p.get("model_output") or []))>=6 for p in treatment)
    gold_ask_ids={sid for sid in ids if any(c.get("name")=="sys.ask_missing_info" for c in label(items[sid]).get("tool_calls",[]))}
    control_by={x["id"]:x for x in baseline_details};adaptive_by={x["id"]:x for x in adaptive_details}
    summary={"experiment":payload["experiment"],"cohort":{"n":309,"gold_ask_n":len(gold_ask_ids),"no_gold_ask_n":309-len(gold_ask_ids)},
        "baseline_truncated_samples":truncated,"control":branch_stats(baseline_details),"adaptive_max6":branch_stats(adaptive_details),
        "adaptive_telemetry":{"switched_samples":switched_samples,"thinking_off_requests":off_requests,
            "samples_using_sixth_tool_call":sixth,"samples_with_successful_ask":successful_asks,"samples_with_any_ask":asks,
            "runtime_errors":sum(x["runtime_error"] for x in adaptive_details)},
        "paired_adaptive_minus_control":paired([adaptive_by[sid] for sid in ids],[control_by[sid] for sid in ids],SEED),
        "subgroups":{},"caveat":payload["caveat"]}
    for name,subset in (("gold_ask_47",[sid for sid in ids if sid in gold_ask_ids]),("no_gold_ask_262",[sid for sid in ids if sid not in gold_ask_ids])):
        c=[control_by[sid] for sid in subset];a=[adaptive_by[sid] for sid in subset]
        summary["subgroups"][name]={"control":branch_stats(c),"adaptive_max6":branch_stats(a),"paired_adaptive_minus_control":paired(a,c,SEED+(1 if name=="gold_ask_47" else 2))}
    writej(OUT/"summary.json",summary)
    b,a=summary["control"],summary["adaptive_max6"];d=summary["paired_adaptive_minus_control"]
    report=["# Final4 ask-stop + adaptive 20s + max6 — full val309","",
        "Control is the archived final4 fixed-thinking max5 val309 run, reconstructed by truncating after the first successful sys.ask_missing_info and replaying the resulting calls through official ToolEnv. Treatment is fresh adaptive 20s inference with max six actions and a successful ask-stop guard. Both use the same locked 309 IDs, Q4 adapter, focused prompt, cached PhoWhisper, and generation settings.","",
        "| Branch | BTC raw | BTC after local 30s gate | >30s | Mean latency | Hash matches | Runtime errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Fixed thinking, max5 + replayed ask-stop | {b['raw']:.2f} | {b['gated']:.2f} | {b['over_30s']}/309 | {b['mean_latency']:.2f}s | {b['hash_matches']}/309 | {b['runtime_errors']} |",
        f"| Adaptive 20s, max6 + successful ask-stop | {a['raw']:.2f} | {a['gated']:.2f} | {a['over_30s']}/309 | {a['mean_latency']:.2f}s | {a['hash_matches']}/309 | {a['runtime_errors']} |","",
        f"Adaptive−control paired delta: raw {d['raw_delta']:+.2f} (95% CI [{d['raw_ci95'][0]:+.2f},{d['raw_ci95'][1]:+.2f}]); gated {d['gated_delta']:+.2f} (95% CI [{d['gated_ci95'][0]:+.2f},{d['gated_ci95'][1]:+.2f}]).",
        f"Thinking switched off in {switched_samples}/309 samples and {off_requests} requests; {sixth}/309 used six tool calls; {successful_asks}/309 had a successful ask-stop. Reconstructed baseline truncated {truncated}/309 archived predictions.",
        "","Paired bootstrap: 10,000 resamples. BTC is official ToolEnv scoring; the 30s mask is a local diagnostic. Full held-out val309; local RTX 4050 timing with cached ASR, not A30 timing or Public LB.",""]
    (OUT/"summary.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":run()
