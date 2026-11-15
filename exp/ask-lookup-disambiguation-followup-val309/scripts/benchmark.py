from __future__ import annotations
import hashlib, importlib.util, json, math, os, random, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/ask-lookup-disambiguation-followup-val309"
OUT=EXP/"output"
PREV=ROOT/"exp/ask-ambiguity-prompt-val309"
PREV_OUT=PREV/"output"
MODEL="lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
DIGEST="fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
BASE_PROMPT_HASH="cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950"
OLD="Call sys.ask_missing_info only when required information is genuinely missing and cannot be safely inferred; ask only for the missing fields and never ask again for information already available."
FOCUSED="After lookup, use an ID only when the request and conversation history leave one clear candidate; if multiple candidates remain plausible, call sys.ask_missing_info for the intended downstream tool and only the unresolved ID field(s), rather than choosing arbitrarily."
SEED=17
BOOTSTRAP_SEED=20261113
BLOCK_SIZE=15

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def read_json(path:Path): return json.loads(path.read_text(encoding="utf-8-sig"))
def read_jsonl(path:Path):
    if not path.exists(): return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def write_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def load_runner():
    p=ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py"
    spec=importlib.util.spec_from_file_location("focused_lookup_runner",p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
runner=load_runner()

def parse_label(item):
    lab=item.get("label",{})
    return json.loads(lab) if isinstance(lab,str) else lab
def original_inject(messages):
    return runner.inject_cot_original(messages)
def focused_inject(messages):
    result=runner.inject_cot_original(messages)
    systems=[m for m in result if m.get("role")=="system"]
    if len(systems)!=1 or systems[0]["content"].count(OLD)!=1:
        raise RuntimeError("Expected one baseline ask_missing_info sentence")
    systems[0]["content"]=systems[0]["content"].replace(OLD,FOCUSED,1)
    return result
def prompt_hash(injector):
    from lifemate.agent import Agent
    os.environ.update({"LIFEMATE_LLM_BACKEND":"ollama","LIFEMATE_OLLAMA_THINK":"true",
        "LIFEMATE_OLLAMA_PROTOCOL":"json","LIFEMATE_PROMPT_LANGUAGE":"en"})
    agent=Agent(max_new_tokens=768,context_limit=8192,ollama_thinking=True,ollama_native_tools=False)
    captured=[]
    def capture(messages):
        captured.append(next(m["content"] for m in injector(messages) if m.get("role")=="system"))
        return '{"action":"stop"}'
    agent._generate=capture
    agent.next_action("prompt hash probe",{},[],[],[])
    if len(captured)!=1: raise RuntimeError("Prompt hash probe did not capture one system prompt")
    return hashlib.sha256(captured[0].encode()).hexdigest()

def build_inputs():
    val_ids=read_json(ROOT/"data/splits/val_ids.json")
    if len(val_ids)!=309: raise RuntimeError(f"Expected 309 val IDs, got {len(val_ids)}")
    base_manifest=read_json(PREV_OUT/"manifest.json")
    if base_manifest["ids"]!=val_ids: raise RuntimeError("Prior baseline cohort order differs")
    data=runner.find_data_root("train+public_test/train+public_test")
    items={r["id"]:r for r in runner.load_jsonl(data/"train/train.jsonl")}
    gold_ask_ids=[sid for sid in val_ids if any(c.get("name")=="sys.ask_missing_info" for c in parse_label(items[sid]).get("tool_calls",[]))]
    if len(gold_ask_ids)!=47: raise RuntimeError(f"Expected 47 audited gold-ask IDs, got {len(gold_ask_ids)}")
    fresh_ids=[sid for sid in val_ids if sid not in set(gold_ask_ids)]
    if len(fresh_ids)!=262: raise RuntimeError(f"Expected 262 complement IDs, got {len(fresh_ids)}")
    transcript_path=Path(base_manifest["transcript_path"])
    transcripts={r["id"]:r["prediction"] for r in runner.read_jsonl(transcript_path)}
    if set(transcripts)!=set(val_ids): raise RuntimeError("Transcript cache IDs differ from val309")
    baseline_rows=read_jsonl(PREV_OUT/"baseline/predictions-COT.jsonl")
    if [r["id"] for r in baseline_rows]!=val_ids: raise RuntimeError("Archived baseline predictions are incomplete or out of order")
    baseline_details=read_jsonl(PREV_OUT/"paired_details.jsonl")
    if [r["id"] for r in baseline_details]!=val_ids: raise RuntimeError("Archived paired detail IDs differ from val309")
    return val_ids,gold_ask_ids,fresh_ids,items,transcripts,transcript_path,base_manifest,baseline_rows,baseline_details

def lock_payload():
    val_ids,gold_ask_ids,fresh_ids,items,transcripts,transcript_path,base_manifest,baseline_rows,baseline_details=build_inputs()
    runner.MODEL=MODEL; runner.EXPECTED_DIGEST=DIGEST
    version,digest=runner.model_fingerprint()
    if digest!=DIGEST or version!=base_manifest["ollama_version"]:
        raise RuntimeError("Ollama version or model digest changed")
    runner.inject_cot_original=runner.inject_cot
    base_hash=prompt_hash(original_inject)
    focused_hash=prompt_hash(focused_inject)
    if base_hash!=BASE_PROMPT_HASH or base_hash!=base_manifest["baseline_prompt_sha256"]:
        raise RuntimeError(f"Baseline prompt hash mismatch: {base_hash}")
    if focused_hash==base_hash: raise RuntimeError("Focused prompt did not change")
    data=runner.find_data_root("train+public_test/train+public_test")
    input_paths=[
        ROOT/"data/splits/val_ids.json",
        data/"train/train.jsonl",
        transcript_path,
        PREV_OUT/"manifest.json",
        PREV_OUT/"baseline/predictions-COT.jsonl",
        PREV_OUT/"paired_details.jsonl",
        ROOT/"src/lifemate/agent.py",
        ROOT/"src/lifemate/pipeline.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        PREV/"scripts/run_prompt.py",
        EXP/"scripts/benchmark.py",
        EXP/"README.md",
    ]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in input_paths}
    ids_hash=lambda xs:hashlib.sha256("\n".join(xs).encode()).hexdigest()
    return {
        "experiment":EXP.name,
        "cohort":{"all_val_ids":val_ids,"all_ids_sha256":ids_hash(val_ids),
            "audited_gold_ask_ids":gold_ask_ids,"audited_47_sha256":ids_hash(gold_ask_ids),
            "primary_fresh_ids":fresh_ids,"primary_262_sha256":ids_hash(fresh_ids),
            "selection":"all val309 excluding the 47 gold ask IDs previously audited; 47 retained as diagnostic only"},
        "model":{"tag":MODEL,"digest":digest,"ollama_version":version},
        "prompt":{"baseline_sha256":base_hash,"focused_sha256":focused_hash,
            "replaced_sentence":OLD,"focused_sentence":FOCUSED},
        "settings":{"thinking":True,"temperature":0,"protocol":"JSON","num_predict":768,
            "num_ctx":8192,"num_batch":512,"seed":SEED,"max_action_turns":5},
        "transcript":{"path":str(transcript_path),"sha256":sha(transcript_path),
            "primary_asr":"QUAN checkpoint-verification-rerun val309","history_asr":"PhoASRModel"},
        "baseline":{"path":str(PREV_OUT/"baseline/predictions-COT.jsonl"),
            "predictions_sha256":sha(PREV_OUT/"baseline/predictions-COT.jsonl"),
            "details_sha256":sha(PREV_OUT/"paired_details.jsonl")},
        "input_sha256":hashes,
    }

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=lock_payload()
    p=OUT/"benchmark_lock.json"
    if p.exists():
        if read_json(p)!=lock: raise RuntimeError("Benchmark lock differs from existing output")
    else: write_json(p,lock)
    write_json(OUT/"cohort_manifest.json",lock["cohort"])
    print(json.dumps({"all_val":309,"fresh_primary":262,"audited_diagnostic":47,
        "prompt_sha256":lock["prompt"]["focused_sha256"],"model_digest":lock["model"]["digest"]},ensure_ascii=False),flush=True)

def run():
    if sys.prefix!=sys.base_prefix: raise RuntimeError("Use native system Python for Ollama")
    lock=read_json(OUT/"benchmark_lock.json")
    current=lock_payload()
    if current!=lock: raise RuntimeError("Inputs, model, or prompt lock changed")
    val_ids,gold_ask_ids,fresh_ids,items,transcripts,transcript_path,base_manifest,baseline_rows,baseline_details=build_inputs()
    runner.MODEL=MODEL; runner.EXPECTED_DIGEST=DIGEST
    runner.inject_cot_original=runner.inject_cot
    runner.inject_cot=focused_inject
    runner.seed_everything(SEED)
    branch_out=OUT/"focused"; branch_out.mkdir(parents=True,exist_ok=True)
    runner.OUT=branch_out
    execution=[{"round":i+1,"ids":val_ids[i:i+BLOCK_SIZE],"branch":"focused"} for i in range(0,len(val_ids),BLOCK_SIZE)]
    write_json(OUT/"execution_order.json",{"block_size":BLOCK_SIZE,"baseline":"reused exact archived baseline",
        "focused_blocks":execution,"paired_by_id":True})
    for ri,block in enumerate((val_ids[i:i+BLOCK_SIZE] for i in range(0,len(val_ids),BLOCK_SIZE)),1):
        done={r["id"] for r in read_jsonl(branch_out/"predictions-COT.jsonl")}
        todo=[sid for sid in block if sid not in done]
        if not todo: continue
        runner.warmup("COT",ri)
        while todo:
            before=len(read_jsonl(branch_out/"predictions-COT.jsonl"))
            try:
                runner.run_block("COT",ri,todo,transcripts,items,cot=True)
            except Exception:
                after=read_jsonl(branch_out/"predictions-COT.jsonl")
                if len(after)<=before or not after[-1].get("error"): raise
            done={r["id"] for r in read_jsonl(branch_out/"predictions-COT.jsonl")}
            todo=[sid for sid in block if sid not in done]
        print(f"focused block {ri}/{len(execution)} complete; total={len(read_jsonl(branch_out/'predictions-COT.jsonl'))}/309",flush=True)
    rows=read_jsonl(branch_out/"predictions-COT.jsonl")
    if [r["id"] for r in rows]!=val_ids: raise RuntimeError("Focused predictions missing or out of canonical order")
    meta_path=branch_out/"run-COT.json"
    if meta_path.exists():
        meta=read_json(meta_path); meta["status"]="complete"; meta["samples_completed"]=len(rows)
        meta["execution_order"]="focused prompt only in 15-ID blocks; archived baseline reused"
        write_json(meta_path,meta)
    print("focused prompt inference complete",flush=True)

def pct_mean(xs): return 100*statistics.mean(xs) if xs else None
def percentile(xs,p):
    if not xs:return None
    xs=sorted(xs);return xs[min(len(xs)-1,math.ceil(p*len(xs))-1)]
def paired_ci(deltas,seed=BOOTSTRAP_SEED,nboot=10000):
    if not deltas:return [None,None]
    rng=random.Random(seed)
    vals=sorted(100*statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(nboot))
    return [vals[int(.025*nboot)],vals[int(.975*nboot)-1]]
def ask_calls(pred):
    return [c for c in (pred.get("tool_calls") or []) if c.get("name")=="sys.ask_missing_info"]
def candidate_multiplicity(pred):
    multi=0;lookups=0
    for obs in pred.get("observations") or []:
        tool=str(obs.get("tool","")).lower()
        if not any(t in tool for t in ("search","list","find","track")): continue
        result=obs.get("result") or {}
        arrays=[v for v in result.values() if isinstance(v,list)]
        if arrays:
            lookups+=1
            if any(len(v)>1 for v in arrays): multi+=1
    return lookups,multi

def analyze():
    val_ids,gold_ask_ids,fresh_ids,items,transcripts,transcript_path,base_manifest,baseline_rows,baseline_details=build_inputs()
    pred_path=OUT/"focused/predictions-COT.jsonl"
    focused={r["id"]:r for r in read_jsonl(pred_path)}
    baseline={r["id"]:r for r in baseline_rows}
    old_details={r["id"]:r for r in baseline_details}
    if set(focused)!=set(val_ids): raise RuntimeError("Focused predictions do not cover val309")
    data=runner.find_data_root("train+public_test/train+public_test")
    from score_btc import score_sample,call_key
    detail_rows=[]
    for sid in val_ids:
        gold=items[sid].get("label",{})
        if isinstance(gold,str): gold=json.loads(gold)
        gold_calls=gold.get("tool_calls",[])
        gold_asks=[c for c in gold_calls if c.get("name")=="sys.ask_missing_info"]
        new=focused[sid]
        s=score_sample(sid,items[sid],new,data)
        new_asks=ask_calls(new)
        exact=any(call_key(a)==call_key(g) for a in new_asks for g in gold_asks)
        lookups,multi=candidate_multiplicity(new)
        old=old_details[sid]["baseline"]
        detail_rows.append({
            "id":sid,"group":"audited_47_diagnostic" if sid in set(gold_ask_ids) else "fresh_262_primary",
            "gold_ask":bool(gold_asks),
            "baseline":{"btc":old["score"],"btc_before_runtime_gate":old["score_before_runtime_gate"],
                "ask":old["ask"],"ask_exact":old["ask_exact"],"runtime_error":old["runtime_error"],
                "hash_match":old["hash_match"],"latency_seconds":old["latency_seconds"]},
            "focused":{"btc":s["official_sample_score"],
                "btc_before_runtime_gate":s["btc_formula_score_before_runtime_gate_diagnostic"],
                "ask":bool(new_asks),"ask_exact":exact,
                "runtime_error":new.get("error"),"hash_match":s["hash_match"],
                "latency_seconds":new.get("latency_seconds"),"lookup_count":lookups,
                "multi_candidate_lookup_count":multi},
        })
    (OUT/"paired_details.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in detail_rows),encoding="utf-8")
    summary={}
    for group_name,group_ids in (("all_val309",val_ids),("fresh_262_primary",fresh_ids),("audited_47_diagnostic",gold_ask_ids)):
        group=[x for x in detail_rows if x["id"] in set(group_ids)]
        if not group: continue
        block={}
        for branch in ("baseline","focused"):
            scores=[x[branch]["btc"] for x in group]
            ungated=[x[branch]["btc_before_runtime_gate"] for x in group]
            lat=[x[branch]["latency_seconds"] for x in group if x[branch]["latency_seconds"] is not None]
            gold_n=sum(x["gold_ask"] for x in group)
            asks=sum(x[branch]["ask"] for x in group)
            exact=sum(x[branch]["ask_exact"] for x in group)
            missed=sum(x["gold_ask"] and not x[branch]["ask_exact"] for x in group)
            extra=sum((not x["gold_ask"]) and x[branch]["ask"] for x in group)
            block[branch]={
                "btc_0_100":pct_mean(scores),"btc_before_runtime_gate_0_100":pct_mean(ungated),
                "hash_matches":sum(x[branch]["hash_match"] for x in group),
                "runtime_errors":sum(bool(x[branch]["runtime_error"]) for x in group),
                "mean_latency_seconds":statistics.mean(lat) if lat else None,
                "p95_latency_seconds":percentile(lat,.95),
                "gold_ask_ids":gold_n,"ask_calls_on_ids":asks,"exact_gold_ask_ids":exact,
                "missed_gold_asks":missed,"ask_calls_outside_gold_ask_ids":extra,
            }
        deltas=[x["focused"]["btc"]-x["baseline"]["btc"] for x in group]
        ungated_delta=[x["focused"]["btc_before_runtime_gate"]-x["baseline"]["btc_before_runtime_gate"] for x in group]
        summary[group_name]={
            **block,
            "paired_focused_minus_baseline":{
                "btc_delta_points":pct_mean(deltas),"bootstrap_95pct_CI_points":paired_ci(deltas),
                "focused_wins":sum(d>0 for d in deltas),"baseline_wins":sum(d<0 for d in deltas),"ties":sum(d==0 for d in deltas),
                "ungated_delta_points":pct_mean(ungated_delta),"ungated_bootstrap_95pct_CI_points":paired_ci(ungated_delta,BOOTSTRAP_SEED+1),
            },
        }
    fresh=[x for x in detail_rows if x["id"] in set(fresh_ids)]
    summary["fresh_262_primary"]["focused"]["lookup_observation_tasks"]=sum(x["focused"]["lookup_count"]>0 for x in fresh)
    summary["fresh_262_primary"]["focused"]["multi_candidate_lookup_tasks"]=sum(x["focused"]["multi_candidate_lookup_count"]>0 for x in fresh)
    summary["fresh_262_primary"]["focused"]["asks_after_any_lookup_tasks"]=sum(x["focused"]["ask"] and x["focused"]["lookup_count"]>0 for x in fresh)
    summary["_method"]={
        "primary":"262 val IDs excluding all 47 gold-ask tasks used during taxonomy; evaluates collateral BTC and false asks only",
        "diagnostic":"47 audited gold-ask IDs are reported separately and are not confirmation evidence",
        "baseline":"reused exact archived predictions from exp/ask-ambiguity-prompt-val309",
        "scoring":"official BTC formula via score_btc.score_sample and ToolEnv gold replay; local 30s gate shown separately from pre-gate diagnostic",
        "bootstrap_draws":10000,"bootstrap_seed":BOOTSTRAP_SEED,
    }
    write_json(OUT/"summary.json",summary)
    render_summary(summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def render_summary(s):
    allm=s["all_val309"];fresh=s["fresh_262_primary"];diag=s["audited_47_diagnostic"]
    pair=fresh["paired_focused_minus_baseline"]
    text=f"""# Post-lookup ambiguity prompt follow-up

## Primary fresh cohort

The primary result uses the 262 val309 IDs outside the 47 gold ask tasks used to form the hypothesis. It is a fresh regression check; it contains no gold ask positives, so it measures collateral BTC and false asks, not ask-recall improvement. The audited 47 are diagnostic only.

| Branch | BTC after local 30s gate | BTC before gate | Hash matches | Runtime errors | Mean seconds | Gold asks exact / total | Extra asks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Archived baseline | {fresh['baseline']['btc_0_100']:.2f} | {fresh['baseline']['btc_before_runtime_gate_0_100']:.2f} | {fresh['baseline']['hash_matches']}/262 | {fresh['baseline']['runtime_errors']} | {fresh['baseline']['mean_latency_seconds']:.2f} | {fresh['baseline']['exact_gold_ask_ids']}/{fresh['baseline']['gold_ask_ids']} | {fresh['baseline']['ask_calls_outside_gold_ask_ids']} |
| Focused lookup rule | {fresh['focused']['btc_0_100']:.2f} | {fresh['focused']['btc_before_runtime_gate_0_100']:.2f} | {fresh['focused']['hash_matches']}/262 | {fresh['focused']['runtime_errors']} | {fresh['focused']['mean_latency_seconds']:.2f} | {fresh['focused']['exact_gold_ask_ids']}/{fresh['focused']['gold_ask_ids']} | {fresh['focused']['ask_calls_outside_gold_ask_ids']} |

Paired BTC difference focused − baseline: **{pair['btc_delta_points']:+.2f} points** (bootstrap 95% CI **[{pair['bootstrap_95pct_CI_points'][0]:+.2f}, {pair['bootstrap_95pct_CI_points'][1]:+.2f}]**); focused wins {pair['focused_wins']}, baseline wins {pair['baseline_wins']}, ties {pair['ties']}. Pre-gate difference: {pair['ungated_delta_points']:+.2f} points (95% CI [{pair['ungated_bootstrap_95pct_CI_points'][0]:+.2f}, {pair['ungated_bootstrap_95pct_CI_points'][1]:+.2f}]).

## Audited 47-task diagnostic

| Branch | BTC after local 30s gate | Exact gold asks | Missing gold asks | Extra asks |
|---|---:|---:|---:|---:|
| Archived baseline | {diag['baseline']['btc_0_100']:.2f} | {diag['baseline']['exact_gold_ask_ids']}/47 | {diag['baseline']['missed_gold_asks']} | {diag['baseline']['ask_calls_outside_gold_ask_ids']} |
| Focused lookup rule | {diag['focused']['btc_0_100']:.2f} | {diag['focused']['exact_gold_ask_ids']}/47 | {diag['focused']['missed_gold_asks']} | {diag['focused']['ask_calls_outside_gold_ask_ids']} |

This subgroup was used to form the hypothesis and is not independent confirmation. Full paired outcomes are in paired_details.jsonl.

## Overall val309 diagnostic

Baseline BTC {allm['baseline']['btc_0_100']:.2f}; focused BTC {allm['focused']['btc_0_100']:.2f}. Full-run artifacts are paired by ID in paired_details.jsonl.

## Caveats

The 30-second gate and latency are measured locally and do not establish A30 runtime. The baseline predictions are reused from the prior locked run rather than reinferred. Ollama was deterministic-configured (temperature 0, seed {SEED}), but local runtime variation is still possible. No production prompt or submission was changed.
"""
    (OUT/"summary.md").write_text(text,encoding="utf-8")
def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ("--prepare","--run","--analyze"):
        raise SystemExit("use --prepare, --run, or --analyze")
    {"--prepare":prepare,"--run":run,"--analyze":analyze}[sys.argv[1]]()

if __name__=="__main__": main()
