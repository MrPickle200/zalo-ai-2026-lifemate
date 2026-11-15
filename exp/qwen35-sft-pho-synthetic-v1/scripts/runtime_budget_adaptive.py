from __future__ import annotations
import hashlib, importlib.util, json, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'exp/btc-scoring/scripts'), str(ROOT / 'exp/quan-continuation-cot-v1/scripts')]
EXP = ROOT / 'exp/qwen35-sft-pho-synthetic-v1'
OUT = EXP / 'output/runtime-budget-ablation'
AB_SCRIPT = EXP / 'scripts/runtime_budget_ablation.py'
THRESHOLD_SEC = 20.0
BRANCH = 'COT'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def read_jsonl(path): return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if path.exists() else []
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def context():
    ab=load_module('runtime_budget_adaptive_support', AB_SCRIPT)
    ev,compare,prompt,runner,ids,items,main_asr,history,data,eval_lock=ab.context()
    lock=read_json(OUT/'ablation_lock.json')
    if len(lock['cohort']['ids'])!=120: raise RuntimeError('Expected locked 120-sample cohort')
    return ab,ev,compare,prompt,runner,ids,items,main_asr,history,data,eval_lock,lock

def run():
    ab,ev,compare,prompt,runner,ids,items,main_asr,history,data,eval_lock,lock=context()
    cohort_ids=lock['cohort']['ids']
    branch_out=OUT/'inference'/'adaptive_switch20'
    branch_out.mkdir(parents=True,exist_ok=True)
    policy_lock={'experiment':'Qwen3.5-4B step120 adaptive thinking by elapsed LLM time',
        'policy':{'initial_thinking':True,'switch_to_thinking_false_when_cumulative_request_latency_seconds_exceeds':THRESHOLD_SEC,
                  'switch_applies':'all subsequent LLM requests for the current sample','num_predict':768},
        'same_model_digest':lock['model']['digest'],'prompt':lock['prompt'],'cohort_ids_sha256':lock['cohort']['cohort_ids_sha256'],
        'common_settings':lock['common_settings'],'source_lock_sha256':sha(OUT/'ablation_lock.json')}
    lock_path=OUT/'adaptive_policy_lock.json'
    if lock_path.exists() and read_json(lock_path)!=policy_lock: raise RuntimeError('Adaptive policy lock differs')
    write_json(lock_path,policy_lock)
    pred_path=branch_out/'predictions-COT.jsonl'
    existing=read_jsonl(pred_path)
    if [r['id'] for r in existing] != cohort_ids[:len(existing)]: raise RuntimeError('Existing adaptive predictions are not a cohort prefix')

    runner.MODEL=ev.MODEL_TAGS['step120']
    runner.EXPECTED_DIGEST=lock['model']['digest']
    runner.CAP=768
    runner.MAX_STEPS=5
    runner.OUT=branch_out
    runner.inject_cot_original=runner.inject_cot
    runner.inject_cot=prompt.focused_inject
    runner.seed_everything(17)
    compare.CachedPhoWhisper.lookup=history
    import lifemate.asr as asr_module
    asr_module.PhoASRModel=compare.CachedPhoWhisper
    import lifemate.agent as agent_module
    original_agent=agent_module.Agent

    def make_adaptive_agent(*args,**kwargs):
        agent=original_agent(*args,**kwargs)
        original_next=agent.next_action
        original_request=agent._ollama_request
        state={'sample_id':None,'start':0,'switched':False,'switch_after_request_count':None,'switch_elapsed':None}
        def adaptive_next(*a,**k):
            sid=agent._telemetry_sample_id
            if sid!=state['sample_id']:
                state.update({'sample_id':sid,'start':len(agent.request_telemetry),'switched':False,
                              'switch_after_request_count':None,'switch_elapsed':None})
                agent.ollama_thinking=True
            elapsed=sum(float(row.get('latency_seconds') or 0) for row in agent.request_telemetry[state['start']:])
            if not state['switched'] and elapsed>=THRESHOLD_SEC:
                agent.ollama_thinking=False
                state['switched']=True
                state['switch_after_request_count']=len(agent.request_telemetry)-state['start']
                state['switch_elapsed']=elapsed
            return original_next(*a,**k)
        def request_with_mode(*a,**k):
            start=len(agent.request_telemetry)
            thinking=bool(agent.ollama_thinking)
            response=original_request(*a,**k)
            for row in agent.request_telemetry[start:]:
                row['adaptive_thinking_enabled']=thinking
                row['adaptive_switch_threshold_seconds']=THRESHOLD_SEC
            return response
        agent.next_action=adaptive_next
        agent._ollama_request=request_with_mode
        agent._adaptive_state=state
        return agent
    agent_module.Agent=make_adaptive_agent
    try:
        for round_no,start in enumerate(range(0,len(cohort_ids),15),1):
            block=cohort_ids[start:start+15]
            done={r['id'] for r in read_jsonl(pred_path)}
            todo=[sid for sid in block if sid not in done]
            if not todo: continue
            warm_started=time.perf_counter()
            warm=runner.api_json('/api/chat',{'model':runner.MODEL,'messages':[{'role':'user','content':'Return {} as JSON.'}],
                'stream':False,'think':True,'format':'json','options':{'temperature':0,'num_predict':8,'num_ctx':8192,'num_batch':512}})
            write_json(branch_out/f'warmup-COT-round{round_no}.json',{'branch':'adaptive_switch20','round':round_no,
                'excluded_from_measurements':True,'wall_seconds':time.perf_counter()-warm_started,
                'eval_count':warm.get('eval_count'),'load_duration_ns':warm.get('load_duration')})
            while todo:
                before=len(read_jsonl(pred_path))
                try: runner.run_block(BRANCH,round_no,todo,main_asr,items,cot=True)
                except Exception:
                    after=read_jsonl(pred_path)
                    if len(after)<=before or not after[-1].get('error'): raise
                done={r['id'] for r in read_jsonl(pred_path)}
                todo=[sid for sid in block if sid not in done]
            meta=read_json(branch_out/'run-COT.json')
            meta.update({'branch':'adaptive_switch20','status':'complete' if len(read_jsonl(pred_path))==start+len(block) else 'in_progress',
                'thinking_policy':policy_lock['policy'],'model_digest':lock['model']['digest'],'prompt_variant':'focused post-lookup disambiguation',
                'cohort_ids':cohort_ids,'executed_rounds':round_no})
            write_json(branch_out/'run-COT.json',meta)
            print(f'adaptive_switch20 block {round_no}/8 complete ({start+len(block)}/120)',flush=True)
    finally:
        agent_module.Agent=original_agent
    analyze()

def analyze():
    from score_btc import score_sample,call_key
    ab,ev,compare,prompt,runner,ids,items,main_asr,history,data,eval_lock,lock=context()
    cohort_ids=lock['cohort']['ids']; slow=set(lock['cohort']['slow_ids'])
    adaptive_rows=read_jsonl(OUT/'inference/adaptive_switch20/predictions-COT.jsonl')
    if [r['id'] for r in adaptive_rows]!=cohort_ids: raise RuntimeError('Adaptive inference incomplete or out of order')
    predmaps={'step120_768_think':{r['id']:r for r in read_jsonl(OUT/'..'/'missing') } }
    orig={r['id']:r for r in read_jsonl(EXP/'output/val309-checkpoint-eval/inference/step120/predictions-COT.jsonl')}
    no_think={r['id']:r for r in read_jsonl(OUT/'inference/no_think_768/predictions-COT.jsonl')}
    adaptive={r['id']:r for r in adaptive_rows}
    predmaps={'step120_768_think':orig,'no_think_768':no_think,'adaptive_switch20':adaptive}
    if any(set(rows)<set(cohort_ids) for rows in predmaps.values()): raise RuntimeError('Comparison branch is missing cohort predictions')
    scored={name:{sid:score_sample(sid,items[sid],rows[sid],data) for sid in cohort_ids} for name,rows in predmaps.items()}
    records=[]
    for sid in cohort_ids:
        gold_asks=[c for c in (json.loads(items[sid]['label']) if isinstance(items[sid].get('label'),str) else items[sid].get('label',{})).get('tool_calls',[]) if c.get('name')=='sys.ask_missing_info']
        rec={'id':sid,'cohort_group':'slow_gt30s' if sid in slow else 'matched_fast_control','gold_ask':bool(gold_asks)}
        for name in predmaps:
            p=predmaps[name][sid]; s=scored[name][sid]
            asks=[c for c in p.get('tool_calls') or [] if c.get('name')=='sys.ask_missing_info']
            rec[name]={'btc':s['official_sample_score'],'btc_before_runtime_gate':s['btc_formula_score_before_runtime_gate_diagnostic'],
                'latency_seconds':p.get('latency_seconds'),'over_30s':bool(s['runtime_over_30s']),'runtime_error':bool(p.get('error')),
                'hash_match':bool(s['hash_match']),'ask_count':len(asks),'ask_exact':any(call_key(a)==call_key(g) for a in asks for g in gold_asks)}
        records.append(rec)
    groups={'all_120':set(cohort_ids),'slow_gt30s':slow,'matched_fast_control':set(cohort_ids)-slow}
    summaries={}
    for gname,members in groups.items():
        group=[r for r in records if r['id'] in members]; summary={}
        for name in predmaps:
            vals=[r[name] for r in group]; lat=[x['latency_seconds'] for x in vals if x['latency_seconds'] is not None]
            summary[name]={'btc_after_30s_gate':ab.pct([x['btc'] for x in vals]),'btc_before_gate':ab.pct([x['btc_before_runtime_gate'] for x in vals]),
                'over_30s':sum(x['over_30s'] for x in vals),'runtime_errors':sum(x['runtime_error'] for x in vals),
                'hash_matches':sum(x['hash_match'] for x in vals),'mean_latency_seconds':sum(lat)/len(lat) if lat else None,
                'median_latency_seconds':__import__('statistics').median(lat) if lat else None,'gold_ask_exact':sum(x['ask_exact'] for x in vals),
                'ask_predictions':sum(x['ask_count'] for x in vals)}
        for other in ('adaptive_switch20','no_think_768'):
            ds=[r[other]['btc']-r['step120_768_think']['btc'] for r in group]
            summary[f'paired_{other}_minus_step120']={'delta_after_gate_points':ab.pct(ds),
                'bootstrap_95pct_CI_after_gate':ab.boot_ci(ds,20261113+(20 if other=='adaptive_switch20' else 768)),
                'wins':sum(d>0 for d in ds),'ties':sum(d==0 for d in ds),'losses':sum(d<0 for d in ds)}
        summaries[gname]=summary
    req=read_jsonl(OUT/'inference/adaptive_switch20/requests-COT.jsonl')
    adaptive_req=[r for r in req if 'adaptive_thinking_enabled' in r]
    samples_switched={r['sample_id'] for r in adaptive_req if not r['adaptive_thinking_enabled']}
    summary={'experiment':'Qwen3.5-4B adaptive thinking switch at 20s','cohort':lock['cohort'],
        'policy':{'start_thinking':True,'switch_after_cumulative_llm_latency_seconds':THRESHOLD_SEC,'num_predict':768},
        'adaptive_requests':{'total':len(adaptive_req),'thinking_enabled':sum(r['adaptive_thinking_enabled'] for r in adaptive_req),
            'thinking_disabled':sum(not r['adaptive_thinking_enabled'] for r in adaptive_req),'samples_with_switch':len(samples_switched)},
        'groups':summaries,'method':{'scoring':'BTC official ToolEnv per sample with runtime gate and pre-gate diagnostic',
            'comparison':'paired on same 120-sample stress cohort','selection_bias':'slow cases oversampled; do not project these scores to full val309/public test'}}
    write_json(OUT/'adaptive_summary.json',summary); write_jsonl(OUT/'adaptive_paired_details.jsonl',records)
    lines=['# Adaptive thinking switch at 20 seconds','',
        'Same 120-sample stress cohort, step120 checkpoint, focused prompt and cached PhoWhisper transcripts. Thinking starts enabled and switches off for later LLM calls after cumulative LLM request time exceeds 20 seconds. Scores use official ToolEnv and the 30-second runtime gate. The cohort intentionally oversamples the 42 slow samples.','',
        '| Group / branch | BTC after gate | Before gate | >30s | Hash matches | Mean seconds | Exact asks |','|---|---:|---:|---:|---:|---:|---:|']
    for g in ('all_120','slow_gt30s','matched_fast_control'):
        for name in predmaps:
            s=summaries[g][name]
            lines.append(f"| {g} / {name} | {s['btc_after_30s_gate']:.2f} | {s['btc_before_gate']:.2f} | {s['over_30s']} | {s['hash_matches']}/{len(groups[g])} | {s['mean_latency_seconds']:.2f} | {s['gold_ask_exact']} |")
    d=summaries['all_120']['paired_adaptive_switch20_minus_step120']; ci=d['bootstrap_95pct_CI_after_gate']
    lines += ['',f"Adaptive vs original step120: {d['delta_after_gate_points']:+.2f} BTC points after gate (95% paired bootstrap CI {ci[0]:+.2f} to {ci[1]:+.2f}); wins/ties/losses {d['wins']}/{d['ties']}/{d['losses']}.",
        f"The policy switched off thinking for {len(samples_switched)}/120 samples and {summary['adaptive_requests']['thinking_disabled']}/{summary['adaptive_requests']['total']} LLM requests.",
        'The cohort is a stress test, not a population estimate. Validate on full val309 before using this as a submission setting.','']
    (OUT/'adaptive_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--run',action='store_true'); p.add_argument('--analyze',action='store_true')
    a=p.parse_args()
    if a.run==a.analyze: p.error('choose --run or --analyze')
    run() if a.run else analyze()

