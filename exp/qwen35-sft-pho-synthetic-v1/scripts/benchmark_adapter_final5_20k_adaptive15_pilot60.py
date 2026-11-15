from __future__ import annotations
import hashlib, importlib.util, json, random, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'exp/qwen35-sft-pho-synthetic-v1'
OUT=EXP/'output/adapter-final5-20k-adaptive15-pilot60'
BASE_TAG='lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest'
NEW_TAG='lifemate-qwen35-phosynth-final5-20k-q4km:latest'
THRESHOLD=15.0
MAX_TURNS=6
CONTEXT=16384
BATCH=512
CAP=768
SEED=17
BLOCK=15
DRAWS=10000
ASR_DIR=ROOT/'exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho'
ASR_MAIN=ASR_DIR/'predictions.jsonl'
ASR_HISTORY=ASR_DIR/'history_predictions.jsonl'
COHORT_PATH=ROOT/'exp/qwen35-thinking-budget-val60/output/cohort_manifest.json'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_jsonl(path):
    p=Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if p.exists() else []
def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def boot_ci(diffs,seed):
    rng=random.Random(seed);n=len(diffs)
    values=sorted(100*statistics.mean(rng.choices(diffs,k=n)) for _ in range(DRAWS))
    return [values[int(.025*DRAWS)],values[int(.975*DRAWS)-1]]

def build():
    base=load_module('run_two_policy_for_final5_adaptive15_pilot60',EXP/'scripts/run_two_policy_val60.py')
    ev,compare,prompt,runner,ids,cohort,items,main,history,data,final4_record,final4,cohort_manifest=base.build()
    if len(cohort)!=60: raise RuntimeError(f'Expected cohort60, got {len(cohort)}')
    main_rows=read_jsonl(ASR_MAIN);history_rows=read_jsonl(ASR_HISTORY)
    main_sec={x['id']:float(x['inference_seconds']) for x in main_rows}
    history_sec={}
    for x in history_rows: history_sec[x['id']]=history_sec.get(x['id'],0.0)+float(x['inference_seconds'])
    asr={sid:main_sec[sid]+history_sec.get(sid,0.0) for sid in cohort}
    final4_manifest=read_json(EXP/'output/adapter-final-4epoch-benchmark/model-provenance/ollama-models.json')
    if final4_manifest.get('tag')!=BASE_TAG or final4_manifest.get('ollama_digest')!=final4_record['digest']:
        raise RuntimeError('Final4 registration manifest/digest mismatch')
    final5=read_json(EXP/'output/adapter-final-5epoch-20k-rows-benchmark/model-provenance/ollama-models.json')
    if final5.get('tag')!=NEW_TAG: raise RuntimeError('Final5 Ollama manifest does not match target tag')
    return base,ev,compare,prompt,runner,ids,cohort,items,main,history,data,final4_record,final5,cohort_manifest,asr

def make_lock(cohort,cohort_manifest,final4_record,final5,prompt,asr):
    inputs=[Path(__file__).resolve(),EXP/'scripts/run_two_policy_val60.py',EXP/'scripts/benchmark_adapter_final4.py',
      EXP/'scripts/tune_adaptive_total_budget_maxturn_val60.py',COHORT_PATH,ASR_MAIN,ASR_HISTORY,
      EXP/'output/adapter-final-4epoch-benchmark/model-provenance/ollama-models.json',
      EXP/'output/adapter-final-5epoch-20k-rows-benchmark/model-provenance/build-manifest.json',
      EXP/'output/adapter-final-5epoch-20k-rows-benchmark/model-provenance/ollama-models.json',
      ROOT/'src/lifemate/pipeline.py',ROOT/'src/lifemate/agent.py',ROOT/'exp/quan-continuation-cot-v1/scripts/run_downstream.py',
      ROOT/'exp/btc-scoring/scripts/score_btc.py']
    return {'experiment':'qwen35-sft-final5-20k-adaptive15-max6-askstop-enriched-pilot60',
      'cohort':{'n':len(cohort),'ids':cohort,'source':cohort_manifest.get('pilot_type','existing enriched fixed pilot60'),
        'manifest_sha256':sha(COHORT_PATH),'ids_sha256':hashlib.sha256('\n'.join(cohort).encode()).hexdigest()},
      'models':{'final4':{'tag':BASE_TAG,'digest':final4_record['digest']},
        'final5_20k':{'tag':NEW_TAG,'digest':final5['ollama_digest'],'checkpoint_sha256':final5['checkpoint_sha256'],
          'gguf_sha256':final5['gguf']['sha256']}},
      'policy':{'adaptive_total_budget_seconds':THRESHOLD,'budget_includes_asr':True,
        'rule':'first Qwen request starts thinking on; before subsequent requests, disable thinking once per-task PhoWhisper ASR plus prior Qwen request latency reaches 15s',
        'max_action_turns':MAX_TURNS,'stop_after_successful_sys_ask_missing_info':True},
      'settings':{'thinking':'adaptive','protocol':'JSON','temperature':0,'num_predict':CAP,'num_ctx':CONTEXT,
        'num_batch':BATCH,'seed':SEED,'prompt':'focused post-lookup disambiguation',
        'asr':'cached PhoWhisper Large v2 main/history; per-task inference_seconds added to Qwen latency before local 30s gate'},
      'asr_timing':{'mean_per_sample':statistics.mean(asr.values()),'main_clips_in_source':309,'history_clips_in_source':692,
        'model_load_startup_excluded':True},
      'prompt_sha256':prompt.prompt_hash(prompt.focused_inject),
      'inputs_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
      'caveat':'Enriched diagnostic cohort60; local RTX 4050 timing with cached ASR, not full-val or A30 estimate.'}

def run_branch(tag,digest,key,base_agent,agent_mod,pipeline_mod,pipeline_source,screen,runner,cohort,main,items,asr):
    branch=OUT/'inference'/key;branch.mkdir(parents=True,exist_ok=True);runner.OUT=branch
    runner.MODEL=tag;runner.EXPECTED_DIGEST=digest;runner.CAP=CAP;runner.CONTEXT=CONTEXT;runner.BATCH=BATCH;runner.SEED=SEED;runner.MAX_STEPS=MAX_TURNS
    # run_two_policy_val60.build() already set the locked focused prompt.
    runner.seed_everything(SEED)
    _,actual=runner.model_fingerprint()
    if actual!=digest: raise RuntimeError(f'Model digest mismatch for {key}: {actual} != {digest}')
    pred=branch/'predictions-COT.jsonl';saved=read_jsonl(pred);saved_ids=[x['id'] for x in saved]
    if saved_ids!=cohort[:len(saved_ids)]: raise RuntimeError(f'{key} output is not a canonical cohort prefix')
    screen.patch_ask_stop(pipeline_mod,pipeline_source)
    agent_mod.Agent=screen.adaptive_factory(base_agent,THRESHOLD,asr)
    done=set(saved_ids)
    for round_no,start in enumerate(range(0,len(cohort),BLOCK),1):
        block=cohort[start:start+BLOCK];todo=[sid for sid in block if sid not in done]
        if not todo:continue
        runner.seed_everything(SEED);runner.warmup('COT',round_no)
        while todo:
            before=len(read_jsonl(pred))
            try:runner.run_block('COT',round_no,todo,main,items,cot=True)
            except Exception:
                after=read_jsonl(pred)
                if len(after)<=before:raise
            done={x['id'] for x in read_jsonl(pred)};todo=[sid for sid in block if sid not in done]
        print(f'{key}: block {round_no}/4 complete ({len(done)}/60)',flush=True)
    rows=read_jsonl(pred)
    if [x['id'] for x in rows]!=cohort:raise RuntimeError(f'{key} inference incomplete or reordered')
    return rows,read_jsonl(branch/'requests-COT.jsonl')

def score_branch(ev,cohort,items,data,asr,pred_rows,request_rows):
    from score_btc import score_sample
    predmap={x['id']:x for x in pred_rows};groups={}
    for x in request_rows:groups.setdefault(x['sample_id'],[]).append(x)
    scored=[]
    for sid in cohort:
        p=dict(predmap[sid]);req=groups.get(sid,[])
        qwen=sum(float(x.get('latency_seconds') or 0) for x in req)
        if qwen==0:qwen=sum(float((x.get('_diagnostics') or {}).get('latency_seconds') or 0) for x in (p.get('model_output') or []))
        p['qwen_latency_seconds']=qwen;p['asr_latency_seconds']=asr[sid];p['latency_seconds']=qwen+asr[sid]
        p['runtime_over_30s']=p['latency_seconds']>30
        s=score_sample(sid,items[sid],p,data)
        scored.append({'id':sid,'raw':s['btc_formula_score_before_runtime_gate_diagnostic'],
          'gated':s['official_sample_score'],'latency_total_seconds':p['latency_seconds'],'qwen_seconds':qwen,'asr_seconds':asr[sid],
          'over_30s':p['latency_seconds']>30,'runtime_error':bool(p.get('error')),
          'hash_match':bool(s['hash_match']),'tool_calls':len(p.get('tool_calls') or []),'turns':len(p.get('model_output') or []),
          'thinking_switched':any(x.get('adaptive_thinking_enabled') is False for x in req)})
    return scored

def summarize(rows):
    return {'n':len(rows),'btc_raw':100*statistics.mean(x['raw'] for x in rows),
      'btc_asr_qwen_gated':100*statistics.mean(x['gated'] for x in rows),
      'mean_total_seconds':statistics.mean(x['latency_total_seconds'] for x in rows),
      'mean_asr_seconds':statistics.mean(x['asr_seconds'] for x in rows),
      'mean_qwen_seconds':statistics.mean(x['qwen_seconds'] for x in rows),
      'over_30s':sum(x['over_30s'] for x in rows),'runtime_errors':sum(x['runtime_error'] for x in rows),
      'hash_matches':sum(x['hash_match'] for x in rows),'mean_tool_calls':statistics.mean(x['tool_calls'] for x in rows),
      'mean_action_turns':statistics.mean(x['turns'] for x in rows),'thinking_switched_samples':sum(x['thinking_switched'] for x in rows)}

def analyze():
    base,ev,compare,prompt,runner,ids,cohort,items,main,history,data,final4_record,final5,cohort_manifest,asr=build()
    lock=read_json(OUT/'comparison_lock.json')
    current=make_lock(cohort,cohort_manifest,final4_record,final5,prompt,asr)
    if current!=lock:raise RuntimeError('Benchmark inputs changed after run')
    results={};details=[]
    for key in ('final4_adaptive15_max6','final5_adaptive15_max6'):
        folder=OUT/'inference'/key
        preds=read_jsonl(folder/'predictions-COT.jsonl');requests=read_jsonl(folder/'requests-COT.jsonl')
        if [x['id'] for x in preds]!=cohort:raise RuntimeError(f'{key} incomplete')
        scored=score_branch(ev,cohort,items,data,asr,preds,requests);results[key]=summarize(scored)
        details.extend({'branch':key,**x} for x in scored)
    by={(x['branch'],x['id']):x for x in details}
    raw=[by[('final5_adaptive15_max6',sid)]['raw']-by[('final4_adaptive15_max6',sid)]['raw'] for sid in cohort]
    gated=[by[('final5_adaptive15_max6',sid)]['gated']-by[('final4_adaptive15_max6',sid)]['gated'] for sid in cohort]
    pair={'raw_delta_points':100*statistics.mean(raw),'raw_ci95_points':boot_ci(raw,20261115),
      'gated_delta_points':100*statistics.mean(gated),'gated_ci95_points':boot_ci(gated,20261116),
      'wins_ties_losses':[sum(x>0 for x in gated),sum(x==0 for x in gated),sum(x<0 for x in gated)]}
    summary={'experiment':'qwen35-sft-final5-20k-adaptive15-max6-askstop-enriched-pilot60','cohort':lock['cohort'],
      'models':lock['models'],'policy':lock['policy'],'settings':lock['settings'],'asr_timing':lock['asr_timing'],
      'results':results,'paired_final5_minus_final4':pair,
      'caveat':lock['caveat']}
    write_json(OUT/'summary.json',summary);write_jsonl(OUT/'paired_details.jsonl',details)
    a=results['final4_adaptive15_max6'];b=results['final5_adaptive15_max6']
    lines=['# Final5 20k-row adapter — adaptive15/max6 enriched pilot60','',
      'Both adapters used the same enriched 60-task cohort, focused prompt, adaptive total ASR+Qwen budget, successful ask-stop, and six-action-turn cap. First Qwen request uses thinking; later requests turn thinking off once cached per-task PhoWhisper time plus prior Qwen latency reaches 15 seconds. The local 30-second gate includes cached ASR timing; ASR model-load startup is excluded.','',
      '| Adapter | BTC raw | BTC after ASR+Qwen 30s gate | Mean end-to-end s | >30s | Errors | Hash match |','|---|---:|---:|---:|---:|---:|---:|',
      f"| final4 control | {a['btc_raw']:.2f} | {a['btc_asr_qwen_gated']:.2f} | {a['mean_total_seconds']:.2f} | {a['over_30s']}/60 | {a['runtime_errors']} | {a['hash_matches']}/60 |",
      f"| final5 20k | {b['btc_raw']:.2f} | {b['btc_asr_qwen_gated']:.2f} | {b['mean_total_seconds']:.2f} | {b['over_30s']}/60 | {b['runtime_errors']} | {b['hash_matches']}/60 |",'',
      f"Paired final5−final4: raw {pair['raw_delta_points']:+.2f} (95% CI {pair['raw_ci95_points']}); after gate {pair['gated_delta_points']:+.2f} (95% CI {pair['gated_ci95_points']}); gated wins/ties/losses {pair['wins_ties_losses']}.",
      f"ASR mean {summary['asr_timing']['mean_per_sample']:.2f}s/task; mean Qwen: final4 {a['mean_qwen_seconds']:.2f}s, final5 {b['mean_qwen_seconds']:.2f}s. This cohort is enriched, not full val/public; local RTX 4050 timing is not A30.",
      '', 'See paired_details.jsonl for per-ID scores, latency, ToolEnv hashes, turns, and thinking switches.', '']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def run():
    if sys.prefix!=sys.base_prefix:raise RuntimeError('Use native system Python so local Ollama is reachable')
    base,ev,compare,prompt,runner,ids,cohort,items,main,history,data,final4_record,final5,cohort_manifest,asr=build()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=make_lock(cohort,cohort_manifest,final4_record,final5,prompt,asr)
    lp=OUT/'comparison_lock.json'
    if lp.exists() and read_json(lp)!=lock:raise RuntimeError('Existing comparison lock differs; refusing to mix runs')
    if not lp.exists():write_json(lp,lock)
    screen=load_module('adaptive_total_budget_helpers_for_final5_pilot60',EXP/'scripts/tune_adaptive_total_budget_maxturn_val60.py')
    import lifemate.pipeline as pipeline_mod
    import lifemate.agent as agent_mod
    source=Path(pipeline_mod.__file__).read_text(encoding='utf-8')
    base_agent=agent_mod.Agent
    try:
        for key,tag,digest in [('final4_adaptive15_max6',BASE_TAG,final4_record['digest']),('final5_adaptive15_max6',NEW_TAG,final5['ollama_digest'])]:
            run_branch(tag,digest,key,base_agent,agent_mod,pipeline_mod,source,screen,runner,cohort,main,items,asr)
    finally:
        agent_mod.Agent=base_agent
        exec(compile(source,str(pipeline_mod.__file__),'exec'),pipeline_mod.__dict__)
    analyze()

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else '--run'
    if mode=='--run':run()
    elif mode=='--analyze':analyze()
    else:raise SystemExit('Use --run or --analyze')

