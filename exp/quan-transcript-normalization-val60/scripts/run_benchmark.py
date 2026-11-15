from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, shutil, sys, time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'exp/quan-transcript-normalization-val60'
OUT=EXP/'output/inference'
RUNNER_PATH=ROOT/'exp/quan-continuation-cot-v1/scripts/run_downstream.py'
ANALYZER_PATH=ROOT/'exp/quan-continuation-cot-v1/scripts/analyze_downstream.py'
BASELINE=ROOT/'exp/qwen35-sft-benchmark-v1/output/inference/config_manifest.json'
SMOKE=ROOT/'exp/qwen35-sft-benchmark-v1/output/smoke/smoke_manifest.json'
TAG='lifemate-qwen35-kagglev1-sft-epoch2-q4km:latest'
DIGEST='973ae821f84016daad8db7ebf494fd2ddf98db7d1f14b5502ffb608cbfc8304d'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write_jsonl(p,rows): Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def init_runner():
    r=load('quan_normalization_cot_runner',RUNNER_PATH)
    r.OUT=OUT; r.MODEL=TAG; r.EXPECTED_DIGEST=DIGEST; r.CAP=768; r.CONTEXT=8192; r.BATCH=512; r.SEED=17; r.MAX_STEPS=5
    return r

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()): raise FileExistsError(f'Refusing to overwrite nonempty output directory {OUT}')
    r=init_runner()
    cfg=read_json(BASELINE)
    if cfg['branches']['C']['model_digest']!=DIGEST or cfg['agent']['cot_prompt_sha256_utf8']!='cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950': raise RuntimeError('SFT/CoT source lock mismatch')
    if read_json(SMOKE).get('status')!='passed': raise RuntimeError('Locked SFT model smoke gate is not passed')
    version,digest=r.model_fingerprint()
    if version!='0.40.15' or digest!=DIGEST: raise RuntimeError(f'Ollama model/version mismatch: {version} {digest}')
    prompt_hash=r.canonical_prompt_hash(True)
    if prompt_hash!=cfg['agent']['cot_prompt_sha256_utf8']: raise RuntimeError(f'Canonical CoT prompt hash mismatch: {prompt_hash}')
    cohort=read_json(EXP/'output/cohort_manifest.json'); ids=cohort['ids']
    expected='87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9'
    id_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if len(ids)!=60 or id_hash!=expected: raise RuntimeError('Fixed cohort order/hash mismatch')
    anchors=read_json(EXP/'output/execution_order.json')['A_repeat_anchor_ids']
    if len(anchors)!=20 or any(sid not in ids for sid in anchors): raise RuntimeError('A-repeat anchors mismatch')
    raw=read_jsonl(EXP/'output/transcripts-raw.jsonl'); normalized=read_jsonl(EXP/'output/transcripts-normalized.jsonl')
    if [x['id'] for x in raw] != [x['id'] for x in normalized]: raise RuntimeError('Transcript cache ID order mismatch')
    raw_map={x['id']:x['transcript'] for x in raw}; norm_map={x['id']:x['transcript'] for x in normalized}
    if any(norm_map[sid]!=raw_map[sid] for sid in ids): raise RuntimeError('Orthography-only B differs for cohort despite audit')
    rounds=read_json(EXP/'output/execution_order.json')['blocks']
    for block in rounds:
        if block['ids']!=ids[(block['block']-1)*15:block['block']*15]: raise RuntimeError('Frozen balanced order differs from cohort order')
    runner_hash=sha(RUNNER_PATH)
    lock=read_json(EXP/'output/input_lock.json'); lock['benchmark_script_sha256']=sha(Path(__file__))
    lock['ollama_native_observed']={'version':version,'model':TAG,'digest':digest,'host':r.HOST,'checked_before_inference':True}
    write_json(EXP/'output/input_lock.json',lock)
    cfgout={'status':'locked','cohort_ids_sha256_newline_joined':id_hash,'samples':60,'branches':['A','B','A-repeat'],
      'treatment':'orthography-only-v1 separate fallback; no number parser; no prediction fields other than current transcript differ',
      'model':{'tag':TAG,'digest':digest,'ollama_version':version},'prompt':{'variant':'observation-grounded-v2 English + current CoT checklist',
      'cot_sha256_utf8':prompt_hash},'runtime':{'thinking':True,'protocol':'JSON','temperature':0,'num_predict':768,'num_ctx':8192,'num_batch':512,'max_action_turns':5,'seed':17,'retrieval':False},
      'history_asr':'PhoASR unchanged','toolenv_prediction_envs_per_sample_branch':1,'source_runner_sha256':runner_hash,
      'normalizer_spec_sha256':sha(EXP/'output/normalizer_spec.json'),'input_lock_sha256':sha(EXP/'output/input_lock.json'),
      'raw_transcript_sha256':sha(EXP/'output/transcripts-raw.jsonl'),'normalized_transcript_sha256':sha(EXP/'output/transcripts-normalized.jsonl'),
      'cohort_manifest_sha256':sha(EXP/'output/cohort_manifest.json'),'execution_order_sha256':sha(EXP/'output/execution_order.json'),
      'prompt_source_manifest_sha256':sha(BASELINE),'pre_inference_payload_check':'B transcript exactly equals A transcript for all 60 IDs; all other inputs loaded from same item/context/history; per-request payload hashes retained.',
      'branches_order':rounds,'A_repeat_anchor_ids':anchors,'A_repeat_anchor_sha256_newline_joined':hashlib.sha256('\n'.join(anchors).encode()).hexdigest(),
      'bootstrap_iterations':20000,'bootstrap_seed':20261109,'inference_started':False,'public_private_test_used':False}
    write_json(OUT/'benchmark_lock.json',cfgout)
    write_json(OUT/'cohort_manifest.json',cohort)
    write_json(OUT/'execution_order.json',read_json(EXP/'output/execution_order.json'))
    print(json.dumps({'status':'locked','ollama_version':version,'model_digest':digest,'cohort_samples':len(ids),'A_repeat_anchors':len(anchors),'normalized_changes_in_cohort':sum(raw_map[sid]!=norm_map[sid] for sid in ids)},ensure_ascii=False,indent=2))

def run_branch_block(r, branch, round_index, ids, transcript_map, items, cot=True):
    # Reuse the audited LifeMate runner while storing each call in an isolated scratch directory.
    scratch=OUT/'_scratch'/f'{branch}-round{round_index}'
    if scratch.exists(): raise FileExistsError(f'Refusing to reuse scratch block {scratch}')
    scratch.mkdir(parents=True)
    original_out=r.OUT; r.OUT=scratch
    try:
        warm=r.warmup('COT',round_index)
        r.run_block('COT',round_index,ids,transcript_map,items,cot=cot)
        preds=read_jsonl(scratch/'predictions-COT.jsonl'); traces=read_jsonl(scratch/'step-traces-COT.jsonl'); requests=read_jsonl(scratch/'requests-COT.jsonl')
    finally: r.OUT=original_out
    for row in traces: row['branch']=branch
    for row in requests:
        row['request_id']=row['request_id'].replace('COT:',branch+':',1)
        row['branch']=branch
    for suffix,rows in [('predictions',preds),('step-traces',traces),('requests',requests)]:
        target=OUT/f'{suffix}-{branch}.jsonl'; prior=read_jsonl(target) if target.exists() else []
        id_key='sample_id' if suffix=='requests' else 'id'
        if {x[id_key] for x in prior} & set(ids): raise RuntimeError(f'Duplicate {branch} IDs in {suffix}')
        write_jsonl(target,prior+rows)
    warm['branch']=branch; write_json(OUT/f'warmup-{branch}-round{round_index}.json',warm)
    shutil.rmtree(scratch)
    print(f'completed branch={branch} round={round_index} samples={len(ids)}',flush=True)

def run():
    if sys.prefix!=sys.base_prefix: raise RuntimeError('Must run with system Python, outside the project .venv, to reach native Ollama.')
    lock=read_json(OUT/'benchmark_lock.json')
    if lock.get('status')!='locked': raise RuntimeError('Benchmark lock is invalid')
    r=init_runner(); version,digest=r.model_fingerprint()
    if version!=lock['model']['ollama_version'] or digest!=lock['model']['digest']: raise RuntimeError('Native Ollama fingerprint changed since prepare')
    if r.canonical_prompt_hash(True)!=lock['prompt']['cot_sha256_utf8']: raise RuntimeError('CoT prompt drift')
    raw=read_jsonl(EXP/'output/transcripts-raw.jsonl'); norm=read_jsonl(EXP/'output/transcripts-normalized.jsonl')
    raw_map={x['id']:x['transcript'] for x in raw}; norm_map={x['id']:x['transcript'] for x in norm}
    ids=read_json(OUT/'cohort_manifest.json')['ids']; anchors=lock['A_repeat_anchor_ids']
    val_ids=json.loads((ROOT/'data/splits/val_ids.json').read_text(encoding='utf-8-sig'))
    if [x['id'] for x in raw]!=val_ids: raise RuntimeError('Raw transcript cache is not exact val309 order')
    data_root=r.find_data_root('train+public_test/train+public_test'); items={x['id']:x for x in r.load_jsonl(data_root/'train'/'train.jsonl')}
    if set(ids)-items.keys(): raise RuntimeError('Missing fixed-cohort train ToolEnv input')
    lock['inference_started']=True; lock['inference_start_epoch']=time.time(); write_json(OUT/'benchmark_lock.json',lock)
    started=time.time()
    for block in lock['branches_order']:
        round_index=block['block']; block_ids=block['ids']
        for branch in block['branch_order']:
            target=OUT/f'predictions-{branch}.jsonl'; existing=read_jsonl(target) if target.exists() else []
            existing_ids={x['id'] for x in existing}
            if set(block_ids)<=existing_ids: continue
            if existing_ids & set(block_ids): raise RuntimeError(f'Partial block for {branch}; inspect artifacts before resuming')
            selected=raw_map if branch=='A' else norm_map
            scratch=OUT/'_scratch'/f'{branch}-round{round_index}'
            if scratch.exists():
                staged=read_jsonl(scratch/'predictions-COT.jsonl')
                if [x['id'] for x in staged]!=block_ids: raise RuntimeError(f'Incomplete staged block {branch}/{round_index}')
                r.OUT=OUT
                preds=staged; traces=read_jsonl(scratch/'step-traces-COT.jsonl'); requests=read_jsonl(scratch/'requests-COT.jsonl')
                for row in traces: row['branch']=branch
                for row in requests: row['request_id']=row['request_id'].replace('COT:',branch+':',1); row['branch']=branch
                for suffix,rows in [('predictions',preds),('step-traces',traces),('requests',requests)]:
                    dest=OUT/f'{suffix}-{branch}.jsonl'; prior=read_jsonl(dest) if dest.exists() else []
                    key='sample_id' if suffix=='requests' else 'id'
                    if {x[key] for x in prior}&set(block_ids): raise RuntimeError(f'Duplicate staged IDs for {branch}/{suffix}')
                    write_jsonl(dest,prior+rows)
                warm=read_json(scratch/'warmup-COT-round'+str(round_index)+'.json'); warm['branch']=branch; write_json(OUT/f'warmup-{branch}-round{round_index}.json',warm)
                shutil.rmtree(scratch)
                print(f'recovered branch={branch} round={round_index} samples={len(block_ids)}',flush=True)
            else:
                run_branch_block(r,branch,round_index,block_ids,selected,items,cot=True)
    if not (OUT/'predictions-A-repeat.jsonl').exists(): run_branch_block(r,'A-repeat',5,anchors,raw_map,items,cot=True)
    for branch,expected_ids in [('A',ids),('B',ids),('A-repeat',anchors)]:
        for kind in ('predictions','step-traces'):
            rows=read_jsonl(OUT/f'{kind}-{branch}.jsonl')
            if [x['id'] for x in rows]!=expected_ids: raise RuntimeError(f'{kind}-{branch} is incomplete or misordered')
    write_json(OUT/'inference-run.json',{'status':'complete','started_epoch':started,'ended_epoch':time.time(),'elapsed_seconds':time.time()-started,'samples_per_branch':{'A':60,'B':60,'A-repeat':20},'public_private_test_used':False,'raw_chain_of_thought_saved':False})
    print(json.dumps(read_json(OUT/'inference-run.json'),indent=2))

def analyze():
    a=load('quan_normalization_analyzer',ANALYZER_PATH); a.OUT=OUT; a.BOOTSTRAPS=20000; a.BOOTSTRAP_SEED=20261109
    lock=read_json(OUT/'benchmark_lock.json'); ids=read_json(OUT/'cohort_manifest.json')['ids']; anchors=lock['A_repeat_anchor_ids']
    data_root=a.find_data_root('train+public_test/train+public_test'); items={x['id']:x for x in a.load_jsonl(data_root/'train'/'train.jsonl')}
    results={}
    for branch,bids in [('A',ids),('B',ids),('A-repeat',anchors)]: results[branch]=a.compute_branch(branch,bids,items,data_root)
    predA,scoreA,metricsA,reqA=results['A']; predB,scoreB,metricsB,reqB=results['B']; predR,scoreR,metricsR,reqR=results['A-repeat']
    rng=random.Random(20261109)
    pAB=a.pair_metrics('A-vs-B',ids,predA,predB,scoreA,scoreB,reqA,reqB,rng)
    sub={sid:predA[sid] for sid in anchors}; subscore={sid:scoreA[sid] for sid in anchors}
    pAR=a.pair_metrics('A-vs-A-repeat',anchors,sub,predR,subscore,scoreR,reqA,reqR,rng)
    pBR=a.pair_metrics('B-vs-A-repeat',anchors,{sid:predB[sid] for sid in anchors},predR,{sid:scoreB[sid] for sid in anchors},scoreR,reqB,reqR,rng)
    # Exact argument changes and transcript/payload checks.
    arg_transitions=[]
    for sid in ids:
        left=predA[sid].get('tool_calls',[]); right=predB[sid].get('tool_calls',[])
        if left!=right: arg_transitions.append({'id':sid,'A_calls':left,'B_calls':right})
    details={'cohort':'fixed enriched held-out val309 cohort60; screening only','treatment':'orthography-only fallback; current transcript field only; numeric parser deferred','branches':{'A':metricsA,'B':metricsB,'A-repeat':metricsR},
      'paired':{'B_minus_A':pAB,'A_repeat_minus_A_on_anchors':pAR,'A_repeat_minus_B_on_anchors':pBR},
      'outcome_changed_samples':arg_transitions[:15],'outcome_changed_sample_count':len(arg_transitions),
      'input_equality':{'raw_equals_normalized_for_cohort':all(raw==norm for raw,norm in zip([x['transcript'] for x in read_jsonl(EXP/'output/transcripts-raw.jsonl') if x['id'] in set(ids)],[x['transcript'] for x in read_jsonl(EXP/'output/transcripts-normalized.jsonl') if x['id'] in set(ids)]))},
      'official_scoring':'BTC hash gate + M>0; exact tool name and arguments; duplicate multiplicity; gold replay through official ToolEnv.',
      'local_runtime_gate':'diagnostic only; local RTX 4050 is not A30.', 'raw_chain_of_thought_saved':False,
      'interpretation':'Orthography-only treatment made no cohort transcript changes, so B-A is not a treatment contrast. Any difference is run variation. No conclusion about numeric normalization.'}
    if any(m['gold_replay_failures'] for m in (metricsA,metricsB,metricsR)): raise RuntimeError('Gold ToolEnv replay failure')
    write_json(OUT/'analysis.json',details)
    (OUT/'summary.md').write_text('# Downstream benchmark summary\n\n'+json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'A':metricsA.get('btc_formula_score_before_runtime_gate_0_100'),'B':metricsB.get('btc_formula_score_before_runtime_gate_0_100'),'A-repeat':metricsR.get('btc_formula_score_before_runtime_gate_0_100'),'B_minus_A':pAB['paired_formula_delta_points_right_minus_left']},ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--prepare',action='store_true'); g.add_argument('--run',action='store_true'); g.add_argument('--analyze',action='store_true'); args=p.parse_args()
    if args.prepare: prepare()
    elif args.run: run()
    else: analyze()
if __name__=='__main__': main()

