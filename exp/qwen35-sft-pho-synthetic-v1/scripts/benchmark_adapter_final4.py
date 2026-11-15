from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, statistics, sys, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'exp/qwen35-sft-pho-synthetic-v1'
OUT=EXP/'output/adapter-final-4epoch-benchmark/val309'
MODEL='lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest'
SEED=17
BLOCK=15

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError(f'Cannot load {path}')
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def read_jsonl(path):return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if path.exists() else []
def write_json(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_jsonl(path,rows):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')

def load_eval():return load_module('locked_qwen35_checkpoint_eval_for_adapter_final4',EXP/'scripts/evaluate_val309_checkpoints.py')

def build():
 ev=load_eval();compare,prompt,runner,ids,items,main,history,data,prior_eval_lock=ev.context()
 final1_dir=EXP/'output/adapter-final-1epoch-benchmark/val309'
 final3_dir=EXP/'output/adapter-final-3epoch-benchmark/val309'
 final1_lock=read_json(final1_dir/'comparison_lock.json');final1_summary=read_json(final1_dir/'summary.json')
 final3_lock=read_json(final3_dir/'comparison_lock.json');final3_summary=read_json(final3_dir/'summary.json')
 final1_pred=final1_dir/'inference/adapter_final1/predictions-COT.jsonl'
 final3_pred=final3_dir/'inference/adapter_final3/predictions-COT.jsonl'
 if [x['id'] for x in read_jsonl(final1_pred)]!=ids or [x['id'] for x in read_jsonl(final3_pred)]!=ids:raise RuntimeError('Fixed adapter prediction rows differ from canonical val309')
 archive=EXP/'output/adapter_final-4-epochs.zip'
 q4=EXP/'output/adapter-final-4epoch-benchmark/gguf/adapter-final-4epoch-Q4_K_M.gguf'
 merge_manifest=EXP/'output/adapter-final-4epoch-benchmark/model-provenance/merge-manifest.json'
 conversion_manifest=EXP/'output/adapter-final-4epoch-benchmark/model-provenance/conversion-manifest.json'
 ollama_manifest=EXP/'output/adapter-final-4epoch-benchmark/model-provenance/ollama-models.json'
 conv=read_json(conversion_manifest);ollama=read_json(ollama_manifest)
 if not q4.is_file() or sha(q4)!=conv['q4_k_m_gguf']['sha256']:raise RuntimeError('Q4 GGUF differs from conversion manifest')
 if ollama['tag']!=MODEL or ollama['gguf']['sha256']!=sha(q4):raise RuntimeError('Ollama tag manifest does not identify current Q4 model')
 with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=30) as r:tags={x['name']:x['digest'] for x in json.load(r)['models']}
 if tags.get(MODEL)!=ollama['ollama_digest']:raise RuntimeError('Current Ollama digest differs from registration manifest')
 if final3_lock['adapter_final3']['digest']!=tags.get(final3_lock['adapter_final3']['model']):raise RuntimeError('3-epoch comparison tag/digest mismatch')
 inputs=[Path(__file__).resolve(),EXP/'scripts/merge_adapter_final4.py',EXP/'scripts/convert_adapter_final4.py',EXP/'scripts/register_ollama_final4.py',archive,merge_manifest,conversion_manifest,ollama_manifest,q4,
         final1_dir/'comparison_lock.json',final1_dir/'summary.json',final1_pred,final3_dir/'comparison_lock.json',final3_dir/'summary.json',final3_pred,
         EXP/'output/val309-checkpoint-eval/comparison_lock.json',EXP/'output/val309-checkpoint-eval/inference/step120/predictions-COT.jsonl',ev.BASELINE]
 inputs_hash={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs}
 lock={'experiment':'qwen35-sft-pho-synthetic-v1-adapter-final4-benchmark','cohort':{'count':len(ids),'ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest()},
  'prior_eval_lock_sha256':sha(EXP/'output/val309-checkpoint-eval/comparison_lock.json'),
  'prompt':{'variant':'focused post-lookup disambiguation','sha256':prompt.prompt_hash(prompt.focused_inject)},
  'settings':{'temperature':0,'thinking':True,'protocol':'JSON','num_predict':768,'num_ctx':8192,'num_batch':512,'seed':SEED,'max_action_turns':5,
              'asr':'PhoWhisper Large v2 cached for main val309 and val692 history'},
  'adapter_final1':{'model':final1_lock['adapter_final1']['model'],'digest':final1_lock['adapter_final1']['digest'],'predictions_sha256':sha(final1_pred),
                    'btc_after_gate':final1_summary['groups']['all_val309']['adapter_final1']['btc_0_100'],'btc_before_gate':final1_summary['groups']['all_val309']['adapter_final1']['btc_before_runtime_gate_0_100']},
  'adapter_final3':{'model':final3_lock['adapter_final3']['model'],'digest':final3_lock['adapter_final3']['digest'],'predictions_sha256':sha(final3_pred),
                    'btc_after_gate':final3_summary['groups']['all_val309']['adapter_final3']['btc_0_100'],'btc_before_gate':final3_summary['groups']['all_val309']['adapter_final3']['btc_before_runtime_gate_0_100']},
  'adapter_final4':{'model':MODEL,'digest':tags[MODEL],'adapter_archive_sha256':sha(archive),'q4_sha256':sha(q4),'merge_manifest_sha256':sha(merge_manifest),
                    'conversion_manifest_sha256':sha(conversion_manifest),'ollama_manifest_sha256':sha(ollama_manifest)},
  'input_sha256':inputs_hash,
  'selection_rule':'For a fixed-thinking public-test candidate, choose between final3 and final4 by paired full-val309 BTC before the local 30-second runtime gate; report after-gate score separately.'}
 return ev,compare,prompt,runner,ids,items,main,history,data,lock

def prepare():
 *_,lock=build();p=OUT/'comparison_lock.json'
 if p.exists() and read_json(p)!=lock:raise RuntimeError('Existing final4 lock differs from current inputs or model')
 if not p.exists():write_json(p,lock)
 write_json(OUT/'cohort_manifest.json',lock['cohort'])
 print(json.dumps({'samples':309,'model':lock['adapter_final4'],'prompt':lock['prompt'],'settings':lock['settings'],'comparison':{'final3':lock['adapter_final3'],'final1':lock['adapter_final1']}},ensure_ascii=False,indent=2),flush=True)

def run():
 if sys.prefix!=sys.base_prefix:raise RuntimeError('Use system Python so local Ollama is reachable')
 ev,compare,prompt,runner,ids,items,main,history,data,lock=build()
 if read_json(OUT/'comparison_lock.json')!=lock:raise RuntimeError('Benchmark lock changed since prepare')
 runner.MODEL=MODEL;runner.EXPECTED_DIGEST=lock['adapter_final4']['digest'];runner.OUT=OUT/'inference/adapter_final4'
 runner.CAP=768;runner.MAX_STEPS=5;runner.inject_cot_original=runner.inject_cot;runner.inject_cot=prompt.focused_inject
 runner.seed_everything(SEED)
 import lifemate.asr as asr_module
 asr_module.PhoASRModel=compare.CachedPhoWhisper;asr_module.PhoASRModel.lookup=history
 runner.OUT.mkdir(parents=True,exist_ok=True);pred=runner.OUT/'predictions-COT.jsonl'
 rows=read_jsonl(pred)
 if [r['id'] for r in rows]!=ids[:len(rows)]:raise RuntimeError('Saved adapter4 rows are not a canonical val309 prefix')
 for round_no,start in enumerate(range(0,len(ids),BLOCK),1):
  block=ids[start:start+BLOCK];done={x['id'] for x in read_jsonl(pred)};todo=[sid for sid in block if sid not in done]
  if not todo:continue
  runner.warmup('COT',round_no)
  while todo:
   before=len(read_jsonl(pred))
   try:runner.run_block('COT',round_no,todo,main,items,cot=True)
   except Exception:
    after=read_jsonl(pred)
    if len(after)<=before or not after[-1].get('error'):raise
   done={x['id'] for x in read_jsonl(pred)};todo=[sid for sid in block if sid not in done]
  meta_path=runner.OUT/'run-COT.json';meta=read_json(meta_path);meta.update({'branch':'adapter_final4','status':'complete' if len(read_jsonl(pred))==start+len(block) else 'in_progress','thinking':True,'num_predict':768,
   'model_digest':lock['adapter_final4']['digest'],'prompt_variant':'focused post-lookup disambiguation','cohort_ids':ids,'executed_rounds':round_no});write_json(meta_path,meta)
  print(f'adapter_final4 block {round_no}/21 complete; {len(read_jsonl(pred))}/309',flush=True)
 if [x['id'] for x in read_jsonl(pred)]!=ids:raise RuntimeError('Adapter4 inference incomplete')
 print('adapter_final4 inference complete: 309/309',flush=True)

def label(item):
 x=item.get('label',{});return json.loads(x) if isinstance(x,str) else x

def pct(x):return 100*statistics.mean(x) if x else None

def analyze():
 ev,compare,prompt,runner,ids,items,main,history,data,lock=build()
 if read_json(OUT/'comparison_lock.json')!=lock:raise RuntimeError('Benchmark lock changed before analysis')
 predictions={'best_baseline':read_jsonl(ev.BASELINE),
  'step120':read_jsonl(EXP/'output/val309-checkpoint-eval/inference/step120/predictions-COT.jsonl'),
  'adapter_final1':read_jsonl(EXP/'output/adapter-final-1epoch-benchmark/val309/inference/adapter_final1/predictions-COT.jsonl'),
  'adapter_final3':read_jsonl(EXP/'output/adapter-final-3epoch-benchmark/val309/inference/adapter_final3/predictions-COT.jsonl'),
  'adapter_final4':read_jsonl(OUT/'inference/adapter_final4/predictions-COT.jsonl')}
 by_id={}
 for key,rows in predictions.items():
  if [x['id'] for x in rows]!=ids:raise RuntimeError(f'{key} predictions incomplete or not canonical')
  by_id[key]={x['id']:x for x in rows}
 asks={sid for sid in ids if any(c.get('name')=='sys.ask_missing_info' for c in label(items[sid]).get('tool_calls',[]))}
 groups={'all_val309':ids,'fresh_262_primary':[s for s in ids if s not in asks],'audited_47_diagnostic':[s for s in ids if s in asks]}
 details=[]
 for sid in ids:
  gold_asks=[c for c in label(items[sid]).get('tool_calls',[]) if c.get('name')=='sys.ask_missing_info']
  rec={'id':sid,'group':'audited_47_diagnostic' if sid in asks else 'fresh_262_primary','gold_ask':sid in asks}
  for key in predictions:
   pred=by_id[key][sid];sc=ev.score_sample(sid,items[sid],pred,data)
   pa=[c for c in (pred.get('tool_calls') or []) if c.get('name')=='sys.ask_missing_info']
   rec[key]={'btc':sc['official_sample_score'],'btc_before_gate':sc['btc_formula_score_before_runtime_gate_diagnostic'],
    'hash_match':bool(sc['hash_match']),'runtime_error':bool(pred.get('error')),'error':pred.get('error'),'latency_seconds':pred.get('latency_seconds'),
    'over_30s':bool(sc['runtime_over_30s']),'gold_ask_exact':any(ev.call_key(a)==ev.call_key(g) for a in pa for g in gold_asks),'ask_count':len(pa)}
  details.append(rec)
 summaries={}
 for group,subset in groups.items():
  recs=[r for r in details if r['id'] in set(subset)];summ={}
  for key in predictions:
   vals=[r[key] for r in recs];lat=[x['latency_seconds'] for x in vals if x['latency_seconds'] is not None]
   summ[key]={'btc_after_30s_gate':pct([x['btc'] for x in vals]),'btc_before_runtime_gate':pct([x['btc_before_gate'] for x in vals]),
    'hash_matches':sum(x['hash_match'] for x in vals),'runtime_errors':sum(x['runtime_error'] for x in vals),'over_30s':sum(x['over_30s'] for x in vals),
    'mean_latency_seconds':statistics.mean(lat) if lat else None,'gold_ask_exact':sum(x['gold_ask_exact'] for x in vals),'ask_count':sum(x['ask_count'] for x in vals)}
  for challenger,inc in [('adapter_final4',4)]:
   for control,seed in [('adapter_final3',20261124),('adapter_final1',20261125),('best_baseline',20261126)]:
    d=[r[challenger]['btc']-r[control]['btc'] for r in recs]
    u=[r[challenger]['btc_before_gate']-r[control]['btc_before_gate'] for r in recs]
    summ[f'paired_{challenger}_minus_{control}']={'delta_after_gate_points':pct(d),'bootstrap_95pct_CI_after_gate':ev.boot_ci(d,seed),
      'wins':sum(x>0 for x in d),'ties':sum(x==0 for x in d),'losses':sum(x<0 for x in d),
      'delta_before_gate_points':pct(u),'bootstrap_95pct_CI_before_gate':ev.boot_ci(u,seed+1)}
  d=[r['adapter_final4']['btc_before_gate']-r['adapter_final3']['btc_before_gate'] for r in recs]
  winner=max(('adapter_final3','adapter_final4'),key=lambda k:summ[k]['btc_before_runtime_gate'])
  summ['paired_final4_minus_final3_raw']={'delta_before_gate_points':pct(d),'bootstrap_95pct_CI_before_gate':ev.boot_ci(d,20261127),
    'final4_wins':sum(x>0 for x in d),'ties':sum(x==0 for x in d),'final3_wins':sum(x<0 for x in d)}
  summ['public_candidate_selection_by_raw_score']=winner
  summaries[group]=summ
 summary={'experiment':lock['experiment'],'cohort':{'count':309,'fresh_primary':len(groups['fresh_262_primary']),'audited_gold_ask_diagnostic':len(groups['audited_47_diagnostic'])},
  'branches':{k:{'model':(lock['adapter_final4']['model'] if k=='adapter_final4' else lock.get(k,{}).get('model')),'status':'complete','samples_completed':309,'predictions_reused':k!='adapter_final4'} for k in predictions},
  'prompt_and_settings':{**lock['prompt'],**lock['settings']},'groups':summaries,
  'selection_rule':lock['selection_rule'],'public_candidate_winner_by_raw_val309':summaries['all_val309']['public_candidate_selection_by_raw_score'],
  'method':{'scoring':'Official BTC ToolEnv per sample; local 30s-gated and before-gate diagnostic reported separately','paired_by_id':True,'bootstrap_draws':10000,'bootstrap_seed_start':20261124,
   'fixed_branches':'Reused locked val309 predictions; only final4 newly inferred.'},
  'caveat':'Runtime measured on local RTX 4050, not competition A30. Public candidate selection follows user policy of fixed full thinking and therefore uses BTC before the local runtime gate; runtime-gated result is reported separately.'}
 write_json(OUT/'summary.json',summary);write_jsonl(OUT/'paired_details.jsonl',details)
 names={'best_baseline':'Best baseline','step120':'Checkpoint step120','adapter_final1':'Adapter 1 epoch','adapter_final3':'Adapter 3 epochs','adapter_final4':'Adapter 4 epochs'}
 allr=summaries['all_val309'];lines=['# Adapter final 4 epochs: fixed-thinking PhoWhisper val309','',
  'Thinking is enabled on every request. Same focused prompt, cached PhoWhisper Large v2 transcripts, JSON, temperature 0, cap768, context8192, batch512, seed17, max five turns. Final1/final3/baseline/step120 predictions are reused; final4 is newly inferred.','',
  '| Branch | BTC before local gate | BTC after 30s gate | Hash matches | Runtime errors | >30s | Mean latency | Exact asks |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for k in predictions:
  x=allr[k];lines.append(f"| {names[k]} | {x['btc_before_runtime_gate']:.2f} | {x['btc_after_30s_gate']:.2f} | {x['hash_matches']}/309 | {x['runtime_errors']} | {x['over_30s']} | {x['mean_latency_seconds']:.2f}s | {x['gold_ask_exact']}/47 |")
 d=allr['paired_final4_minus_final3_raw'];ci=d['bootstrap_95pct_CI_before_gate']
 lines+=['',f"4 epochs − 3 epochs before runtime gate: {d['delta_before_gate_points']:+.2f} (paired 95% CI {ci[0]:+.2f} to {ci[1]:+.2f}); wins/ties/losses {d['final4_wins']}/{d['ties']}/{d['final3_wins']}.",
  f"Selected public candidate by fixed-thinking raw val309 score: **{summary['public_candidate_winner_by_raw_val309']}**. No public-test rows have been generated yet.",'',summary['caveat'],'']
 (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({'public_candidate_winner_by_raw_val309':summary['public_candidate_winner_by_raw_val309'],'all_val309':allr},ensure_ascii=False,indent=2),flush=True)

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--analyze',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.run:run()
 else:analyze()
if __name__=='__main__':main()
