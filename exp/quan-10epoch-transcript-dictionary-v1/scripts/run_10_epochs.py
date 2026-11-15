from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'exp/quan-10epoch-transcript-dictionary-v1'
OUT=EXP/'output'
VENV=ROOT/'.venv/Scripts/python.exe'
TRAIN=ROOT/'exp/quan-continuation-cot-v1/scripts/finetune_quan_continuation.py'
EVAL=ROOT/'exp/asr-modeling/scripts/inference_quan_transcribe.py'
BASE=ROOT/'models/quan-transcribe'
DATA=ROOT/'train+public_test/train+public_test'
TRAIN_IDS=ROOT/'data/splits/train_ids.json'; VAL_IDS=ROOT/'data/splits/val_ids.json'
SEED=17; EPOCHS=10; LR_READER=5e-5; LR_ENCODER=1e-6; ACCUM=4
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path:Path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path:Path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def validate_env():
 if not VENV.is_file(): raise FileNotFoundError(VENV)
 r=subprocess.run([str(VENV),'-c','import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'],cwd=ROOT,text=True,capture_output=True,check=True)
 lines=r.stdout.strip().splitlines()
 if not lines or lines[0].strip().lower()!='true': raise RuntimeError('Project .venv does not have a usable CUDA runtime')
 return {'python':str(VENV),'cuda':True,'gpu':lines[1] if len(lines)>1 else 'unknown'}
def hashes():
 files=[ROOT/'TASK.md',ROOT/'COMPETITION.md',TRAIN,ROOT/'exp/asr-modeling/scripts/asr_finetune_common.py',EVAL,ROOT/'exp/asr-modeling/scripts/inference_common.py',ROOT/'models/quan-transcribe/config.json',ROOT/'models/quan-transcribe/units.json',BASE/'model.safetensors',BASE/'best.pt',BASE/'lm/lm_config.json',TRAIN_IDS,VAL_IDS,DATA/'train/train.jsonl',DATA/'public_test/public_test.jsonl']
 return {str(p.relative_to(ROOT)):sha(p) for p in files}
def preflight():
 OUT.mkdir(parents=True,exist_ok=True)
 plan_path=OUT/'training_plan.json'
 if plan_path.exists(): raise FileExistsError(f'Preflight already exists: {plan_path}')
 env=validate_env(); train_ids=load(TRAIN_IDS); val_ids=load(VAL_IDS)
 if len(train_ids)!=1236 or len(set(train_ids))!=1236 or len(val_ids)!=309 or len(set(val_ids))!=309 or set(train_ids)&set(val_ids): raise RuntimeError('Unexpected train/val split or overlap')
 rows={}
 for line in (DATA/'train/train.jsonl').read_text(encoding='utf-8-sig').splitlines():
  if line.strip():
   row=json.loads(line)
   if row['id'] in set(train_ids): rows[row['id']]={'audio':row['audio'],'audio_text':row['audio_text']}
 if set(rows)!=set(train_ids): raise RuntimeError('Train IDs not covered by train manifest')
 missing=[sid for sid,row in rows.items() if not row['audio_text'].strip() or not (DATA/'train'/row['audio']).is_file()]
 if missing: raise RuntimeError(f'Missing transcript/audio for {len(missing)} train rows')
 public=[json.loads(line) for line in (DATA/'public_test/public_test.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
 if len(public)!=634 or len({r['id'] for r in public})!=634: raise RuntimeError('Unexpected public-test manifest size')
 plan={'status':'preflight_locked','objective':'Train QUAN for 10 chained epochs from the same original pretrained QUAN checkpoint; evaluate each epoch on val309 and select lowest corpus WER, CER tie-break.',
  'starting_checkpoint':'models/quan-transcribe','starting_model_sha256':sha(BASE/'model.safetensors'),'starting_best_pt_sha256':sha(BASE/'best.pt'),
  'train_ids':len(train_ids),'train_ids_sha256_newline_joined':hashlib.sha256('\n'.join(train_ids).encode()).hexdigest(),
  'val_ids':len(val_ids),'val_ids_sha256_newline_joined':hashlib.sha256('\n'.join(val_ids).encode()).hexdigest(),
  'train_validation_overlap':0,'public_test_samples':len(public),'dictionary_source':'train_ids.json only (1236); val309 excluded from error-pair extraction',
  'schedule':[{'epoch':1,'reader_epochs':1,'encoder_epochs':0,'reader_lr':LR_READER,'encoder_lr':LR_ENCODER}]+[{'epoch':e,'reader_epochs':0,'encoder_epochs':1,'reader_lr':LR_READER,'encoder_lr':LR_ENCODER} for e in range(2,EPOCHS+1)],
  'seed':SEED,'accumulation':ACCUM,'decoder':'KenLM CTC beam 32; alpha/beta read from locked lm_config','gpu':env,'input_sha256':hashes(),
  'data_policy':'Training consumes audio/transcripts for train IDs only. val309 labels are used only for WER/CER evaluation and checkpoint selection. No *_environment/ files are accessed.',
  'public_test_policy':'Transcribe audio only; no labels used. Suggested dictionary edits remain unaccepted and unapplied.'}
 write(plan_path,plan)
 write(OUT/'run_state.json',{'status':'preflight_locked','created_epoch':time.time(),'completed_epochs':[]})
 print(json.dumps({'status':'preflight_locked','gpu':env['gpu'],'train':len(train_ids),'validation':len(val_ids),'public_test':len(public),'estimated_train_minutes':77},ensure_ascii=False,indent=2))
def run_logged(command:list[str],log_path:Path,stage:str):
 print(f'[{stage}] starting: {" ".join(command)}',flush=True)
 env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['PYTHONUTF8']='1'
 with log_path.open('a',encoding='utf-8') as log:
  p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,env=env)
  assert p.stdout is not None
  for line in p.stdout:
   log.write(line);log.flush();print(line.rstrip(),flush=True)
  code=p.wait()
 if code: raise subprocess.CalledProcessError(code,command)
def eval_checkpoint(model_dir:Path,label:str):
 output=OUT/'validation'/label
 if (output/'summary.json').is_file(): return load(output/'summary.json')
 command=[str(VENV),str(EVAL),'--val-ids',str(VAL_IDS),'--data-root',str(DATA),'--model',str(model_dir),'--lm',str(BASE/'lm'),
  '--output-dir',str(output),'--device','cuda','--beam-width','32','--overwrite','--phoasr-predictions',str(OUT/'no_phoasr_comparison.jsonl')]
 run_logged(command,OUT/'logs'/f'{label}-eval.log',f'{label} val309')
 summary=load(output/'summary.json')
 if summary['samples']!=309: raise RuntimeError(f'Incomplete validation evaluation for {label}')
 return summary
def run():
 plan=load(OUT/'training_plan.json'); state=load(OUT/'run_state.json')
 if plan.get('status')!='preflight_locked': raise RuntimeError('Run plan is not locked')
 now=hashes()
 if now!=plan['input_sha256']: raise RuntimeError('Locked training input hashes changed')
 validate_env()
 state['status']='running';state['started_epoch']=state.get('started_epoch',time.time());write(OUT/'run_state.json',state)
 baseline=eval_checkpoint(BASE,'epoch-00-base')
 results=load(OUT/'epoch_metrics.json') if (OUT/'epoch_metrics.json').exists() else []
 if not results: results.append({'epoch':0,'checkpoint':'models/quan-transcribe','model_sha256':sha(BASE/'model.safetensors'),'corpus_wer':baseline['metrics']['corpus_wer'],'corpus_cer':baseline['metrics']['corpus_cer'],'role':'pretrained baseline; comparison only, not fine-tuned candidate'})
 for epoch in range(1,EPOCHS+1):
  ckpt=OUT/'checkpoints'/f'epoch-{epoch:02d}'
  val_dir=OUT/'validation'/f'epoch-{epoch:02d}'
  if any(r.get('epoch')==epoch for r in results): continue
  if ckpt.exists() and (not (ckpt/'training-summary.json').is_file()): raise FileExistsError(f'Incomplete epoch output exists: {ckpt}')
  if not (ckpt/'training-summary.json').is_file():
   base=BASE if epoch==1 else OUT/'checkpoints'/f'epoch-{epoch-1:02d}'
   reader_epochs=1 if epoch==1 else 0; encoder_epochs=0 if epoch==1 else 1
   command=[str(VENV),str(TRAIN),'--base-model',str(base),'--inference-code',str(BASE/'inference'),'--output',str(ckpt),
    '--seed',str(SEED),'--epoch-index',str(epoch-1),'--reader-epochs',str(reader_epochs),'--encoder-epochs',str(encoder_epochs),
    '--reader-learning-rate',str(LR_READER),'--encoder-learning-rate',str(LR_ENCODER),'--accumulation',str(ACCUM)]
   run_logged(command,OUT/'logs'/f'epoch-{epoch:02d}-train.log',f'epoch {epoch}/10 training')
  summary=eval_checkpoint(ckpt,f'epoch-{epoch:02d}')
  row={'epoch':epoch,'checkpoint':str(ckpt.relative_to(ROOT)),'model_sha256':sha(ckpt/'model.safetensors'),
   'corpus_wer':summary['metrics']['corpus_wer'],'corpus_cer':summary['metrics']['corpus_cer'],
   'mean_inference_seconds':summary['metrics']['mean_inference_seconds'],'training_summary':str((ckpt/'training-summary.json').relative_to(ROOT))}
  results=[r for r in results if r.get('epoch')!=epoch]+[row]; results.sort(key=lambda r:r['epoch'])
  write(OUT/'epoch_metrics.json',results)
  state['completed_epochs']=sorted(set(state.get('completed_epochs',[])+[epoch]));state['last_completed_epoch']=epoch;write(OUT/'run_state.json',state)
  print(json.dumps(row,ensure_ascii=False),flush=True)
 # Choose best among trained checkpoints by WER, then CER, then earliest epoch.
 candidates=[r for r in results if r['epoch']>0]
 best=min(candidates,key=lambda r:(r['corpus_wer'],r['corpus_cer'],r['epoch']))
 selection={'status':'complete','selection_metric_priority':['corpus_wer','corpus_cer','epoch'],'baseline_epoch0':results[0],
  'evaluated_epochs':candidates,'selected_checkpoint':best['checkpoint'],'selected_epoch':best['epoch'],
  'selected_model_sha256':best['model_sha256'],'selection_rule':'lowest val309 corpus WER; lower CER breaks exact WER ties; earliest epoch breaks remaining ties.',
  'validation_policy':'val309 labels used only to evaluate/select checkpoint; excluded from training targets and dictionary extraction.',
  'public_test_labels_used':False}
 write(OUT/'checkpoint-selection.json',selection)
 state.update({'status':'complete','finished_epoch':time.time(),'selected_epoch':best['epoch']});write(OUT/'run_state.json',state)
 print(json.dumps({'status':'complete','selected_epoch':best['epoch'],'wer':best['corpus_wer'],'cer':best['corpus_cer']},ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
 if a.preflight:preflight()
 else:run()
if __name__=='__main__':main()
