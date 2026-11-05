from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import torch
from transformers import PreTrainedTokenizerFast,AutoModelForSequenceClassification

def scores(y,pred):
 labels=sorted(set(y)|set(pred)); out={};
 for c in labels:
  tp=sum(a==c and b==c for a,b in zip(y,pred));fp=sum(a!=c and b==c for a,b in zip(y,pred));fn=sum(a==c and b!=c for a,b in zip(y,pred));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0
  errors=Counter(b for a,b in zip(y,pred) if a==c and b!=c)
  out[c]={'support':sum(a==c for a in y),'precision':pr,'recall':re,'f1':2*pr*re/(pr+re) if pr+re else 0,'common_confusions':[{'predicted':x,'count':n} for x,n in errors.most_common(3)]}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default='exp/tool-router-v1/output/training-data');p.add_argument('--model-dir',default='exp/tool-router-v1/output/training-data/model');p.add_argument('--output-dir',default='exp/tool-router-v1-val309/output');p.add_argument('--qwen-predictions',default='exp/baseline-v1-val309/output/predictions.jsonl');a=p.parse_args()
 d=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[json.loads(s) for s in (d/'val.jsonl').read_text(encoding='utf-8').splitlines() if s]
 tok=PreTrainedTokenizerFast.from_pretrained(a.model_dir,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(a.model_dir,local_files_only=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(dev).eval(); records=[]
 training_summary=json.loads((Path(a.model_dir)/'training_summary.json').read_text(encoding='utf-8')); max_length=training_summary['max_length']
 with torch.inference_mode():
  for r in rows:
   x=tok(r['input_text'],truncation=True,max_length=max_length,return_tensors='pt');x={k:v.to(dev) for k,v in x.items()};logits=model(**x).logits[0]; vals,idx=logits.topk(min(3,len(model.config.id2label))); pred=model.config.id2label[int(idx[0])];records.append({**r,'prediction':pred,'top3':[model.config.id2label[int(i)] for i in idx],'confidence':float(torch.softmax(logits,-1)[idx[0]])})
 y=[r['target_tool'] for r in records];pred=[r['prediction'] for r in records];per=scores(y,pred);overall={'examples':len(y),'top1_accuracy':sum(a==b for a,b in zip(y,pred))/len(y),'top3_accuracy':sum(r['target_tool'] in r['top3'] for r in records)/len(y),'macro_f1':sum(x['f1'] for x in per.values())/len(per),'weighted_f1':sum(x['f1']*x['support'] for x in per.values())/len(y),'accuracy_excluding_STOP':sum(a==b and a!='STOP' for a,b in zip(y,pred))/max(1,sum(a!='STOP' for a in y)),'stop':per.get('STOP',{})}
 bystep=defaultdict(list); bytraj=defaultdict(list)
 for r in records:bystep['STOP' if r['target_tool']=='STOP' else ('first' if r['step']==0 else 'second' if r['step']==1 else 'third_plus')].append(r);bytraj[r['id']].append(r)
 step_metrics={k:{'support':len(v),'accuracy':sum(z['prediction']==z['target_tool'] for z in v)/len(v)} for k,v in bystep.items()}
 trajectories=list(bytraj.values());traj={'all_tool_names_correct':sum(all(z['prediction']==z['target_tool'] for z in t if z['target_tool']!='STOP') for t in trajectories)/len(trajectories),'first_correct_second_wrong':sum(len(t)>1 and t[0]['prediction']==t[0]['target_tool'] and t[1]['prediction']!=t[1]['target_tool'] for t in trajectories)/len(trajectories),'wrong_first':sum(t[0]['prediction']!=t[0]['target_tool'] for t in trajectories)/len(trajectories)}
 confusion=Counter((r['target_tool'],r['prediction']) for r in records if r['target_tool']!=r['prediction']);(out/'per_class_metrics.json').write_text(json.dumps(per,ensure_ascii=False,indent=2),encoding='utf-8')
 with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
  classes=sorted(set(y)|set(pred));w=csv.writer(f);w.writerow(['gold\\pred',*classes]);
  for c in classes:w.writerow([c,*[sum(g==c and pr==x for g,pr in zip(y,pred)) for x in classes]])
 (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
 qwen={json.loads(s)['id']:json.loads(s) for s in Path(a.qwen_predictions).read_text(encoding='utf-8').splitlines() if s}; qgold=Counter();qok=Counter();stop_tp=stop_fp=stop_fn=0
 for r in records:
  q=qwen.get(r['id'],{}); calls=q.get('tool_calls',[]);step=r['step'];gold=r['target_tool'];got=calls[step].get('name') if step<len(calls) else 'STOP'
  if gold=='STOP': stop_tp+=got=='STOP';stop_fn+=got!='STOP';continue
  bucket='first' if step==0 else 'second' if step==1 else 'later';qgold[bucket]+=1;qok[bucket]+=got==gold
  if got=='STOP': stop_fp+=1
 qcomp={k:{'support':qgold[k],'accuracy':qok[k]/qgold[k] if qgold[k] else None} for k in qgold};qcomp['stop_behavior']={'precision':stop_tp/(stop_tp+stop_fp) if stop_tp+stop_fp else 0,'recall':stop_tp/(stop_tp+stop_fn) if stop_tp+stop_fn else 0,'support':stop_tp+stop_fn};summary={'overall':overall,'by_step':step_metrics,'trajectory':traj,'qwen_baseline_routing':qcomp,'common_confusions':[{'gold':g,'predicted':p,'count':n} for (g,p),n in confusion.most_common(20)],'model':model.config._name_or_path,'parameter_count':sum(p.numel() for p in model.parameters()),'matching':'exact tool name at aligned gold timestep; Qwen may diverge after wrong tool'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 md=['# Tool router evaluation: fixed val309','','## Overall',f"- Top-1: {overall['top1_accuracy']:.4f}",f"- Top-3: {overall['top3_accuracy']:.4f}",f"- Macro F1: {overall['macro_f1']:.4f}",f"- Weighted F1: {overall['weighted_f1']:.4f}",f"- Non-STOP accuracy: {overall['accuracy_excluding_STOP']:.4f}",f"- STOP F1: {overall['stop'].get('f1',0):.4f}",'','## By step']+[f"- {k}: {v['accuracy']:.4f} (n={v['support']})" for k,v in step_metrics.items()]+['','## Qwen baseline aligned routing']+[f"- {k}: {v['accuracy']:.4f} (n={v['support']})" for k,v in qcomp.items() if 'accuracy' in v]+[f"- STOP precision/recall: {qcomp['stop_behavior']['precision']:.4f}/{qcomp['stop_behavior']['recall']:.4f}",'','## Common confusions']+[f"- {c['gold']} → {c['predicted']}: {c['count']}" for c in summary['common_confusions']]
 decision='GO' if overall['top1_accuracy']>=.70 and step_metrics.get('first',{}).get('accuracy',0)>=.75 and step_metrics.get('second',{}).get('accuracy',0)>=.5 else 'NO-GO' if overall['top1_accuracy']<.5 or step_metrics.get('second',{}).get('accuracy',0)<.3 else 'BORDERLINE';summary['decision']=decision;(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');md+=['',f'## Decision: {decision}'];(out/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
