from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import PreTrainedTokenizerFast,AutoModelForSequenceClassification

MODEL='models/BamiBERT'
class Rows(Dataset):
 def __init__(self,rows,tok,labels,max_len): self.rows=rows;self.tok=tok;self.labels=labels;self.max_len=max_len
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; x=self.tok(r['input_text'],truncation=True,max_length=self.max_len,return_tensors='pt')
  return {k:v[0] for k,v in x.items()},self.labels[r['target_tool']]
def collate(batch,tok):
 xs,ys=zip(*batch); features=[{k:v[k].tolist() for k in v} for v in xs]; return tok.pad(features,padding=True,return_tensors='pt'),torch.tensor(ys)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default='exp/tool-router-v1/output/training-data');p.add_argument('--output-dir',default='exp/tool-router-v1/output/training-data/model');p.add_argument('--model',default=MODEL);p.add_argument('--tool-catalog',default='train+public_test/train+public_test/train/train_tools_catalog.json');p.add_argument('--max-length',type=int,default=512);p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--seed',type=int,default=20261102);p.add_argument('--token-file',default='.env');a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 d=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 train=[json.loads(s) for s in (d/'train.jsonl').read_text(encoding='utf-8').splitlines() if s]; val=[json.loads(s) for s in (d/'val.jsonl').read_text(encoding='utf-8').splitlines() if s]
 catalog=json.loads(Path(a.tool_catalog).read_text(encoding='utf-8')); labels=sorted({'STOP'}|{t['name'] for app in catalog.get('apps',[]) for t in app.get('tools',[])});mapping={x:i for i,x in enumerate(labels)}
 model_path=Path(a.model)
 revision=next((x.stem for x in (model_path/'.cache/huggingface/trees').iterdir() if x.is_file()),'local-unknown') if (model_path/'.cache/huggingface/trees').is_dir() else 'local-unknown'
 # The released tokenizer.json is BPE while tokenizer_config.json names the
 # slow XLM-R tokenizer (which expects Unigram). Load the declared BPE graph directly.
 tok=PreTrainedTokenizerFast(tokenizer_file=str(model_path/'tokenizer.json'),bos_token='<s>',eos_token='</s>',unk_token='<unk>',sep_token='</s>',pad_token='<pad>',cls_token='<s>',mask_token='<mask>')
 model=AutoModelForSequenceClassification.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,num_labels=len(labels),id2label={i:x for x,i in mapping.items()},label2id=mapping)
 params=sum(x.numel() for x in model.parameters()); print(json.dumps({'model':a.model,'revision':revision,'parameters':params,'max_length':a.max_length,'tokenizer_class':type(tok).__name__,'tokenizer_vocab':len(tok),'device':'cuda' if torch.cuda.is_available() else 'cpu'},indent=2))
 if params>=6_000_000_000: raise ValueError('competition total parameter cap exceeded')
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(dev)
 # Measure truncation once before training.
 lengths=[]
 for row in train+val: lengths.append(len(tok(row['input_text'],truncation=False)['input_ids']))
 trunc_rate=sum(n>a.max_length for n in lengths)/len(lengths)
 tr=DataLoader(Rows(train,tok,mapping,a.max_length),batch_size=a.batch_size,shuffle=True,collate_fn=lambda b:collate(b,tok))
 va=DataLoader(Rows(val,tok,mapping,a.max_length),batch_size=a.batch_size,shuffle=False,collate_fn=lambda b:collate(b,tok))
 counts=np.bincount([mapping[r['target_tool']] for r in train],minlength=len(labels)); class_weights=np.zeros(len(labels),dtype=np.float32);present=counts>0;class_weights[present]=1/np.sqrt(counts[present]);class_weights[present]/=class_weights[present].mean();weight=torch.tensor(class_weights,device=dev)
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01); total=len(tr)*a.epochs; warm=max(1,int(total*.1))
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warm, max(0.,(total-s)/max(1,total-warm))))
 use_amp=dev.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=use_amp);best=-1;bad=0;history=[];start=time.time();torch.cuda.reset_peak_memory_stats() if use_amp else None
 for epoch in range(a.epochs):
  model.train(); losses=[]
  for x,y in tr:
   x={k:v.to(dev) for k,v in x.items()};y=y.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=use_amp): logits=model(**x).logits; loss=torch.nn.functional.cross_entropy(logits,y,weight=weight)
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();scheduler.step();losses.append(float(loss.detach()))
  model.eval();correct=0;n=0
  with torch.inference_mode():
   for x,y in va:
    x={k:v.to(dev) for k,v in x.items()}; y=y.to(dev)
    with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=use_amp): logits=model(**x).logits
    correct+=(logits.argmax(-1)==y).sum().item();n+=len(y)
  acc=correct/n; row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_accuracy':acc};history.append(row);print(json.dumps(row),flush=True)
  if acc>best:
   best=acc;bad=0;model.save_pretrained(out);tok.save_pretrained(out)
  else: bad+=1
  if bad>=2: break
 summary={'model':'Qualcomm-AI-Research/BamiBERT','revision':revision,'parameters':params,'train_examples':len(train),'val_examples':len(val),'max_length':a.max_length,'batch_size':a.batch_size,'learning_rate':a.lr,'class_weighting':'inverse square-root frequency, normalized over observed classes','epochs_completed':len(history),'best_val_accuracy':best,'truncation_rate':trunc_rate,'tokenizer_class':type(tok).__name__,'tokenizer_config':tok.init_kwargs,'seed':a.seed,'history':history,'train_seconds':time.time()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated() if use_amp else None,'checkpoint':str(out)}
 (out/'training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='tokenizer_config'},indent=2))
if __name__=='__main__':main()
