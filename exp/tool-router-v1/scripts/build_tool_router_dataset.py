from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import _bootstrap  # noqa: F401
from lifemate.tool_router import serialize_router_input
from lifemate.tool_executor import available_tool_names, create_tool_env
from lifemate.utils import find_data_root, load_jsonl, resolve_audio
from lifemate.asr import PhoASRModel


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root'); p.add_argument('--split-dir', default='data/splits')
    p.add_argument('--source-dir', default='exp/qwen-next-action-v1/output/training-data')
    p.add_argument('--val-predictions', default='exp/baseline-v1-val309/output/predictions.jsonl')
    p.add_argument('--output-dir', default='exp/tool-router-v1/output/training-data')
    a = p.parse_args(); root=find_data_root(a.data_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    train_ids=json.loads((Path(a.split_dir)/'train_ids.json').read_text(encoding='utf-8'))
    val_ids=json.loads((Path(a.split_dir)/'val_ids.json').read_text(encoding='utf-8'))
    if len(train_ids)!=1236 or len(val_ids)!=309 or set(train_ids)&set(val_ids): raise ValueError('fixed split mismatch/leak')
    src=read_jsonl(Path(a.source_dir)/'sft_train.jsonl'); train_set=set(train_ids)
    train=[]
    for r in src:
        if r['sample_id'] not in train_set: continue
        payload=json.loads(r['messages'][1]['content']); t=r['target']
        label='STOP' if t['action']=='stop' else t['tool']
        train.append({'id':r['sample_id'],'step':r['step_index'],'input_text':serialize_router_input(payload,r['step_index']),'target_tool':label})
    if len({x['id'] for x in train}) != 1236: raise ValueError('train source missing fixed train trajectories')
    # Validation transcripts come from the already completed baseline ASR run. Gold calls are replayed
    # once per trajectory through the official ToolEnv to reconstruct exactly the observation at each step.
    pred={r['id']:r.get('asr_text') or r.get('transcript_used','') for r in read_jsonl(a.val_predictions)}
    tr=root/'train'; items={r['id']:r for r in load_jsonl(tr/'train.jsonl')}; catalog=json.loads((tr/'train_tools_catalog.json').read_text(encoding='utf-8'))
    specs={t['name']:t for app in catalog.get('apps',[]) for t in app.get('tools',[])}
    val=[]; failures=[]; asr=PhoASRModel()
    for i,sid in enumerate(val_ids,1):
        item=items[sid]
        try:
            transcript=pred[sid]; available=available_tool_names(item)
            history=[]
            for turn in item.get('history',[]):
                converted=dict(turn)
                if converted.get('audio'):
                    converted['text']=asr.transcribe(str(resolve_audio(root,tr,item,converted['audio'])))
                    converted.pop('audio',None)
                history.append(converted)
            compact=[{'name':n} for n in available]
            context={k:v for k,v in item.get('context',{}).items() if k!='installed_apps'}
            env=create_tool_env('train',sid,root); obs=[]
            for step,call in enumerate(item.get('label',{}).get('tool_calls',[])):
                name=call['name']; args=call.get('arguments',{})
                if name not in available or name not in specs or not env.validate(name,args).get('ok'): raise ValueError(f'invalid gold call at {step}: {name}')
                payload={'transcript':transcript,'context':context,'history':history,'available_tools':compact,'observations':obs}
                val.append({'id':sid,'step':step,'input_text':serialize_router_input(payload,step),'target_tool':name})
                result=env.call(name,args); obs.append({'tool':name,'arguments':args,'result':result})
            step=len(item.get('label',{}).get('tool_calls',[])); payload={'transcript':transcript,'context':context,'history':history,'available_tools':compact,'observations':obs}
            val.append({'id':sid,'step':step,'input_text':serialize_router_input(payload,step),'target_tool':'STOP'})
        except Exception as e: failures.append({'id':sid,'error':f'{type(e).__name__}: {e}'})
        if i%50==0: print(f'validation trajectories {i}/{len(val_ids)}; examples={len(val)}; failures={len(failures)}',flush=True)
    for name,rows in [('train.jsonl',train),('val.jsonl',val)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    (out/'replay_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    counts=Counter(x['target_tool'] for x in train); classes=sorted({'STOP'}|set(specs)); (out/'class_mapping.json').write_text(json.dumps({c:i for i,c in enumerate(classes)},indent=2),encoding='utf-8')
    def step_buckets(rows):
        c=Counter('STOP' if r['target_tool']=='STOP' else 'step_0' if r['step']==0 else 'step_1' if r['step']==1 else 'step_2_plus' for r in rows)
        return dict(sorted(c.items()))
    summary={'train_trajectories':len(train_ids),'validation_trajectories':len(val_ids),'train_examples':len(train),'validation_examples':len(val),'train_validation_overlap':0,'train_class_counts':dict(sorted(counts.items())),'val_class_counts':dict(sorted(Counter(x['target_tool'] for x in val).items())),'train_step_counts':step_buckets(train),'val_step_counts':step_buckets(val),'validation_replay_failures':failures,'train_source':'gold ToolEnv replay dataset qwen-next-action-v1; arguments removed from labels','validation_transcripts':'baseline-v1-val309 PhoASR outputs','split_dir':str(a.split_dir)}
    (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if not failures else 2
if __name__=='__main__': raise SystemExit(main())
