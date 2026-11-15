from __future__ import annotations
import json, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'exp/qwen35-sft-pho-synthetic-v1'
sys.path.insert(0, str(EXP / 'scripts'))
import tune_adaptive_total_budget_maxturn_val60 as screen
OUT = EXP / 'output/adaptive-totalbudget-maxturn-fullval309'
THRESHOLDS = [15.0, 20.0]
CAPS = [4, 5, 6, 7]
BLOCK = 15

def setup():
    bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds = screen.build()
    archive = screen.read_jsonl(screen.ARCHIVED)
    return bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds, archive

def analyze():
    bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds, archive = setup()
    out = {}
    details = []
    per_threshold = {}
    for threshold in THRESHOLDS:
        branch = OUT / f'total-{int(threshold)}s'
        preds = screen.read_jsonl(branch / 'predictions-COT.jsonl')
        if [r['id'] for r in preds] != ids:
            raise RuntimeError(f'Incomplete or noncanonical full val for {threshold}s: {len(preds)}/{len(ids)}')
        pred_map = {r['id']: r for r in preds}
        requests = screen.read_jsonl(branch / 'requests-COT.jsonl')
        groups = {}
        for r in requests:
            groups.setdefault(r['sample_id'], []).append(r)
        per_threshold[str(int(threshold))] = {}
        for cap in CAPS:
            scored = screen.summarize(pred_map, ids, items, data, asr_seconds, groups, cap)
            per_threshold[str(int(threshold))][str(cap)] = screen.stats(scored)
            details.extend({'threshold_total_seconds': threshold, 'max_action_turns': cap, **r} for r in scored)
        out[str(int(threshold))] = (pred_map, groups)
    score_rows = {(float(d['threshold_total_seconds']), int(d['max_action_turns']), d['id']): d for d in details}
    paired = []
    for a,b,cap in [(15.0,20.0,6)]:
        diffs = [score_rows[(a,cap,sid)]['gated'] - score_rows[(b,cap,sid)]['gated'] for sid in ids]
        ci = screen.ci_mean_delta(diffs, 20261115)
        rawdiffs = [score_rows[(a,cap,sid)]['raw'] - score_rows[(b,cap,sid)]['raw'] for sid in ids]
        paired.append({'comparison': f'{int(a)}s minus {int(b)}s, cap{cap}', 'gated_delta_points': 100*statistics.mean(diffs), 'gated_bootstrap_95_ci_points': ci, 'raw_delta_points': 100*statistics.mean(rawdiffs), 'n': len(ids)})
    for cap_a,cap_b in [(6,5),(7,6)]:
        diffs = [score_rows[(15.0,cap_a,sid)]['gated'] - score_rows[(15.0,cap_b,sid)]['gated'] for sid in ids]
        ci = screen.ci_mean_delta(diffs, 20261116+cap_a)
        rawdiffs = [score_rows[(15.0,cap_a,sid)]['raw'] - score_rows[(15.0,cap_b,sid)]['raw'] for sid in ids]
        paired.append({'comparison': f'15s cap{cap_a} minus cap{cap_b}', 'gated_delta_points': 100*statistics.mean(diffs), 'gated_bootstrap_95_ci_points': ci, 'raw_delta_points': 100*statistics.mean(rawdiffs), 'n': len(ids)})
    summary = {
        'experiment': 'qwen35-final4-adaptive-total-budget-max-turn-full-val309',
        'model_digest': model_lock['adapter_final4']['digest'],
        'thresholds_are_end_to_end_seconds_including_asr': THRESHOLDS,
        'asr_timing': {'mean_per_sample': statistics.mean(asr_seconds.values()), 'median_per_sample': statistics.median(asr_seconds.values()), 'p95_per_sample': sorted(asr_seconds.values())[int(.95*(len(ids)-1))], 'main_clips': 309, 'history_clips': 692, 'model_load_startup_excluded': True},
        'caps_scored_from_nested_trajectories': CAPS,
        'results': per_threshold,
        'paired_comparisons': paired,
        'note': 'Local RTX 4050/Q4 evaluation; val PhoWhisper inference_seconds added to Qwen request latency before 30s official gate. Revalidate on newly fine-tuned adapter and target GPU.'
    }
    screen.write_json(OUT/'summary.json', summary)
    screen.write_jsonl(OUT/'paired_details.jsonl', details)
    lines=['# Adaptive total-budget threshold and max-turn full val309','',
      'Each threshold is applied to PhoWhisper per-task inference seconds plus cumulative Qwen request time. First Qwen request uses thinking. Up to seven action turns were generated once per threshold; caps 4–7 are scored from nested prefixes and repair requests within each included action turn are counted.']
    lines += ['',f"ASR timing: mean {summary['asr_timing']['mean_per_sample']:.2f}s/task, median {summary['asr_timing']['median_per_sample']:.2f}s, p95 {summary['asr_timing']['p95_per_sample']:.2f}s; load startup excluded."]
    for th in THRESHOLDS:
        lines += ['',f'## Total threshold {int(th)} seconds','', '| Max turns | BTC raw | BTC with ASR+Qwen 30s gate | Mean end-to-end sec | >30s | Mean tool calls | runtime errors |','|---:|---:|---:|---:|---:|---:|---:|']
        for cap in CAPS:
            m=per_threshold[str(int(th))][str(cap)]
            lines.append(f"| {cap} | {m['btc_raw']:.2f} | {m['btc_asr_aware_gated']:.2f} | {m['mean_total_seconds']:.2f} | {m['over_30s']}/{m['n']} | {m['mean_tool_calls']:.2f} | {m['runtime_errors']} |")
    lines += ['', '## Paired comparisons','', '| Comparison | Gated score delta (points) | Bootstrap 95% CI | Raw delta |','|---|---:|---:|---:|']
    for x in paired: lines.append(f"| {x['comparison']} | {x['gated_delta_points']:.2f} | [{x['gated_bootstrap_95_ci_points'][0]:.2f}, {x['gated_bootstrap_95_ci_points'][1]:.2f}] | {x['raw_delta_points']:.2f} |")
    lines += ['', 'Local Q4/RTX 4050 latency is not target A30 timing; ASR load startup is excluded. Confirm the winning policy after the new fine-tune finishes.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'analyzed','summary':str(OUT/'summary.json')}, ensure_ascii=False), flush=True)

def run():
    if sys.prefix != sys.base_prefix:
        raise RuntimeError('Use native system Python so local Ollama is reachable.')
    bench, compare, prompt, runner, ids, items, main, history, data, model_lock, asr_seconds, archive = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = {
        'experiment':'qwen35-final4-adaptive-total-budget-max-turn-full-val309', 'ids':ids,
        'thresholds_seconds_include_asr':THRESHOLDS,'caps_scored':CAPS,'run_cap':max(CAPS),
        'model_digest':model_lock['adapter_final4']['digest'],
        'asr_seconds':{'mean_per_sample':statistics.mean(asr_seconds.values()),'main_clips':309,'history_clips':692},
        'settings':{'temperature':0,'num_predict':screen.GEN_CAP,'num_ctx':screen.CONTEXT,'num_batch':screen.BATCH,'seed':17,'successful_ask_missing_info_harness_stop':True,'prompt':'focused post-lookup'},
        'script_sha256':screen.sha(Path(__file__)),
    }
    lock_path=OUT/'comparison_lock.json'
    if lock_path.exists() and screen.read_json(lock_path)!=lock:
        raise RuntimeError('Full val lock differs from existing run; refusing to mix configs.')
    if not lock_path.exists(): screen.write_json(lock_path,lock)
    runner.MODEL=screen.MODEL
    runner.EXPECTED_DIGEST=model_lock['adapter_final4']['digest']
    runner.CAP=screen.GEN_CAP; runner.CONTEXT=screen.CONTEXT; runner.BATCH=screen.BATCH; runner.SEED=17; runner.MAX_STEPS=max(CAPS)
    runner.inject_cot_original=runner.inject_cot; runner.inject_cot=prompt.focused_inject
    runner.seed_everything(17); runner.model_fingerprint()
    import lifemate.pipeline as pipeline
    import lifemate.agent as agent_module
    import lifemate.asr as asr_module
    source=Path(pipeline.__file__).read_text(encoding='utf-8')
    original_agent=agent_module.Agent
    asr_module.PhoASRModel=compare.CachedPhoWhisper; asr_module.PhoASRModel.lookup=history
    try:
        screen.patch_ask_stop(pipeline,source)
        for threshold in THRESHOLDS:
            branch=OUT/f'total-{int(threshold)}s'; branch.mkdir(parents=True,exist_ok=True); runner.OUT=branch
            agent_module.Agent=screen.adaptive_factory(original_agent,threshold,asr_seconds)
            pred_path=branch/'predictions-COT.jsonl'
            done=screen.read_jsonl(pred_path); done_ids=[r['id'] for r in done]
            if done_ids!=ids[:len(done_ids)]: raise RuntimeError(f'Noncanonical saved prefix {threshold}s')
            for round_no,start in enumerate(range(0,len(ids),BLOCK),1):
                block=ids[start:start+BLOCK]; completed={r['id'] for r in screen.read_jsonl(pred_path)}; todo=[sid for sid in block if sid not in completed]
                if not todo: continue
                runner.seed_everything(17); runner.warmup('COT',round_no)
                while todo:
                    before=len(screen.read_jsonl(pred_path))
                    try: runner.run_block('COT',round_no,todo,main,items,cot=True)
                    except Exception:
                        after=screen.read_jsonl(pred_path)
                        if len(after)<=before: raise
                    completed={r['id'] for r in screen.read_jsonl(pred_path)}; todo=[sid for sid in block if sid not in completed]
                print(f'[threshold={int(threshold)}s] completed {min(start+len(block),len(ids))}/{len(ids)}',flush=True)
            final=screen.read_jsonl(pred_path)
            if [r['id'] for r in final]!=ids: raise RuntimeError(f'Incomplete threshold {threshold}')
    finally:
        agent_module.Agent=original_agent
        exec(compile(source,str(pipeline.__file__),'exec'),pipeline.__dict__)
    analyze()

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else '--run'
    if mode=='--run': run()
    elif mode=='--analyze': analyze()
    else: raise SystemExit('Use --run or --analyze')
