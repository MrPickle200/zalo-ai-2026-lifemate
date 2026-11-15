from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, random, statistics, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'exp/btc-scoring/scripts'), str(ROOT / 'exp/quan-continuation-cot-v1/scripts')]
EXP = ROOT / 'exp/qwen35-sft-pho-synthetic-v1'
OUT = EXP / 'output/runtime-budget-ablation'
EVAL_SCRIPT = EXP / 'scripts/evaluate_val309_checkpoints.py'
STEP120 = EXP / 'output/val309-checkpoint-eval/inference/step120/predictions-COT.jsonl'
DETAILS = EXP / 'output/val309-checkpoint-eval/paired_details.jsonl'
BASELINE = ROOT / 'exp/ask-lookup-phowhisper-asr-val309/output/phowhisper-large/predictions-COT.jsonl'
SEED = 20261113
N_SLOW = 42
N_CONTROL = 78
HOST = 'http://127.0.0.1:11434'


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if path.exists() else []

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def context():
    ev = module('runtime_ablation_eval', EVAL_SCRIPT)
    compare, prompt, runner, ids, items, main_asr, history, data, eval_lock = ev.context()
    if len(ids) != 309 or [r['id'] for r in read_jsonl(STEP120)] != ids:
        raise RuntimeError('step120 predictions are not the canonical complete val309')
    tags = ev.context()[0]  # context() already verifies Ollama tags, prompt and locked inputs.
    return ev, compare, prompt, runner, ids, items, main_asr, history, data, eval_lock

def parse_label(item):
    label = item.get('label', {})
    return json.loads(label) if isinstance(label, str) else label

def turns(row):
    return len(row.get('model_output') or [])

def choose_controls(ids, items, main_asr, by_step, slow_ids):
    rng = random.Random(SEED)
    slow_set = set(slow_ids)
    fast_ids = [sid for sid in ids if sid not in slow_set]
    ask = {sid: any(c.get('name') == 'sys.ask_missing_info' for c in parse_label(items[sid]).get('tool_calls', [])) for sid in ids}
    slow_ask = sum(ask[sid] for sid in slow_ids)
    target_ask = round(N_CONTROL * slow_ask / len(slow_ids))
    quotas = {True: target_ask, False: N_CONTROL - target_ask}
    picked = []
    available = {flag: [sid for sid in fast_ids if ask[sid] == flag] for flag in (True, False)}
    for flag in (True, False):
        candidates = available[flag][:]
        rng.shuffle(candidates)
        pairs = []
        for slow_sid in slow_ids:
            if ask[slow_sid] != flag:
                continue
            for candidate in candidates:
                step_turn_gap = abs(turns(by_step[slow_sid]) - turns(by_step[candidate]))
                transcript_gap = abs(len(main_asr[slow_sid]) - len(main_asr[candidate])) / 100.0
                pairs.append((step_turn_gap * 10.0 + transcript_gap, rng.random(), slow_sid, candidate))
        pairs.sort()
        used_slow, used_candidates = set(), set()
        count = 0
        for _, _, slow_sid, candidate in pairs:
            if candidate in used_candidates:
                continue
            # Greedy nearest-neighbor controls; allowing multiple controls per slow sample.
            used_candidates.add(candidate)
            picked.append(candidate)
            count += 1
            if count == quotas[flag]:
                break
        if count < quotas[flag]:
            # If one stratum lacks enough fast examples, fill from the other stratum and record actual counts.
            pass
    if len(picked) < N_CONTROL:
        rest = [sid for sid in fast_ids if sid not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[:N_CONTROL-len(picked)])
    if len(picked) != N_CONTROL or len(set(picked)) != N_CONTROL:
        raise RuntimeError(f'Could not select {N_CONTROL} unique fast controls')
    return picked, ask

def prepare():
    ev, compare, prompt, runner, ids, items, main_asr, history, data, eval_lock = context()
    step_rows = read_jsonl(STEP120)
    by_step = {r['id']: r for r in step_rows}
    slow_ids = [sid for sid in ids if float(by_step[sid].get('latency_seconds') or 0) > 30.0]
    if len(slow_ids) != N_SLOW:
        raise RuntimeError(f'Expected all 42 step120 samples >30s, found {len(slow_ids)}')
    controls, ask = choose_controls(ids, items, main_asr, by_step, slow_ids)
    cohort_ids = [sid for sid in ids if sid in set(slow_ids + controls)]
    if len(cohort_ids) != N_SLOW + N_CONTROL:
        raise RuntimeError('Cohort does not contain 120 unique IDs')
    baseline_rows = read_jsonl(BASELINE)
    if [r['id'] for r in baseline_rows] != ids:
        raise RuntimeError('Best baseline predictions do not cover canonical val309')
    with urllib.request.urlopen(HOST + '/api/tags', timeout=30) as response:
        tags = {r['name']: r['digest'] for r in json.load(response)['models']}
    model = ev.MODEL_TAGS['step120']
    if tags.get(model) != eval_lock['checkpoints']['step120']['digest']:
        raise RuntimeError('Local step120 Ollama model digest differs from locked validation')
    lock = {
        'experiment': 'Qwen3.5-4B step120 runtime/thinking ablation',
        'cohort': {'n': 120, 'ids': cohort_ids, 'slow_n': len(slow_ids), 'slow_ids': slow_ids,
                   'fast_controls_n': len(controls), 'fast_control_ids': controls,
                   'all_val309_ids_sha256': eval_lock['cohort']['ids_sha256'],
                   'cohort_ids_sha256': hashlib.sha256('\n'.join(cohort_ids).encode()).hexdigest(),
                   'selection': 'All step120 samples >30 sec plus 78 <30 sec controls greedily matched within gold-ask stratum by action turns and PhoWhisper transcript length; seed=20261113',
                   'slow_gold_ask_n': sum(ask[s] for s in slow_ids),
                   'control_gold_ask_n': sum(ask[s] for s in controls)},
        'model': {'tag': model, 'digest': tags[model]},
        'prompt': eval_lock['prompt'],
        'common_settings': {'temperature': 0, 'protocol': 'JSON', 'num_ctx': 8192, 'num_batch': 512,
                            'seed': 17, 'max_action_turns': 5, 'main_and_history_asr': 'PhoWhisper Large v2 cached',
                            'toolenv': 'official BTC ToolEnv per sample'},
        'branches': {'step120_768_think': {'num_predict': 768, 'thinking': True, 'source': str(STEP120.relative_to(ROOT)).replace('\\','/')},
                     'cap384_think': {'num_predict': 384, 'thinking': True},
                     'no_think_768': {'num_predict': 768, 'thinking': False}},
        'input_sha256': {'step120_predictions': sha(STEP120), 'paired_details': sha(DETAILS),
                         'best_baseline_predictions': sha(BASELINE), 'btc_val309_lock': sha(ev.OUT / 'comparison_lock.json'),
                         'step120_evaluator_source': sha(EVAL_SCRIPT)},
        'method_note': 'This intentionally enriched diagnostic cohort is not an unbiased estimate of full val309/public score.'}
    lock_path = OUT / 'ablation_lock.json'
    if lock_path.exists() and load_json(lock_path) != lock:
        raise RuntimeError('Existing ablation lock differs; refusing to replace it')
    write_json(lock_path, lock)
    write_json(OUT / 'cohort_manifest.json', lock['cohort'])
    print(json.dumps({'cohort_n': 120, 'slow_n': len(slow_ids), 'controls_n': len(controls),
                      'slow_gold_ask_n': lock['cohort']['slow_gold_ask_n'],
                      'control_gold_ask_n': lock['cohort']['control_gold_ask_n'],
                      'model': model, 'digest': tags[model], 'prompt_sha256': eval_lock['prompt']['sha256']}, ensure_ascii=False, indent=2), flush=True)

def run():
    ev, compare, prompt, runner, ids, items, main_asr, history, data, eval_lock = context()
    lock_path = OUT / 'ablation_lock.json'
    if not lock_path.exists():
        raise FileNotFoundError('Run --prepare first')
    lock = load_json(lock_path)
    if sha(STEP120) != lock['input_sha256']['step120_predictions'] or sha(BASELINE) != lock['input_sha256']['best_baseline_predictions']:
        raise RuntimeError('Locked baseline or step120 predictions changed')
    cohort_ids = lock['cohort']['ids']
    if hashlib.sha256('\n'.join(cohort_ids).encode()).hexdigest() != lock['cohort']['cohort_ids_sha256']:
        raise RuntimeError('Cohort manifest hash mismatch')
    model = ev.MODEL_TAGS['step120']
    runner.MODEL = model
    runner.EXPECTED_DIGEST = lock['model']['digest']
    runner.inject_cot_original = runner.inject_cot
    runner.inject_cot = prompt.focused_inject
    runner.seed_everything(17)
    import lifemate.asr as asr_module
    compare.CachedPhoWhisper.lookup = history
    asr_module.PhoASRModel = compare.CachedPhoWhisper
    import lifemate.agent as agent_module
    base_agent = agent_module.Agent
    branches = [('cap384_think', 'CAP384', 384, True), ('no_think_768', 'NO_THINK', 768, False)]
    for round_no, start in enumerate(range(0, len(cohort_ids), 15), 1):
        block = cohort_ids[start:start+15]
        order = branches if round_no % 2 else list(reversed(branches))
        for key, branch, cap, thinking in order:
            branch_out = OUT / 'inference' / key
            branch_out.mkdir(parents=True, exist_ok=True)
            runner.OUT = branch_out
            runner.CAP = cap
            runner.MAX_STEPS = 5
            if thinking:
                agent_module.Agent = base_agent
            else:
                def make_no_think(*args, **kwargs):
                    kwargs['ollama_thinking'] = False
                    return base_agent(*args, **kwargs)
                agent_module.Agent = make_no_think
            body = {'model': model, 'messages': [{'role':'user','content':'Return {} as JSON.'}], 'stream':False,
                    'think': thinking, 'format':'json', 'options':{'temperature':0,'num_predict':8,'num_ctx':8192,'num_batch':512}}
            warm_started = time.perf_counter()
            warm = runner.api_json('/api/chat', body)
            write_json(branch_out / f'warmup-COT-round{round_no}.json', {'branch':key,'round':round_no,'excluded_from_measurements':True,'wall_seconds':time.perf_counter()-warm_started,'prompt_eval_count':warm.get('prompt_eval_count'),'eval_count':warm.get('eval_count'),'load_duration_ns':warm.get('load_duration'),'total_duration_ns':warm.get('total_duration')})
            todo = list(block)
            while todo:
                pred_path = branch_out / 'predictions-COT.jsonl'
                before = len(read_jsonl(pred_path))
                try:
                    runner.run_block('COT', round_no, todo, main_asr, items, cot=True)
                except Exception:
                    after = read_jsonl(pred_path)
                    if len(after) <= before or not after[-1].get('error'):
                        raise
                done_ids = {row['id'] for row in read_jsonl(pred_path)}
                todo = [sid for sid in block if sid not in done_ids]
            run_meta = load_json(branch_out / 'run-COT.json')
            run_meta.update({'branch': key, 'status':'complete' if len(read_jsonl(branch_out / 'predictions-COT.jsonl')) == start+len(block) else 'in_progress',
                             'thinking': thinking, 'num_predict': cap, 'model_digest': lock['model']['digest'],
                             'prompt_variant':'focused post-lookup disambiguation', 'cohort_ids':cohort_ids,
                             'executed_rounds': round_no})
            write_json(branch_out / 'run-COT.json', run_meta)
        print(f'paired block {round_no}/8 complete ({start+len(block)}/120)', flush=True)
    agent_module.Agent = base_agent
    analyze()

def pct(values):
    return 100 * statistics.mean(values) if values else None

def percentile(values, q):
    if not values: return None
    v = sorted(values)
    return v[min(len(v)-1, max(0, math.ceil(q*len(v))-1))]

def boot_ci(values, seed, draws=10000):
    rng = random.Random(seed)
    result = sorted(100*statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [result[int(.025*draws)], result[int(.975*draws)-1]] if result else [None, None]

def analyze():
    from score_btc import score_sample, call_key
    ev, compare, prompt, runner, ids, items, main_asr, history, data, eval_lock = context()
    lock = load_json(OUT / 'ablation_lock.json')
    cohort_ids = lock['cohort']['ids']
    base = {r['id']:r for r in read_jsonl(BASELINE)}
    step = {r['id']:r for r in read_jsonl(STEP120)}
    predictions = {'best_baseline':base, 'step120_768_think':step}
    for key in ('cap384_think','no_think_768'):
        path = OUT / 'inference' / key / 'predictions-COT.jsonl'
        rows = read_jsonl(path)
        if [r['id'] for r in rows] != cohort_ids:
            raise RuntimeError(f'{key} incomplete or IDs differ from locked cohort: {len(rows)}/{len(cohort_ids)}')
        predictions[key] = {r['id']:r for r in rows}
    slow = set(lock['cohort']['slow_ids'])
    scored = {name:{sid:score_sample(sid, items[sid], rows[sid], data) for sid in cohort_ids} for name,rows in predictions.items()}
    records=[]
    for sid in cohort_ids:
        gold_asks=[c for c in parse_label(items[sid]).get('tool_calls',[]) if c.get('name')=='sys.ask_missing_info']
        row={'id':sid,'cohort_group':'slow_gt30s' if sid in slow else 'matched_fast_control',
             'gold_ask':bool(gold_asks)}
        for name in predictions:
            pred=predictions[name][sid]; score=scored[name][sid]
            pred_asks=[c for c in (pred.get('tool_calls') or []) if c.get('name')=='sys.ask_missing_info']
            row[name]={'btc':score['official_sample_score'], 'btc_before_runtime_gate':score['btc_formula_score_before_runtime_gate_diagnostic'],
                       'latency_seconds':pred.get('latency_seconds'), 'over_30s':bool(score['runtime_over_30s']),
                       'runtime_error':bool(pred.get('error')), 'hash_match':bool(score['hash_match']),
                       'matched_calls':score['matched_calls'],'gold_calls':score['gold_calls'],
                       'ask':bool(pred_asks),'ask_exact':any(call_key(a)==call_key(g) for a in pred_asks for g in gold_asks)}
        records.append(row)
    groups={'all_120':set(cohort_ids),'slow_gt30s':slow,'matched_fast_control':set(cohort_ids)-slow}
    summaries={}
    for group_name, members in groups.items():
        group=[r for r in records if r['id'] in members]
        summary={}
        for name in predictions:
            lat=[r[name]['latency_seconds'] for r in group if r[name]['latency_seconds'] is not None]
            summary[name]={'btc_after_30s_gate':pct([r[name]['btc'] for r in group]),
                           'btc_before_gate':pct([r[name]['btc_before_runtime_gate'] for r in group]),
                           'n_over_30s':sum(r[name]['over_30s'] for r in group),
                           'runtime_errors':sum(r[name]['runtime_error'] for r in group),
                           'hash_matches':sum(r[name]['hash_match'] for r in group),
                           'mean_latency_seconds':statistics.mean(lat) if lat else None,
                           'median_latency_seconds':statistics.median(lat) if lat else None,
                           'p90_latency_seconds':percentile(lat,.90),
                           'gold_ask_exact':sum(r[name]['ask_exact'] for r in group),
                           'ask_count':sum(r[name]['ask'] for r in group)}
        for name in ('cap384_think','no_think_768'):
            ds=[r[name]['btc']-r['step120_768_think']['btc'] for r in group]
            us=[r[name]['btc_before_runtime_gate']-r['step120_768_think']['btc_before_runtime_gate'] for r in group]
            summary[f'paired_{name}_minus_step120']={'delta_after_gate_points':pct(ds),
                'delta_before_gate_points':pct(us),'bootstrap_95pct_CI_after_gate':boot_ci(ds,SEED+(384 if name=='cap384_think' else 768)),
                'wins':sum(d>0 for d in ds),'ties':sum(d==0 for d in ds),'losses':sum(d<0 for d in ds)}
        summaries[group_name]=summary
    write_jsonl(OUT/'paired_details.jsonl',records)
    summary={'experiment':lock['experiment'],'cohort':lock['cohort'],'branches':lock['branches'],
             'groups':summaries,'method':{'scoring':'BTC official ToolEnv per sample; official 30s gate and pre-gate diagnostic both reported',
                'paired_against':'fresh step120 inference with original 768/thinking=true settings',
                'baseline_reference':'best public baseline predictions reused for context',
                'selection_bias':'all >30s cases oversampled by design; scores do not project to val309/public population'},
             'runtime_excludes_ASR':'PhoWhisper main/history transcripts were cached/reused'}
    write_json(OUT/'summary.json',summary)
    labels={'best_baseline':'Best baseline (cached)','step120_768_think':'Step120 768 + thinking (cached original)',
            'cap384_think':'Step120 384 + thinking','no_think_768':'Step120 768 + thinking off'}
    lines=['# Qwen3.5-4B step120 runtime/thinking ablation','',
           'Ablation cohort is intentionally enriched with the 42 step120 samples that exceeded 30 seconds and 78 matched fast controls. This is diagnostic, not an unbiased val309/public score estimate. All branches use the same step120 model, focused prompt, cached PhoWhisper main/history transcripts, ToolEnv, temperature 0, context 8192, batch 512 and five-turn cap.','',
           '| Cohort / branch | BTC after 30s gate | Before gate | >30s | Hash matches | Mean sec | Median sec | P90 sec | Exact asks |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for grp in ('all_120','slow_gt30s','matched_fast_control'):
        for name in predictions:
            s=summaries[grp][name]
            lines.append(f"| {grp} / {labels[name]} | {s['btc_after_30s_gate']:.2f} | {s['btc_before_gate']:.2f} | {s['n_over_30s']} | {s['hash_matches']}/{len(groups[grp])} | {s['mean_latency_seconds']:.2f} | {s['median_latency_seconds']:.2f} | {s['p90_latency_seconds']:.2f} | {s['gold_ask_exact']} |")
    lines += ['', '## Paired deltas vs original step120 settings','']
    for name in ('cap384_think','no_think_768'):
        s=summaries['all_120'][f'paired_{name}_minus_step120']; ci=s['bootstrap_95pct_CI_after_gate']
        lines.append(f"- {labels[name]}: {s['delta_after_gate_points']:+.2f} BTC points after gate (95% paired bootstrap CI {ci[0]:+.2f} to {ci[1]:+.2f}); before gate {s['delta_before_gate_points']:+.2f}; wins/ties/losses {s['wins']}/{s['ties']}/{s['losses']}.")
    lines += ['', 'Interpret the full-cohort score only as a stress-test result because slow samples were deliberately oversampled. The primary diagnostic is whether each setting lowers timeouts while preserving pre-gate tool/state accuracy.', '']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true'); g.add_argument('--run',action='store_true'); g.add_argument('--analyze',action='store_true')
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.run: run()
    else: analyze()
if __name__=='__main__': main()




