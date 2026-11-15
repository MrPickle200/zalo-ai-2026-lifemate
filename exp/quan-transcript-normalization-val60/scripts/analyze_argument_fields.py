from __future__ import annotations
import json,re
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; EXP=ROOT/'exp/quan-transcript-normalization-val60'; OUT=EXP/'output/inference'
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def flatten(obj,prefix=''):
 out={}
 if isinstance(obj,dict):
  for k,v in obj.items(): out.update(flatten(v,f'{prefix}.{k}' if prefix else str(k)))
 elif isinstance(obj,list):
  for i,v in enumerate(obj): out.update(flatten(v,f'{prefix}[{i}]'))
 else: out[prefix]=obj
 return out
def field_relevant(path,value):
 name=path.lower()
 if re.search(r'(^|[._\[])('r'number|count|amount|price|temperature|temp|duration|minute|hour|date|time|start|end|day|month|year|percent|percentage|degree|currency|value)([._\[]|$)',name): return True
 return isinstance(value,(int,float)) and not isinstance(value,bool)
def numeric_value(v):
 if isinstance(v,bool) or v is None: return None
 if isinstance(v,(int,float,Decimal)):
  try:return Decimal(str(v))
  except InvalidOperation:return None
 if isinstance(v,str):
  q=v.strip().replace(',','.')
  if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',q):
   try:return Decimal(q)
   except InvalidOperation:return None
 return None
def main():
 data_root=ROOT/'train+public_test/train+public_test'; train=rows(data_root/'train/train.jsonl'); items={r['id']:r for r in train}
 ids=json.loads((OUT/'cohort_manifest.json').read_text(encoding='utf-8'))['ids']
 pred={b:{r['id']:r for r in rows(OUT/f'predictions-{b}.jsonl')} for b in ('A','B')}
 scores={b:{r['id']:r for r in rows(OUT/f'btc-per-sample-{b}.jsonl')} for b in ('A','B')}
 transcripts={r['id']:r['transcript'] for r in rows(EXP/'output/transcripts-raw.jsonl')}
 transitions=Counter(); repr_only=0; actual_value_changes=0; examples=[]; outcome_changes=[]
 for sid in ids:
  gold=items[sid].get('label',{}).get('tool_calls',[])
  ac=pred['A'][sid].get('tool_calls',[]); bc=pred['B'][sid].get('tool_calls',[])
  if ac!=bc or pred['A'][sid].get('state_hash')!=pred['B'][sid].get('state_hash'):
   outcome_changes.append({'id':sid,'btc_A':scores['A'][sid]['btc_formula_score_before_runtime_gate_diagnostic'],'btc_B':scores['B'][sid]['btc_formula_score_before_runtime_gate_diagnostic'],'hash_A':scores['A'][sid]['hash_match'],'hash_B':scores['B'][sid]['hash_match'],'transcript_raw':transcripts[sid],'transcript_B':transcripts[sid],'A_calls':ac,'B_calls':bc})
  # Align occurrences by exact tool name; only after scoring, for argument diagnosis.
  ga=defaultdict(list); aa=defaultdict(list); ba=defaultdict(list)
  for c in gold: ga[c.get('name')].append(c)
  for c in ac: aa[c.get('name')].append(c)
  for c in bc: ba[c.get('name')].append(c)
  for name,gcalls in ga.items():
   for index,gc in enumerate(gcalls):
    a=aa[name][index].get('arguments',{}) if index<len(aa[name]) else {}
    b=ba[name][index].get('arguments',{}) if index<len(ba[name]) else {}
    gf=flatten(gc.get('arguments',{})); af=flatten(a); bf=flatten(b)
    for field in sorted(set(gf)|set(af)|set(bf)):
     gv=gf.get(field); av=af.get(field); bv=bf.get(field)
     if not field_relevant(field,gv) and not field_relevant(field,av) and not field_relevant(field,bv): continue
     if av==bv: continue
     aok=av==gv; bok=bv==gv
     label=('correct_to_wrong' if aok and not bok else 'wrong_to_correct' if not aok and bok else 'wrong_value_changed' if not aok and not bok else 'other')
     transitions[label]+=1
     na,nb,ng=numeric_value(av),numeric_value(bv),numeric_value(gv)
     if na is not None and nb is not None and na==nb and (type(av)!=type(bv) or av!=bv): repr_only+=1
     elif av!=bv: actual_value_changes+=1
     if len(examples)<30:
      examples.append({'id':sid,'tool':name,'occurrence':index,'field':field,'gold':gv,'A':av,'B':bv,'A_exact':aok,'B_exact':bok,'transition':label,
       'same_numeric_value_A_B':bool(na is not None and nb is not None and na==nb),'btc_A':scores['A'][sid]['btc_formula_score_before_runtime_gate_diagnostic'],'btc_B':scores['B'][sid]['btc_formula_score_before_runtime_gate_diagnostic']})
 out={'classification':'diagnostic after frozen treatment and official BTC scoring; no tuning/transform change made','paired_argument_field_transitions':dict(transitions),'same_numeric_value_notation_only_changes':repr_only,'actual_value_changes':actual_value_changes,
  'numeric_time_field_change_examples':examples,'samples_with_outcome_or_hash_change':len(outcome_changes),'outcome_change_examples_max15':outcome_changes[:15],
  'limitation':'The selected treatment changed no transcript in the cohort, so any call/argument changes are model run variation, not a transcript treatment effect.'}
 (OUT/'argument-field-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 # Attach diagnostics to the main analysis and rewrite concise report.
 ap=OUT/'analysis.json'; analysis=json.loads(ap.read_text(encoding='utf-8')); analysis['argument_field_audit']=out; ap.write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 summary=f'''# Downstream benchmark summary\n\n## Scope and decision\n\nThis is the separate orthography-only fallback treatment. The numeric parser was deferred, and the spelling rules changed zero transcripts in the 60-ID cohort, so A/B are identical-input repeat runs. The benchmark cannot establish an effect of transcript normalization.\n\n## Official BTC and paired results\n\n- A: {analysis['branches']['A']['btc_formula_score_before_runtime_gate_0_100']:.2f}/100; B: {analysis['branches']['B']['btc_formula_score_before_runtime_gate_0_100']:.2f}/100. Both are official BTC formula scores with hash gate and M>0; exact tool-name/argument matching with duplicate multiplicity.\n- Paired B−A: {analysis['paired']['B_minus_A']['paired_formula_delta_points_right_minus_left']['mean']:.2f} points, 95% bootstrap CI [{analysis['paired']['B_minus_A']['paired_formula_delta_points_right_minus_left']['ci95_low']:.2f}, {analysis['paired']['B_minus_A']['paired_formula_delta_points_right_minus_left']['ci95_high']:.2f}], 20,000 resamples, seed 20261109. Outcomes: {analysis['paired']['B_minus_A']['paired_outcomes_formula_score']}.\n- A-repeat on 20 anchors: {analysis['branches']['A-repeat']['btc_formula_score_before_runtime_gate_0_100']:.2f}/100. A-repeat−A on anchors: {analysis['paired']['A_repeat_minus_A_on_anchors']['paired_formula_delta_points_right_minus_left']['mean']:.2f} points, CI [{analysis['paired']['A_repeat_minus_A_on_anchors']['paired_formula_delta_points_right_minus_left']['ci95_low']:.2f}, {analysis['paired']['A_repeat_minus_A_on_anchors']['paired_formula_delta_points_right_minus_left']['ci95_high']:.2f}]. A-repeat−B on anchors: {analysis['paired']['A_repeat_minus_B_on_anchors']['paired_formula_delta_points_right_minus_left']['mean']:.2f} points, CI [{analysis['paired']['A_repeat_minus_B_on_anchors']['paired_formula_delta_points_right_minus_left']['ci95_low']:.2f}, {analysis['paired']['A_repeat_minus_B_on_anchors']['paired_formula_delta_points_right_minus_left']['ci95_high']:.2f}].\n- Local 30-second-gated scores are diagnostics only: A {analysis['branches']['A']['local_30s_gated_score_diagnostic_0_100']:.2f}, B {analysis['branches']['B']['local_30s_gated_score_diagnostic_0_100']:.2f}, A-repeat {analysis['branches']['A-repeat']['local_30s_gated_score_diagnostic_0_100']:.2f}. RTX 4050 timings are not A30 benchmark timings.\n\n## Runtime and call diagnostics\n\nAll gold ToolEnv replays passed; prediction trace runtime errors: 0. Hash matches A/B/A-repeat: {analysis['branches']['A']['hash_matches']}/{analysis['branches']['B']['hash_matches']}/{analysis['branches']['A-repeat']['hash_matches']}. M/U/G: A {analysis['branches']['A']['M_matched_calls']}/{analysis['branches']['A']['U_unmatched_submitted_calls']}/{analysis['branches']['A']['G_gold_calls']}; B {analysis['branches']['B']['M_matched_calls']}/{analysis['branches']['B']['U_unmatched_submitted_calls']}/{analysis['branches']['B']['G_gold_calls']}; A-repeat {analysis['branches']['A-repeat']['M_matched_calls']}/{analysis['branches']['A-repeat']['U_unmatched_submitted_calls']}/{analysis['branches']['A-repeat']['G_gold_calls']}. Repair attempts/failures: A {analysis['branches']['A']['repair_turns_attempted']}/{analysis['branches']['A']['repair_turns_failed']}, B {analysis['branches']['B']['repair_turns_attempted']}/{analysis['branches']['B']['repair_turns_failed']}, repeat {analysis['branches']['A-repeat']['repair_turns_attempted']}/{analysis['branches']['A-repeat']['repair_turns_failed']}. Requests/prompt-eval/generated tokens: A {analysis['branches']['A']['ollama_requests']}/{analysis['branches']['A']['total_prompt_eval_tokens']}/{analysis['branches']['A']['total_generated_tokens_including_thinking']}; B {analysis['branches']['B']['ollama_requests']}/{analysis['branches']['B']['total_prompt_eval_tokens']}/{analysis['branches']['B']['total_generated_tokens_including_thinking']}; A-repeat {analysis['branches']['A-repeat']['ollama_requests']}/{analysis['branches']['A-repeat']['total_prompt_eval_tokens']}/{analysis['branches']['A-repeat']['total_generated_tokens_including_thinking']}.\n\nFirst-request payload hashes matched across A/B on all {analysis['paired']['B_minus_A']['first_request_payload_hash_compared']} samples; identical first payload but different prediction on {analysis['paired']['B_minus_A']['same_first_request_payload_but_different_prediction_count']} samples. This confirms measurable run variation.\n\nArgument-field details are in `argument-field-audit.json`; no normalization-driven numeric/time change is possible because A and B transcript strings are equal for this cohort.\n\n## Decision\n\nNo evidence about the requested numeric-plus-spelling treatment. The apparent +{analysis['paired']['B_minus_A']['paired_formula_delta_points_right_minus_left']['mean']:.2f} A/B point delta is a run-variance observation with a CI including zero; A-repeat shows similar variance. Cohort60 is enriched screening, not a representative estimate for val309 or test. No submission or Public LB action was performed.\n'''
 (OUT/'summary.md').write_text(summary,encoding='utf-8')
 print(json.dumps({'field_transitions':out['paired_argument_field_transitions'],'notation_only':repr_only,'numeric_value_changes':actual_value_changes,'outcome_change_samples':len(outcome_changes)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

