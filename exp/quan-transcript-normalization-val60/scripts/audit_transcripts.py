from __future__ import annotations
import hashlib, json, re, unicodedata, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'exp/quan-transcript-normalization-val60'
OUT = EXP / 'output'
RAW_SOURCE = ROOT / 'exp/quan-transcribe-lifemate-domain-v1-val309/output/predictions.jsonl'
REFERENCE_SOURCE = ROOT / 'exp/current-config-error-analysis-val309/output/transcript_cache_val309.jsonl'
COHORT_A = ROOT / 'exp/inference-latency-optimization-val60/output/cohort_manifest.json'
COHORT_B = ROOT / 'exp/qwen35-sft-benchmark-v1/output/inference/cohort_manifest.json'
RULES = {'unicode': 'NFC', 'orthographic_map': {'hoà':'hòa','huỷ':'hủy','khoá':'khóa','hoá':'hóa'}}
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
NUM_WORDS = {'không','một','hai','ba','bốn','tư','năm','sáu','bảy','tám','chín','mười','mươi','trăm','nghìn','ngàn','triệu','tỷ','rưỡi','lẻ','linh','mốt','lăm'}
NEGATIONS = {'không','chẳng','chưa','đừng','chớ','khỏi'}
TIME_MARKERS = {'ngày','tháng','năm','giờ','phút','sáng','trưa','chiều','tối','đêm','hôm','mai','mốt','tuần','thứ','lịch','ngàymai','ngàymốt'}

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path: Path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def write_json(path: Path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_jsonl(path: Path, values): path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')
def norm(text): return ' '.join(PUNCT.sub(' ',str(text).lower()).split())
def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def number_tokens(text): return [t for t in norm(text).split() if t in NUM_WORDS or re.fullmatch(r'\d+(?:[.,]\d+)?',t)]
def critical_slots(ref,hyp):
    rt,ht=norm(ref).split(),norm(hyp).split(); rn,hn=number_tokens(ref),number_tokens(hyp)
    ref_neg=bool(set(rt)&NEGATIONS); hyp_neg=bool(set(ht)&NEGATIONS)
    ref_time=[t for t in rt if t in TIME_MARKERS or re.fullmatch(r'\d{1,2}:\d{2}',t)]
    hyp_time=[t for t in ht if t in TIME_MARKERS or re.fullmatch(r'\d{1,2}:\d{2}',t)]
    return {'numeric_tokens_reference':rn,'numeric_tokens_raw':hn,'numeric_tokens_normalized':number_tokens(hyp),
      'numeric_slot_present':bool(rn),'numeric_slot_exact_raw':rn==hn if rn else None,'numeric_slot_exact_normalized':rn==number_tokens(hyp) if rn else None,
      'negation_reference':ref_neg,'negation_raw':hyp_neg,'negation_normalized':bool(set(norm(hyp).split())&NEGATIONS),
      'negation_polarity_exact_raw':ref_neg==hyp_neg,'negation_polarity_exact_normalized':ref_neg==bool(set(norm(hyp).split())&NEGATIONS),
      'time_markers_reference':ref_time,'time_markers_raw':hyp_time,'time_markers_normalized':[t for t in norm(hyp).split() if t in TIME_MARKERS or re.fullmatch(r'\d{1,2}:\d{2}',t)],
      'time_slot_present':bool(ref_time),'time_slot_exact_raw':ref_time==hyp_time if ref_time else None,
      'time_slot_exact_normalized':ref_time==[t for t in norm(hyp).split() if t in TIME_MARKERS or re.fullmatch(r'\d{1,2}:\d{2}',t)] if ref_time else None}
def transform(s):
    s=unicodedata.normalize('NFC',s)
    for a,b in RULES['orthographic_map'].items(): s=re.sub(r'(?<!\w)'+re.escape(a)+r'(?!\w)',b,s)
    return s

def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True,exist_ok=True)
    cohort_a=json.loads(COHORT_A.read_text(encoding='utf-8')); cohort_b=json.loads(COHORT_B.read_text(encoding='utf-8'))
    ids=cohort_a['ids']; ids2=cohort_b['ids']; digest=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if ids != ids2 or len(ids)!=60 or digest!='87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9': raise SystemExit('Cohort lock mismatch')
    source=rows(RAW_SOURCE)
    if len(source)!=309 or len({r['id'] for r in source})!=309: raise SystemExit('Raw QUAN cache must contain exactly 309 unique IDs')
    raw=[{'id':r['id'],'transcript':str(r.get('prediction') or r.get('transcript') or '')} for r in source]
    if any(not r['transcript'] for r in raw): raise SystemExit('Empty raw transcript')
    write_jsonl(OUT/'transcripts-raw.jsonl',raw)
    normalized=[{'id':r['id'],'transcript':transform(r['transcript'])} for r in raw]
    if any(transform(transform(r['transcript']))!=transform(r['transcript']) for r in raw): raise SystemExit('Transform is not idempotent')
    write_jsonl(OUT/'transcripts-normalized.jsonl',normalized)
    # References are first read only after transform/rule and raw/normalized cache have been fixed.
    refs={r['id']:r.get('reference','') for r in rows(REFERENCE_SOURCE)}
    if set(refs)!=set(r['id'] for r in raw): raise SystemExit('Reference cache IDs do not match QUAN 309 IDs')
    by_norm={r['id']:r['transcript'] for r in normalized}
    audit=[]; changes=Counter(); examples=[]; rejected=[]
    word_err=char_err=word_ref=char_ref=exact=0
    norm_word_err=norm_char_err=norm_exact=0
    numeric=neg=time=0; numeric_n=neg_n=time_n=0
    for r in raw:
        sid,rt=r['id'],r['transcript']; nt=by_norm[sid]; ref=refs[sid]
        a,b=norm(ref).split(),norm(rt).split(); c=norm(nt).split()
        word_err+=edit_distance(a,b); word_ref+=len(a); char_err+=edit_distance(list(''.join(a)),list(''.join(b))); char_ref+=len(''.join(a)); exact+=a==b
        norm_word_err+=edit_distance(a,c); norm_char_err+=edit_distance(list(''.join(a)),list(''.join(c))); norm_exact+=a==c
        slots=critical_slots(ref,rt)
        if slots['numeric_slot_present']:
            numeric_n+=1; numeric+=bool(slots['numeric_slot_exact_normalized'])
        neg_n+=1; neg+=bool(slots['negation_polarity_exact_normalized'])
        if slots['time_slot_present']:
            time_n+=1; time+=bool(slots['time_slot_exact_normalized'])
        edits=[]
        for before,after in RULES['orthographic_map'].items():
            n=len(re.findall(r'(?<!\w)'+re.escape(before)+r'(?!\w)',rt))
            if n: changes[before+'->'+after]+=n; edits.extend([{'rule':before+'->'+after,'count':n}])
        changed=rt!=nt
        # Any character-level difference outside the predeclared replacements is a scope breach.
        residue=rt
        for before,after in RULES['orthographic_map'].items(): residue=re.sub(r'(?<!\w)'+re.escape(before)+r'(?!\w)',after,residue)
        residue=unicodedata.normalize('NFC',residue)
        scope_ok=(residue==nt)
        if not scope_ok: rejected.append(sid)
        audit.append({'id':sid,'raw':rt,'normalized':nt,'changed':changed,'applied_rules':edits,'scope_check_passed':scope_ok,'slots':slots})
    if rejected: raise SystemExit('Unexpected edits outside frozen rule table: '+','.join(rejected))
    write_jsonl(OUT/'audit-per-sample-309.jsonl',audit)
    changed=[x for x in audit if x['changed']]
    examples=changed[:8]+[x for x in audit if not x['changed'] and (number_tokens(x['raw']) or re.search(r'\b[A-Z]{1,5}\d+[A-Z0-9]*\b',x['raw']))][:7]
    summary={'samples':len(raw),'changed_transcripts':len(changed),'unchanged_transcripts':len(raw)-len(changed),'span_counts_by_rule':dict(changes),
      'number_parser':'not implemented; numeric treatment deferred per TASK.md safe-parser fallback; no number spans changed and no parser rejections are claimed',
      'out_of_scope_semantic_or_structural_changes':len(rejected),'scope_violations':rejected,
      'raw':{'corpus_WER':word_err/max(1,word_ref),'corpus_CER':char_err/max(1,char_ref),'normalized_exact_match_rate':exact/len(raw)},
      'normalized':{'corpus_WER':norm_word_err/max(1,word_ref),'corpus_CER':norm_char_err/max(1,char_ref),'normalized_exact_match_rate':norm_exact/len(raw)},
      'critical_slots':{'numeric_token_sequence_exact':{'raw_correct':numeric,'tested':numeric_n,'normalized_correct':numeric,'note':'the same frozen implementation used for current-config-error-analysis-val309; orthography-only transform cannot change these tokens'},
        'negation_polarity_exact':{'correct':neg,'tested':neg_n},'time_marker_sequence_exact':{'correct':time,'tested':time_n}},
      'examples':examples,'interpretation':'Exploratory validation audit. The orthography-only fallback is a separate treatment; this is not the full numeric+spelling B from TASK.md.'}
    write_json(OUT/'audit-summary.json',summary)
    write_json(OUT/'normalizer_spec.json',{'version':'orthography-only-v1','status':'numeric parser deferred; inference prohibited until a separately approved safe parser is available','unicode_normalization':'NFC','mapping':RULES['orthographic_map'],'number_rules':[],'input_scope':'Only current QUAN transcript field; no reference, gold action, ToolEnv state, or evaluation label is read by transform','idempotent':True})
    script_hash=sha(Path(__file__))
    write_json(OUT/'input_lock.json',{'status':'offline_preflight_locked_inference_blocked','source_hashes':{
      'quan_raw_predictions':sha(RAW_SOURCE),'quan_prediction_summary':sha(ROOT/'exp/quan-transcribe-lifemate-domain-v1-val309/output/summary.json'),
      'cohort_manifest_latency':sha(COHORT_A),'cohort_manifest_sft':sha(COHORT_B),
      'benchmark_config_manifest':sha(ROOT/'exp/qwen35-sft-benchmark-v1/output/inference/config_manifest.json'),
      'benchmark_lock':sha(ROOT/'exp/qwen35-sft-benchmark-v1/output/inference/benchmark_lock.json'),
      'prompt_source':sha(ROOT/'src/lifemate/agent.py'),'normalizer_script_sha256':script_hash,
      'rule_table_sha256':hashlib.sha256(json.dumps(RULES['orthographic_map'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()},
      'raw_transcripts_sha256':sha(OUT/'transcripts-raw.jsonl'),'normalized_transcripts_sha256':sha(OUT/'transcripts-normalized.jsonl'),
      'cohort_ids_sha256_newline_joined':digest,'cohort_samples':60,'normalizer_version':'orthography-only-v1','inference_started':False})
    print(json.dumps({'cohort_sha256':digest,'raw_sha256':sha(OUT/'transcripts-raw.jsonl'),'normalized_sha256':sha(OUT/'transcripts-normalized.jsonl'),'summary':summary},ensure_ascii=False,indent=2))
if __name__=='__main__': main()



