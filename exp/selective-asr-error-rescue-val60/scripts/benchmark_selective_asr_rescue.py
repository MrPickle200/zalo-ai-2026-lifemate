from __future__ import annotations
import hashlib, importlib.util, json, random, re, statistics, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"exp/btc-scoring/scripts"), str(ROOT/"exp/quan-continuation-cot-v1/scripts")]
EXP = ROOT/"exp/qwen35-sft-pho-synthetic-v1"
NEW = ROOT/"exp/selective-asr-error-rescue-val60"
OUT = NEW/"output"
BASE = EXP/"output/two-policy-val60"
TAG = "lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest"
PHO = ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output/pho/predictions.jsonl"
QUAN = ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output/quan/predictions.jsonl"
SEED, DRAWS = 20261117, 10000
ACTIONS = set("tắt bật mở đóng hủy xóa dời đổi gọi nhắn nhắc gửi chuyển giữ bỏ khóa đặt trả mua thêm lấy thanh toán hẹn".split())
POLARITY = set("không chẳng chưa đừng nếu thôi trừ ngoại trừ".split())
NUMBERS = {"không":"0","một":"1","hai":"2","ba":"3","bốn":"4","tư":"4","năm":"5","sáu":"6","bảy":"7","tám":"8","chín":"9","mười":"10","mốt":"1","lăm":"5","trăm":"100","nghìn":"1000","ngàn":"1000","triệu":"1000000","tỷ":"1000000000","rưỡi":"0.5","lẻ":"0"}
TIMES = set("sáng trưa chiều tối đêm hôm mai ngày tháng giờ phút tuần thứ".split())
ORTHO = {"huỷ":"hủy","khoá":"khóa","xoá":"xóa"}

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def read_json(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def read_jsonl(path): return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if path.exists() else []
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def risk(text):
    text=unicodedata.normalize("NFC",text).lower()
    for old,new in ORTHO.items(): text=text.replace(old,new)
    tokens=re.findall(r"\d+(?:[.,]\d+)?|[^\W_]+",text,flags=re.UNICODE)
    out=set()
    for t in tokens:
        if t in ACTIONS: out.add("action:"+t)
        if t in POLARITY: out.add("polarity:"+t)
        if t in TIMES: out.add("time:"+t)
        if t.isdigit(): out.add("number:"+t)
        elif t in NUMBERS: out.add("number:"+NUMBERS[t])
    return out

def build():
    base=load_module("two_policy_for_asr_rescue",EXP/"scripts/run_two_policy_val60.py")
    ev,compare,prompt,runner,ids,cohort,items,main,history,data,model_record,final4,manifest=base.build()
    pho_rows,quan_rows=read_jsonl(PHO),read_jsonl(QUAN)
    pho={x["id"]:x["prediction"] for x in pho_rows}; quan={x["id"]:x["prediction"] for x in quan_rows}
    if len(pho_rows)!=309 or len(quan_rows)!=309 or set(cohort)-pho.keys() or set(cohort)-quan.keys():
        raise RuntimeError("Cached PhoWhisper/QUAN transcripts do not cover val309 and pilot60.")
    if any(pho[sid]!=main[sid] for sid in cohort): raise RuntimeError("PhoWhisper text differs from baseline.")
    routing=[]; treatment=dict(main); selected=[]
    for sid in cohort:
        delta=sorted(risk(pho[sid]) ^ risk(quan[sid]))
        flag=bool(delta)
        routing.append({"id":sid,"selected":flag,"critical_disagreements":delta,
                        "primary_transcript":pho[sid],"secondary_transcript":quan[sid]})
        if flag:
            selected.append(sid)
            treatment[sid]=("[Primary ASR transcript — PhoWhisper Large v2]\n"+pho[sid]
                +"\n\n[Secondary ASR transcript — QUAN]\n"+quan[sid]
                +"\n\nBoth transcripts describe the same audio and may contain recognition errors. "
                "Use the secondary only to resolve a clear disagreement in an action, negation, number, or time. "
                "Prefer the primary when the secondary is garbled. Do not invent information.")
    if not selected: raise RuntimeError("No samples met the label-free rescue trigger.")
    return base,ev,prompt,runner,cohort,items,main,history,data,model_record,routing,treatment,selected

def boot_ci(diffs):
    rng=random.Random(SEED); n=len(diffs)
    vals=sorted(100*statistics.mean(rng.choices(diffs,k=n)) for _ in range(DRAWS))
    return [vals[int(.025*DRAWS)],vals[int(.975*DRAWS)-1]]

def run():
    if sys.prefix!=sys.base_prefix: raise RuntimeError("Use system Python so local Ollama is reachable.")
    base,ev,prompt,runner,cohort,items,main,history,data,record,routing,treatment,selected=build()
    OUT.mkdir(parents=True,exist_ok=True)
    baseline_path=BASE/"inference/fixed_ask_stop/predictions-COT.jsonl"
    baseline_rows=read_jsonl(baseline_path)
    if [x["id"] for x in baseline_rows]!=cohort: raise RuntimeError("Archived Q4 control differs from pilot60.")
    baseline={x["id"]:x for x in baseline_rows}; selected_set=set(selected)
    paths=[Path(__file__).resolve(),EXP/"scripts/run_two_policy_val60.py",baseline_path,BASE/"comparison_lock.json",
        PHO,QUAN,ROOT/"src/lifemate/agent.py",ROOT/"src/lifemate/pipeline.py",
        ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        ROOT/"exp/qwen35-thinking-budget-val60/output/cohort_manifest.json",
        EXP/"output/adapter-final-4epoch-benchmark/val309/comparison_lock.json"]
    lock={"experiment":"selective-asr-error-rescue-val60",
        "cohort":{"n":60,"ids":cohort,"source":"existing enriched fixed pilot60"},
        "model":{"tag":TAG,"digest":record["digest"]},
        "asr":{"primary":"cached PhoWhisper Large v2","secondary":"cached QUAN epoch 06",
            "route":"Only when normalized action, polarity, number, or time tokens differ"},
        "settings":{"thinking":True,"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,
            "num_batch":512,"seed":17,"max_steps":5,"focused_prompt":True,"stop_after_successful_ask":True},
        "baseline_lock_sha256":sha(BASE/"comparison_lock.json"),
        "inputs_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in paths},
        "selected_ids":selected,"selection_is_label_free":True,
        "caveat":"Enriched diagnostic pilot60; secondary transcript is cached, so incremental ASR latency is excluded."}
    lp=OUT/"comparison_lock.json"
    if lp.exists() and read_json(lp)!=lock and (OUT/"inference/selective_rescue/predictions-COT.jsonl").exists(): raise RuntimeError("Existing run outputs differ from current inputs.")
    write_json(lp,lock); write_jsonl(OUT/"routing.jsonl",routing)
    branch=OUT/"inference/selective_rescue"; branch.mkdir(parents=True,exist_ok=True)
    predpath=branch/"predictions-COT.jsonl"; done={x["id"] for x in read_jsonl(predpath)}
    if done-selected_set: raise RuntimeError("Rescue output has unselected IDs.")
    runner.MODEL=TAG; runner.EXPECTED_DIGEST=record["digest"]
    version,digest=runner.model_fingerprint()
    if digest!=record["digest"]: raise RuntimeError("Q4 model digest mismatch.")
    import lifemate.pipeline as pipeline
    source=Path(pipeline.__file__).read_text(encoding="utf-8")
    try:
        base.patch_pipeline(pipeline,source,True)
        runner.CAP,runner.CONTEXT,runner.BATCH=768,8192,512
        runner.SEED,runner.MAX_STEPS=17,5

        runner.OUT=branch
        for rn,start in enumerate(range(0,60,15),1):
            todo=[sid for sid in cohort[start:start+15] if sid in selected_set and sid not in done]
            if not todo: continue
            runner.warmup("COT",rn); runner.run_block("COT",rn,todo,treatment,items,cot=True)
            done.update(todo); print(f"rescue block {rn}/4; {len(done)}/{len(selected)} selected",flush=True)
    finally: base.patch_pipeline(pipeline,source,False)
    rescue_rows=read_jsonl(predpath); rescue={x["id"]:x for x in rescue_rows}
    if set(rescue)!=selected_set: raise RuntimeError("Selective inference incomplete.")
    from score_btc import score_sample
    details=[]
    for r in routing:
        sid=r["id"]; pred=rescue[sid] if r["selected"] else baseline[sid]
        b=score_sample(sid,items[sid],baseline[sid],data); t=score_sample(sid,items[sid],pred,data)
        details.append({"id":sid,"selected":r["selected"],"critical_disagreements":r["critical_disagreements"],
            "baseline":{"raw":b["btc_formula_score_before_runtime_gate_diagnostic"],"gated":b["official_sample_score"],"latency":baseline[sid].get("latency_seconds")},
            "treatment":{"raw":t["btc_formula_score_before_runtime_gate_diagnostic"],"gated":t["official_sample_score"],"latency":pred.get("latency_seconds"),"rerun":r["selected"]}})
    raw=[x["treatment"]["raw"]-x["baseline"]["raw"] for x in details]
    gated=[x["treatment"]["gated"]-x["baseline"]["gated"] for x in details]
    summary={"experiment":lock["experiment"],"cohort":lock["cohort"],
        "route":{"selected":len(selected),"not_selected":60-len(selected),"selected_ids":selected},
        "baseline":{"raw":100*statistics.mean(x["baseline"]["raw"] for x in details),
            "gated":100*statistics.mean(x["baseline"]["gated"] for x in details),
            "mean_latency_seconds":statistics.mean(x["baseline"]["latency"] for x in details)},
        "treatment":{"raw":100*statistics.mean(x["treatment"]["raw"] for x in details),
            "gated":100*statistics.mean(x["treatment"]["gated"] for x in details),
            "mean_latency_seconds":statistics.mean(x["treatment"]["latency"] for x in details),
            "runtime_errors":sum(bool(rescue.get(x["id"],{}).get("error")) for x in details),
            "over_30s":sum(bool(x["treatment"]["latency"] and x["treatment"]["latency"]>30) for x in details)},
        "paired":{"raw_delta":100*statistics.mean(raw),"raw_ci95":boot_ci(raw),
            "gated_delta":100*statistics.mean(gated),"gated_ci95":boot_ci(gated),
            "wins_ties_losses_raw":[sum(x>0 for x in raw),sum(x==0 for x in raw),sum(x<0 for x in raw)]},
        "runtime_caveat":"Qwen+ToolEnv latency only; QUAN transcript was cached.",
        "caveat":lock["caveat"]}
    write_json(OUT/"paired_details.json",details); write_json(OUT/"summary.json",summary)
    q,b=summary["treatment"],summary["baseline"]; p=summary["paired"]
    report=["# Selective ASR error rescue — enriched pilot60","",
        "Same Q4 final4 adapter, focused prompt, fixed thinking, cached PhoWhisper main/history, JSON/temp0/cap768/context8192/batch512/seed17/five turns, and the successful sys.ask_missing_info stop guard. QUAN is exposed only when normalized action, polarity, number, or time tokens disagree. Unflagged samples reuse the archived Q4 ask-stop control.",
        "",f"Label-free route selected {len(selected)}/60. The QUAN transcript is cached, so measured latency excludes second-ASR inference.","",
        "| Branch | BTC before gate | BTC after local 30s gate | Mean Qwen+ToolEnv seconds |","|---|---:|---:|---:|",
        f"| Q4 ask-stop control | {b['raw']:.2f} | {b['gated']:.2f} | {b['mean_latency_seconds']:.2f} |",
        f"| Selective rescue | {q['raw']:.2f} | {q['gated']:.2f} | {q['mean_latency_seconds']:.2f} |","",
        f"Paired rescue−baseline: raw {p['raw_delta']:+.2f} (95% CI {p['raw_ci95']}); after gate {p['gated_delta']:+.2f} (95% CI {p['gated_ci95']}); raw wins/ties/losses {p['wins_ties_losses_raw']}.","",
        "This enriched 60-task screen is diagnostic, not a full-val or Public LB estimate. The current public candidate and policy were not changed.",""]
    (OUT/"summary.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": run()
