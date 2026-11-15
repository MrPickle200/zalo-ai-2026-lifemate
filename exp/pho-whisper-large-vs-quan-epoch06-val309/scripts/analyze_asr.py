from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"exp/asr-modeling/scripts"))
sys.path.insert(0,str(ROOT/"exp/quan-10epoch-transcript-dictionary-v1/scripts"))
from inference_common import normalize, edit_distance, read_jsonl, score_rows
from analyze_verb_errors import align, verb_flags, unaccent
from vietnamese_number_normalizer import normalize_number_phrases
OUT=ROOT/"exp/pho-whisper-large-vs-quan-epoch06-val309/output"
ACTION_WORDS={"bật","tắt","mở","đóng","xóa","xoá","hủy","huỷ","đổi","dời","chuyển","trả","nạp","mua","bán","gửi","gọi","nhắn","đặt","tạo","giữ","tăng","giảm"}
def analyze(branch,kind):
    path=OUT/branch/("predictions.jsonl" if kind=="main" else "history_predictions.jsonl")
    rows=read_jsonl(path)
    counts=Counter(); affected=set(); events=[]; risky=set()
    for row in rows:
        key=row["id"] if kind=="main" else row["key"]
        ref,_,_=normalize_number_phrases(normalize(row.get("reference","")))
        hyp,_,_=normalize_number_phrases(normalize(row.get("prediction","")))
        rf,hf=verb_flags(ref),verb_flags(hyp)
        for op,ri,hi in align(ref,hyp):
            if op=="match":continue
            rw=ref[ri] if ri is not None else None; hw=hyp[hi] if hi is not None else None
            if op=="substitution" and unaccent(rw)==unaccent(hw):continue
            if not ((ri is not None and rf[ri]) or (hi is not None and hf[hi])):continue
            counts[op]+=1;affected.add(key)
            risk=rw in ACTION_WORDS or hw in ACTION_WORDS
            if risk:risky.add(key)
            events.append({"key":key,"operation":op,"reference_token":rw,"prediction_token":hw,"reference":row.get("reference"),"prediction":row.get("prediction"),"potential_action_change":risk})
    events.sort(key=lambda x:(not x["potential_action_change"],x["key"]))
    result={"branch":branch,"kind":kind,"samples":len(rows),"metrics":score_rows(rows),"verb_error_tokens":sum(counts.values()),"verb_errors_by_operation":dict(counts),"samples_with_verb_error":len(affected),"samples_with_potential_action_verb_error":len(risky),"note":"Lexical screening; potential action impact is not proven downstream causality. Number phrases normalized for verb alignment; WER/CER use shared raw normalization."}
    (OUT/branch/f"{kind}_verb_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/branch/f"{kind}_verb_events.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in events),encoding="utf-8")
    return result
def main():
    reports=[]
    for kind in ("main","history"):
        for branch in ("quan","pho"):
            path=OUT/branch/("predictions.jsonl" if kind=="main" else "history_predictions.jsonl")
            if path.exists(): reports.append(analyze(branch,kind))
    for kind in ("main","history"):
        name="predictions.jsonl" if kind=="main" else "history_predictions.jsonl"
        qp=OUT/"quan"/name; pp=OUT/"pho"/name
        if not (qp.exists() and pp.exists()): continue
        key="id" if kind=="main" else "key"
        qrows=read_jsonl(qp); prows=read_jsonl(pp)
        if len(qrows)!=(309 if kind=="main" else 692) or len(prows)!=len(qrows): continue
        pmap={r[key]:r for r in prows}
        paired=[]
        for q in qrows:
            p=pmap[q[key]]
            ref=normalize(q["reference"]); rc=list("".join(ref))
            data={"key":q[key],"reference":q["reference"]}
            for branch,row in (("quan",q),("pho",p)):
                hyp=normalize(row["prediction"]);hc=list("".join(hyp))
                data[branch]={"prediction":row["prediction"],"word_edits":edit_distance(ref,hyp),"word_count":len(ref),"wer":edit_distance(ref,hyp)/max(1,len(ref)),"char_edits":edit_distance(rc,hc),"char_count":len(rc),"cer":edit_distance(rc,hc)/max(1,len(rc)),"inference_seconds":row.get("inference_seconds")}
            paired.append(data)
        (OUT/f"per_id_{kind}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in paired),encoding="utf-8")
    (OUT/"asr_comparison.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([{k:v for k,v in x.items() if k!="metrics"} for x in reports],ensure_ascii=False,indent=2))
if __name__=="__main__":main()



