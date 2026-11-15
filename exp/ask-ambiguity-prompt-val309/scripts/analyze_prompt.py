from __future__ import annotations
import json, math, random, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"exp/ask-ambiguity-prompt-val309/output"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"exp/btc-scoring/scripts")]
from lifemate.utils import find_data_root,load_jsonl
from score_btc import score_sample,call_key

def rows(path):
    return {r["id"]:r for r in (json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip())}
def summarize(vals):
    v=sorted(vals);return {"mean":statistics.mean(v),"median":statistics.median(v),"p95":v[min(len(v)-1,math.ceil(.95*len(v))-1)]}
def main():
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    ids=manifest["ids"]; data=find_data_root("train+public_test/train+public_test")
    items=rows(data/"train/train.jsonl")
    preds={b:rows(OUT/b/"predictions-COT.jsonl") for b in ("baseline","variant")}
    assert all(set(p)==set(ids) for p in preds.values())
    details=[]; scores={"baseline":[],"variant":[]}
    for sid in ids:
        item=items[sid]; gold=item.get("label",{}).get("tool_calls",[])
        gold_ask_calls=[c for c in gold if c.get("name")=="sys.ask_missing_info"]
        row={"id":sid,"gold_ask":bool(gold_ask_calls)}
        for b in preds:
            p=preds[b][sid]; s=score_sample(sid,item,p,data)
            calls=p.get("tool_calls") or []
            ask_calls=[c for c in calls if c.get("name")=="sys.ask_missing_info"]
            ask=bool(ask_calls)
            ask_exact=bool({call_key(c) for c in ask_calls} & {call_key(c) for c in gold_ask_calls})
            row[b]={"score":s["official_sample_score"],"score_before_runtime_gate":s["btc_formula_score_before_runtime_gate_diagnostic"],"ask":ask,"ask_exact":ask_exact,"latency_seconds":p.get("latency_seconds"),"runtime_error":p.get("error"),"hash_match":s["hash_match"],"matched_calls":s["matched_calls"],"gold_calls":s["gold_calls"],"unmatched_submitted_calls":s["unmatched_submitted_calls"],"schema_rejections":s["schema_rejections"]}
            scores[b].append(s["official_sample_score"])
        details.append(row)
    (OUT/"paired_details.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in details),encoding="utf-8")
    metrics={}
    gold_ask=sum(x["gold_ask"] for x in details)
    for b in preds:
        m={"btc_0_100":100*statistics.mean(scores[b]),"btc_without_runtime_gate_0_100":100*statistics.mean(x[b]["score_before_runtime_gate"] for x in details),"gold_ask_ids":gold_ask,"ask_true_positive":sum(x["gold_ask"] and x[b]["ask"] for x in details),"ask_exact_gold_call":sum(x[b]["ask_exact"] for x in details),"ask_missing":sum(x["gold_ask"] and not x[b]["ask"] for x in details),"ask_extra":sum(not x["gold_ask"] and x[b]["ask"] for x in details),"hash_matches":sum(x[b]["hash_match"] for x in details),"latency":summarize([x[b]["latency_seconds"] for x in details]),"over_30_seconds":sum(x[b]["latency_seconds"]>30 for x in details),"runtime_errors":sum(bool(x[b]["runtime_error"]) for x in details),"schema_rejections":sum(x[b]["schema_rejections"] for x in details)}
        req=[json.loads(line) for line in (OUT/b/"requests-COT.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        m["ollama_requests"]=len(req)
        m["requests_at_generation_cap"]=sum(x.get("done_reason")=="length" or (x.get("eval_count") or 0)>=768 for x in req)
        m["context_limit_reached_ids"]=sum(any(term in str(x[b]["runtime_error"]).lower() for term in ("context","overflow","maximum input length")) for x in details)
        m["requests_prompt_tokens_at_or_above_8192"]=sum((x.get("prompt_eval_count") or 0)>=8192 for x in req)
        metrics[b]=m
    deltas=[x["variant"]["score"]-x["baseline"]["score"] for x in details]
    rng=random.Random(17)
    boots=sorted(100*statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(2000))
    ungated=[x["variant"]["score_before_runtime_gate"]-x["baseline"]["score_before_runtime_gate"] for x in details]
    ungated_boots=sorted(100*statistics.mean(rng.choices(ungated,k=len(ungated))) for _ in range(2000))
    metrics["paired"]={"btc_delta_points":100*statistics.mean(deltas),"bootstrap_95pct_CI_points":[boots[49],boots[1949]],"variant_wins":sum(d>0 for d in deltas),"baseline_wins":sum(d<0 for d in deltas),"ties":sum(d==0 for d in deltas),"ungated_btc_delta_points":100*statistics.mean(ungated),"ungated_bootstrap_95pct_CI_points":[ungated_boots[49],ungated_boots[1949]],"ask_exact_baseline_only":sum(x["baseline"]["ask_exact"] and not x["variant"]["ask_exact"] for x in details),"ask_exact_variant_only":sum(x["variant"]["ask_exact"] and not x["baseline"]["ask_exact"] for x in details)}
    (OUT/"summary.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=="__main__":main()





