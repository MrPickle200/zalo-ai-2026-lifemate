from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/deepseek-vs-current-public-test"
OUT=EXP/"output"
BASE=ROOT/"outputs/submissions"
FILES={
"deepseek":"deepseek-api.jsonl",
"current_en":"qwen35-sft-epoch3-en-cot-quan-beam32-num-predict-768-num-batch-512-public-test.jsonl",
"current_vi":"qwen35-sft-epoch3-cot-quan-beam32-num-predict-768-num-batch-512-public-test.jsonl",
"current_sft2":"qwen35-sft-cot-quan-beam32-num-predict-768-num-batch-512-public-test.jsonl",
}
def load(name):
    rows=[json.loads(x) for x in (BASE/FILES[name]).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if len(rows)!=634 or len({r["id"] for r in rows})!=634: raise RuntimeError(f"Invalid coverage: {name}")
    return {r["id"]:r for r in rows}
def calls(r):return json.dumps(r["tool_calls"],ensure_ascii=False,sort_keys=True,separators=(",",":"))
def names(r):return [c["name"] for c in r["tool_calls"]]
data={name:load(name) for name in FILES}
ids=list(data["deepseek"])
if any(set(rows)!=set(ids) for rows in data.values()): raise RuntimeError("ID mismatch")
stats={}
details=[]
consensus_ids=[]
for name in list(FILES)[1:]:
    stats[name]=Counter()
for sid in ids:
    d=data["deepseek"][sid]
    trio=[data[n][sid] for n in list(FILES)[1:]]
    consensus=len({calls(x) for x in trio})==1
    row={"id":sid,"current_three_consensus":consensus,"deepseek_calls":d["tool_calls"],"deepseek_hash":d["state_hash"],"comparisons":{}}
    for name,pred in zip(list(FILES)[1:],trio):
        same_calls=calls(d)==calls(pred)
        same_names=names(d)==names(pred)
        same_hash=d["state_hash"]==pred["state_hash"]
        category="same_calls" if same_calls else ("same_names_different_arguments" if same_names else "different_tool_sequence")
        row["comparisons"][name]={"category":category,"same_hash":same_hash,"calls":pred["tool_calls"],"state_hash":pred["state_hash"]}
        stats[name][category]+=1
        stats[name]["same_hash"]+=same_hash
    if consensus and calls(d)!=calls(trio[0]):
        consensus_ids.append(sid)
    details.append(row)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"per_id_differences.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in details),encoding="utf-8")
report={"cohort_ids":len(ids),"files":FILES,"by_current_submission":{k:dict(v) for k,v in stats.items()},
        "current_three_consensus":sum(x["current_three_consensus"] for x in details),
        "deepseek_differs_from_current_three_consensus":len(consensus_ids),
        "consensus_disagreement_ids":consensus_ids,
        "priority_examples":consensus_ids[:30],
        "interpretation":"Public test has no per-ID gold; differences alone do not identify the correct pipeline."}
(OUT/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"summary.md").write_text(
    "# DeepSeek versus three current submissions\n\n"
    f"All {len(ids)} public-test IDs align. Current three agree on full calls for {report['current_three_consensus']} IDs; "
    f"DeepSeek differs on {len(consensus_ids)} of those. Detailed calls and hashes: per_id_differences.jsonl. "
    "No per-ID public gold is available, so disagreement is not a correctness label.\n",encoding="utf-8")
(EXP/"README.md").write_text("# DeepSeek versus current submissions\n\nRun scripts/compare.py. Results: output/comparison.json, output/per_id_differences.jsonl, output/summary.md.\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in {"consensus_disagreement_ids","priority_examples"}},ensure_ascii=True))
