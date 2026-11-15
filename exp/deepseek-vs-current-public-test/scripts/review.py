from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"exp/deepseek-vs-current-public-test/output"
def rows(path):
    return {r["id"]:r for r in (json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip())}
manifest=rows(ROOT/"train+public_test/train+public_test/public_test/public_test.jsonl")
pred=rows(ROOT/"outputs/submissions/qwen35-sft-epoch3-en-cot-quan-beam32-num-predict-768-num-batch-512-public-test_predictions.jsonl")
diff=rows(OUT/"per_id_differences.jsonl")
notes={
"public_0002":"ASR says book now, yet current branch used a scheduled booking after calendar lookup; DeepSeek booked immediately. Destination specificity also differs.",
"public_0004":"Both set an alarm; current branch first queried flight bookings to resolve the flight time. DeepSeek's 07:00 time is not supported by the ASR transcript alone.",
"public_0006":"Both queried weather; current branch omitted location while DeepSeek supplied Hanoi from profile. The conditional alarm action is absent in both saved call sequences.",
"public_0015":"Same two tool names and order. Alarm label differs; current branch omitted it.",
"public_0016":"ASR text is corrupted around the amount/source phrase. Branches explored different banking/contact lookups. No reliable correctness judgment from text alone.",
"public_0021":"After topping up 100000, current branch transferred 100000 to Minh although request asks for another amount; DeepSeek asked for transfer amount. This is a concrete amount carryover risk.",
"public_0022":"Both paid the TV bill from bank; current branch included an extra balance lookup. No individual gold score is available.",
"public_0023":"Current branch created a reminder in addition to the requested alarm. DeepSeek only called and set the alarm. Current call used phone_number while DeepSeek used contact_id.",
"public_0029":"Both eventually transferred 912000; lookup order and transfer note differ. State impact cannot be inferred without gold.",
"public_0030":"Current branch scoped status to floor 2 and turned off three lights; DeepSeek queried unscoped status and turned off two. Final ASR phrase is corrupted.",
}
result=[]
for sid,note in notes.items():
    a=manifest[sid]; d=diff[sid]; p=pred[sid]
    result.append({"id":sid,"audio":a.get("audio"),"asr_transcript":p.get("transcript_used"),"current_time":a.get("context",{}).get("current_time"),"history_audio_count":sum(bool(h.get("audio")) for h in a.get("history",[])),"deepseek_calls":d["deepseek_calls"],"current_en_calls":d["comparisons"]["current_en"]["calls"],"review_note":note,"review_basis":"ASR transcript, public context and saved tool calls; audio not manually listened to; no per-ID gold"})
(OUT/"review_examples.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in result),encoding="utf-8")
(OUT/"summary.md").write_text("""# DeepSeek versus three current submissions

All 634 public-test IDs align. Current three agree on full calls for 182 IDs; DeepSeek differs on 115 of those. The newest English branch differs from DeepSeek in tool sequence on 406 IDs, and in arguments with the same tool sequence on 129 IDs.

Ten selected examples are in review_examples.jsonl. Repeated differences include extra lookup/action calls, missing or mismatched arguments, amount carryover between separate requests, and decisions under unclear ASR text. For public_0021, current Qwen reused a 100000 top-up amount for a separate transfer while DeepSeek asked for that amount. For public_0004, current Qwen looked up flight time whereas DeepSeek set an alarm using a time absent from the ASR text. Thus disagreement is not a correctness label.

Review used saved ASR transcripts, public context and tool calls; original audio has not been manually listened to. The reported public score 49 for DeepSeek is user-provided; no per-ID public labels are available.
""",encoding="utf-8")
print(len(result))
if __name__=="__main__": pass

