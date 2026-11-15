from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from lifemate.pipeline import LifeMatePipeline
from lifemate.utils import resolve_audio
class SpyASR:
    def __init__(self):self.paths=[]
    def transcribe(self,path):
        self.paths.append(str(Path(path).resolve()))
        return "probe transcript"
class StopAgent:
    def next_action(self,*args,**kwargs):return {"action":"stop"}
data=ROOT/"train+public_test/train+public_test"
rows=[json.loads(x) for x in (data/"public_test/public_test.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
asr=SpyASR();pipe=LifeMatePipeline(data,"public_test",asr=asr,agent=StopAgent(),max_steps=1)
errors=[];matched=0
for row in rows:
    start=len(asr.paths)
    result=pipe.run(row,transcript_override="probe")
    expected=[str(resolve_audio(data,data/"public_test",row,h["audio"]).resolve()) for h in row.get("history",[]) if h.get("audio")]
    actual=asr.paths[start:]
    if actual!=expected:errors.append({"id":row["id"],"expected":expected,"actual":actual})
    matched+=len(actual)
out={"tasks":len(rows),"history_asr_calls_observed":matched,"expected_history_audio":sum(bool(h.get("audio")) for r in rows for h in r.get("history",[])),"mismatch_tasks":errors,"main_asr_calls_with_override":0,"mode":"spy ASR and stop agent through real LifeMatePipeline + ToolEnv; no audio decoded"}
target=ROOT/"exp/audio-coverage-public-test-v1/output/runtime_spy.json"
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="mismatch_tasks"},ensure_ascii=False),len(errors))

