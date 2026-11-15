from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/audio-coverage-public-test-v1"
OUT=EXP/"output"
DATA=ROOT/"train+public_test/train+public_test/public_test"
rows=[json.loads(x) for x in (DATA/"public_test.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
ids=[r["id"] for r in rows]
if len(ids)!=len(set(ids)):
    raise RuntimeError("Duplicate public-test IDs")
main=[r["audio"] for r in rows]
history=[(r["id"],t["audio"]) for r in rows for t in r.get("history",[]) if t.get("audio")]
refs=main+[x for _,x in history]
disk={str(p.relative_to(DATA)).replace("\\","/") for p in (DATA/"audio").rglob("*.wav")}
missing=sorted(set(refs)-disk)
unreferenced=sorted(disk-set(refs))
transcript=ROOT/"exp/quan-10epoch-transcript-dictionary-v1/output/transcripts/public_test.jsonl"
tr=[json.loads(x) for x in transcript.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
tr_ids=[r["id"] for r in tr]
manifest=ROOT/"outputs/submissions/qwen35-sft-epoch3-en-cot-quan-beam32-num-predict-768-num-batch-512-public-test_manifest.json"
settings=json.loads(manifest.read_text(encoding="utf-8-sig"))["settings"]
pipeline=(ROOT/"src/lifemate/pipeline.py").read_text(encoding="utf-8")
history_path_present='converted["text"] = self.asr.transcribe' in pipeline
report={"samples":len(rows),"main_audio":len(main),"history_audio":len(history),
        "samples_with_history_audio":sum(any(t.get("audio") for t in r.get("history",[])) for r in rows),
        "max_history_audio_per_sample":max((sum(bool(t.get("audio")) for t in r.get("history",[])) for r in rows),default=0),
        "referenced_audio_unique":len(set(refs)),"disk_wav":len(disk),"missing_audio_refs":missing,
        "unreferenced_wav":unreferenced,"quan_main_transcripts":len(tr),
        "quan_missing_ids":sorted(set(ids)-set(tr_ids)),"quan_extra_ids":sorted(set(tr_ids)-set(ids)),
        "history_asr_from_latest_manifest":settings.get("history_asr"),
        "pipeline_transcribes_history_at_runtime":history_path_present,
        "history_audio_count_by_task":{r["id"]:sum(bool(t.get("audio")) for t in r.get("history",[])) for r in rows}}
counts=list(report["history_audio_count_by_task"].values())
report["history_audio_count_distribution"]={str(k):v for k,v in sorted(Counter(counts).items())}
report["mean_history_audio_per_task"]=sum(counts)/len(counts)
report["mean_asr_calls_per_task_including_main"]=1+report["mean_history_audio_per_task"]
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"coverage.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
summary=(f"# Public-test audio coverage\n\n{len(rows)} tasks, {len(main)} main audio and {len(history)} history audio; "
         f"{report['samples_with_history_audio']} tasks contain history audio. "
         f"Referenced missing: {len(missing)}; unreferenced WAV: {len(unreferenced)}. "
         f"QUAN main transcripts: {len(tr)}; missing IDs: {len(report['quan_missing_ids'])}. "
         f"Latest submission uses {settings.get('history_asr')} for history; pipeline runtime path present: {history_path_present}.\n")
summary+=(f"History audio adds one PhoASR decode per clip: mean {report['mean_history_audio_per_task']:.2f} extra ASR calls per task, maximum 12; these are call counts, not measured runtime.\n")
(OUT/"summary.md").write_text(summary,encoding="utf-8")
(EXP/"README.md").write_text("# Public-test audio coverage\n\nRun scripts/audit.py. Results: output/coverage.json and output/summary.md.\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="history_audio_count_by_task"},ensure_ascii=True))

