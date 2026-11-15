from __future__ import annotations
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py"
SPEC=importlib.util.spec_from_file_location("base_runner",SOURCE)
runner=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)
OUT=ROOT/"exp/ask-ambiguity-prompt-val309/output"
MODEL="lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest"
DIGEST="fed86638cefacb100fbf704a1f4099fde3a70112bc54bf0542cbee005bf81caf"
EXPECTED_BASE="cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950"
OLD="Call sys.ask_missing_info only when required information is genuinely missing and cannot be safely inferred; ask only for the missing fields and never ask again for information already available."
NEW="Use request details in lookup queries. If a required value is missing or several results fit and context cannot distinguish them, call sys.ask_missing_info with the intended downstream tool_name and only the unresolved params; never choose an ID arbitrarily or ask for known values."
runner.MODEL=MODEL; runner.EXPECTED_DIGEST=DIGEST
original_inject=runner.inject_cot

def variant_inject(messages):
    result=original_inject(messages)
    systems=[m for m in result if m.get("role")=="system"]
    assert len(systems)==1 and systems[0]["content"].count(OLD)==1
    systems[0]["content"]=systems[0]["content"].replace(OLD,NEW,1)
    return result

def prompt_hash(inject):
    from lifemate.agent import Agent
    os.environ.update({"LIFEMATE_LLM_BACKEND":"ollama","LIFEMATE_OLLAMA_THINK":"true","LIFEMATE_OLLAMA_PROTOCOL":"json","LIFEMATE_PROMPT_LANGUAGE":"en"})
    agent=Agent(max_new_tokens=768,context_limit=8192,ollama_thinking=True,ollama_native_tools=False)
    captured=[]
    def capture(messages):
        captured.append(next(m["content"] for m in inject(messages) if m.get("role")=="system"))
        return '{"action":"stop"}'
    agent._generate=capture
    agent.next_action("prompt hash probe",{},[],[],[])
    assert len(captured)==1
    return hashlib.sha256(captured[0].encode()).hexdigest()

def main():
    if sys.prefix!=sys.base_prefix: raise RuntimeError("Use native system Python for Ollama")
    sys.path.insert(0,str(ROOT/"src"))
    sys.path.insert(0,str(ROOT/"exp/btc-scoring/scripts"))
    from lifemate.utils import find_data_root, load_jsonl
    version,digest=runner.model_fingerprint()
    base_hash=prompt_hash(original_inject)
    variant_hash=prompt_hash(variant_inject)
    assert base_hash==EXPECTED_BASE,base_hash
    assert base_hash!=variant_hash
    ids=json.loads((ROOT/"data/splits/val_ids.json").read_text(encoding="utf-8-sig"))
    transcript_path=ROOT/"exp/quan-continuation-cot-v1/output/asr/checkpoint-verification-rerun-val309/predictions.jsonl"
    transcripts={r["id"]:r["prediction"] for r in runner.read_jsonl(transcript_path)}
    assert len(ids)==309 and set(ids)==set(transcripts)
    data=find_data_root("train+public_test/train+public_test")
    items={r["id"]:r for r in load_jsonl(data/"train/train.jsonl")}
    assert set(ids)<=set(items)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={"ids":ids,"model":MODEL,"digest":digest,"ollama_version":version,"baseline_prompt_sha256":base_hash,"variant_prompt_sha256":variant_hash,"old_sentence":OLD,"replacement_sentence":NEW,"transcript_path":str(transcript_path),"transcript_sha256":runner.sha256(transcript_path),"settings":{"thinking":True,"temperature":0,"protocol":"JSON","num_predict":768,"num_ctx":8192,"num_batch":512,"seed":17,"max_action_turns":5,"history_asr":"PhoASRModel"}}
    mp=OUT/"manifest.json"
    if mp.exists():
        assert runner.read_json(mp)==manifest,"Locked inputs changed"
    else: runner.write_json(mp,manifest)
    for block_index,start in enumerate(range(0,len(ids),15),1):
        block=ids[start:start+15]
        for branch in (("baseline","variant") if block_index%2 else ("variant","baseline")):
            branch_out=OUT/branch; branch_out.mkdir(exist_ok=True)
            runner.OUT=branch_out
            runner.inject_cot=original_inject if branch=="baseline" else variant_inject
            prior=runner.read_jsonl(branch_out/"predictions-COT.jsonl")
            done={r["id"] for r in prior}
            assert len(done)==len(prior) and done<=set(ids)
            todo=[sid for sid in block if sid not in done]
            if not todo: continue
            runner.warmup("COT",block_index)
            while todo:
                before=len(runner.read_jsonl(branch_out/"predictions-COT.jsonl"))
                try:
                    runner.run_block("COT",block_index,todo,transcripts,items,cot=True)
                except Exception:
                    after=runner.read_jsonl(branch_out/"predictions-COT.jsonl")
                    if len(after)<=before or not after[-1].get("error"): raise
                done={r["id"] for r in runner.read_jsonl(branch_out/"predictions-COT.jsonl")}
                todo=[sid for sid in block if sid not in done]
            print(f"prompt {branch} {len(done)}/309",flush=True)
    for branch in ("baseline","variant"):
        rows=runner.read_jsonl(OUT/branch/"predictions-COT.jsonl")
        assert [r["id"] for r in rows]==ids,(branch,len(rows))
    print("prompt inference complete",flush=True)

if __name__=="__main__": main()


