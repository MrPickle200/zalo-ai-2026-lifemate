from __future__ import annotations
import hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'exp/qwen35-sft-pho-synthetic-v1'
OUT=EXP/'output/adapter-final-4epoch-benchmark'
PROV=OUT/'model-provenance'
GGUF=OUT/'gguf/adapter-final-4epoch-Q4_K_M.gguf'
TAG='lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest'
HOST='http://127.0.0.1:11434'
OLLAMA=Path(os.environ.get('OLLAMA_EXE',r'C:\Users\PC\AppData\Local\Programs\Ollama\ollama.exe'))
REF=EXP/'output/adapter-final-3epoch-benchmark/val309/comparison_lock.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def api(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(HOST+path,data=data,headers={'Content-Type':'application/json'},method='POST' if payload is not None else 'GET')
 with urllib.request.urlopen(req,timeout=90) as r:return json.loads(r.read())
def main():
 if sys.prefix!=sys.base_prefix: raise RuntimeError('Use system Python so Ollama local API is reachable')
 if not OLLAMA.is_file(): raise FileNotFoundError(OLLAMA)
 conv=json.loads((PROV/'conversion-manifest.json').read_text(encoding='utf-8'))
 artifact=conv['q4_k_m_gguf']
 if conv.get('status')!='converted_and_quantized' or (ROOT/artifact['path']).resolve()!=GGUF.resolve() or sha(GGUF)!=artifact['sha256'] or GGUF.stat().st_size!=artifact['size_bytes']:
  raise RuntimeError('Q4_K_M GGUF differs from verified conversion manifest')
 ref=json.loads(REF.read_text(encoding='utf-8-sig'))
 ref3=ref['adapter_final3']
 version=api('/api/version')['version']
 if any(x.get('name')==TAG for x in api('/api/tags').get('models',[])):raise FileExistsError(f'Tag already exists: {TAG}')
 modelfile=OUT/'ollama/adapter-final-4epoch.Modelfile'; log=OUT/'ollama/adapter-final-4epoch-create.log'
 if modelfile.exists() or log.exists() or (PROV/'ollama-models.json').exists():raise FileExistsError('Refusing to overwrite Ollama registration artifacts')
 modelfile.parent.mkdir(parents=True,exist_ok=True)
 modelfile.write_text('\n'.join((f'FROM {GGUF.as_posix()}','PARAMETER num_ctx 8192','PARAMETER num_batch 512','PARAMETER num_predict 768','PARAMETER temperature 0','')),encoding='utf-8')
 started=time.time(); result=subprocess.run([str(OLLAMA),'create',TAG,'-f',str(modelfile)],cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=3600)
 log.write_text((result.stdout or '')+(result.stderr or ''),encoding='utf-8')
 if result.returncode:raise RuntimeError(f'ollama create failed ({result.returncode}); see {log}')
 found=next((x for x in api('/api/tags').get('models',[]) if x.get('name')==TAG),None)
 if not found:raise RuntimeError('Ollama tag not visible after create')
 shown=api('/api/show',{'model':TAG}); info=shown.get('model_info',{})
 if info.get('general.architecture')!='qwen35' or info.get('general.quantization_version')!=2:raise RuntimeError(f'Unexpected model info: {info}')
 manifest={'status':'registered','tag':TAG,'ollama_digest':found.get('digest'),'ollama_size_bytes':found.get('size'),'ollama_version':version,'host':HOST,
  'gguf':artifact,'modelfile':str(modelfile.relative_to(ROOT)).replace('\\','/'),'modelfile_sha256':sha(modelfile),'create_log':str(log.relative_to(ROOT)).replace('\\','/'),'create_seconds':time.time()-started,
  'model_info':info,'template':shown.get('template'),'parameters':shown.get('parameters'),'settings':{'thinking':True,'temperature':0,'protocol':'JSON','num_predict':768,'num_ctx':8192,'num_batch':512},
  'reference_3epoch':{'tag':ref3['model'],'digest':ref3['digest'],'same_ollama_version':True}}
 (PROV/'ollama-models.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:manifest[k] for k in ('status','tag','ollama_digest','ollama_version','ollama_size_bytes')},indent=2))
if __name__=='__main__':main()
