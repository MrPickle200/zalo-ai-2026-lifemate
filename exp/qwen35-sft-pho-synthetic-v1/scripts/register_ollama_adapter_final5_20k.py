from __future__ import annotations
import hashlib, json, os, subprocess, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'exp/qwen35-sft-pho-synthetic-v1'
OUT=EXP/'output/adapter-final-5epoch-20k-rows-benchmark'
PROV=OUT/'model-provenance'
BUILD=PROV/'build-manifest.json'
GGUF=OUT/'gguf/adapter-final-5epoch-20k-rows-Q4_K_M.gguf'
TAG='lifemate-qwen35-phosynth-final5-20k-q4km:latest'
HOST='http://127.0.0.1:11434'
OLLAMA=Path(os.environ.get('OLLAMA_EXE',r'C:\Users\PC\AppData\Local\Programs\Ollama\ollama.exe'))
REF=EXP/'output/adapter-final-4epoch-benchmark/model-provenance/ollama-models.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def api(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(HOST+path,data=data,headers={'Content-Type':'application/json'},method='POST' if payload is not None else 'GET')
 with urllib.request.urlopen(req,timeout=90) as r:return json.loads(r.read())
def main():
 if not OLLAMA.is_file():raise FileNotFoundError(OLLAMA)
 if not GGUF.is_file() or not BUILD.is_file():raise FileNotFoundError('Verified final5 Q4 GGUF/build manifest is missing')
 build=json.loads(BUILD.read_text(encoding='utf-8-sig'));record=build['gguf']
 if sha(GGUF)!=record['sha256'] or GGUF.stat().st_size!=record['size_bytes']:
  raise RuntimeError('Final5 GGUF differs from its build manifest')
 if (ROOT/record['path']).resolve()!=GGUF.resolve():raise RuntimeError('Build manifest points to a different GGUF')
 ref=json.loads(REF.read_text(encoding='utf-8-sig'))
 version=api('/api/version')['version']
 if any(x.get('name')==TAG for x in api('/api/tags').get('models',[])):
  raise FileExistsError(f'Refusing to overwrite existing Ollama tag {TAG}')
 modelfile=OUT/'ollama/adapter-final5-20k.Modelfile';log=OUT/'ollama/adapter-final5-20k-create.log'
 if modelfile.exists() or log.exists() or (PROV/'ollama-models.json').exists():
  raise FileExistsError('Refusing to overwrite final5 Ollama registration artifacts')
 modelfile.parent.mkdir(parents=True,exist_ok=True)
 modelfile.write_text('\n'.join((f'FROM {GGUF.as_posix()}','PARAMETER num_ctx 16384','PARAMETER num_batch 512','PARAMETER num_predict 768','PARAMETER temperature 0','')),encoding='utf-8')
 started=time.time()
 result=subprocess.run([str(OLLAMA),'create',TAG,'-f',str(modelfile)],cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=3600)
 log.write_text((result.stdout or '')+(result.stderr or ''),encoding='utf-8')
 if result.returncode:raise RuntimeError(f'ollama create failed ({result.returncode}); see {log}')
 found=next((x for x in api('/api/tags').get('models',[]) if x.get('name')==TAG),None)
 if not found:raise RuntimeError('Ollama tag not visible after create')
 shown=api('/api/show',{'model':TAG});info=shown.get('model_info',{})
 if info.get('general.architecture')!='qwen35' or info.get('general.quantization_version')!=2:
  raise RuntimeError(f'Unexpected model info: {info}')
 manifest={'status':'registered','tag':TAG,'ollama_digest':found.get('digest'),'ollama_size_bytes':found.get('size'),
  'ollama_version':version,'host':HOST,'checkpoint_path':build['checkpoint_path'],'checkpoint_sha256':build['checkpoint_sha256'],
  'gguf':record,'build_manifest_sha256':sha(BUILD),'modelfile':str(modelfile.relative_to(ROOT)).replace('\\','/'),
  'modelfile_sha256':sha(modelfile),'create_log':str(log.relative_to(ROOT)).replace('\\','/'),
  'create_seconds':time.time()-started,'model_info':info,'template':shown.get('template'),'parameters':shown.get('parameters'),
  'settings':{'thinking':True,'temperature':0,'protocol':'JSON','num_predict':768,'num_ctx':16384,'num_batch':512},
  'reference_final4':{'tag':ref['tag'],'digest':ref['ollama_digest'],'same_ollama_version':True}}
 (PROV/'ollama-models.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:manifest[k] for k in ('status','tag','ollama_digest','ollama_version','ollama_size_bytes','create_seconds')},indent=2),flush=True)
if __name__=='__main__':main()
