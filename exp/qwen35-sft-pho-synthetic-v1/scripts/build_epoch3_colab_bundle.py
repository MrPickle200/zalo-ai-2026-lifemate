from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/"exp/qwen35-sft-pho-synthetic-v1"; DATA=EXP/"output/data"
AD=ROOT/"exp/qwen35-sft-epoch3-en-cot-val60/output/model-provenance/adapter-final"
REPORT=ROOT/"exp/qwen35-sft-epoch3-en-cot-val60/output/model-provenance/training/training_report.json"
INFER=ROOT/"exp/qwen35-sft-epoch3-en-cot-val60/output/inference/config_manifest.json"
README=EXP/"README-colab-epoch3-continuation.md"
DEST=EXP/"output/colab-resources/lifemate-qwen35-pho-synthetic-epoch3-continue-colab-v1.zip"
PKG="lifemate-qwen35-pho-synthetic-epoch3-continue-colab-v1"
EA="a15c99d66b1bc2c14846428ef0efd84865110d084fbf17f7206517ad69977048"
ED="6203ce35636f65a34b629b0398f12cc2bf1b8bdbd3c266a1abf5a5602ffc16d1"
FILES=("dataset_manifest.json","synthetic_utterances_1236.jsonl","btc_train_1236.jsonl","original_phowhisper_sft_3873.jsonl","synthetic_direct_sft_3873.jsonl","combined_sft_7746.jsonl","toolenv_replay_train_1236.jsonl","synthetic_quality_audit.json")
def sha(x):
 h=hashlib.sha256()
 if isinstance(x,Path):
  with x.open("rb") as f:
   for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 else:h.update(x)
 return h.hexdigest()
def main():
 if sha(DATA/"combined_sft_7746.jsonl")!=ED or sha(AD/"adapter_model.safetensors")!=EA:raise RuntimeError("Checksum mismatch")
 src=json.loads((AD/"adapter_config.json").read_text(encoding="utf8")); cfg=dict(src);cfg["base_model_name_or_path"]="Qwen/Qwen3.5-4B"
 DEST.parent.mkdir(parents=True,exist_ok=True);tmp=DEST.with_suffix(".tmp")
 with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  z.write(README,f"{PKG}/README.md")
  for f in FILES:z.write(DATA/f,f"{PKG}/data/{f}")
  z.write(AD/"adapter_model.safetensors",f"{PKG}/initial_adapter/adapter_model.safetensors")
  z.writestr(f"{PKG}/initial_adapter/adapter_config.json",json.dumps(cfg,indent=2)+"\n")
  z.writestr(f"{PKG}/provenance/epoch3_adapter_source_config.json",json.dumps(src,indent=2)+"\n")
  z.write(REPORT,f"{PKG}/provenance/epoch3_training_report.json");z.write(INFER,f"{PKG}/provenance/epoch3_inference_config_manifest.json")
  z.writestr(f"{PKG}/provenance/bundle_manifest.json",json.dumps({"adapter_epoch":3,"adapter_sha256":EA,"combined_sft_sha256":ED,"prompt_variant":"english-system-prompt-v1","model_id":"Qwen/Qwen3.5-4B","rows":7746},indent=2)+"\n")
 tmp.replace(DEST)
 with zipfile.ZipFile(DEST) as z:
  if z.testzip():raise RuntimeError("ZIP corrupt")
  rows=[x for x in z.read(f"{PKG}/data/combined_sft_7746.jsonl").splitlines() if x.strip()]
  if len(rows)!=7746 or sha(z.read(f"{PKG}/initial_adapter/adapter_model.safetensors"))!=EA:raise RuntimeError("ZIP content mismatch")
 print(json.dumps({"archive":str(DEST),"bytes":DEST.stat().st_size,"rows":len(rows),"adapter_sha256":EA,"data_sha256":ED},indent=2))
if __name__=="__main__":main()
