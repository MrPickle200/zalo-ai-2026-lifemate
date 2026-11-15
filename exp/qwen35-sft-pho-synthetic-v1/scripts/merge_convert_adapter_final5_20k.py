from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'exp/qwen35-sft-pho-synthetic-v1'
CHECKPOINT = EXP / 'output/adapter_final-5-epochs-20k-rows.zip'
OUT = EXP / 'output/adapter-final-5epoch-20k-rows-benchmark'
BASE = ROOT / 'models/kaggle-qwen3-5-4b-v1/Qwen3.5-4B'
MERGE_SCRIPT = EXP / 'scripts/merge_adapter_final4.py'
CONVERT_SCRIPT = EXP / 'scripts/convert_adapter_final4.py'

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not BASE.joinpath('config.json').is_file():
        raise FileNotFoundError(BASE / 'config.json')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty output: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    PROV = OUT / 'model-provenance'

    merge = load_module('merge_final5_20k_reusing_verified_pipeline', MERGE_SCRIPT)
    merge.ARCHIVE = CHECKPOINT
    merge.OUT = OUT
    merge.PROVENANCE = PROV
    saved_argv = sys.argv
    try:
        sys.argv = [str(MERGE_SCRIPT), '--base-dir', str(BASE)]
        if merge.main() != 0:
            raise RuntimeError('Adapter merge failed')
    finally:
        sys.argv = saved_argv
    merge_manifest_path = PROV / 'merge-manifest.json'
    merge_manifest = read_json(merge_manifest_path)
    checkpoint_sha = sha(CHECKPOINT)
    if merge_manifest.get('adapter_archive_sha256') != checkpoint_sha:
        raise RuntimeError('Merge manifest does not identify the requested 5-epoch checkpoint')
    if not merge_manifest.get('merge', {}).get('probe_same_argmax'):
        raise RuntimeError('Merge fidelity probe did not preserve argmax')

    convert = load_module('convert_final5_20k_reusing_pinned_pipeline', CONVERT_SCRIPT)
    convert.OUT = OUT
    convert.PROV = PROV
    convert.MERGED = OUT / 'merged-hf'
    convert.GGUF = OUT / 'gguf'
    convert.LOGS = OUT / 'conversion-logs'
    if convert.main() != 0:
        raise RuntimeError('GGUF conversion/quantization failed')
    conversion_manifest = read_json(PROV / 'conversion-manifest.json')
    artifact = conversion_manifest['q4_k_m_gguf']
    gguf_path = ROOT / artifact['path']
    if sha(gguf_path) != artifact['sha256'] or gguf_path.stat().st_size != artifact['size_bytes']:
        raise RuntimeError('Q4_K_M GGUF failed final hash/size validation')
    build = {
        'status': 'merged_and_quantized',
        'checkpoint_path': str(CHECKPOINT.relative_to(ROOT)).replace('\\', '/'),
        'checkpoint_sha256': checkpoint_sha,
        'base_model_path': str(BASE.relative_to(ROOT)).replace('\\', '/'),
        'base_manifest_sha256': merge_manifest['base_manifest_sha256'],
        'adapter_config': merge_manifest['adapter_files_sha256'],
        'merge_manifest_sha256': sha(merge_manifest_path),
        'merge_tree_sha256': merge_manifest['merged_tree_sha256'],
        'conversion_manifest_sha256': sha(PROV / 'conversion-manifest.json'),
        'quantization': conversion_manifest['quantization'],
        'gguf': artifact,
        'source_merge_script': str(MERGE_SCRIPT.relative_to(ROOT)).replace('\\', '/'),
        'source_conversion_script': str(CONVERT_SCRIPT.relative_to(ROOT)).replace('\\', '/'),
        'note': 'Reused the verified Qwen3.5-4B base, PEFT safe merge, pinned llama.cpp converter and Q4_K_M quantizer used for final4; source adapter is the 5-epoch 20k-row checkpoint.',
    }
    (PROV / 'build-manifest.json').write_text(json.dumps(build, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': build['status'], 'checkpoint_sha256': checkpoint_sha,
                      'merge_tree_sha256': build['merge_tree_sha256'], 'quantization': build['quantization'],
                      'gguf': artifact}, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
