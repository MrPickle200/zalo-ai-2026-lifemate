from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / 'exp' / 'qwen35-sft-next-action-v1'
SOURCE_DIR = EXP_DIR / 'kaggle-upload'
OUTPUT_DIR = EXP_DIR / 'kaggle-upload-prompt-en-v1'
ZIP_PATH = EXP_DIR / 'kaggle-upload-prompt-en-v1.zip'
AGENT_PATH = ROOT / 'src' / 'lifemate' / 'agent.py'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def english_production_prompt() -> str:
    tree = ast.parse(AGENT_PATH.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_english_branch = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == 'prompt_language'
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == 'en'
        )
        if not is_english_branch:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == 'system' for target in statement.targets):
                continue
            prompt = ast.literal_eval(statement.value)
            if not isinstance(prompt, str) or not prompt.endswith(' /no_think'):
                raise ValueError('Unexpected Agent.next_action English prompt format')
            # Match Agent.next_action when thinking is enabled.
            prompt = prompt[:-len(' /no_think')]
            if not prompt.endswith('}.'):
                raise ValueError('Unexpected end of production English prompt')
            return prompt
    raise LookupError('Could not extract the English branch from Agent.next_action')


def newline_of(raw_line: str) -> str:
    if raw_line.endswith('\r\n'):
        return '\r\n'
    if raw_line.endswith('\n'):
        return '\n'
    if raw_line.endswith('\r'):
        return '\r'
    return ''


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(SOURCE_DIR)
    if OUTPUT_DIR.exists() or ZIP_PATH.exists():
        raise FileExistsError(f'Refusing to overwrite: {OUTPUT_DIR} or {ZIP_PATH}')
    required = ['sft_train.jsonl', 'train_ids.json', 'val309_ids.json', 'val309_inputs.jsonl', 'dataset_manifest.json']
    missing = [name for name in required if not (SOURCE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Source SFT package is incomplete: {missing}')

    source_manifest = json.loads((SOURCE_DIR / 'dataset_manifest.json').read_text(encoding='utf-8'))
    prompt = english_production_prompt()
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    source_sft_hash = sha256_file(SOURCE_DIR / 'sft_train.jsonl')
    if source_sft_hash != source_manifest['sft_train_sha256']:
        raise ValueError('Source SFT JSONL hash does not match its manifest')

    OUTPUT_DIR.mkdir(parents=True)
    try:
        for name in ['train_ids.json', 'val309_ids.json', 'val309_inputs.jsonl']:
            shutil.copy2(SOURCE_DIR / name, OUTPUT_DIR / name)

        row_count = 0
        target_counts: Counter[str] = Counter()
        seen_keys: set[tuple[str, int]] = set()
        source_prompt_hashes: set[str] = set()
        with (SOURCE_DIR / 'sft_train.jsonl').open('r', encoding='utf-8', newline='') as source, \
             (OUTPUT_DIR / 'sft_train.jsonl').open('w', encoding='utf-8', newline='') as output:
            for raw_line in source:
                ending = newline_of(raw_line)
                text = raw_line[:-len(ending)] if ending else raw_line
                row = json.loads(text)
                if len(row.get('messages', [])) != 2 or row['messages'][0].get('role') != 'system':
                    raise ValueError(f'Unexpected SFT message layout at line {row_count + 1}')
                source_prompt_hashes.add(hashlib.sha256(row['messages'][0]['content'].encode('utf-8')).hexdigest())
                row['messages'][0]['content'] = prompt
                key = (row['sample_id'], row['step_index'])
                if key in seen_keys:
                    raise ValueError(f'Duplicate trajectory step: {key}')
                seen_keys.add(key)
                target = row['target']
                target_counts['STOP' if target.get('action') == 'stop' else target.get('tool', '<missing>')] += 1
                output.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + ending)
                row_count += 1

        if row_count != source_manifest['sft_examples']:
            raise ValueError(f'Expected {source_manifest["sft_examples"]} rows, found {row_count}')
        if source_prompt_hashes != {source_manifest['system_prompt_sha256']}:
            raise ValueError('Source JSONL has unexpected/mixed system prompts')
        if dict(sorted(target_counts.items())) != source_manifest['target_counts']:
            raise ValueError('Target distribution changed during prompt-only transformation')

        new_manifest = dict(source_manifest)
        new_manifest.update({
            'dataset_variant': 'english-system-prompt-v1',
            'parent_dataset_dir': 'kaggle-upload',
            'parent_sft_train_sha256': source_sft_hash,
            'system_prompt_sha256': prompt_hash,
            'sft_train_sha256': sha256_file(OUTPUT_DIR / 'sft_train.jsonl'),
            'prompt_language': 'en',
            'prompt_source': 'src/lifemate/agent.py Agent.next_action() English branch; /no_think removed for thinking-enabled training',
            'input_serialization': 'Agent.next_action English production prompt; Vietnamese transcript and unchanged context, history, compact tool schemas, and ToolEnv observations',
            'prompt_variant_changes_only_system_prompt': True,
            'cot_targets_added': False,
        })
        (OUTPUT_DIR / 'dataset_manifest.json').write_text(
            json.dumps(new_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        readme = '''# LifeMate Qwen3.5 SFT — English system prompt variant v1

This is a separate prompt-language variant of `kaggle-upload`. The parent package is unchanged.

## What changed

Only `messages[0].content` (the system prompt) in `sft_train.jsonl` changed. It now uses the English branch from `Agent.next_action()` in `src/lifemate/agent.py`, with `/no_think` removed because the Qwen chat template is trained with thinking enabled. This is a language-alignment experiment; no CoT checklist or CoT target was added.

The Vietnamese user transcript, context, conversation history, available-tool schemas, ToolEnv observations, next-action targets, row ordering, split IDs, and val309 inference inputs are unchanged. The training set remains 3,873 action-only examples from the fixed 1,236 train trajectories; the 309 validation labels remain excluded.

## Files

- `sft_train.jsonl`: English-system-prompt SFT rows.
- `train_ids.json`, `val309_ids.json`: unchanged split IDs.
- `val309_inputs.jsonl`: unchanged label-free validation inputs.
- `dataset_manifest.json`: source/variant provenance and hashes.

## Kaggle

Upload this folder as a separate private Kaggle Dataset, suggested slug `daoviet/lifemate-qwen35-sft-data-en-v1`. When running the continuation notebook, attach this dataset and detach the Vietnamese `lifemate-qwen35-sft-data-v1` dataset so that only one `sft_train.jsonl` is discovered. The current Vietnamese continuation notebook pins the old dataset hash; its English variant must use the new manifest hash.
'''
        (OUTPUT_DIR / 'README.md').write_text(readme, encoding='utf-8')

        with (SOURCE_DIR / 'sft_train.jsonl').open('r', encoding='utf-8', newline='') as before, \
             (OUTPUT_DIR / 'sft_train.jsonl').open('r', encoding='utf-8', newline='') as after:
            for line_number, (before_line, after_line) in enumerate(zip(before, after), start=1):
                before_row = json.loads(before_line)
                after_row = json.loads(after_line)
                expected_after = copy.deepcopy(before_row)
                expected_after['messages'][0]['content'] = prompt
                if expected_after != after_row:
                    raise ValueError(f'Unexpected non-prompt change at SFT line {line_number}')
            if next(before, None) is not None or next(after, None) is not None:
                raise ValueError('Source and English variant row counts differ')

        for name in ['train_ids.json', 'val309_ids.json', 'val309_inputs.jsonl']:
            if sha256_file(SOURCE_DIR / name) != sha256_file(OUTPUT_DIR / name):
                raise ValueError(f'Unexpected change in copied file: {name}')

        with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(OUTPUT_DIR.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(OUTPUT_DIR).as_posix())
        print(json.dumps({
            'output_dir': str(OUTPUT_DIR),
            'zip': str(ZIP_PATH),
            'rows': row_count,
            'source_sft_sha256': source_sft_hash,
            'english_system_prompt_sha256': prompt_hash,
            'english_sft_sha256': new_manifest['sft_train_sha256'],
            'train_ids_unchanged': True,
            'val_inputs_unchanged': True,
            'targets_and_user_inputs_unchanged': True,
        }, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        raise


if __name__ == '__main__':
    main()