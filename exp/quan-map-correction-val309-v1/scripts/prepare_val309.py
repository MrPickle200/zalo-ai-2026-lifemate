from __future__ import annotations
import hashlib, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "exp/quan-map-correction-val60-v1/scripts"))
import prepare_benchmark as prep

EXP = ROOT / "exp/quan-map-correction-val309-v1"
OUT = EXP / "output" / "benchmark"
VAL_IDS = ROOT / "data/splits/val_ids.json"
A_PRED = ROOT / "exp/quan-10epoch-transcript-dictionary-v1/output/validation/epoch-06/predictions.jsonl"
D_PRED = ROOT / "exp/quan-continuation-cot-v1/output/asr/epoch-02-val309/predictions.jsonl"
A_CKPT = ROOT / "exp/quan-10epoch-transcript-dictionary-v1/output/checkpoints/epoch-06/model.safetensors"
D_CKPT = ROOT / "exp/quan-continuation-cot-v1/output/checkpoints/epoch-02/model.safetensors"
MAP_LOCK = ROOT / "exp/quan-map-correction-val60-v1/output/benchmark/map_rules.json"
NORMALIZER = ROOT / "exp/quan-10epoch-transcript-dictionary-v1/scripts/vietnamese_number_normalizer.py"
NORMALIZATION_RUNNER = ROOT / "exp/quan-transcript-normalization-val60/scripts/run_benchmark.py"
DATA_ROOT_HINT = ROOT / "train+public_test" / "train+public_test"
BLOCK_SIZE = 20
BRANCHES = ("A", "B", "C", "D")

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty val309 benchmark output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    ids = prep.read_json(VAL_IDS)
    if len(ids) != 309 or len(set(ids)) != 309:
        raise RuntimeError(f"Expected exactly 309 unique validation IDs, got {len(ids)}")
    train_ids = prep.read_json(ROOT / "data/splits/train_ids.json")
    if set(ids) & set(train_ids):
        raise RuntimeError("Validation IDs overlap train IDs")

    a_rows, d_rows = prep.read_jsonl(A_PRED), prep.read_jsonl(D_PRED)
    if [row["id"] for row in a_rows] != ids or [row["id"] for row in d_rows] != ids:
        raise RuntimeError("A/D transcript caches are not exact val309 order")
    a_map = {row["id"]: row["prediction"] for row in a_rows}
    d_map = {row["id"]: row["prediction"] for row in d_rows}
    if any(not a_map[sid] or not d_map[sid] for sid in ids):
        raise RuntimeError("A/D transcript cache contains an empty prediction")

    # Reuse the exact train-only rules frozen for the preceding val60 screen.
    frozen_map = prep.read_json(MAP_LOCK)
    if frozen_map["source_sha256"] != digest(ROOT / "exp/quan-10epoch-transcript-dictionary-v1/output/dictionary/train_candidates.json"):
        raise RuntimeError("Train dictionary changed since the val60 map was frozen")
    rules = frozen_map["rules"]
    if len(rules) != 17:
        raise RuntimeError(f"Expected the previously locked 17 map rules, got {len(rules)}")

    runner_module = prep.module("quan_val309_prepare_runner", NORMALIZATION_RUNNER)
    runner_module.OUT = OUT
    runner = runner_module.init_runner()
    runner.OUT = OUT
    data_root = runner.find_data_root(DATA_ROOT_HINT)
    manifest = data_root / "train" / "train.jsonl"
    items = {row["id"]: row for row in runner.load_jsonl(manifest)}
    if set(ids) - items.keys():
        raise RuntimeError("BTC train manifest is missing validation IDs")
    refs = {sid: items[sid].get("audio_text", "") for sid in ids}
    if any(not refs[sid] for sid in ids):
        raise RuntimeError("BTC script is empty for at least one validation ID")
    for sid in ids:
        for rows in (a_rows, d_rows):
            row = rows[ids.index(sid)]
            if row.get("reference") is not None and row["reference"] != refs[sid]:
                raise RuntimeError(f"Cached ASR reference differs from BTC script for {sid}")

    normalizer = prep.module("quan_val309_number_normalizer", NORMALIZER)
    transcripts, map_applications = [], []
    project_rows = {branch: [] for branch in BRANCHES}
    number_rows = {branch: [] for branch in BRANCHES}
    normalized_spans = 0
    for sid in ids:
        raw_a, raw_d, ref = a_map[sid], d_map[sid], refs[sid]
        a_words, a_spans = prep.normalize(raw_a, normalizer)
        d_words, _ = prep.normalize(raw_d, normalizer)
        ref_words, _ = prep.normalize(ref, normalizer)
        b_words, apps = prep.apply_map(a_words, rules)
        b_text = " ".join(b_words)
        checked = prep.check_applications(a_words, ref_words, apps, sid)
        map_applications.extend(checked)
        normalized_spans += a_spans

        values = {
            "A_quan_epoch06_raw": raw_a,
            "B_epoch06_number_normalized_then_train_map": b_text,
            "C_btc_original_script_oracle": ref,
            "D_quan_epoch02_raw": raw_d,
        }
        transcripts.append({"id": sid, **values})
        project_refs = prep.tokens(ref)
        project_hypotheses = {
            "A": prep.tokens(raw_a), "B": prep.tokens(b_text),
            "C": prep.tokens(ref), "D": prep.tokens(raw_d),
        }
        number_hypotheses = {"A": a_words, "B": b_words, "C": ref_words, "D": d_words}
        for branch in BRANCHES:
            project_rows[branch].append((sid, project_refs, project_hypotheses[branch]))
            number_rows[branch].append((sid, ref_words, number_hypotheses[branch]))

    n_apps = len(map_applications)
    n_correct = sum(row["exactly_correct"] for row in map_applications)
    from collections import Counter
    map_report = {
        "definition": "exact map output equals BTC reference phrase at the original QUAN-to-reference aligned span",
        "applications": n_apps, "exactly_correct": n_correct, "incorrect_or_unaligned": n_apps - n_correct,
        "precision": n_correct / n_apps if n_apps else None,
        "changed_utterances": len({row["id"] for row in map_applications}),
        "by_type": dict(Counter(row["candidate_type"] for row in map_applications)),
        "number_spans_normalized_before_map": normalized_spans,
        "rules_source": "frozen train-only map from quan-map-correction-val60-v1",
        "selected_rules": len(rules),
    }
    transcript_report = {
        "split": "entire held-out validation set", "samples": len(ids),
        "ids_sha256_newline_joined": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "WER_project_normalization": {
            "rule": "lowercase Unicode word tokens; punctuation removed; number words retained",
            **{branch: prep.corpus_wer(project_rows[branch]) for branch in BRANCHES},
        },
        "WER_number_value_normalization": {
            "rule": "same Vietnamese cardinal parser on references and transcripts before WER",
            **{branch: prep.corpus_wer(number_rows[branch]) for branch in BRANCHES},
        },
        "map_accuracy": map_report,
        "BTC_reference_source": str(manifest.relative_to(ROOT)),
        "heldout_label_use": "Scoring/reporting only; the map was frozen from train-only data before this val309 comparison.",
    }

    val_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    blocks = []
    for start in range(0, len(ids), BLOCK_SIZE):
        round_index = len(blocks) + 1
        block_ids = ids[start:start + BLOCK_SIZE]
        order = [BRANCHES[(i + round_index - 1) % len(BRANCHES)] for i in range(len(BRANCHES))]
        blocks.append({"round": round_index, "ids": block_ids, "branch_order": order})
    execution_order = {
        "block_size": BLOCK_SIZE, "blocks": blocks,
        "branch_order_rotation": list(BRANCHES),
        "paired_by_id": True, "all_validation_ids": True,
    }

    version, model_digest = runner.model_fingerprint()
    config = prep.read_json(runner_module.BASELINE)
    prompt_hash = runner.canonical_prompt_hash(True)
    expected_digest = config["branches"]["C"]["model_digest"]
    expected_prompt = config["agent"]["cot_prompt_sha256_utf8"]
    if model_digest != expected_digest or version != "0.40.15":
        raise RuntimeError(f"Locked Qwen/Ollama fingerprint mismatch: {version}, {model_digest}")
    if prompt_hash != expected_prompt:
        raise RuntimeError("Qwen CoT prompt hash differs from preceding benchmark")
    if prep.read_json(runner_module.SMOKE).get("status") != "passed":
        raise RuntimeError("Qwen model smoke gate is not passed")

    project_root_rel = [
        VAL_IDS, A_PRED, D_PRED, A_CKPT, D_CKPT, MAP_LOCK, NORMALIZER,
        NORMALIZATION_RUNNER, ROOT / "exp/quan-continuation-cot-v1/scripts/run_downstream.py",
        ROOT / "exp/quan-continuation-cot-v1/scripts/analyze_downstream.py",
        ROOT / "exp/btc-scoring/scripts/score_btc.py", runner_module.BASELINE,
        runner_module.SMOKE, manifest,
    ]
    lock = {
        "status": "prepared", "experiment": "QUAN map correction full val309, four-branch comparison",
        "split": {"name": "val", "samples": 309, "ids_sha256_newline_joined": val_hash,
                  "source": str(VAL_IDS.relative_to(ROOT))},
        "branches": {
            "A": {"input": "raw QUAN epoch-06 transcript", "prediction_file": str(A_PRED.relative_to(ROOT)),
                  "checkpoint": str(A_CKPT.relative_to(ROOT)), "checkpoint_sha256": digest(A_CKPT)},
            "B": {"input": "A transcript; Vietnamese number normalization then frozen train-only map",
                  "map_source": str(MAP_LOCK.relative_to(ROOT)), "map_source_sha256": digest(MAP_LOCK),
                  "map_rules": len(rules)},
            "C": {"input": "original BTC audio_text reference script (oracle transcript)"},
            "D": {"input": "raw QUAN-transcribe continuation epoch-02 transcript",
                  "prediction_file": str(D_PRED.relative_to(ROOT)), "checkpoint": str(D_CKPT.relative_to(ROOT)),
                  "checkpoint_sha256": digest(D_CKPT)},
        },
        "qwen": {"model": runner.MODEL, "digest": model_digest, "ollama_version": version,
                 "thinking": True, "protocol": "JSON", "temperature": 0, "num_predict": runner.CAP,
                 "num_ctx": runner.CONTEXT, "num_batch": runner.BATCH, "max_action_turns": runner.MAX_STEPS,
                 "prompt": "observation-grounded-v2 English + existing CoT checklist",
                 "prompt_sha256_utf8": prompt_hash, "history_asr": "PhoASR unchanged"},
        "toolenv": {"one_prediction_ToolEnv_per_sample_per_branch": True,
                    "official_score_replay": "exp/btc-scoring/scripts/score_btc.py"},
        "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in project_root_rel},
        "execution_environment": "native system Python outside project .venv; local Ollama",
        "data_policy": "val labels used only for reported WER, map audit, oracle input and official BTC evaluation; no training/tuning",
        "raw_chain_of_thought_saved": False,
    }

    write_json(OUT / "cohort_manifest.json", {"ids": ids, "samples": len(ids), "split": "val",
        "source": str(VAL_IDS.relative_to(ROOT)), "ids_sha256_newline_joined": val_hash, "is_entire_split": True})
    write_json(OUT / "execution_order.json", execution_order)
    write_json(OUT / "transcript_analysis.json", transcript_report)
    write_jsonl(OUT / "branch_transcripts.jsonl", transcripts)
    write_jsonl(OUT / "map_applications.jsonl", map_applications)
    write_jsonl(OUT / "gold_references.jsonl", [{"id": sid, "reference": refs[sid]} for sid in ids])
    write_json(OUT / "benchmark_lock.json", lock)
    print(json.dumps({"status": "prepared", "samples": len(ids), "blocks": len(blocks),
        "branches": list(BRANCHES), "map_rules": len(rules), "map_applications": n_apps,
        "map_correct": n_correct, "Qwen": runner.MODEL, "Ollama": version}, ensure_ascii=True, indent=2))

if __name__ == "__main__":
    main()

