from __future__ import annotations
import hashlib, importlib.util, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp" / "quan-map-correction-val60-v1"
OUT = EXP / "output" / "benchmark"
COHORT = ROOT / "exp/inference-latency-optimization-val60/output/cohort_manifest.json"
PREDICTIONS = ROOT / "exp/quan-10epoch-transcript-dictionary-v1/output/validation/epoch-06/predictions.jsonl"
DICTIONARY = ROOT / "exp/quan-10epoch-transcript-dictionary-v1/output/dictionary/train_candidates.json"
NORMALIZER = ROOT / "exp/quan-10epoch-transcript-dictionary-v1/scripts/vietnamese_number_normalizer.py"
NORMALIZATION_RUNNER = ROOT / "exp/quan-transcript-normalization-val60/scripts/run_benchmark.py"
BATCH_ORDER = [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot load {path}")
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result; spec.loader.exec_module(result)
    return result

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tokens(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

def normalize(text, normalizer):
    result, _, spans = normalizer.normalize_number_phrases(tokens(text))
    return result, spans

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]

def align(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]), dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    out, i, j = [], n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append(("match" if a[i - 1] == b[j - 1] else "substitution", i - 1, j - 1)); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            out.append(("deletion", i - 1, None)); i -= 1
        else:
            out.append(("insertion", None, j - 1)); j -= 1
    return list(reversed(out))

def corpus_wer(rows):
    errors = sum(edit_distance(hyp, ref) for _, ref, hyp in rows)
    words = sum(len(ref) for _, ref, _ in rows)
    macro = sum(edit_distance(hyp, ref) / max(1, len(ref)) for _, ref, hyp in rows) / max(1, len(rows))
    return {"corpus_wer": errors / max(1, words), "word_errors": errors, "reference_words": words,
            "utterances": len(rows), "macro_utterance_wer": macro}

def choose_rules(data):
    meta = data.get("metadata", {})
    if meta.get("train_ids") != 1236 or meta.get("validation_ids_excluded") != 309:
        raise RuntimeError("Map is not the train-only 1236/309 candidate artifact")
    possible = [r for r in data["candidates"] if r.get("eligible_for_context_review")
                and r.get("confidence") in {"high", "medium"}
                and r.get("candidate_type") in {"lexical_substitution", "numeric_value_substitution"}]
    grouped = defaultdict(list)
    for row in possible:
        source, target = tuple(tokens(row["wrong_phrase"])), tuple(tokens(row["correct_phrase"]))
        if source and target: grouped[source].append({**row, "source_tokens": list(source), "target_tokens": list(target)})
    rules, excluded = [], []
    for source, rows in grouped.items():
        targets = {tuple(row["target_tokens"]) for row in rows}
        if len(targets) != 1:
            excluded.extend({"wrong_phrase": row["wrong_phrase"], "reason": "ambiguous_target"} for row in rows)
        else:
            rules.append(max(rows, key=lambda r: (r["occurrences"], r["dominance_for_wrong_phrase"])))
    rules.sort(key=lambda r: (-len(r["source_tokens"]), -r["occurrences"], r["wrong_phrase"]))
    return rules, excluded

def apply_map(words, rules):
    lookup = {tuple(r["source_tokens"]): r for r in rules}
    lengths = sorted({len(key) for key in lookup}, reverse=True)
    output, apps, i = [], [], 0
    while i < len(words):
        rule = next((lookup[tuple(words[i:i+n])] for n in lengths
                     if len(words[i:i+n]) == n and tuple(words[i:i+n]) in lookup), None)
        if rule is None:
            output.append(words[i]); i += 1; continue
        end = i + len(rule["source_tokens"])
        output.extend(rule["target_tokens"])
        apps.append({"start": i, "end": end, "wrong_phrase": rule["wrong_phrase"],
            "correct_phrase": rule["correct_phrase"], "candidate_type": rule["candidate_type"],
            "occurrences_in_train": rule["occurrences"], "confidence": rule["confidence"],
            "target_tokens": rule["target_tokens"]})
        i = end
    return output, apps

def check_applications(source, reference, apps, sid):
    hyp_to_ref = {hyp_index: ref_index for op, hyp_index, ref_index in align(source, reference) if hyp_index is not None and ref_index is not None}
    checked = []
    for app in apps:
        ri = [hyp_to_ref.get(i) for i in range(app["start"], app["end"])]
        contiguous = all(x is not None for x in ri) and all(ri[i] + 1 == ri[i+1] for i in range(len(ri)-1))
        ref_span = reference[ri[0]:ri[-1]+1] if contiguous and ri else []
        checked.append({"id": sid, **app, "reference_span_at_original_alignment": " ".join(ref_span),
                        "exactly_correct": bool(contiguous and ref_span == app["target_tokens"])})
    return checked

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inference_artifacts = [path.name for path in OUT.iterdir() if path.name.startswith(("predictions-", "step-traces-", "requests-")) or path.name in {"inference_run.json", "run-progress.json"}]
    if inference_artifacts: raise FileExistsError(f"Refusing to overwrite inference artifacts: {inference_artifacts}")
    normalizer = module("val60_number_normalizer", NORMALIZER)
    cohort = read_json(COHORT); ids = cohort["ids"]
    val_ids = json.loads((ROOT / "data/splits/val_ids.json").read_text(encoding="utf-8-sig"))
    train_ids = json.loads((ROOT / "data/splits/train_ids.json").read_text(encoding="utf-8-sig"))
    if len(ids) != 60 or len(set(ids)) != 60 or set(ids) - set(val_ids) or set(ids) & set(train_ids):
        raise RuntimeError("Cohort is not a disjoint 60-ID held-out validation subset")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != cohort["ids_sha256_newline_joined"]:
        raise RuntimeError("Locked cohort ID hash/order mismatch")
    pred_rows = read_jsonl(PREDICTIONS)
    if [r["id"] for r in pred_rows] != val_ids: raise RuntimeError("QUAN epoch-6 predictions are not exact val309 order")
    pred = {r["id"]: r["prediction"] for r in pred_rows}
    if any(not pred.get(sid) for sid in ids): raise RuntimeError("Empty QUAN prediction in the cohort")

    dictionary = read_json(DICTIONARY)
    rules, excluded = choose_rules(dictionary)
    if not rules: raise RuntimeError("No eligible unambiguous train-only map rules")
    write_json(OUT / "map_rules.json", {
        "source": str(DICTIONARY.relative_to(ROOT)), "source_sha256": sha(DICTIONARY),
        "train_ids_sha256": dictionary["metadata"]["train_ids_sha256"],
        "train_ids": dictionary["metadata"]["train_ids"], "validation_ids_excluded": 309,
        "policy": "Only eligible medium/high-confidence lexical/numeric candidates; unique target per source; longest exact phrase first.",
        "number_normalization_before_map": dictionary["metadata"]["number_normalization"],
        "rules": [{k:r[k] for k in ("wrong_phrase","correct_phrase","candidate_type","occurrences","distinct_train_ids",
                                    "dominance_for_wrong_phrase","confidence","source_tokens","target_tokens")} for r in rules],
        "excluded_ambiguous_rules": excluded})

    # Freeze QUAN-derived branch A and its number-normalized/map-corrected branch B before loading held-out references.
    pre_map, corrected, applications, n_spans = {}, {}, {}, 0
    for sid in ids:
        word_tokens, count = normalize(pred[sid], normalizer)
        out, apps = apply_map(word_tokens, rules)
        pre_map[sid], corrected[sid], applications[sid] = word_tokens, out, apps
        n_spans += count

    runner_mod = module("val60_prepare_runner", NORMALIZATION_RUNNER)
    runner = runner_mod.init_runner()
    data_root = runner.find_data_root(ROOT / "train+public_test" / "train+public_test")
    manifest = data_root / "train" / "train.jsonl"
    items = {r["id"]: r for r in runner.load_jsonl(manifest)}
    if set(ids) - items.keys(): raise RuntimeError("Official BTC manifest is missing cohort IDs")
    references = {sid: items[sid].get("audio_text", "") for sid in ids}
    if any(not references[sid] for sid in ids): raise RuntimeError("Empty BTC reference script")
    by_pred = {r["id"]: r for r in pred_rows}
    if any(by_pred[sid].get("reference") != references[sid] for sid in ids):
        raise RuntimeError("BTC script mismatch between QUAN val output and official train manifest")

    project_A, project_B, project_C, value_A, value_B, value_C = [], [], [], [], [], []
    map_rows, branch_rows = [], []
    for sid in ids:
        ref = references[sid]
        ref_project = tokens(ref)
        ref_value, _ = normalize(ref, normalizer)
        apps = check_applications(pre_map[sid], ref_value, applications[sid], sid)
        map_rows.extend(apps)
        raw = pred[sid]
        map_text = " ".join(corrected[sid])
        project_A.append((sid, ref_project, tokens(raw)))
        project_B.append((sid, ref_project, tokens(map_text)))
        project_C.append((sid, ref_project, tokens(ref)))
        value_A.append((sid, ref_value, pre_map[sid]))
        value_B.append((sid, ref_value, corrected[sid]))
        value_C.append((sid, ref_value, ref_value))
        branch_rows.append({"id": sid, "A_quan_raw": raw, "B_quan_number_normalized_then_map": map_text,
                            "C_btc_original_script": ref, "number_spans_normalized_before_map":
                            sum(1 for _ in range(0)) + normalize(raw, normalizer)[1],
                            "map_application_count": len(apps)})
    correct = sum(x["exactly_correct"] for x in map_rows)
    types = Counter(x["candidate_type"] for x in map_rows)
    n_apps = len(map_rows)
    transcript_report = {
        "cohort": "fixed enriched subset of held-out val309; screening cohort, not unbiased val estimate",
        "cohort_ids_sha256_newline_joined": cohort["ids_sha256_newline_joined"],
        "WER_project_normalization": {"rule": "lowercase Unicode word tokens; punctuation removed; number words retained",
            "A_QUAN_raw": corpus_wer(project_A), "B_number_normalization_plus_map": corpus_wer(project_B),
            "C_BTC_script": corpus_wer(project_C)},
        "WER_number_value_normalization": {"rule": "same Vietnamese cardinal parser on references and transcripts before WER",
            "A_QUAN_raw": corpus_wer(value_A), "B_number_normalization_plus_map": corpus_wer(value_B),
            "C_BTC_script": corpus_wer(value_C)},
        "map_accuracy": {"definition": "exact map output equals BTC reference phrase at the original QUAN-to-reference aligned span",
            "applications": n_apps, "exactly_correct": correct, "incorrect_or_unaligned": n_apps-correct,
            "precision": correct/n_apps if n_apps else None, "changed_utterances": len({x["id"] for x in map_rows}),
            "by_type": dict(types), "number_spans_normalized_before_map": n_spans},
        "map_rule_selection": "train_ids only; eligible medium/high-confidence lexical or numeric-value pairs with unique targets; val text not used to create/select rules",
        "BTC_reference_source": str(manifest.relative_to(ROOT)),
        "heldout_label_use": "Only for WER, map correction precision, oracle branch C, and requested BTC scoring."
    }
    write_json(OUT/"transcript_analysis.json", transcript_report)
    write_jsonl(OUT/"map_applications.jsonl", map_rows)
    write_jsonl(OUT/"branch_transcripts.jsonl", branch_rows)
    write_jsonl(OUT/"gold_references.jsonl", [{"id":sid,"reference":references[sid]} for sid in ids])

    # Lock the same SFT epoch-2 + CoT downstream settings used by the prior QUAN normalization val60 screen.
    runner = runner_mod.init_runner()
    version, digest = runner.model_fingerprint()
    base_cfg = read_json(runner_mod.BASELINE)
    prompt_hash = runner.canonical_prompt_hash(True)
    if digest != runner.EXPECTED_DIGEST or base_cfg["branches"]["C"]["model_digest"] != digest:
        raise RuntimeError("Qwen SFT epoch-2 model digest mismatch")
    if prompt_hash != base_cfg["agent"]["cot_prompt_sha256_utf8"]:
        raise RuntimeError("Qwen CoT prompt hash changed since the earlier val60 benchmark")
    if read_json(runner_mod.SMOKE).get("status") != "passed": raise RuntimeError("Locked Qwen smoke gate is not passed")

    lock = {"status":"prepared","experiment":"QUAN map correction and BTC downstream val60",
        "cohort":{"samples":60,"ids":ids,"source":str(COHORT.relative_to(ROOT)),
                  "ids_sha256_newline_joined":cohort["ids_sha256_newline_joined"],"heldout_val309_subset":True},
        "asr":{"checkpoint":"exp/quan-10epoch-transcript-dictionary-v1/output/checkpoints/epoch-06",
               "predictions":str(PREDICTIONS.relative_to(ROOT)),"sha256":sha(PREDICTIONS)},
        "map":{"source":str(DICTIONARY.relative_to(ROOT)),"sha256":sha(DICTIONARY),"selected_rules":len(rules),
               "number_parser_before_map":True,"map_accuracy_selection_uses_val":False},
        "qwen":{"model":runner.MODEL,"digest":digest,"ollama_version":version,"thinking":True,"protocol":"JSON",
                "temperature":0,"num_predict":runner.CAP,"num_ctx":runner.CONTEXT,"num_batch":runner.BATCH,
                "max_action_turns":runner.MAX_STEPS,"prompt":"observation-grounded-v2 English + existing CoT checklist",
                "prompt_sha256_utf8":prompt_hash,"history_asr":"PhoASR unchanged"},
        "toolenv":{"one_prediction_ToolEnv_per_sample_per_branch":True,
                   "official_score_replay":"exp/btc-scoring/scripts/score_btc.py"},
        "number_parser":{"script":str(NORMALIZER.relative_to(ROOT)),"sha256":sha(NORMALIZER),
                         "examples":{"hai mươi lăm":"25","bốn mươi lăm":"45","năm mươi":"50"}},
        "source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [
            COHORT,PREDICTIONS,DICTIONARY,NORMALIZER,NORMALIZATION_RUNNER,
            ROOT/"exp/quan-continuation-cot-v1/scripts/run_downstream.py",
            ROOT/"exp/quan-continuation-cot-v1/scripts/analyze_downstream.py",
            ROOT/"exp/btc-scoring/scripts/score_btc.py",runner_mod.BASELINE,runner_mod.SMOKE,manifest]},
        "execution_environment":"native system Python outside project .venv; local Ollama",
        "public_private_test_used":False,"raw_chain_of_thought_saved":False,"branch_order":[["A","B","C"],["B","C","A"],["C","A","B"]]}
    write_json(OUT/"benchmark_lock.json",lock)
    write_json(OUT/"cohort_manifest.json",cohort)
    write_json(OUT/"execution_order.json",[{"round":i+1,"ids":ids[i*20:(i+1)*20],"branch_order":BATCH_ORDER[i]} for i in range(3)])
    print(json.dumps({"status":"prepared","samples":60,"map_rules":len(rules),"map_applications":n_apps,
        "map_correct":correct,"map_precision":correct/n_apps if n_apps else None,
        "WER_A_number_normalized":transcript_report["WER_number_value_normalization"]["A_QUAN_raw"]["corpus_wer"],
        "WER_B_number_normalized":transcript_report["WER_number_value_normalization"]["B_number_normalization_plus_map"]["corpus_wer"],
        "Qwen":runner.MODEL,"Ollama":version},ensure_ascii=True,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())

