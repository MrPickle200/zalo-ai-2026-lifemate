from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
EXP = ROOT / "exp" / "qwen35-sft-epoch3-cot-val60"
OUT = EXP / "output"
INFER = OUT / "inference"
PROV = OUT / "model-provenance"
RUNNER_PATH = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "run_downstream.py"
ANALYZER_PATH = ROOT / "exp" / "quan-continuation-cot-v1" / "scripts" / "analyze_downstream.py"
OLD_EXP = ROOT / "exp" / "qwen35-sft-benchmark-v1"
OLD_OUT = OLD_EXP / "output" / "inference"
OLD_PROV = OLD_EXP / "output" / "model-provenance"
MODEL_TAG = "lifemate-qwen35-kagglev1-sft-epoch3-q4km:latest"
SFT_TRAIN = ROOT / "exp" / "qwen35-sft-next-action-v1" / "kaggle-upload" / "sft_train.jsonl"
SMOKE_ID = "train_0050"
EXPECTED_COT_HASH = "cc3dc621c74578bc4bda30fa74ea4cc59e916644487bdba4964b652ece4d9950"
EXPECTED_QUAN_HASH = "e31be8516152587134fb62e0d1690954221a0f02c53b6549cf79db58c8eb18ae"
EXPECTED_IDS_HASH = "87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def load_runner():
    spec = importlib.util.spec_from_file_location("epoch3_cot_shared_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    ollama = read_json(PROV / "ollama-models.json")
    runner.OUT = INFER
    runner.MODEL = ollama["tag"]
    runner.EXPECTED_DIGEST = ollama["ollama_digest"]
    return runner, ollama


def prepare() -> int:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use system Python outside .venv to reach local Ollama")
    if (INFER / "run_lock.json").exists():
        raise FileExistsError("Epoch-3 inference lock already exists; refusing to overwrite")
    audit = read_json(PROV / "local-audit.json")
    merge = read_json(PROV / "merge-manifest.json")
    conversion = read_json(PROV / "conversion-manifest.json")
    ollama = read_json(PROV / "ollama-models.json")
    old_config = read_json(OLD_OUT / "config_manifest.json")
    old_run = read_json(OLD_OUT / "run-COT.json")
    old_metrics = read_json(OLD_OUT / "metrics-COT.json")
    if merge.get("status") != "merged" or not merge.get("merge", {}).get("probe_same_argmax"):
        raise RuntimeError("Epoch-3 PEFT merge/logit gate did not pass")
    if conversion.get("status") != "converted_and_quantized" or ollama.get("status") != "registered":
        raise RuntimeError("Epoch-3 conversion or Ollama registration is incomplete")
    if old_run.get("status") != "complete" or old_metrics.get("samples") != 60:
        raise RuntimeError("The reusable epoch-2 branch C artifacts are incomplete")
    if old_metrics.get("btc_formula_score_before_runtime_gate_0_100") != 26.0:
        raise RuntimeError("Unexpected reused branch-C baseline score")

    config_path = INFER / "config_manifest.json"
    cohort = read_json(INFER / "cohort_manifest.json")
    runner, model = load_runner()
    if config_path.exists():
        config = read_json(config_path)
        if config.get("experiment") != "quan-continuation-cot-v1" or config.get("status") != "downstream_config_locked":
            raise RuntimeError("Existing partial preflight config is not the expected shared-runner output")
    else:
        runner.prepare()
        config = read_json(config_path)
    old_ids = read_json(OLD_OUT / "cohort_manifest.json")["ids"]
    ids_hash = hashlib.sha256("\n".join(cohort["ids"]).encode("utf-8")).hexdigest()
    if cohort["ids"] != old_ids or ids_hash != EXPECTED_IDS_HASH:
        raise RuntimeError("New C+CoT cohort/order differs from the completed epoch-2 branch C cohort")
    if (config["agent"].get("cot_prompt_sha256_utf8") != old_config["agent"].get("cot_prompt_sha256_utf8")
            or config["agent"]["cot_prompt_sha256_utf8"] != EXPECTED_COT_HASH):
        raise RuntimeError("The exact screened CoT prompt differs from old branch C")
    if config["agent"].get("current_prompt_sha256_utf8") != old_config["agent"].get("current_prompt_sha256_utf8"):
        raise RuntimeError("The English production base prompt differs from old branch C")
    for key in ("selected_checkpoint_sha256", "transcript_file_sha256", "kenlm_config_sha256"):
        if config["asr"].get(key) != old_config["asr"].get(key):
            raise RuntimeError(f"QUAN ASR input changed versus old C: {key}")
    if config["ollama"].get("version") != old_config["ollama"].get("version"):
        raise RuntimeError("Ollama version changed versus old branch C")
    if {key: config["ollama"][key] for key in ("thinking", "temperature", "protocol", "num_predict", "num_ctx", "num_batch", "seed_python")} != {
        key: old_config["ollama"][key] for key in ("thinking", "temperature", "protocol", "num_predict", "num_ctx", "num_batch", "seed_python")
    }:
        raise RuntimeError("Inference settings differ from the completed epoch-2 branch C")

    config["experiment"] = EXP.name
    config["status"] = "epoch3_cot_config_locked"
    config["branches"] = {
        "epoch2_C_reused": {"model": old_run["model"], "model_digest": old_run["model_digest"],
            "official_btc_0_100": old_metrics["btc_formula_score_before_runtime_gate_0_100"],
            "prediction_artifact": "exp/qwen35-sft-benchmark-v1/output/inference/predictions-COT.jsonl"},
        "epoch3_COT": {"model": model["tag"], "model_digest": model["ollama_digest"],
            "gguf_sha256": conversion["q4_k_m_gguf"]["sha256"], "base": audit["base"]["source"],
            "adapter_sha256": audit["adapter"]["final"]["sha256"],
            "prompt": "same English production prompt + exact screened CoT checklist"},
    }
    config["comparison"] = {"left": "reused old epoch-2 SFT + CoT branch C",
        "right": "new one-epoch continuation adapter + CoT", "paired_by_id": True,
        "old_C_rerun": False, "cohort_ids_sha256_newline_joined": ids_hash,
        "old_C_btc_formula_score_before_runtime_gate_0_100": 26.0}
    config.setdefault("model_provenance", {}).update({
        "new_adapter_archive_sha256": audit["archive"]["sha256"],
        "new_adapter_sha256": audit["adapter"]["final"]["sha256"],
        "continuation_source_adapter_sha256": audit["training_report"]["continuation_source_adapter_sha256"],
        "continuation_source_matches_epoch2": audit["adapter"]["matches_continuation_source_archive"],
        "sft_train_sha256": audit["training_report"]["sft_train_sha256"],
        "merge_manifest_sha256": sha256(PROV / "merge-manifest.json"),
        "conversion_manifest_sha256": sha256(PROV / "conversion-manifest.json"),
        "epoch3_gguf_sha256": conversion["q4_k_m_gguf"]["sha256"],
        "epoch2_C_reused_model_digest": old_run["model_digest"],
        "epoch3_model_digest": model["ollama_digest"],
        "remote_training_report_has_exact_base_weight_hash": False,
    })
    input_paths = [Path(__file__).resolve(), Path(__file__).with_name("analyze_epoch3.py"), RUNNER_PATH, ANALYZER_PATH,
        EXP / "scripts" / "audit_extract.py", EXP / "scripts" / "merge_adapter.py",
        EXP / "scripts" / "convert_model.py", EXP / "scripts" / "register_ollama.py",
        PROV / "local-audit.json", PROV / "kaggle-version1-sha256.json",
        PROV / "merge-manifest.json", PROV / "conversion-manifest.json", PROV / "ollama-models.json",
        OLD_OUT / "config_manifest.json", OLD_OUT / "cohort_manifest.json", OLD_OUT / "run-COT.json",
        OLD_OUT / "metrics-COT.json", OLD_OUT / "predictions-COT.jsonl",
        OLD_OUT / "btc-per-sample-COT.jsonl", OLD_OUT / "requests-COT.jsonl"]
    config["input_sha256"].update({str(path.relative_to(ROOT)): sha256(path) for path in input_paths})
    runner.write_json(config_path, config)
    lock = {"status": "locked", "created_epoch": time.time(), "cohort_samples": 60,
        "cohort_ids_sha256_newline_joined": ids_hash,
        "config_manifest_sha256": sha256(config_path), "cohort_manifest_sha256": sha256(INFER / "cohort_manifest.json"),
        "old_C_artifact_reused": True, "old_C_official_btc_0_100": 26.0,
        "epoch3_model_tag": model["tag"], "epoch3_model_digest": model["ollama_digest"],
        "cot_prompt_sha256_utf8": EXPECTED_COT_HASH, "ollama_version": config["ollama"]["version"],
        "settings": {key: config["ollama"][key] for key in ("thinking", "temperature", "protocol", "num_predict", "num_ctx", "num_batch", "seed_python")},
        "prediction_toolenvs_per_sample": 1, "public_private_test_used": False}
    write_json(INFER / "run_lock.json", lock)
    print(json.dumps({"status": lock["status"], "samples": 60, "ids_sha256": ids_hash,
        "epoch2_C_reused_btc": 26.0, "epoch3_tag": model["tag"],
        "epoch3_digest": model["ollama_digest"], "cot_prompt_sha256": EXPECTED_COT_HASH}, indent=2))
    return 0


def make_agent(runner, sample_id: str, cot: bool, round_index: int):
    from lifemate.agent import Agent
    from lifemate.utils import seed_everything

    os.environ.update({"OLLAMA_HOST": runner.HOST, "LIFEMATE_LLM_BACKEND": "ollama",
        "LIFEMATE_OLLAMA_MODEL": runner.MODEL, "LIFEMATE_OLLAMA_CONTEXT": str(runner.CONTEXT),
        "LIFEMATE_OLLAMA_NUM_PREDICT": str(runner.CAP), "LIFEMATE_OLLAMA_NUM_BATCH": str(runner.BATCH),
        "LIFEMATE_OLLAMA_THINK": "true", "LIFEMATE_OLLAMA_PROTOCOL": "json",
        "LIFEMATE_PROMPT_LANGUAGE": "en", "LIFEMATE_OLLAMA_TELEMETRY": "1"})
    seed_everything(runner.SEED)
    agent = Agent(max_new_tokens=runner.CAP, context_limit=runner.CONTEXT,
                  ollama_thinking=True, ollama_native_tools=False)
    agent._telemetry_sample_id = sample_id
    if cot:
        original_generate = agent._generate
        agent._generate = lambda messages: original_generate(runner.inject_cot(messages))
    original_request = agent._ollama_request

    def request_with_prompt_hash(messages, tools=None):
        start = len(agent.request_telemetry)
        system_message = next((message.get("content", "") for message in messages if message.get("role") == "system"), "")
        prompt_hash = hashlib.sha256(system_message.encode("utf-8")).hexdigest()
        try:
            return original_request(messages, tools)
        finally:
            for row in agent.request_telemetry[start:]:
                row["system_prompt_sha256_utf8"] = prompt_hash
                row["prompt_variant"] = "cot"
                row["model_digest"] = runner.EXPECTED_DIGEST
                row["round"] = round_index

    agent._ollama_request = request_with_prompt_hash
    return agent


def run_one(runner, agent, pipeline, item: dict[str, Any], transcript: str, round_index: int):
    import lifemate.pipeline as pipeline_module

    envs = []
    original_factory = pipeline_module.create_tool_env

    def tracked_factory(split, sample_id, data_root):
        env = original_factory(split, sample_id, data_root)
        envs.append(env)
        return env

    pipeline_module.create_tool_env = tracked_factory
    telemetry_start = len(agent.request_telemetry)
    action_turn = [0]
    captured_actions: list[dict[str, Any]] = []
    original_next = agent.next_action

    def tracked_next(*args, **kwargs):
        action_turn[0] += 1
        agent._telemetry_action_turn = action_turn[0]
        action = original_next(*args, **kwargs)
        captured_actions.append({key: value for key, value in action.items() if key != "_native_message"})
        return action

    agent.next_action = tracked_next
    started = time.perf_counter()
    error = None
    result = None
    try:
        result = pipeline.run(item, transcript_override=transcript)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        pipeline_module.create_tool_env = original_factory
        agent.next_action = original_next

    if len(envs) == 1:
        env = envs[0]
        executed = env.get_tool_calls()
        state_hash = env.get_state_hash()
        final_state = env.get_state()
    else:
        executed = result.get("tool_calls", []) if result else []
        state_hash = result.get("state_hash") if result else None
        final_state = result.get("final_state", {}) if result else {}
        if error is None:
            error = f"ToolEnv creation count was {len(envs)}, expected exactly one"
    observations = result.get("observations", []) if result else []
    model_output = result.get("model_output", []) if result else captured_actions
    latency = result.get("latency_seconds", time.perf_counter() - started) if result else time.perf_counter() - started
    calls = [{"name": call.get("name"), "arguments": call.get("arguments", {})} for call in executed]
    prediction = {"id": item["id"], "asr_text": transcript, "transcript_used": transcript,
        "tool_calls": calls, "state_hash": state_hash, "observations": observations,
        "final_state": final_state, "model_output": model_output, "latency_seconds": latency,
        "error": error, "toolenv_count": len(envs)}
    usage = agent.request_telemetry[telemetry_start:]
    for index, row in enumerate(usage, 1):
        row["request_id"] = f"E3-COT:{item['id']}:r{index}"
        row["sample_request_index"] = index
    trace = {"id": item["id"], "branch": "E3-COT", "round": round_index,
        "latency_seconds": latency, "runtime_error": error, "turns": model_output,
        "request_count": len(usage), "action_turns": len(model_output),
        "repair_request_count": sum(row.get("request_kind") == "repair" for row in usage),
        "toolenv_count": len(envs)}
    return prediction, trace, usage, error


def smoke() -> int:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use system Python outside .venv to reach local Ollama")
    runner, model = load_runner()
    ollama_version, digest = runner.model_fingerprint()
    if digest != model["ollama_digest"]:
        raise RuntimeError("Registered epoch-3 model digest changed")
    smoke_path = OUT / "smoke" / f"cot-{SMOKE_ID}.json"
    if smoke_path.exists():
        raise FileExistsError(smoke_path)
    dataset_row = next((json.loads(line) for line in SFT_TRAIN.read_text(encoding="utf-8").splitlines()
                        if json.loads(line).get("sample_id") == SMOKE_ID), None)
    if dataset_row is None:
        raise RuntimeError(f"Fixed smoke row {SMOKE_ID} is missing")
    user = [message for message in dataset_row.get("messages", []) if message.get("role") == "user"]
    if len(user) != 1:
        raise RuntimeError("Smoke row does not have exactly one user message")
    payload = json.loads(user[0]["content"])
    transcript = payload.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise RuntimeError("Smoke transcript is absent")
    from lifemate.asr import PhoASRModel
    from lifemate.pipeline import LifeMatePipeline
    from lifemate.utils import find_data_root, load_jsonl

    data_root = find_data_root("train+public_test/train+public_test")
    item = next((row for row in load_jsonl(data_root / "train" / "train.jsonl") if row["id"] == SMOKE_ID), None)
    if item is None:
        raise RuntimeError(f"Fixed official train ToolEnv row {SMOKE_ID} is missing")
    agent = make_agent(runner, SMOKE_ID, cot=True, round_index=0)
    pipeline = LifeMatePipeline(data_root, "train", asr=PhoASRModel(), agent=agent, max_steps=runner.MAX_STEPS)
    pred, trace, requests, error = run_one(runner, agent, pipeline, item, transcript, 0)
    if error:
        raise RuntimeError(f"Epoch-3 CoT smoke failed: {error}")
    if pred["toolenv_count"] != 1:
        raise RuntimeError(f"Smoke created {pred['toolenv_count']} ToolEnv instances")
    if not pred["tool_calls"]:
        raise RuntimeError("Fixed real-ToolEnv smoke emitted no call; inspect before cohort run")
    record = {"status": "passed", "sample_id": SMOKE_ID, "split": "official train",
        "target_used_for_prompt": False, "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "model_tag": model["tag"], "model_digest": digest, "ollama_version": ollama_version,
        "prompt_variant": "English production + exact CoT checklist", "toolenv_count": pred["toolenv_count"],
        "executed_tool_names": [call["name"] for call in pred["tool_calls"]],
        "state_hash": pred["state_hash"], "latency_seconds": pred["latency_seconds"],
        "request_count": len(requests), "raw_chain_of_thought_saved": False}
    write_json(smoke_path, record)
    print(json.dumps(record, indent=2))
    return 0


def run() -> int:
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use system Python outside .venv to reach local Ollama")
    runner, model = load_runner()
    config_path = INFER / "config_manifest.json"
    lock_path = INFER / "run_lock.json"
    config = read_json(config_path)
    lock = read_json(lock_path)
    if config.get("status") != "epoch3_cot_config_locked" or lock.get("status") != "locked":
        raise RuntimeError("Epoch-3 configuration is not locked")
    if lock.get("config_manifest_sha256") != sha256(config_path):
        raise RuntimeError("Configuration lock hash mismatch")
    if not (OUT / "smoke" / f"cot-{SMOKE_ID}.json").is_file():
        raise RuntimeError("Run the real ToolEnv smoke before cohort inference")
    runner.verify_locked_inputs(config)
    version, digest = runner.model_fingerprint()
    if version != config["ollama"]["version"] or digest != model["ollama_digest"]:
        raise RuntimeError("Ollama fingerprint changed after preflight")
    ids = config["cohort"]["ids"]
    if len(ids) != 60 or hashlib.sha256("\n".join(ids).encode()).hexdigest() != EXPECTED_IDS_HASH:
        raise RuntimeError("Locked epoch-3 cohort is not the expected ordered 60 IDs")
    for name in ("predictions-COT.jsonl", "step-traces-COT.jsonl", "requests-COT.jsonl", "run-COT.json"):
        if (INFER / name).exists():
            raise FileExistsError(f"Refusing to overwrite {INFER / name}")

    transcripts = {row["id"]: row["prediction"] for row in runner.read_jsonl(runner.ASR_PREDICTIONS)}
    if [row["id"] for row in runner.read_jsonl(runner.ASR_PREDICTIONS)] != json.loads(
            (ROOT / "data" / "splits" / "val_ids.json").read_text(encoding="utf-8-sig")):
        raise RuntimeError("QUAN transcript validation IDs/order changed")
    data_root = runner.find_data_root("train+public_test/train+public_test")
    items = {row["id"]: row for row in runner.load_jsonl(data_root / "train" / "train.jsonl")}
    if set(ids) - transcripts.keys() or set(ids) - items.keys():
        raise RuntimeError("Cohort input rows or transcript coverage is incomplete")

    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    run_started = time.time()
    blocks = [ids[index * 15:(index + 1) * 15] for index in range(4)]
    write_json(INFER / "epoch3-execution-order.json", {"branch": "E3-COT", "ids": ids,
        "blocks": [{"round": i + 1, "ids": block} for i, block in enumerate(blocks)],
        "paired_with_reused_old_C_by_id": True, "warmup_before_each_block": True,
        "old_C_rerun": False, "settings": lock["settings"]})
    for round_index, block in enumerate(blocks, 1):
        current_version, current_digest = runner.model_fingerprint()
        if current_version != version or current_digest != digest:
            raise RuntimeError("Epoch-3 Ollama fingerprint changed between blocks")
        warmup = runner.warmup("E3-COT", round_index)
        agent = make_agent(runner, block[0], cot=True, round_index=round_index)
        from lifemate.asr import PhoASRModel
        from lifemate.pipeline import LifeMatePipeline
        pipeline = LifeMatePipeline(data_root, "train", asr=PhoASRModel(), agent=agent,
                                    max_steps=runner.MAX_STEPS)
        block_started = time.time()
        for offset, sid in enumerate(block, 1):
            agent._telemetry_sample_id = sid
            pred, trace, usage, error = run_one(runner, agent, pipeline, items[sid], transcripts[sid], round_index)
            predictions.append(pred)
            traces.append(trace)
            requests.extend(usage)
            write_jsonl_atomic(INFER / "predictions-COT.jsonl", predictions)
            write_jsonl_atomic(INFER / "step-traces-COT.jsonl", traces)
            write_jsonl_atomic(INFER / "requests-COT.jsonl", requests)
            print(f"E3-COT round={round_index} [{offset}/{len(block)}] {sid} "
                  f"{pred['latency_seconds']:.2f}s requests={len(usage)} "
                  f"calls={len(pred['tool_calls'])} toolenv={pred['toolenv_count']} error={error}", flush=True)
        write_json(INFER / "run-COT.json", {"branch": "E3-COT", "status": "in_progress",
            "rounds_completed": round_index, "samples_completed": len(predictions), "block_ids": block,
            "block_started_epoch": block_started, "block_ended_epoch": time.time(),
            "warmup": warmup, "model": model["tag"], "model_digest": digest})

    if [row["id"] for row in predictions] != ids:
        raise RuntimeError("Epoch-3 predictions are not a complete ordered 60-ID set")
    if any(row.get("toolenv_count") != 1 for row in predictions):
        raise RuntimeError("At least one prediction did not create exactly one ToolEnv")
    run_manifest = {"branch": "E3-COT", "status": "complete", "started_epoch": run_started,
        "ended_epoch": time.time(), "elapsed_seconds": time.time() - run_started,
        "samples_completed": len(predictions), "toolenv_per_sample": 1,
        "runtime_errors": sum(bool(row.get("error")) for row in predictions),
        "samples_over_30s_local": sum(float(row["latency_seconds"]) > 30 for row in predictions),
        "model": model["tag"], "model_digest": digest, "ollama_version": version,
        "num_predict": runner.CAP, "num_ctx": runner.CONTEXT, "num_batch": runner.BATCH,
        "temperature": 0, "thinking": True, "protocol": "JSON", "prompt_variant": "cot",
        "cohort_ids_sha256_newline_joined": EXPECTED_IDS_HASH,
        "raw_chain_of_thought_saved": False, "public_private_test_used": False}
    write_json(INFER / "run-COT.json", run_manifest)
    print(json.dumps(run_manifest, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        return prepare()
    if args.smoke:
        return smoke()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
