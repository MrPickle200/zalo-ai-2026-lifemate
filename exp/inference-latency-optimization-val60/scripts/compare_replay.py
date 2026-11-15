from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/inference-latency-optimization-val60/output"
OLD = ROOT / "exp/num-predict-512-vs-768-val/output"


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def calls(row: dict) -> Counter[str]:
    return Counter(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for x in row.get("tool_calls", []))


def main() -> None:
    cohort = json.loads((OUT / "cohort_manifest.json").read_text(encoding="utf-8"))["ids"]
    archived = {x["id"]: x for x in read(OLD / "predictions-768.jsonl")}
    current = {x["id"]: x for x in read(OUT / "predictions-baseline.jsonl")}
    old_req: dict[str, list[dict]] = {x: [] for x in cohort}
    new_req: dict[str, list[dict]] = {x: [] for x in cohort}
    for row in read(OLD / "token_usage-768.jsonl"):
        old_req[row["sample_id"]].append(row)
    for row in read(OUT / "requests-baseline.jsonl"):
        new_req[row["sample_id"]].append(row)
    ordered = [sid for sid in current if current[sid].get("tool_calls") != archived[sid].get("tool_calls")]
    multiset = [sid for sid in current if calls(current[sid]) != calls(archived[sid])]
    hashes = [sid for sid in current if current[sid].get("state_hash") != archived[sid].get("state_hash")]
    hash_sequences_equal, hash_sample_mismatch = 0, []
    first_hash_equal, first_action_equal = 0, 0
    per_sample_hash_equal = []
    for sid in current:
        a = [x.get("payload_sha256") for x in old_req[sid]]
        b = [x.get("payload_sha256") for x in new_req[sid]]
        if a == b:
            hash_sequences_equal += 1
        else:
            hash_sample_mismatch.append(sid)
        if a and b and a[0] == b[0]:
            first_hash_equal += 1
        old_actions = archived[sid].get("model_output", [])
        new_actions = current[sid].get("model_output", [])
        if old_actions and new_actions:
            action_fields = ("action", "tool", "arguments")
            if {k: old_actions[0].get(k) for k in action_fields} == {k: new_actions[0].get(k) for k in action_fields}:
                first_action_equal += 1
        per_sample_hash_equal.append({"id": sid, "old_request_count": len(a), "new_request_count": len(b),
                                      "same_request_payload_hash_sequence": a == b,
                                      "old_hashes": a, "new_hashes": b})
    result = {"checked_samples": len(current), "ordered_call_mismatches": len(ordered),
              "ordered_mismatch_ids": ordered, "call_multiset_mismatches": len(multiset),
              "call_multiset_mismatch_ids": multiset, "state_hash_mismatches": len(hashes),
              "state_hash_mismatch_ids": hashes, "identical_request_payload_hash_sequences": hash_sequences_equal,
              "identical_first_request_payload_hashes": first_hash_equal,
              "identical_first_actions": first_action_equal,
              "request_hash_sequence_mismatch_ids": hash_sample_mismatch,
              "per_sample_request_hashes": per_sample_hash_equal,
              "effective_num_batch_archived": 512,
              "effective_num_batch_current": 1024,
              "interpretation": "Payload hashes establish request identity. The effective batch setting is recovered from the old/current llama-server command lines; it is not a field in the client JSON when omitted."}
    branch_comparisons = {}
    for branch in ("batch512", "batch1024"):
        candidate_path = OUT / f"predictions-{branch}.jsonl"
        if not candidate_path.exists():
            continue
        candidate = {x["id"]: x for x in read(candidate_path)}
        checked = [sid for sid in cohort if sid in candidate]
        branch_comparisons[branch] = {
            "checked_samples": len(checked),
            "ordered_call_mismatch_samples": sum(candidate[sid].get("tool_calls") != archived[sid].get("tool_calls") for sid in checked),
            "call_multiset_mismatch_samples": sum(calls(candidate[sid]) != calls(archived[sid]) for sid in checked),
            "state_hash_mismatch_samples": sum(candidate[sid].get("state_hash") != archived[sid].get("state_hash") for sid in checked),
        }
    result["archived_output_comparison_by_explicit_batch"] = branch_comparisons
    default_comparison = {}
    explicit1024_path = OUT / "predictions-batch1024.jsonl"
    if explicit1024_path.exists():
        explicit1024 = {x["id"]: x for x in read(explicit1024_path)}
        default_comparison["implicit_default_vs_explicit1024"] = {
            "ordered_call_mismatch_samples": sum(current[sid].get("tool_calls") != explicit1024[sid].get("tool_calls") for sid in cohort),
            "call_multiset_mismatch_samples": sum(calls(current[sid]) != calls(explicit1024[sid]) for sid in cohort),
            "state_hash_mismatch_samples": sum(current[sid].get("state_hash") != explicit1024[sid].get("state_hash") for sid in cohort),
        }
    explicit512_path = OUT / "predictions-batch512.jsonl"
    if explicit512_path.exists():
        explicit512 = {x["id"]: x for x in read(explicit512_path)}
        default_comparison["archived_vs_explicit512"] = {
            "ordered_call_mismatch_samples": sum(explicit512[sid].get("tool_calls") != archived[sid].get("tool_calls") for sid in cohort),
            "call_multiset_mismatch_samples": sum(calls(explicit512[sid]) != calls(archived[sid]) for sid in cohort),
            "state_hash_mismatch_samples": sum(explicit512[sid].get("state_hash") != archived[sid].get("state_hash") for sid in cohort),
        }
    result["effective_batch_control_comparisons"] = default_comparison
    (OUT / "baseline-reproducibility.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample_request_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
