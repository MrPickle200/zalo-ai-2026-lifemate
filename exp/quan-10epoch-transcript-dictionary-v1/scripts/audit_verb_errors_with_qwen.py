from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string", "enum": ["verb_error", "not_verb_error", "uncertain"]},
                },
                "required": ["index", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def request_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask local Qwen to audit train-only verb-error candidates.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    events_path, output_path, summary_path = map(Path, (args.events, args.output, args.summary))
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tags = request_json(args.base_url + "/api/tags")
    model_row = next((row for row in tags.get("models", []) if row.get("name") == args.model), None)
    if model_row is None:
        raise RuntimeError(f"Model tag not installed: {args.model}")

    system = (
        "Classify Vietnamese ASR alignment errors by grammatical function in context. "
        "A verb_error means at least one mismatched token is used as a lexical action/state verb in that sentence. "
        "A not_verb_error means neither token is a lexical verb there. Use uncertain if context is insufficient. "
        "For insertions/deletions, judge the present token in its context. Do not infer from spelling alone. "
        "Return exactly one label for every supplied index, with no explanation."
    )
    results: dict[int, str] = {}
    request_count = 0
    for start in range(0, len(events), args.batch_size):
        chunk = events[start:start + args.batch_size]
        indexed = []
        for index, row in enumerate(chunk, start):
            indexed.append({
                "index": index,
                "operation": row["operation"],
                "reference_token": row["reference_token"],
                "transcript_token": row["transcript_token"],
                "reference_context": row["reference_context"],
                "transcript_context": row["transcript_context"],
            })
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(indexed, ensure_ascii=False)},
            ],
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024, "num_batch": 512},
        }
        reply = request_json(args.base_url + "/api/chat", payload)
        request_count += 1
        content = reply.get("message", {}).get("content", "")
        parsed = json.loads(content)
        returned = parsed.get("items", [])
        expected = {item["index"] for item in indexed}
        actual: dict[int, str] = {}
        for item in returned:
            index = item.get("index")
            label = item.get("label")
            if index in expected and label in {"verb_error", "not_verb_error", "uncertain"}:
                actual[index] = label
        if set(actual) != expected:
            raise RuntimeError(f"Incomplete/invalid Qwen batch at row {start}: expected {len(expected)}, got {len(actual)}")
        results.update(actual)

    output_rows = []
    by_label = Counter()
    by_operation = Counter()
    for index, row in enumerate(events):
        label = results[index]
        by_label[label] += 1
        by_operation[(row["operation"], label)] += 1
        output_rows.append({**row, "qwen_label": label, "qwen_model": args.model})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "method": "local Qwen context audit of the rule-based verb-error candidate events; does not edit transcripts",
        "model": args.model,
        "model_digest": model_row.get("digest"),
        "server_version": request_json(args.base_url + "/api/version").get("version"),
        "thinking": False,
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 1024,
        "num_batch": 512,
        "batch_size": args.batch_size,
        "requests": request_count,
        "candidate_events": len(events),
        "labels": dict(by_label),
        "by_operation_and_label": {f"{op}:{label}": count for (op, label), count in sorted(by_operation.items())},
        "input_events_sha256": sha(events_path),
        "output_events_sha256": sha(output_path),
        "interpretation_limit": "Qwen reviewed only rule-based candidate events; it is an independent context audit, not a POS gold label or an exhaustive scan for missed verbs.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
