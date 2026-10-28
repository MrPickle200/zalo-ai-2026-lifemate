from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
MISSING_INFO = "sys.ask_missing_info"
DEFAULT_TOP_K = 3
MAX_CANDIDATE_TOOLS = 5
MIN_CONFIDENCE = 0.06


def _tokens(text: str) -> list[str]:
    words = TOKEN_RE.findall((text or "").casefold())
    return words + [f"bi:{a}_{b}" for a, b in zip(words, words[1:])]


class TrainRetriever:
    """Small deterministic TF-IDF retriever built only from the fixed training IDs."""

    def __init__(self, data_root: str | Path, split_dir: str | Path = "data/splits"):
        data_root = Path(data_root)
        train_ids_path = Path(split_dir) / "train_ids.json"
        train_ids = set(json.loads(train_ids_path.read_text(encoding="utf-8")))
        manifest = data_root / "train" / "train.jsonl"
        self.documents: list[dict[str, Any]] = []
        for line in manifest.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("id") not in train_ids:
                continue
            label_calls = row.get("label", {}).get("tool_calls", [])
            expected_tools = sorted({call.get("name") for call in label_calls if call.get("name")})
            self.documents.append({"sample_id": row["id"], "transcript": row.get("audio_text", ""),
                                   "expected_tools": expected_tools,
                                   "expected_apps": sorted({name.split(".", 1)[0] for name in expected_tools}),
                                   "expected_tool_calls": label_calls})
        if len(self.documents) != len(train_ids):
            raise ValueError(f"Training retrieval corpus mismatch: got {len(self.documents)} docs for {len(train_ids)} train IDs")

        tokenized = [_tokens(doc["transcript"]) for doc in self.documents]
        document_frequency: Counter[str] = Counter()
        term_counts: list[Counter[str]] = []
        for tokens in tokenized:
            counts = Counter(tokens)
            term_counts.append(counts)
            document_frequency.update(counts.keys())
        n_docs = max(len(self.documents), 1)
        self.idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1.0
                    for term, freq in document_frequency.items()}
        self.vectors: list[dict[str, float]] = []
        self.inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, counts in enumerate(term_counts):
            raw = {term: (1.0 + math.log(count)) * self.idf[term] for term, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
            vector = {term: value / norm for term, value in raw.items()}
            self.vectors.append(vector)
            for term, value in vector.items():
                self.inverted[term].append((i, value))
        self.missing_info_demos = self._select_missing_info_demos()

    def _select_missing_info_demos(self) -> list[dict[str, Any]]:
        categories = {
            "recipient": {"contact_id", "beneficiary_id", "to", "phone_number", "assignee"},
            "time": {"time", "date", "start", "end", "schedule"},
            "location": {"address", "origin", "destination", "pickup", "dropoff", "location"},
            "amount": {"amount", "value", "price", "quantity"},
        }
        selected = []
        for category, fields in categories.items():
            candidates = []
            for doc in self.documents:
                if not any(call.get("name") == MISSING_INFO for call in doc["expected_tool_calls"]):
                    continue
                missing = {str(param) for call in doc["expected_tool_calls"] if call.get("name") == MISSING_INFO
                           for param in (call.get("arguments", {}).get("params", []))}
                if missing & fields:
                    candidates.append(doc)
            if candidates:
                doc = min(candidates, key=lambda d: (len(d["transcript"]), len(d["expected_tool_calls"]), d["sample_id"]))
                selected.append({"category": category, "sample_id": doc["sample_id"],
                                 "transcript": doc["transcript"], "expected_tool_calls": doc["expected_tool_calls"]})
        return selected

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        counts = Counter(_tokens(query))
        raw = {term: (1.0 + math.log(count)) * self.idf[term]
               for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
        query_vector = {term: value / norm for term, value in raw.items()}
        scores: dict[int, float] = defaultdict(float)
        for term, query_weight in query_vector.items():
            for index, doc_weight in self.inverted.get(term, ()):
                scores[index] += query_weight * doc_weight
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], self.documents[pair[0]]["sample_id"]))
        return [{**self.documents[index], "similarity_score": score}
                for index, score in ranked[:max(0, top_k)] if score > 0.0]

    @staticmethod
    def shortlist(results: list[dict[str, Any]], available_tools: list[str],
                  top_tools: int = MAX_CANDIDATE_TOOLS,
                  min_confidence: float = MIN_CONFIDENCE) -> tuple[list[str] | None, float]:
        confidence = results[0]["similarity_score"] if results else 0.0
        if confidence < min_confidence:
            return None, confidence
        allowed = set(available_tools)
        votes: Counter[str] = Counter()
        for rank, result in enumerate(results, 1):
            weight = result["similarity_score"] / math.sqrt(rank)
            for name in {c.get("name") for c in result["expected_tool_calls"]}:
                if name in allowed and name != MISSING_INFO:
                    votes[name] += weight
        ranked_tools = sorted(votes, key=lambda name: (-votes[name], name))[:top_tools]
        if not ranked_tools:
            return None, confidence
        # Missing-info is a system capability and must remain available in routed prompts.
        if MISSING_INFO in allowed:
            ranked_tools.append(MISSING_INFO)
        return ranked_tools, confidence
