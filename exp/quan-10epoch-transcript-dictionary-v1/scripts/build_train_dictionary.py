from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "lifemate").is_dir()) / "exp" / "asr-modeling" / "scripts"))
import _bootstrap
from inference_common import read_jsonl
from vietnamese_number_normalizer import normalize_number_phrases

def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

def align(ref: list[str], hyp: list[str]):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j] = min(dp[i-1][j-1]+(ref[i-1] != hyp[j-1]), dp[i-1][j]+1, dp[i][j-1]+1)
    ops=[]; i=n; j=m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1]+(ref[i-1] != hyp[j-1]):
            same=ref[i-1] == hyp[j-1]
            ops.append(("match" if same else "substitution", ref[i-1], hyp[j-1])); i-=1; j-=1
        elif i and dp[i][j] == dp[i-1][j]+1:
            ops.append(("deletion", ref[i-1], None)); i-=1
        else:
            ops.append(("insertion", None, hyp[j-1])); j-=1
    return list(reversed(ops))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unaccent(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))

NUMBER_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu",
                "bảy", "tám", "chín", "mười", "mươi", "trăm", "nghìn", "ngàn",
                "triệu", "tỷ", "tỉ"}
SPOKEN_VARIANTS = {("nha", "nhé"), ("nhé", "nha"), ("không", "hông"),
                   ("nhất", "nhứt"), ("nay", "ni"), ("dù", "dùm"), ("dù", "giùm")}

def candidate_type(wrong: str, correct: str) -> str:
    if unaccent(wrong) == unaccent(correct):
        return "orthographic_variant"
    if wrong.isdigit() and correct.isdigit():
        return "numeric_value_substitution"
    def number_like(text: str) -> bool:
        return all(token.isdigit() or token in NUMBER_WORDS for token in text.split())
    if number_like(wrong) and number_like(correct):
        return "number_rendering_variant"
    if (wrong, correct) in SPOKEN_VARIANTS:
        return "spoken_variant"
    return "lexical_substitution"

def marked_context(words: list[str], surfaces: list[str], start: int, size: int, radius: int, normalized_phrase: str) -> str:
    left = surfaces[max(0,start-radius):start]
    center = " ".join(surfaces[start:start+size])
    right = surfaces[start+size:start+size+radius]
    return " ".join(left + ["⟦"+center+"⟧"] + right)

def normalized_context(words: list[str], start: int, size: int, radius: int) -> str:
    left = words[max(0,start-radius):start]
    center = " ".join(words[start:start+size])
    right = words[start+size:start+size+radius]
    return " ".join(left + ["⟦"+center+"⟧"] + right)

def main() -> int:
    project=_bootstrap.ROOT
    parser=argparse.ArgumentParser(description="Build train-only substitution candidates with whole-phrase number normalization.")
    parser.add_argument("--train-ids",default=str(project/"data"/"splits"/"train_ids.json"))
    parser.add_argument("--manifest",default=str(project/"train+public_test"/"train+public_test"/"train"/"train.jsonl"))
    parser.add_argument("--transcripts",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--context-words",type=int,default=4)
    a=parser.parse_args()
    ids_path,manifest_path,transcript_path=map(Path,(a.train_ids,a.manifest,a.transcripts))
    train_ids=json.loads(ids_path.read_text(encoding="utf-8-sig")); train_set=set(train_ids)
    manifest={r["id"]:r for r in read_jsonl(manifest_path) if r["id"] in train_set}
    transcripts={r["id"]:r for r in read_jsonl(transcript_path) if r["id"] in train_set}
    if train_set-set(manifest) or train_set-set(transcripts):
        raise ValueError("Missing train-only references or transcripts")
    val_path=ids_path.with_name("val_ids.json")
    val_ids=set(json.loads(val_path.read_text(encoding="utf-8-sig"))) if val_path.exists() else set()
    if train_set & val_ids: raise ValueError("train_ids and val_ids overlap")

    evidence=defaultdict(lambda: {"count":0,"ids":set(),"contexts":[]})
    wrong_totals=Counter(); edit_counts=Counter()
    normalized_ref_spans=normalized_hyp_spans=0
    for sid in train_ids:
        ref_raw=tokens(manifest[sid].get("audio_text") or "")
        hyp_raw=tokens(transcripts[sid].get("prediction") or "")
        ref,ref_surface,ref_count=normalize_number_phrases(ref_raw)
        hyp,hyp_surface,hyp_count=normalize_number_phrases(hyp_raw)
        normalized_ref_spans += ref_count; normalized_hyp_spans += hyp_count
        ops=align(ref,hyp)
        edit_counts.update(op for op,_,_ in ops if op in ("substitution","insertion","deletion"))
        k=0
        while k<len(ops):
            if ops[k][0]!="substitution": k+=1; continue
            start=k
            while k<len(ops) and ops[k][0]=="substitution": k+=1
            run=ops[start:k]
            correct=" ".join(x[1] for x in run); wrong=" ".join(x[2] for x in run)
            row=evidence[(wrong,correct)]; row["count"]+=1; row["ids"].add(sid); wrong_totals[wrong]+=1
            ri=sum(op in ("match","substitution","deletion") for op,_,_ in ops[:start])
            hi=sum(op in ("match","substitution","insertion") for op,_,_ in ops[:start])
            size=len(run); radius=a.context_words
            if len(row["contexts"])<8:
                row["contexts"].append({
                    "id":sid,
                    "reference_context":marked_context(ref,ref_surface,ri,size,radius,correct),
                    "transcript_context":marked_context(hyp,hyp_surface,hi,size,radius,wrong),
                    "normalized_reference_context":normalized_context(ref,ri,size,radius),
                    "normalized_transcript_context":normalized_context(hyp,hi,size,radius),
                })
    candidates=[]
    for (wrong,correct),row in sorted(evidence.items(),key=lambda p:(-p[1]["count"],p[0])):
        count=row["count"]; ids=sorted(row["ids"]); dominance=count/max(1,wrong_totals[wrong])
        confidence=("high" if count>=5 and len(ids)>=3 and dominance>=.8 else
                    "medium" if count>=2 and len(ids)>=2 and dominance>=.6 else "low")
        kind=candidate_type(wrong,correct)
        candidates.append({"wrong_phrase":wrong,"correct_phrase":correct,"candidate_type":kind,
                           "occurrences":count,"distinct_train_ids":len(ids),"ids":ids,
                           "dominance_for_wrong_phrase":round(dominance,4),"confidence":confidence,
                           "eligible_for_context_review":kind in ("lexical_substitution","numeric_value_substitution") and confidence in ("high","medium"),
                           "review_required":True,"contexts":row["contexts"]})
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    metadata={
        "source":"train split only; references restricted to data/splits/train_ids.json",
        "train_ids":len(train_ids),"validation_ids_excluded":len(val_ids),
        "text_tokenization":"Unicode regex \\w+ after lowercase; punctuation removed; diacritics retained",
        "number_normalization":"Parse complete Vietnamese cardinal spans (units, tens, hundreds, thousands, millions, billions) into one integer token before alignment; keep original surface phrase in contexts",
        "number_spans_normalized_in_references":normalized_ref_spans,
        "number_spans_normalized_in_transcripts":normalized_hyp_spans,
        "number_parser_examples":{"hai mươi lăm":"25","bốn mươi lăm":"45","năm mươi":"50"},
        "alignment":"word-level Levenshtein after number normalization; emit only contiguous equal-length substitution runs",
        "insertions_excluded_from_dictionary":edit_counts["insertion"],
        "deletions_excluded_from_dictionary":edit_counts["deletion"],
        "substitution_operations":edit_counts["substitution"],
        "confidence_scope":"confidence measures repeated alignment consistency only, not semantic correctness",
        "type_policy":"orthographic/format variants are not correction rules; numeric value substitutions and lexical candidates require context review; spoken variants require human context review",
        "confidence_heuristic":"high: >=5 occurrences, >=3 IDs, dominance >=0.8; medium: >=2 occurrences, >=2 IDs, dominance >=0.6; otherwise low",
        "train_ids_sha256":sha(ids_path),"reference_manifest_sha256":sha(manifest_path),
        "transcript_file_sha256":sha(transcript_path),"candidate_pairs":len(candidates),
    }
    output.write_text(json.dumps({"metadata":metadata,"candidates":candidates},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=True,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
