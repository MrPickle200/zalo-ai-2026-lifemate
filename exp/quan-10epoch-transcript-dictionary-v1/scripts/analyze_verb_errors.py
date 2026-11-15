from __future__ import annotations
import argparse, hashlib, json, re, sys, unicodedata
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "lifemate").is_dir()) / "exp" / "asr-modeling" / "scripts"))
import _bootstrap
from inference_common import read_jsonl
from vietnamese_number_normalizer import normalize_number_phrases

# Conservative lexical verb inventory for LifeMate instructions and everyday requests.
VERBS = set("""
ăn uống nấu đặt đổi dời chuyển trả nạp rút mua bán gửi lấy thêm xóa xoá hủy huỷ gọi nhắn nhắc báo hỏi xem tìm đọc viết ghi lưu soạn dịch sửa chỉnh bật tắt mở đóng tăng giảm giữ chọn chờ nhận giao đón chạy đi về nghỉ ngủ học làm nói phát tính tra giúp nhờ hẹn hoãn tạo gặp đợi mời xin dùng xài kiếm thay đánh lập cất xếp bỏ đưa mang đem nộp tham dự tham gia
""".split())

# Compound predicates: mark their parts only when the complete phrase is present.
VERB_PHRASES = [
    ("thanh", "toán"), ("kiểm", "tra"), ("cập", "nhật"), ("điều", "chỉnh"),
    ("điều", "khiển"), ("xác", "nhận"), ("hoàn", "tất"), ("hoàn", "thành"),
    ("trả", "lời"), ("hủy", "bỏ"), ("huỷ", "bỏ"), ("xóa", "bỏ"), ("xoá", "bỏ"),
    ("nhắc", "nhở"), ("đăng", "ký"), ("sắp", "xếp"), ("thay", "đổi"),
    ("gợi", "ý"), ("cài", "đặt"), ("mua", "sắm"), ("tham", "gia"),
]
def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
def verb_flags(words: list[str]) -> list[bool]:
    flags = [word in VERBS for word in words]
    for phrase in VERB_PHRASES:
        size = len(phrase)
        for i in range(len(words)-size+1):
            if tuple(words[i:i+size]) == phrase:
                flags[i:i+size] = [True] * size
    listening_objects = {"nhạc", "radio", "podcast", "tin", "lời", "tiếng", "máy",
                         "bài", "trên", "nina"}
    for i, word in enumerate(words):
        nxt = words[i+1] if i+1 < len(words) else None
        prev = words[i-1] if i > 0 else None
        # Sentence-final "nghe" is often a Vietnamese discourse particle; count it
        # only in clear listening constructions or when followed by a known name.
        if word == "nghe":
            flags[i] = nxt in listening_objects
        if word == "báo" and nxt == "thức":
            flags[i] = False  # noun phrase "báo thức"
        if word == "giao" and nxt == "dịch":
            flags[i] = False  # noun phrase "giao dịch"
        if word == "đi" and prev == "mục" and nxt == "lại":
            flags[i] = False  # nominalized "đi lại"
        if word == "đi" and prev in {"sáng", "trưa", "chiều", "tối"} and nxt in {"nghe", "nha", "nhé", "nghen", "à"}:
            flags[i] = False  # "sáng ni nghe" misrecognized as "sáng đi nghe"
        if word == "đi" and prev == "đó" and nxt in {"giờ", "lúc"}:
            flags[i] = False
        if word == "đi" and prev == "nhất" and nxt in {"rồi", "thôi"}:
            flags[i] = False  # imperative particle, not motion verb
        if word == "đi" and prev in VERBS and nxt in {None, "nha", "nhé", "nghen"}:
            flags[i] = False  # post-verbal particle, e.g. "xóa đi"
    return flags

def unaccent(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(ch))

def align(ref: list[str], hyp: list[str]):
    n,m=len(ref),len(hyp)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j-1]+(ref[i-1]!=hyp[j-1]),dp[i-1][j]+1,dp[i][j-1]+1)
    ops=[];i=n;j=m
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
            op="match" if ref[i-1]==hyp[j-1] else "substitution"
            ops.append((op,i-1,j-1));i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:
            ops.append(("deletion",i-1,None));i-=1
        else:
            ops.append(("insertion",None,j-1));j-=1
    return list(reversed(ops))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def context(words: list[str], i: int | None, radius: int=4) -> str:
    if i is None: return ""
    return " ".join(words[max(0,i-radius):min(len(words),i+radius+1)])

def main() -> int:
    project=_bootstrap.ROOT
    parser=argparse.ArgumentParser(description="Count train-only verb error tokens with an auditable lexical inventory.")
    parser.add_argument("--train-ids",default=str(project/"data"/"splits"/"train_ids.json"))
    parser.add_argument("--manifest",default=str(project/"train+public_test"/"train+public_test"/"train"/"train.jsonl"))
    parser.add_argument("--transcripts",required=True)
    parser.add_argument("--summary",required=True)
    parser.add_argument("--events",required=True)
    a=parser.parse_args()
    ids_path,manifest_path,transcript_path=map(Path,(a.train_ids,a.manifest,a.transcripts))
    ids=json.loads(ids_path.read_text(encoding="utf-8-sig")); idset=set(ids)
    refs={r["id"]:r for r in read_jsonl(manifest_path) if r["id"] in idset}
    hyps={r["id"]:r for r in read_jsonl(transcript_path) if r["id"] in idset}
    if idset-set(refs) or idset-set(hyps): raise ValueError("Missing train-only reference or transcript")
    counts={"substitution":0,"insertion":0,"deletion":0}
    normalized_ref_spans=normalized_hyp_spans=0
    by_pair={}
    affected=set(); events=[]; orthographic_excluded=0
    for sid in ids:
        ref_raw=tokens(refs[sid].get("audio_text") or "")
        hyp_raw=tokens(hyps[sid].get("prediction") or "")
        ref,ref_surface,ref_count=normalize_number_phrases(ref_raw)
        hyp,hyp_surface,hyp_count=normalize_number_phrases(hyp_raw)
        normalized_ref_spans += ref_count
        normalized_hyp_spans += hyp_count
        rf=verb_flags(ref);hf=verb_flags(hyp)
        for op,ri,hi in align(ref,hyp):
            if op=="match": continue
            if op=="substitution" and unaccent(ref[ri]) == unaccent(hyp[hi]):
                orthographic_excluded += 1
                continue
            rv=ri is not None and rf[ri]
            hv=hi is not None and hf[hi]
            if not (rv or hv): continue
            counts[op]+=1; affected.add(sid)
            rword=ref[ri] if ri is not None else None
            hword=hyp[hi] if hi is not None else None
            pair=(rword or "<deleted>",hword or "<inserted>")
            by_pair[pair]=by_pair.get(pair,0)+1
            events.append({
                "id":sid,"operation":op,"reference_token":rword,"transcript_token":hword,
                "verb_side":"both" if rv and hv else ("reference" if rv else "transcript"),
                "reference_context":context(ref_surface,ri),"transcript_context":context(hyp_surface,hi),
                "counting_rule":"count one word-level error when either aligned side matches the lexical verb inventory"
            })
    summary={
        "source":"train IDs only; val IDs excluded",
        "train_ids":len(ids),
        "validation_ids_excluded":309,
        "normalization":"lowercase Unicode regex \\w+; punctuation removed; diacritics retained; complete Vietnamese cardinal phrases normalized before alignment",
        "count_definition":"word-level Levenshtein error event where the reference token or transcript token is in the rule-based LifeMate verb inventory; multiword predicates listed in the script are also recognized",
        "number_spans_normalized_in_references":normalized_ref_spans,
        "number_spans_normalized_in_transcripts":normalized_hyp_spans,
        "verb_lexicon_size":len(VERBS),
        "verb_error_tokens_total":sum(counts.values()),
        "orthographic_spelling_variants_excluded":orthographic_excluded,
        "by_operation":counts,
        "affected_train_utterances":len(affected),
        "distinct_reference_transcript_pairs":len(by_pair),
        "top_pairs":[{"reference_token":k[0],"transcript_token":k[1],"count":v} for k,v in sorted(by_pair.items(),key=lambda x:(-x[1],x[0]))[:100]],
        "train_ids_sha256":sha(ids_path),
        "reference_manifest_sha256":sha(manifest_path),
        "transcript_file_sha256":sha(transcript_path),
        "interpretation_limit":"Rule-based screening, not Vietnamese POS tagging. Ambiguous items should be reviewed in the saved event contexts."
    }
    summary_path=Path(a.summary);summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    event_path=Path(a.events);event_path.parent.mkdir(parents=True,exist_ok=True)
    event_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in events),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("top_pairs",)},ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
