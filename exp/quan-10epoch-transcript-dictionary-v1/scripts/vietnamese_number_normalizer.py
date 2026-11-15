from __future__ import annotations

from collections.abc import Sequence

UNITS = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3,
    "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6,
    "bảy": 7, "tám": 8, "chín": 9,
}
TENS = {"mười", "mươi", "chục"}
CONNECTORS = {"linh", "lẻ"}
SCALES = {"nghìn": 1_000, "ngàn": 1_000, "triệu": 1_000_000,
          "tỷ": 1_000_000_000, "tỉ": 1_000_000_000}
NUMBER_WORDS = set(UNITS) | TENS | CONNECTORS | {"trăm"} | set(SCALES)
UNIT_CONTEXT = {
    "giờ", "phút", "giây", "ngày", "tuần", "tháng", "năm", "độ", "phần",
    "đồng", "người", "lần", "vé", "cái", "chỗ", "bước", "km", "kg", "mét",
    "ly", "cốc", "món", "đơn", "phòng", "lít", "tấn", "tiếng", "chuyến",
    "quả", "hộp", "chai", "tờ", "đêm", "buổi",
}
NUMBER_PREFIXES = {"thứ", "số", "mã", "tầng", "phòng", "chuyến", "ngày", "tháng"}


def _under_100(words: Sequence[str]) -> int | None:
    if not words:
        return 0
    if len(words) == 1:
        return 10 if words[0] == "mười" else UNITS.get(words[0])
    if words[0] == "mười":
        rest = list(words[1:])
        if rest and rest[0] in CONNECTORS:
            rest = rest[1:]
        if not rest:
            return 10
        if len(rest) == 1 and rest[0] in UNITS and UNITS[rest[0]] > 0:
            return 10 + UNITS[rest[0]]
        return None
    if len(words) >= 2 and words[0] in UNITS and UNITS[words[0]] > 0 and words[1] in {"mươi", "chục"}:
        rest = list(words[2:])
        if rest and rest[0] in CONNECTORS:
            rest = rest[1:]
        if not rest:
            return UNITS[words[0]] * 10
        if len(rest) == 1 and rest[0] in UNITS:
            return UNITS[words[0]] * 10 + UNITS[rest[0]]
    return None


def _under_1000(words: Sequence[str]) -> int | None:
    if words.count("trăm") > 1:
        return None
    try:
        hundred_index = words.index("trăm")
    except ValueError:
        return _under_100(words)
    before = list(words[:hundred_index])
    if len(before) > 1 or (before and (before[0] not in UNITS or UNITS[before[0]] <= 0)):
        return None
    hundreds = UNITS[before[0]] if before else 1
    rest = list(words[hundred_index + 1:])
    if rest and rest[0] in CONNECTORS:
        rest = rest[1:]
    tail = _under_100(rest) if rest else 0
    if tail is None:
        return None
    return hundreds * 100 + tail


def parse_vietnamese_number(words: Sequence[str]) -> int | None:
    """Parse a Vietnamese cardinal phrase into an integer; return None for invalid sequences."""
    if not words or any(word not in NUMBER_WORDS for word in words):
        return None
    largest_scale = max((SCALES.get(word, 0) for word in words), default=0)
    if largest_scale:
        scale_word = next(word for word in words if SCALES.get(word, 0) == largest_scale)
        index = words.index(scale_word)
        left_words = words[:index]
        right_words = words[index + 1:]
        left = _under_1000(left_words) if left_words else None
        right = parse_vietnamese_number(right_words) if right_words else 0
        if left is None or right is None or left <= 0 or right >= largest_scale:
            return None
        return left * largest_scale + right
    return _under_1000(words)


def normalize_number_phrases(words: Sequence[str]) -> tuple[list[str], list[str], int]:
    """Replace complete cardinal spans with digit tokens and retain each raw span as a surface."""
    normalized: list[str] = []
    surfaces: list[str] = []
    normalized_spans = 0
    i = 0
    while i < len(words):
        if words[i].isdigit():
            normalized.append(words[i])
            surfaces.append(words[i])
            i += 1
            continue
        best: tuple[int, int] | None = None
        limit = min(len(words), i + 12)
        for end in range(limit, i, -1):
            span = list(words[i:end])
            if not all(word in NUMBER_WORDS for word in span):
                continue
            # Final "năm" is commonly a year unit: "mười năm" means ten years;
            # fifteen is written "mười lăm". Leave the unit outside the numeral.
            if len(span) > 1 and span[-1] == "năm" and span[-2] in {"mười", "mươi", "chục"}:
                shorter = span[:-1]
                value = parse_vietnamese_number(shorter)
                if value is not None and (end == len(words) or words[end] in {"nữa", "qua", "trước", "sau", "nay", "ngoái"}):
                    end -= 1
                    span = shorter
                    span = shorter
            value = parse_vietnamese_number(span)
            if value is None:
                continue
            eligible = (
                len(span) > 1
                or (end < len(words) and words[end] in UNIT_CONTEXT)
                or (i > 0 and words[i - 1] in NUMBER_PREFIXES)
            )
            if eligible:
                best = (end, value)
                break
        if best is None:
            normalized.append(words[i])
            surfaces.append(words[i])
            i += 1
            continue
        end, value = best
        normalized.append(str(value))
        surfaces.append(" ".join(words[i:end]))
        normalized_spans += 1
        i = end
    return normalized, surfaces, normalized_spans

