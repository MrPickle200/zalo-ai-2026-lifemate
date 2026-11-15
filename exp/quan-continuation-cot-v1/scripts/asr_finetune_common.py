from __future__ import annotations

import re

_ONES = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_ORTHOGRAPHY = {
    "hoà": "hòa", "hoá": "hóa", "hoả": "hỏa", "hoã": "hõa", "hoạ": "họa",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}


def _under_100(value: int) -> str:
    if value < 10:
        return _ONES[value]
    if value < 20:
        unit = value % 10
        return "mười" if unit == 0 else "mười " + ("lăm" if unit == 5 else _ONES[unit])
    tens, unit = divmod(value, 10)
    result = _ONES[tens] + " mươi"
    if unit:
        word = "mốt" if unit == 1 and tens > 1 else "tư" if unit == 4 and tens > 1 else "lăm" if unit == 5 else _ONES[unit]
        result += " " + word
    return result


def _under_1000(value: int) -> str:
    if value < 100:
        return _under_100(value)
    hundreds, remainder = divmod(value, 100)
    result = _ONES[hundreds] + " trăm"
    if remainder:
        result += " " + ("linh " + _ONES[remainder] if remainder < 10 else _under_100(remainder))
    return result


def vietnamese_number_words(value: int) -> str:
    if value < 0:
        return "âm " + vietnamese_number_words(-value)
    if value < 1000:
        return _under_1000(value)
    for scale, name in ((10**9, "tỷ"), (10**6, "triệu"), (1000, "nghìn")):
        if value >= scale:
            major, remainder = divmod(value, scale)
            result = vietnamese_number_words(major) + " " + name
            if remainder:
                if remainder < 100:
                    result += " không trăm linh " + vietnamese_number_words(remainder)
                elif remainder < 1000:
                    result += " không trăm " + vietnamese_number_words(remainder)
                else:
                    result += " " + vietnamese_number_words(remainder)
            return result
    return str(value)


def normalize_quan_text(text: str) -> str:
    """Match QUAN's lowercase, punctuation-free text style and spell numeric tokens."""
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    normalized: list[str] = []
    for word in words:
        word = _ORTHOGRAPHY.get(word, word)
        if word.isascii() and word.isdigit():
            if len(word) >= 7 or (len(word) > 1 and word.startswith("0")):
                normalized.extend(_ONES[int(digit)] for digit in word)
            else:
                normalized.extend(vietnamese_number_words(int(word)).split())
        else:
            normalized.append(word)
    return " ".join(normalized)
