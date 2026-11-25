from __future__ import annotations

import re

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]+\s*条")


def chinese_number_to_int(text: str) -> int:
    """把常见中文数字（一至九千九百九十九）转换为整数。"""
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if unit == 10000:
                total += (section + number) * unit
                section = 0
                number = 0
            else:
                section += (number or 1) * unit
                number = 0
        elif char in {"第", "条", " ", "\u3000"}:
            continue
    return total + section + number


def normalize_article_number(text: str) -> str:
    """把“第二百六十三条”“第263条”统一成可比较的数字字符串。"""
    raw = str(text or "").strip()
    match = _ARTICLE_RE.search(raw)
    if match:
        raw = match.group(0)
    core = re.sub(r"[第条\s\u3000]", "", raw)
    if not core:
        return ""
    if core.isdigit():
        return str(int(core))
    value = chinese_number_to_int(core)
    return str(value) if value else core


def extract_article_numbers(text: str) -> set[str]:
    """提取文本中出现的全部法条编号，并统一为数字字符串。"""
    return {
        normalize_article_number(match.group(0))
        for match in _ARTICLE_RE.finditer(text or "")
        if normalize_article_number(match.group(0))
    }
