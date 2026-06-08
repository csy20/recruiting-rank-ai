import re
from typing import List, Set


def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    text = text.lower()
    tokens = set(re.findall(r"[a-z0-9_+.-]+", text))
    return tokens


def count_keyword_matches(
    text: str, keywords: List[str], case_sensitive: bool = False
) -> int:
    if not text:
        return 0
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def count_ngram_matches(text: str, keywords: List[str]) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    matches = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            matches += 1
    if not keywords:
        return 0.0
    return matches / len(keywords)


def extract_years(text: str) -> float:
    if not text:
        return 0.0
    years = re.findall(r"(\d+\.?\d*)\s*years?", text.lower())
    if years:
        return float(years[0])
    return 0.0


def compute_text_similarity(text_a: str, text_b: str) -> float:
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def has_production_indicators(text: str) -> bool:
    indicators = [
        r"\bprod(uction)?\b",
        r"\bdeploy(ed|ment|ing)?\b",
        r"\bship(ped|ping)?\b",
        r"\blive\b",
        r"\bscal(e|ing|ed)?\b",
        r"\breal.?user",
        r"\bproduction.?grade\b",
    ]
    for pattern in indicators:
        if re.search(pattern, text.lower()):
            return True
    return False
