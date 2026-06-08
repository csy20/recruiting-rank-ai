import re
from typing import Dict, List, Optional, Pattern, Set, Tuple

_PROD_INDICATORS: List[Pattern] = [
    re.compile(r"\bprod(uction)?\b", re.IGNORECASE),
    re.compile(r"\bdeploy(ed|ment|ing)?\b", re.IGNORECASE),
    re.compile(r"\bship(ped|ping)?\b", re.IGNORECASE),
    re.compile(r"\blive\b", re.IGNORECASE),
    re.compile(r"\bscal(e|ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\breal[\s-]?user", re.IGNORECASE),
    re.compile(r"\bproduction[\s-]?grade\b", re.IGNORECASE),
]

_WHITESPACE_RE = re.compile(r"[a-z0-9_+.\-]+", re.IGNORECASE)
_YEARS_RE = re.compile(r"(\d+\.?\d*)\s*years?", re.IGNORECASE)


def tokenize(text: Optional[str]) -> Set[str]:
    if not text:
        return set()
    return set(_WHITESPACE_RE.findall(text.lower()))


def _make_boundary_pattern(keyword: str) -> Pattern:
    escaped = re.escape(keyword)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def compile_keyword_patterns(keywords: List[str]) -> List[Pattern]:
    return [_make_boundary_pattern(kw) for kw in keywords]


def count_pattern_matches(text: str, patterns: List[Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for p in patterns if p.search(text))


def count_keyword_matches(text: Optional[str], keywords: List[str]) -> int:
    if not text:
        return 0
    patterns = compile_keyword_patterns(keywords)
    return count_pattern_matches(text, patterns)


def count_ngram_matches(text: Optional[str], keywords: List[str]) -> float:
    if not text:
        return 0.0
    if not keywords:
        return 0.0
    patterns = compile_keyword_patterns(keywords)
    matches = count_pattern_matches(text, patterns)
    return matches / len(keywords)


def extract_years(text: Optional[str]) -> float:
    if not text:
        return 0.0
    match = _YEARS_RE.search(text)
    return float(match.group(1)) if match else 0.0


def has_production_indicators(text: Optional[str]) -> bool:
    if not text:
        return False
    pattern = re.compile(
        r"\b(prod(uction)?|deploy(ed|ment|ing)?|ship(ped|ping)?|"
        r"live|scal(e|ing|ed)?|real[\s-]?user|production[\s-]?grade)\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def extract_key_phrases(text: str, top_n: int = 50) -> List[Tuple[str, float]]:
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{2,}", text.lower())
    if not tokens:
        return []
    from collections import Counter

    counts = Counter(tokens)
    total = sum(counts.values())
    return [(word, count / total) for word, count in counts.most_common(top_n)]


def extract_technical_terms(text: str) -> Dict[str, float]:
    tech_keywords = {
        "python",
        "java",
        "scala",
        "go",
        "rust",
        "c++",
        "typescript",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "xgboost",
        "spark",
        "kafka",
        "flink",
        "airflow",
        "hadoop",
        "kubernetes",
        "docker",
        "terraform",
        "jenkins",
        "aws",
        "gcp",
        "azure",
        "gcs",
        "s3",
        "lambda",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "graphql",
        "rest",
        "grpc",
        "microservices",
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "llm",
        "rag",
        "transformers",
        "embedding",
    }
    result = {}
    for term in tech_keywords:
        pattern = _make_boundary_pattern(term)
        matches = pattern.findall(text.lower())
        if matches:
            result[term] = float(len(matches))
    return result


def compute_jd_candidate_overlap(
    jd_terms: Dict[str, float], candidate_terms: Dict[str, float]
) -> float:
    if not jd_terms:
        return 0.0
    jd_set = set(jd_terms.keys())
    cand_set = set(candidate_terms.keys())
    if not jd_set:
        return 0.0
    intersection = jd_set & cand_set
    return len(intersection) / len(jd_set)
