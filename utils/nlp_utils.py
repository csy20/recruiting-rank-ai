import re
from collections import Counter
from re import Pattern

_PROD_INDICATORS: list[Pattern] = [
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


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_WHITESPACE_RE.findall(text.lower()))


def _make_boundary_pattern(keyword: str) -> Pattern:
    escaped = re.escape(keyword)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def compile_keyword_patterns(keywords: list[str]) -> list[Pattern]:
    return [_make_boundary_pattern(kw) for kw in keywords]


_TECH_KEYWORDS = {
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
_TECH_PATTERNS: list[Pattern] = [_make_boundary_pattern(t) for t in _TECH_KEYWORDS]


def count_pattern_matches(text: str, patterns: list[Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for p in patterns if p.search(text))


def count_keyword_matches(text: str | None, keywords: list[str]) -> int:
    if not text:
        return 0
    patterns = compile_keyword_patterns(keywords)
    return count_pattern_matches(text, patterns)


def count_ngram_matches(text: str | None, keywords: list[str]) -> float:
    if not text:
        return 0.0
    if not keywords:
        return 0.0
    patterns = compile_keyword_patterns(keywords)
    matches = count_pattern_matches(text, patterns)
    return matches / len(keywords)


def extract_years(text: str | None) -> float:
    if not text:
        return 0.0
    match = _YEARS_RE.search(text)
    return float(match.group(1)) if match else 0.0


def has_production_indicators(text: str | None) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _PROD_INDICATORS)


def extract_key_phrases(text: str, top_n: int = 50) -> list[tuple[str, float]]:
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{2,}", text.lower())
    if not tokens:
        return []
    counts = Counter(tokens)
    total = sum(counts.values())
    return [(word, count / total) for word, count in counts.most_common(top_n)]


def extract_technical_terms(text: str) -> dict[str, float]:
    result = {}
    text_lower = text.lower()
    for term, pattern in zip(_TECH_KEYWORDS, _TECH_PATTERNS, strict=True):
        matches = pattern.findall(text_lower)
        if matches:
            result[term] = float(len(matches))
    return result


def compute_jd_candidate_overlap(
    jd_terms: dict[str, float], candidate_terms: dict[str, float]
) -> float:
    if not jd_terms:
        return 0.0
    jd_set = set(jd_terms.keys())
    cand_set = set(candidate_terms.keys())
    if not jd_set:
        return 0.0
    intersection = jd_set & cand_set
    return len(intersection) / len(jd_set)
