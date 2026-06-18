import re
from collections import Counter
from typing import Any

from config import JD_KEYWORDS
from utils.nlp_utils import (
    compile_keyword_patterns,
    count_pattern_matches,
    extract_technical_terms,
)

_SECTION_PATTERNS = {
    "about": re.compile(r"(about|overview|summary|intro)", re.IGNORECASE),
    "requirements": re.compile(
        r"(requirements|qualifications|what you need|required skills|"
        r"basic qualifications|minimum qualifications)",
        re.IGNORECASE,
    ),
    "preferred": re.compile(
        r"(preferred|nice to have|bonus points|good to have|plus)",
        re.IGNORECASE,
    ),
    "responsibilities": re.compile(
        r"(responsibilities|what you will do|role|key duties|"
        r"day-to-day|the opportunity)",
        re.IGNORECASE,
    ),
    "benefits": re.compile(
        r"(benefits|perks|compensation|what we offer)",
        re.IGNORECASE,
    ),
}

_JD_KEYWORD_PATTERNS: dict[str, list[Any]] = {
    dim: compile_keyword_patterns(cfg["terms"]) for dim, cfg in JD_KEYWORDS.items()
}

_EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+)\+?\s*years?\s*(of\s*)?(experience|work)"),
    re.compile(r"(\d+)\s*-\s*(\d+)\s*years?"),
    re.compile(r"at\s*least\s*(\d+)\s*years?"),
]


def parse_jd(jd_text: str) -> dict[str, Any]:
    if not jd_text:
        return {
            "keywords": {},
            "experience_years": (0, 20),
            "sections": {},
            "raw_terms": {},
        }

    sections = _split_sections(jd_text)
    required_terms = _extract_section_terms(sections.get("requirements", ""))
    preferred_terms = _extract_section_terms(sections.get("preferred", ""))
    all_terms = _extract_section_terms(jd_text)

    exp_range = _extract_experience_range(jd_text)
    tech_terms = extract_technical_terms(jd_text)

    dimension_scores = {}
    for dim, cfg in JD_KEYWORDS.items():
        patterns = _JD_KEYWORD_PATTERNS[dim]
        req_matches = count_pattern_matches(sections.get("requirements", ""), patterns)
        pref_matches = count_pattern_matches(sections.get("preferred", ""), patterns)
        all_matches = count_pattern_matches(jd_text, patterns)

        weight = cfg["weight"]
        if req_matches > 0:
            weight *= 1.5
        if pref_matches > 0:
            weight *= 1.2
        dimension_scores[dim] = {
            "base_weight": cfg["weight"],
            "adjusted_weight": weight,
            "required_matches": req_matches,
            "preferred_matches": pref_matches,
            "total_matches": all_matches,
        }

    return {
        "keywords": JD_KEYWORDS,
        "dimension_analysis": dimension_scores,
        "experience_years": exp_range,
        "sections": {k: v for k, v in sections.items() if v},
        "required_terms": required_terms,
        "preferred_terms": preferred_terms,
        "technical_terms": tech_terms,
        "raw_terms": all_terms,
        "text_length": len(jd_text),
    }


def _split_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    sections: dict[str, str] = {}
    current_section = "about"
    current_lines: list[str] = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        matched_section = None
        for section_name, pattern in _SECTION_PATTERNS.items():
            if pattern.search(line_stripped) and len(line_stripped) < 100:
                matched_section = section_name
                break
        if matched_section:
            if current_lines:
                sections[current_section] = " ".join(current_lines)
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line_stripped)

    if current_lines:
        sections[current_section] = " ".join(current_lines)

    return sections


def _extract_section_terms(text: str) -> dict[str, float]:
    if not text:
        return {}
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{2,}", text.lower())
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {word: count / total for word, count in counts.most_common(30)}


def _extract_experience_range(text: str) -> tuple[float, float]:
    matches = []
    for pattern in _EXPERIENCE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if groups[0]:
                if groups[1]:
                    try:
                        matches.append((float(groups[0]), float(groups[1])))
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        val = float(groups[0])
                        matches.append((val, val + 2))
                    except (ValueError, IndexError):
                        pass

    if not matches:
        return (0.0, 20.0)

    min_years = min(m[0] for m in matches)
    max_years = max(m[1] for m in matches)
    return (min_years, max_years)


def get_jd_dimension_weights(jd_profile: dict[str, Any]) -> dict[str, float]:
    dim_analysis = jd_profile.get("dimension_analysis", {})
    weights = {}
    for dim, info in dim_analysis.items():
        weights[dim] = info["adjusted_weight"]
    if not weights:
        for dim, cfg in JD_KEYWORDS.items():
            weights[dim] = cfg["weight"]
    total = sum(weights.values()) or 1
    return {k: v / total for k, v in weights.items()}


def get_jd_experience_score(exp_years: float, jd_profile: dict[str, Any]) -> float:
    exp_range = jd_profile.get("experience_years", (0.0, 20.0))
    min_exp, max_exp = exp_range
    spread = max(max_exp - min_exp, 1.0)

    if min_exp <= exp_years <= max_exp:
        return 1.0
    distance = min(abs(exp_years - min_exp), abs(exp_years - max_exp))
    score = max(0.0, 1.0 - (distance / spread))
    return score
