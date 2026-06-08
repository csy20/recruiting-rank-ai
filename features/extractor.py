import json
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, date

from config import (
    JD_KEYWORDS,
    JD_ANTI_PATTERNS,
    CONSULTING_COMPANIES,
    CONSULTING_WITH_ML_BOOKS,
    COMPANY_TIERS,
    SKILL_CATEGORIES,
    EDUCATION_PATTERNS,
    EDUCATION_FIELDS,
    CERTIFICATION_PATTERNS,
    REFERENCE_DATE,
    SENIORITY_KEYWORDS,
    PREFERRED_LOCATIONS,
    NON_TECH_TITLES,
    HONEYPOT_THRESHOLD,
    INFLATED_DURATION_MONTHS,
    INFLATED_SKILL_EXPERT_THRESHOLD,
    INFLATED_LOW_DURATION_THRESHOLD,
    INFLATED_SKILL_COUNT_THRESHOLD,
    SHORT_STINT_MONTHS,
    BEHAVIORAL_WEIGHTS,
)
from utils.nlp_utils import (
    tokenize,
    count_keyword_matches,
    count_ngram_matches,
    has_production_indicators,
    compile_keyword_patterns,
    count_pattern_matches,
)
from scoring.skill_graph import compute_skill_match, compute_skill_breadth

_SKILL_CATEGORY_PATTERNS: Dict[str, Any] = {
    cat: compile_keyword_patterns(list(terms))
    for cat, terms in SKILL_CATEGORIES.items()
}

_EDUCATION_LEVEL_PATTERNS: Dict[str, Any] = {
    level: compile_keyword_patterns(info["keywords"])
    for level, info in EDUCATION_PATTERNS.items()
}

_EDUCATION_FIELD_PATTERNS: Dict[str, Any] = {
    field: compile_keyword_patterns(info["keywords"])
    for field, info in EDUCATION_FIELDS.items()
}

_CERT_PATTERNS = compile_keyword_patterns(CERTIFICATION_PATTERNS)

_AI_TERMS = [
    "embedding",
    "vector",
    "retrieval",
    "ranking",
    "ranker",
    "nlp",
    "transformer",
    "attention",
    "neural network",
    "deep learning",
    "machine learning",
    "recommendation",
    "classifier",
    "xgboost",
    "lightgbm",
    "gradient boosting",
    "fine-tune",
    "lora",
    "rag",
]

_RETRIEVAL_TERMS = [
    "elasticsearch",
    "opensearch",
    "solr",
    "lucene",
    "bm25",
    "tf-idf",
    "tfidf",
    "inverted index",
    "vector search",
    "hybrid search",
    "semantic search",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "annoy",
    "hnsw",
    "approximate nearest neighbor",
    "embedding search",
    "dense retrieval",
    "sparse retrieval",
    "colbert",
    "dpr",
    "dense passage",
]

_EVAL_TERMS = [
    "ndcg",
    "mrr",
    "map",
    "precision",
    "recall",
    "evaluation",
    "a/b test",
    "ab test",
    "offline eval",
    "online eval",
    "relevance",
    "ranking metric",
    "cross-validation",
    "holdout",
    "test set",
]

_ML_TITLES = [
    "machine learning",
    "ml engineer",
    "ai engineer",
    "data scientist",
    "nlp engineer",
    "research engineer",
    "applied scientist",
]

_AI_KEYWORDS_IN_SUMMARY = [
    "ai",
    "machine learning",
    "deep learning",
    "llm",
    "artificial intelligence",
    "nlp",
]


def load_candidates(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        with open(path) as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                if not isinstance(data, list):
                    return []
                return data[:limit] if limit else data
    except (json.JSONDecodeError, IOError, OSError) as e:
        raise ValueError(f"Failed to load candidates from {path}: {e}") from e

    candidates: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if limit is not None and i >= limit:
                    break
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError) as e:
        raise ValueError(f"Failed to read candidates file {path}: {e}") from e

    return candidates


def _get_combined_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) or {}
    texts: List[str] = []
    headline = profile.get("headline", "") or ""
    summary = profile.get("summary", "") or ""
    if headline:
        texts.append(headline)
    if summary:
        texts.append(summary)
    for role in candidate.get("career_history") or []:
        if isinstance(role, dict):
            desc = role.get("description", "") or ""
            title = role.get("title", "") or ""
            if desc:
                texts.append(desc)
            if title:
                texts.append(title)
    for skill in candidate.get("skills") or []:
        if isinstance(skill, dict):
            name = skill.get("name", "") or ""
            if name:
                texts.append(name)
    return " ".join(texts)


def _get_all_skill_names(candidate: Dict[str, Any]) -> List[str]:
    return [
        s.get("name", "").lower()
        for s in (candidate.get("skills") or [])
        if isinstance(s, dict) and s.get("name")
    ]


def _get_skill_experience_months(candidate: Dict[str, Any]) -> int:
    total = 0
    for s in candidate.get("skills") or []:
        if isinstance(s, dict):
            total += s.get("duration_months", 0) or 0
    return total


def _get_skill_proficiencies(candidate: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in candidate.get("skills") or []:
        if isinstance(s, dict):
            prof = s.get("proficiency", "beginner") or "beginner"
            counts[prof] = counts.get(prof, 0) + 1
    return counts


def _analyze_career_history(candidate: Dict[str, Any]) -> Dict[str, Any]:
    roles = candidate.get("career_history") or []
    role_count = len(roles)

    if role_count == 0:
        return {
            "distinct_companies": 0,
            "total_experience_months": 0,
            "role_count": 0,
            "entirely_consulting": False,
            "at_consulting": False,
            "has_product_exp": False,
            "avg_tenure_months": 0.0,
            "short_stint_count": 0,
            "career_progression": 0.5,
            "career_seniority": 0.0,
            "max_company_prestige": 0.0,
            "consulting_with_ml": False,
        }

    companies: Set[str] = set()
    total_months = 0
    consulting_roles = 0
    product_roles = 0
    max_prestige = 0.0
    seniority_counts: List[int] = []
    tenures: List[int] = []
    short_stints = 0
    has_ml_consulting = False

    for role in roles:
        if not isinstance(role, dict):
            continue
        company = (role.get("company", "") or "").lower().strip()
        title = role.get("title", "") or ""
        industry = (role.get("industry", "") or "").lower().strip()
        dur = role.get("duration_months", 0) or 0

        if company:
            companies.add(role.get("company", "") or "")
        total_months += dur

        if dur > 0:
            tenures.append(dur)
        if 0 < dur < SHORT_STINT_MONTHS:
            short_stints += 1

        if company in CONSULTING_COMPANIES:
            consulting_roles += 1
            if company in CONSULTING_WITH_ML_BOOKS:
                has_ml_consulting = True
        else:
            if role.get("is_current") or True:
                if "services" not in industry:
                    product_roles += 1

        score = sum(1 for kw in SENIORITY_KEYWORDS if kw.lower() in title.lower())
        seniority_counts.append(score)

        for _tier_name, tier_info in COMPANY_TIERS.items():
            if company in tier_info["companies"]:
                max_prestige = max(max_prestige, tier_info["score"])

    entirely_consulting = consulting_roles == role_count if role_count > 0 else False
    at_consulting = consulting_roles > 0

    distinct_companies = len(companies)
    avg_tenure = (sum(tenures) / len(tenures)) if tenures else 0.0

    progression = 0.5
    normalized_seniority = 0.0
    n = len(seniority_counts)
    if n >= 2:
        increases = sum(
            1 for i in range(1, n) if seniority_counts[i] >= seniority_counts[i - 1]
        )
        progression = increases / (n - 1)
    if n > 0:
        normalized_seniority = min(sum(seniority_counts) / n / 2.0, 1.0)

    return {
        "distinct_companies": distinct_companies,
        "total_experience_months": total_months,
        "role_count": role_count,
        "entirely_consulting": entirely_consulting,
        "at_consulting": at_consulting,
        "has_product_exp": product_roles > 0,
        "avg_tenure_months": avg_tenure,
        "short_stint_count": short_stints,
        "career_progression": progression,
        "career_seniority": normalized_seniority,
        "max_company_prestige": max_prestige,
        "consulting_with_ml": has_ml_consulting,
    }


def _compute_jd_technical_match(text: str) -> Dict[str, float]:
    scores = {}
    for dim, cfg in JD_KEYWORDS.items():
        scores[dim] = count_ngram_matches(text, cfg["terms"])
    return scores


def _detect_anti_patterns(text: str) -> int:
    return count_keyword_matches(text, JD_ANTI_PATTERNS)


def _extract_ai_depth(text: str, candidate: Dict[str, Any]) -> float:
    match_count = count_keyword_matches(text, _AI_TERMS)
    depth = min(match_count / 5.0, 1.0)
    if has_production_indicators(text):
        depth = min(depth * 1.5, 1.0)
    profile = candidate.get("profile") or {}
    current_title = (profile.get("current_title") or "").lower()
    if any(t in current_title for t in _ML_TITLES):
        depth = min(depth * 1.3, 1.0)
    return depth


def _extract_retrieval_depth(text: str) -> float:
    match_count = count_keyword_matches(text, _RETRIEVAL_TERMS)
    depth = min(match_count / 4.0, 1.0)
    if has_production_indicators(text):
        depth = min(depth * 2.0, 1.0)
    return depth


def _extract_evaluation_depth(text: str) -> float:
    match_count = count_keyword_matches(text, _EVAL_TERMS)
    return min(match_count / 3.0, 1.0)


def _compute_behavioral_score(signals: Any) -> Dict[str, float]:
    components: Dict[str, float] = {}
    signals = signals or {}

    rr = signals.get("recruiter_response_rate", 0.0) or 0.0
    components["recruiter_response_rate"] = min(float(rr), 1.0)

    icr = signals.get("interview_completion_rate", 0.0) or 0.0
    components["interview_completion_rate"] = min(float(icr), 1.0)

    components["open_to_work_flag"] = (
        1.0 if signals.get("open_to_work_flag", False) else 0.0
    )

    saved = float(signals.get("saved_by_recruiters_30d", 0) or 0)
    components["saved_by_recruiters_30d"] = min(saved / 20.0, 1.0)

    search_app = float(signals.get("search_appearance_30d", 0) or 0)
    components["search_appearance_30d"] = min(search_app / 50.0, 1.0)

    gh = signals.get("github_activity_score", -1)
    if gh is None:
        gh = -1
    gh = float(gh)
    components["github_activity_score"] = max(0.0, gh / 100.0) if gh >= 0 else 0.0

    last_active = signals.get("last_active_date", "") or ""
    if last_active:
        try:
            la_date = datetime.strptime(str(last_active)[:10], "%Y-%m-%d").date()
            days_since = (REFERENCE_DATE - la_date).days
            components["recent_activity"] = max(0.0, 1.0 - days_since / 365.0)
        except (ValueError, TypeError):
            components["recent_activity"] = 0.0
    else:
        components["recent_activity"] = 0.0

    pcs = float(signals.get("profile_completeness_score", 0) or 0)
    components["profile_completeness"] = min(pcs / 100.0, 1.0)

    oar = signals.get("offer_acceptance_rate", 0.5)
    if oar is None:
        oar = 0.5
    components["offer_acceptance_rate"] = max(0.0, min(float(oar), 1.0))

    art = float(signals.get("avg_response_time_hours", 24) or 24)
    components["avg_response_time_hours"] = max(0.0, 1.0 - art / 72.0)

    verified = 0.0
    if signals.get("verified_email", False):
        verified += 0.5
    if signals.get("verified_phone", False):
        verified += 0.5
    components["verified_contact"] = verified

    connections = int(signals.get("connection_count", 0) or 0)
    components["connection_density"] = min(connections / 500.0, 1.0)

    return components


def _compute_retention_score(
    candidate: Dict[str, Any], career: Dict[str, Any]
) -> Dict[str, float]:
    avg_tenure = career.get("avg_tenure_months", 0.0) or 0.0
    tenure_years = avg_tenure / 12.0 if avg_tenure > 0 else 0.0

    tenure_score = 0.2
    if tenure_years >= 2.0 and tenure_years <= 5.0:
        tenure_score = 1.0
    elif tenure_years > 5.0:
        tenure_score = 0.8
    elif tenure_years > 1.0:
        tenure_score = tenure_years / 2.0

    role_count = career.get("role_count", 0) or 0
    job_hop_penalty = 0.0
    if role_count >= 3:
        short_stint_count = career.get("short_stint_count", 0) or 0
        ratio = short_stint_count / role_count
        job_hop_penalty = min(ratio, 1.0) * 0.5

    notice_period_str = candidate.get("redrob_signals", {}).get(
        "notice_period_days", 30
    )
    try:
        notice_period = (
            float(notice_period_str) if notice_period_str is not None else 30.0
        )
    except (ValueError, TypeError):
        notice_period = 30.0
    if notice_period < 30:
        notice_period = 30.0
    notice_score = max(0.0, 1.0 - (notice_period - 30) / 150.0)

    stability_score = max(0.0, 1.0 - job_hop_penalty)
    overall = max(
        0.0, min(1.0, 0.6 * tenure_score + 0.4 * stability_score - job_hop_penalty)
    )

    return {
        "tenure_score": tenure_score,
        "job_hop_penalty": job_hop_penalty,
        "notice_score": notice_score,
        "overall": overall,
    }


def _detect_honeypot(candidate: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    risk_score = 0.0

    for role in candidate.get("career_history") or []:
        if not isinstance(role, dict):
            continue
        if role.get("is_current", False):
            continue
        start = role.get("start_date", "") or ""
        dur = role.get("duration_months", 0) or 0
        if start and dur > 0:
            try:
                start_dt = datetime.strptime(str(start)[:10], "%Y-%m-%d")
                end_dt = role.get("end_date")
                if end_dt:
                    datetime.strptime(str(end_dt)[:10], "%Y-%m-%d")
                if dur > INFLATED_DURATION_MONTHS:
                    risk_score += 0.2
                    flags.append("inflated_duration")
            except (ValueError, TypeError):
                pass

    skills = candidate.get("skills") or []
    exp_years_profile = candidate.get("profile", {}) or {}
    total_exp_years = float(exp_years_profile.get("years_of_experience", 0) or 0)
    total_exp_months = total_exp_years * 12

    expert_skills = [
        s for s in skills if isinstance(s, dict) and s.get("proficiency") == "expert"
    ]
    low_duration_expert = [
        s
        for s in expert_skills
        if (s.get("duration_months", 0) or 0) < INFLATED_SKILL_EXPERT_THRESHOLD
    ]
    if (
        len(expert_skills) >= INFLATED_SKILL_COUNT_THRESHOLD
        and len(low_duration_expert) >= INFLATED_LOW_DURATION_THRESHOLD
    ):
        risk_score += 0.3
        flags.append("inflated_skills")

    if total_exp_months > 0:
        all_durations = [
            s.get("duration_months", 0) or 0
            for s in skills
            if isinstance(s, dict) and s.get("duration_months")
        ]
        if all_durations:
            max_skill_dur = max(all_durations)
            if max_skill_dur > total_exp_months * 1.5:
                risk_score += 0.3
                flags.append("skill_exceeds_career")

    profile = candidate.get("profile") or {}
    summary = (profile.get("summary") or "").lower()
    current_title = (profile.get("current_title") or "").lower()

    has_ai_in_text = any(kw in summary for kw in _AI_KEYWORDS_IN_SUMMARY)
    is_non_tech_title = any(t in current_title for t in NON_TECH_TITLES)
    if has_ai_in_text and is_non_tech_title:
        has_ml_background = any(
            s.get("name", "").lower()
            in {"python", "machine learning", "data science", "tensorflow", "pytorch"}
            for s in skills
            if isinstance(s, dict)
        )
        if not has_ml_background:
            risk_score += 0.3
            flags.append("keyword_stuffer")

    return {
        "risk_score": min(risk_score, 1.0),
        "flags": flags,
        "is_honeypot": risk_score >= HONEYPOT_THRESHOLD,
    }


def _compute_location_score(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile") or {}
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()

    for _tier_name, tier_info in PREFERRED_LOCATIONS.items():
        cities = tier_info.get("cities")
        if cities and any(city in location for city in cities):
            return tier_info["score"]

    signals = candidate.get("redrob_signals") or {}
    if signals.get("willing_to_relocate", False):
        return 0.70
    if country == "india":
        return PREFERRED_LOCATIONS.get("global_other", {}).get("score", 0.40)
    return PREFERRED_LOCATIONS.get("global_other", {}).get("score", 0.40)


def _detect_education(candidate: Dict[str, Any]) -> Dict[str, float]:
    profile = candidate.get("profile") or {}
    summary = (profile.get("summary") or "").lower()
    text = summary
    for role in candidate.get("career_history") or []:
        if isinstance(role, dict):
            text += " " + ((role.get("title") or "") or "").lower()
    for skill in candidate.get("skills") or []:
        if isinstance(skill, dict):
            text += " " + ((skill.get("name") or "") or "").lower()

    detected_level = 0.0
    for level, patterns in _EDUCATION_LEVEL_PATTERNS.items():
        if count_pattern_matches(text, patterns) > 0:
            level_info = EDUCATION_PATTERNS.get(level, {})
            detected_level = max(detected_level, level_info.get("score", 0.0))

    detected_field = 0.0
    for field, patterns in _EDUCATION_FIELD_PATTERNS.items():
        if count_pattern_matches(text, patterns) > 0:
            field_info = EDUCATION_FIELDS.get(field, {})
            detected_field = max(detected_field, field_info.get("score", 0.0))

    return {
        "education_level": detected_level,
        "education_field": detected_field,
    }


def _detect_certifications(text: str) -> float:
    return float(count_pattern_matches(text, _CERT_PATTERNS))


def _compute_skill_categories(skills: List[str]) -> Dict[str, float]:
    result = {}
    combined = " ".join(skills)
    for category, patterns in _SKILL_CATEGORY_PATTERNS.items():
        result[f"skill_cat_{category}"] = float(
            count_pattern_matches(combined, patterns)
        )
    return result


def _compute_keyword_diversity(text: str) -> float:
    jd_dims = _compute_jd_technical_match(text)
    non_zero = sum(1 for v in jd_dims.values() if v > 0)
    return non_zero / max(len(jd_dims), 1)


def _compute_growth_rate(career: Dict[str, Any]) -> float:
    total_months = career.get("total_experience_months", 0) or 0
    seniority = career.get("career_seniority", 0.0) or 0.0
    role_count = career.get("role_count", 0) or 0
    if total_months <= 0 or role_count <= 0:
        return 0.5
    years = total_months / 12.0
    if years <= 0:
        return 0.5
    return min(seniority * 2.0 / years, 1.0)


def extract_all_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(candidate, dict):
        return {}

    text = _get_combined_text(candidate)
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    skills = _get_all_skill_names(candidate)
    career = _analyze_career_history(candidate)

    jd_scores = _compute_jd_technical_match(text)
    retrieval_depth = _extract_retrieval_depth(text)
    ai_depth = _extract_ai_depth(text, candidate)
    eval_depth = _extract_evaluation_depth(text)
    anti_patterns = _detect_anti_patterns(text)
    keyword_diversity = _compute_keyword_diversity(text)

    behavioral = _compute_behavioral_score(signals)
    retention = _compute_retention_score(candidate, career)
    honeypot = _detect_honeypot(candidate)
    education = _detect_education(candidate)
    certifications = _detect_certifications(text)
    skill_cats = _compute_skill_categories(skills)
    growth_rate = _compute_growth_rate(career)

    features: Dict[str, float] = {
        "years_experience": float(profile.get("years_of_experience", 0) or 0),
        "num_skills": float(len(skills)),
        "expert_skills": float(
            sum(
                1
                for s in (candidate.get("skills") or [])
                if isinstance(s, dict) and s.get("proficiency") == "expert"
            )
        ),
        "num_career_entries": float(career["role_count"]),
        "num_companies": float(career["distinct_companies"]),
        "entirely_consulting": 1.0 if career["entirely_consulting"] else 0.0,
        "has_product_exp": 1.0 if career["has_product_exp"] else 0.0,
        "is_at_consulting": 1.0 if career["at_consulting"] else 0.0,
        "avg_tenure_months": career["avg_tenure_months"],
        "skill_total_months": float(_get_skill_experience_months(candidate)),
        "text_length": float(len(text)),
        "location_score": _compute_location_score(candidate),
        "anti_pattern_count": float(anti_patterns),
        "company_prestige": career["max_company_prestige"],
        "keyword_diversity": keyword_diversity,
        "growth_rate": growth_rate,
    }

    for dim, score in jd_scores.items():
        features[f"jd_match_{dim}"] = score
    features["ai_depth"] = ai_depth
    features["retrieval_depth"] = retrieval_depth
    features["eval_depth"] = eval_depth
    features["career_progression"] = career["career_progression"]
    features["career_seniority"] = career["career_seniority"]
    features["consulting_with_ml"] = 1.0 if career["consulting_with_ml"] else 0.0
    for k, v in behavioral.items():
        features[f"beh_{k}"] = v
    for k, v in retention.items():
        features[f"ret_{k}"] = v
    features["risk_score"] = honeypot["risk_score"]
    features["is_honeypot"] = 1.0 if honeypot["is_honeypot"] else 0.0
    features["education_level"] = education["education_level"]
    features["education_field"] = education["education_field"]
    features["certifications"] = certifications
    features.update(skill_cats)

    skill_match_score, exact_matches, transferable_matches = compute_skill_match(
        skills,
        list(JD_KEYWORDS.keys()),
    )
    features["skill_match_score"] = skill_match_score
    features["skill_exact_matches"] = float(len(exact_matches))
    features["skill_transferable_matches"] = float(len(transferable_matches))

    breadth = compute_skill_breadth(skills)
    features["skill_category_count"] = breadth.get("category_count", 0.0)
    for cat_key, cat_val in breadth.items():
        if cat_key.startswith("cat_"):
            features[f"skill_breadth_{cat_key.replace('cat_', '')}"] = cat_val

    return features
