import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from collections import Counter

from config import JD_KEYWORDS, JD_ANTI_PATTERNS, CONSULTING_COMPANIES
from utils.nlp_utils import (
    tokenize,
    count_keyword_matches,
    count_ngram_matches,
    has_production_indicators,
)


REFERENCE_DATE = date(2026, 6, 1)


def load_candidates(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if limit:
                return data[:limit]
            return data
        else:
            candidates = []
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if limit and i >= limit:
                    break
                candidates.append(json.loads(line))
            return candidates


def _get_combined_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    texts = []
    texts.append(profile.get("headline", ""))
    texts.append(profile.get("summary", ""))
    for role in candidate.get("career_history", []):
        texts.append(role.get("description", ""))
        texts.append(role.get("title", ""))
    for skill in candidate.get("skills", []):
        texts.append(skill.get("name", ""))
    return " ".join(texts)


def _get_all_skill_names(candidate: Dict[str, Any]) -> List[str]:
    return [s.get("name", "").lower() for s in candidate.get("skills", [])]


def _get_skill_experience_months(candidate: Dict[str, Any]) -> int:
    total = 0
    for s in candidate.get("skills", []):
        total += s.get("duration_months", 0) or 0
    return total


def _is_at_consulting(candidate: Dict[str, Any]) -> bool:
    profile = candidate.get("profile", {})
    company = (profile.get("current_company", "") or "").lower().strip()
    return company in CONSULTING_COMPANIES


def _has_product_experience(candidate: Dict[str, Any]) -> bool:
    for role in candidate.get("career_history", []):
        company = (role.get("company", "") or "").lower().strip()
        industry = (role.get("industry", "") or "").lower().strip()
        if company not in CONSULTING_COMPANIES:
            if "product" not in industry and "services" not in industry:
                return True
            if "product" in industry:
                return True
    return False


def _career_entirely_consulting(candidate: Dict[str, Any]) -> bool:
    for role in candidate.get("career_history", []):
        company = (role.get("company", "") or "").lower().strip()
        if company not in CONSULTING_COMPANIES:
            return False
    return True


def _count_distinct_companies(candidate: Dict[str, Any]) -> int:
    companies = set()
    for role in candidate.get("career_history", []):
        companies.add(role.get("company", ""))
    return len(companies)


def _avg_tenure_months(candidate: Dict[str, Any]) -> float:
    tenures = []
    for role in candidate.get("career_history", []):
        dur = role.get("duration_months", 0)
        if dur and dur > 0:
            tenures.append(dur)
    if not tenures:
        return 0.0
    return sum(tenures) / len(tenures)


def _title_has_keywords(title: str, keywords: List[str]) -> bool:
    title_lower = title.lower()
    for kw in keywords:
        if kw in title_lower:
            return True
    return False


def _compute_jd_technical_match(text: str) -> Dict[str, float]:
    scores = {}
    for dimension, config in JD_KEYWORDS.items():
        score = count_ngram_matches(text, config["terms"])
        scores[dimension] = score
    return scores


def _detect_anti_patterns(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for pattern in JD_ANTI_PATTERNS:
        if pattern.lower() in text_lower:
            count += 1
    return count


def _extract_ai_depth(candidate: Dict[str, Any]) -> float:
    text = _get_combined_text(candidate)
    ai_terms = [
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
        "fine-tun",
        "lora",
        "rag",
    ]
    match_count = count_keyword_matches(text, ai_terms)
    depth = min(match_count / 5.0, 1.0)
    prod = has_production_indicators(text)
    if prod:
        depth = min(depth * 1.5, 1.0)
    has_ml_title = _title_has_keywords(
        candidate.get("profile", {}).get("current_title", ""),
        [
            "machine learning",
            "ml engineer",
            "ai engineer",
            "data scientist",
            "nlp engineer",
            "research engineer",
            "applied scientist",
        ],
    )
    if has_ml_title:
        depth = min(depth * 1.3, 1.0)
    return depth


def _extract_retrieval_depth(candidate: Dict[str, Any]) -> float:
    text = _get_combined_text(candidate)
    retrieval_terms = [
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
    match_count = count_keyword_matches(text, retrieval_terms)
    depth = min(match_count / 4.0, 1.0)
    prod = has_production_indicators(text)
    if prod:
        depth = min(depth * 2.0, 1.0)
    return depth


def _extract_evaluation_depth(candidate: Dict[str, Any]) -> float:
    text = _get_combined_text(candidate)
    eval_terms = [
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
    match_count = count_keyword_matches(text, eval_terms)
    depth = min(match_count / 3.0, 1.0)
    return depth


def _compute_career_progression(candidate: Dict[str, Any]) -> Tuple[float, float]:
    roles = candidate.get("career_history", [])
    if len(roles) < 2:
        return 0.5, 0.5
    seniority_keywords = [
        "senior",
        "staff",
        "principal",
        "lead",
        "head",
        "chief",
        "manager",
        "director",
        "architect",
        "fellow",
    ]
    scores = []
    for role in roles:
        title = role.get("title", "").lower()
        score = sum(1 for kw in seniority_keywords if kw in title)
        scores.append(score)
    progression = 0.0
    if len(scores) >= 2:
        increases = sum(1 for i in range(1, len(scores)) if scores[i] >= scores[i - 1])
        progression = increases / (len(scores) - 1)
    avg_seniority = sum(scores) / len(scores) if scores else 0.0
    normalized_seniority = min(avg_seniority / 2.0, 1.0)
    return progression, normalized_seniority


def _compute_behavioral_score(signals: Dict[str, Any]) -> Dict[str, float]:
    components = {}
    signals = signals or {}

    rr = signals.get("recruiter_response_rate", 0.0) or 0.0
    components["recruiter_response_rate"] = rr

    icr = signals.get("interview_completion_rate", 0.0) or 0.0
    components["interview_completion_rate"] = icr

    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    components["open_to_work_flag"] = otw

    saved = signals.get("saved_by_recruiters_30d", 0) or 0
    components["saved_by_recruiters_30d"] = min(saved / 20.0, 1.0)

    search_app = signals.get("search_appearance_30d", 0) or 0
    components["search_appearance_30d"] = min(search_app / 50.0, 1.0)

    gh = signals.get("github_activity_score", -1) or -1
    if gh < 0:
        gh_score = 0.0
    else:
        gh_score = gh / 100.0
    components["github_activity_score"] = gh_score

    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            la_date = datetime.strptime(last_active[:10], "%Y-%m-%d").date()
            days_since = (REFERENCE_DATE - la_date).days
            recency = max(0.0, 1.0 - days_since / 365.0)
        except (ValueError, TypeError):
            recency = 0.0
    else:
        recency = 0.0
    components["recent_activity"] = recency

    pcs = signals.get("profile_completeness_score", 0) or 0
    components["profile_completeness"] = pcs / 100.0

    return components


def _compute_retention_score(candidate: Dict[str, Any]) -> Dict[str, float]:
    roles = candidate.get("career_history", [])
    if len(roles) <= 1:
        avg_tenure = _avg_tenure_months(candidate)
    else:
        avg_tenure = _avg_tenure_months(candidate)

    tenure_years = avg_tenure / 12.0
    if 2.0 <= tenure_years <= 5.0:
        tenure_score = 1.0
    elif tenure_years > 5.0:
        tenure_score = 0.8
    elif tenure_years > 1.0:
        tenure_score = tenure_years / 2.0
    else:
        tenure_score = 0.2

    job_hop_penalty = 0.0
    if len(roles) >= 3:
        short_stints = sum(1 for r in roles if (r.get("duration_months", 0) or 0) < 12)
        job_hop_penalty = min(short_stints / len(roles), 1.0) * 0.5

    notice_period = (
        candidate.get("redrob_signals", {}).get("notice_period_days", 30) or 30
    )
    notice_score = max(0.0, 1.0 - (notice_period - 30) / 150.0)

    overall = max(0.0, min(1.0, tenure_score - job_hop_penalty))
    return {
        "tenure_score": tenure_score,
        "job_hop_penalty": job_hop_penalty,
        "notice_score": notice_score,
        "overall": overall,
    }


def _detect_honeypot(candidate: Dict[str, Any]) -> Dict[str, Any]:
    flags = []
    risk_score = 0.0

    career = candidate.get("career_history", [])
    for role in career:
        if role.get("is_current", False):
            continue
        start = role.get("start_date", "")
        dur = role.get("duration_months", 0) or 0
        if start and dur > 0:
            try:
                start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
                end_dt = role.get("end_date")
                if end_dt:
                    actual_end = datetime.strptime(end_dt[:10], "%Y-%m-%d")
                else:
                    actual_end = datetime(2026, 6, 1)
                if dur > 60:
                    risk_score += 0.2
                    flags.append("inflated_duration")
            except (ValueError, TypeError):
                pass

    skills = candidate.get("skills", [])
    total_exp_years = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    total_exp_months = total_exp_years * 12

    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    low_duration_expert = [
        s for s in expert_skills if (s.get("duration_months", 0) or 0) < 6
    ]
    if len(expert_skills) >= 6 and len(low_duration_expert) >= 3:
        risk_score += 0.3
        flags.append("inflated_skills")

    if total_exp_months > 0:
        all_durations = [
            s.get("duration_months", 0) or 0 for s in skills if s.get("duration_months")
        ]
        if all_durations:
            max_skill_dur = max(all_durations)
            if max_skill_dur > total_exp_months * 1.5:
                risk_score += 0.3
                flags.append("skill_exceeds_career")

    profile = candidate.get("profile", {})
    summary = profile.get("summary", "").lower()
    current_title = profile.get("current_title", "").lower()

    ai_keywords_in_summary = [
        "ai",
        "machine learning",
        "deep learning",
        "llm",
        "artificial intelligence",
        "nlp",
    ]
    has_ai_in_text = any(kw in summary for kw in ai_keywords_in_summary)
    is_non_tech_title = any(
        t in current_title
        for t in [
            "marketing",
            "manager",
            "hr",
            "sales",
            "accountant",
            "content",
            "writer",
            "designer",
            "support",
            "executive",
        ]
    )
    if has_ai_in_text and is_non_tech_title:
        risk_score += 0.3
        flags.append("keyword_stuffer")

    return {
        "risk_score": min(risk_score, 1.0),
        "flags": flags,
        "is_honeypot": risk_score >= 0.6,
    }


def _compute_location_score(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()

    preferred_cities = [
        "pune",
        "noida",
        "delhi",
        "gurgaon",
        "gurugram",
        "hyderabad",
        "mumbai",
        "bangalore",
        "bengaluru",
        "chennai",
        "kolkata",
        "ahmedabad",
    ]
    if any(city in location for city in preferred_cities):
        return 1.0
    signals = candidate.get("redrob_signals", {})
    if signals.get("willing_to_relocate", False):
        return 0.7
    if country == "india":
        return 0.5
    return 0.2


def extract_all_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    text = _get_combined_text(candidate)
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = _get_all_skill_names(candidate)

    jd_scores = _compute_jd_technical_match(text)
    ai_depth = _extract_ai_depth(candidate)
    retrieval_depth = _extract_retrieval_depth(candidate)
    eval_depth = _extract_evaluation_depth(candidate)
    anti_patterns = _detect_anti_patterns(text)

    progression, seniority = _compute_career_progression(candidate)
    behavioral = _compute_behavioral_score(signals)
    retention = _compute_retention_score(candidate)
    honeypot = _detect_honeypot(candidate)

    features = {
        "years_experience": profile.get("years_of_experience", 0) or 0,
        "num_skills": len(skills),
        "expert_skills": sum(
            1 for s in candidate.get("skills", []) if s.get("proficiency") == "expert"
        ),
        "num_career_entries": len(candidate.get("career_history", [])),
        "num_companies": _count_distinct_companies(candidate),
        "entirely_consulting": 1.0 if _career_entirely_consulting(candidate) else 0.0,
        "has_product_exp": 1.0 if _has_product_experience(candidate) else 0.0,
        "is_at_consulting": 1.0 if _is_at_consulting(candidate) else 0.0,
        "avg_tenure_months": _avg_tenure_months(candidate),
        "skill_total_months": _get_skill_experience_months(candidate),
        "text_length": len(text),
        "location_score": _compute_location_score(candidate),
        "anti_pattern_count": anti_patterns,
    }

    for dim, score in jd_scores.items():
        features[f"jd_match_{dim}"] = score
    features["ai_depth"] = ai_depth
    features["retrieval_depth"] = retrieval_depth
    features["eval_depth"] = eval_depth
    features["career_progression"] = progression
    features["career_seniority"] = seniority
    for k, v in behavioral.items():
        features[f"beh_{k}"] = v
    for k, v in retention.items():
        features[f"ret_{k}"] = v
    features["risk_score"] = honeypot["risk_score"]
    features["is_honeypot"] = 1.0 if honeypot["is_honeypot"] else 0.0

    return features
