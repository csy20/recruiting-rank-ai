import numpy as np
from typing import Dict, List, Tuple

from config import DIMENSION_WEIGHTS


def compute_dimension_scores(features: Dict[str, float]) -> Dict[str, float]:
    scores = {}

    jd_dims = {
        "jd_match_embeddings": 3.0,
        "jd_match_vector_db": 3.0,
        "jd_match_ranking": 3.0,
        "jd_match_ml_production": 2.5,
        "jd_match_nlp_ir": 2.0,
        "jd_match_python": 1.5,
        "jd_match_llm": 1.5,
        "jd_match_distributed_systems": 1.0,
        "jd_match_data_engineering": 0.5,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for dim, w in jd_dims.items():
        weighted_sum += features.get(dim, 0.0) * w
        total_weight += w
    core_tech_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    retrieval_boost = features.get("retrieval_depth", 0.0)
    ai_boost = features.get("ai_depth", 0.0)
    eval_boost = features.get("eval_depth", 0.0)

    depth_weighted = (
        0.45 * min(retrieval_boost * 1.5, 1.0)
        + 0.35 * min(ai_boost * 1.3, 1.0)
        + 0.20 * min(eval_boost * 1.5, 1.0)
    )

    tech_match = 0.55 * core_tech_score + 0.45 * depth_weighted
    tech_match = min(tech_match, 1.0)
    scores["technical_match"] = tech_match

    has_product = features.get("has_product_exp", 0.0)
    is_consulting = features.get("entirely_consulting", 0.0)
    consulting_penalty = is_consulting * 0.5

    retrieval_depth = features.get("retrieval_depth", 0.0)
    ai_depth = features.get("ai_depth", 0.0)
    eval_depth = features.get("eval_depth", 0.0)

    semantic_transfer = max(retrieval_depth, ai_depth, eval_depth) * 0.6
    exp_years = features.get("years_experience", 0.0)
    exp_band = (
        1.0 if 4.0 <= exp_years <= 10.0 else max(0.0, 1.0 - abs(exp_years - 7.0) / 10.0)
    )

    semantic_match = max(
        0.0,
        min(1.0, semantic_transfer + 0.2 * exp_band - consulting_penalty),
    )
    scores["semantic_match"] = semantic_match

    progression = features.get("career_progression", 0.5)
    seniority = features.get("career_seniority", 0.5)
    has_product = features.get("has_product_exp", 0.0)
    is_consulting = features.get("entirely_consulting", 0.0)

    career_quality = (
        0.35 * progression
        + 0.25 * seniority
        + 0.25 * has_product
        - 0.15 * is_consulting
    )
    career_quality = max(0.0, min(1.0, career_quality))
    scores["career_quality"] = career_quality

    beh_rr = features.get("beh_recruiter_response_rate", 0.0)
    beh_icr = features.get("beh_interview_completion_rate", 0.0)
    beh_otw = features.get("beh_open_to_work_flag", 0.0)
    beh_saved = features.get("beh_saved_by_recruiters_30d", 0.0)
    beh_search = features.get("beh_search_appearance_30d", 0.0)
    beh_gh = features.get("beh_github_activity_score", 0.0)
    beh_recent = features.get("beh_recent_activity", 0.0)
    beh_completeness = features.get("beh_profile_completeness", 0.0)

    behavioral_score = (
        0.25 * beh_rr
        + 0.15 * beh_icr
        + 0.15 * beh_otw
        + 0.10 * beh_saved
        + 0.10 * beh_search
        + 0.10 * beh_gh
        + 0.10 * beh_recent
        + 0.05 * beh_completeness
    )
    behavioral_score = max(0.0, min(1.0, behavioral_score))
    scores["behavioral"] = behavioral_score

    retention_overall = features.get("ret_overall", 0.5)
    notice_score = features.get("ret_notice_score", 0.5)
    retention_score = 0.6 * retention_overall + 0.4 * notice_score
    retention_score = max(0.0, min(1.0, retention_score))
    scores["retention"] = retention_score

    risk_score = features.get("risk_score", 0.0)
    anti_patterns = features.get("anti_pattern_count", 0)
    is_honeypot = features.get("is_honeypot", 0.0)

    risk_penalty = risk_score * 0.5 + min(anti_patterns * 0.15, 0.3) + is_honeypot * 0.8
    risk_penalty = min(risk_penalty, 1.0)
    risk_adjustment = 1.0 - risk_penalty
    scores["risk_adjustment"] = risk_adjustment

    return scores


def compute_final_score(dim_scores: Dict[str, float]) -> float:
    score = (
        DIMENSION_WEIGHTS["technical_match"] * dim_scores.get("technical_match", 0.0)
        + DIMENSION_WEIGHTS["semantic_match"] * dim_scores.get("semantic_match", 0.0)
        + DIMENSION_WEIGHTS["career_quality"] * dim_scores.get("career_quality", 0.0)
        + DIMENSION_WEIGHTS["behavioral"] * dim_scores.get("behavioral", 0.0)
        + DIMENSION_WEIGHTS["retention"] * dim_scores.get("retention", 0.0)
    )
    risk_adj = dim_scores.get("risk_adjustment", 1.0)
    score = score * risk_adj
    score = max(0.0, min(100.0, score * 100.0))
    return score


def rank_candidates(
    candidate_ids: List[str],
    features_list: List[Dict[str, float]],
) -> List[Tuple[str, float, int, Dict[str, float]]]:
    results = []
    for cid, feats in zip(candidate_ids, features_list):
        dim_scores = compute_dimension_scores(feats)
        final_score = compute_final_score(dim_scores)
        results.append((cid, final_score, dim_scores))

    results.sort(key=lambda x: (-x[1], x[0]))

    ranked = []
    for rank, (cid, score, dims) in enumerate(results, start=1):
        ranked.append((cid, score, rank, dims))
    return ranked


def generate_reasoning(
    cid: str,
    score: float,
    rank: int,
    dim_scores: Dict[str, float],
    features: Dict[str, float],
) -> str:
    parts = []

    tech = dim_scores.get("technical_match", 0.0)
    beh = dim_scores.get("behavioral", 0.0)
    risk = dim_scores.get("risk_adjustment", 1.0)

    if tech > 0.5:
        retrieval = features.get("retrieval_depth", 0.0)
        if retrieval > 0.2:
            parts.append(f"Strong retrieval/vector search background")
        else:
            ai_d = features.get("ai_depth", 0.0)
            if ai_d > 0.2:
                parts.append(f"Solid ML/AI engineering experience")
            else:
                parts.append(f"Relevant technical background")
    elif tech > 0.25:
        parts.append(f"Adjacent technical skills")
    else:
        is_consulting = features.get("entirely_consulting", 0.0)
        if is_consulting:
            parts.append(f"Consulting background, limited product ML experience")
        else:
            parts.append(f"Weak technical match to JD")

    if beh > 0.7:
        parts.append(f"highly engaged on platform")
    elif beh > 0.4:
        parts.append(f"moderate engagement")
    else:
        parts.append(f"low platform activity")

    exp = features.get("years_experience", 0)
    parts.append(f"{exp:.1f}yr exp")

    risk_score = features.get("risk_score", 0.0)
    if risk_score > 0.3:
        parts.append(f"risk flags detected")
    anti = features.get("anti_pattern_count", 0)
    if anti > 0:
        parts.append(f"buzzword concern")

    if features.get("is_honeypot", 0.0) > 0.5:
        parts.append(f"HONEYPOT")

    reasoning = "; ".join(parts) if parts else "No strong signals"
    return reasoning
