from typing import Any

from config import DIMENSION_WEIGHTS


def compute_feature_contributions(
    features: dict[str, float],
    dim_scores: dict[str, float],
) -> dict[str, float]:
    contributions: dict[str, float] = {}

    tech_dim = dim_scores.get("technical_match", 0.0)
    sem_dim = dim_scores.get("semantic_match", 0.0)
    career_dim = dim_scores.get("career_quality", 0.0)
    beh_dim = dim_scores.get("behavioral", 0.0)
    ret_dim = dim_scores.get("retention", 0.0)
    risk_dim = dim_scores.get("risk_adjustment", 1.0)

    for dim_name, dim_value, weight in [
        ("technical_match", tech_dim, DIMENSION_WEIGHTS.get("technical_match", 0.0)),
        ("semantic_match", sem_dim, DIMENSION_WEIGHTS.get("semantic_match", 0.0)),
        ("career_quality", career_dim, DIMENSION_WEIGHTS.get("career_quality", 0.0)),
        ("behavioral", beh_dim, DIMENSION_WEIGHTS.get("behavioral", 0.0)),
        ("retention", ret_dim, DIMENSION_WEIGHTS.get("retention", 0.0)),
    ]:
        raw_contrib = dim_value * weight * 100.0
        contributions[dim_name] = round(raw_contrib, 2)

    risk_weight = DIMENSION_WEIGHTS.get("risk_adjustment", 0.0)
    contributions["risk_penalty"] = round(risk_dim * risk_weight * 100.0, 2)

    tech_contrib = contributions.get("technical_match", 0.0)
    core_portion = 0.50
    depth_portion = 0.35
    diversity_portion = 0.15
    contributions["tech_core_keywords"] = round(tech_contrib * core_portion, 2)
    contributions["tech_depth_ai"] = round(
        tech_contrib * depth_portion * 0.35 * (features.get("ai_depth", 0.0) or 0.0), 2
    )
    contributions["tech_depth_retrieval"] = round(
        tech_contrib * depth_portion * 0.40 * (features.get("retrieval_depth", 0.0) or 0.0), 2
    )
    contributions["tech_depth_eval"] = round(
        tech_contrib * depth_portion * 0.15 * (features.get("eval_depth", 0.0) or 0.0), 2
    )
    contributions["tech_diversity"] = round(
        tech_contrib * diversity_portion * (features.get("keyword_diversity", 0.0) or 0.0), 2
    )

    beh_contrib = contributions.get("behavioral", 0.0)
    contributions["beh_engagement"] = round(beh_contrib * 0.5, 2)
    contributions["beh_activity"] = round(beh_contrib * 0.3, 2)
    contributions["beh_verification"] = round(beh_contrib * 0.2, 2)

    career_contrib = contributions.get("career_quality", 0.0)
    prestige = features.get("company_prestige", 0.0)
    contributions["career_prestige_bonus"] = round(career_contrib * prestige * 0.3, 2)

    edu_level = features.get("education_level", 0.0)
    contributions["education_boost"] = round(edu_level * 2.0, 2)

    certs = features.get("certifications", 0.0)
    contributions["certification_boost"] = round(min(certs * 1.0, 3.0), 2)

    risk_score = features.get("risk_score", 0.0)
    anti = features.get("anti_pattern_count", 0)
    contributions["risk_penalty_detail"] = round(-(risk_score * 20.0 + anti * 3.0), 2)

    sem_val = features.get("semantic_similarity", 0.0)
    contributions["semantic_similarity"] = round(sem_val * 10.0, 2)

    total = sum(v for v in contributions.values() if v > 0)
    if total > 0:
        for k in contributions:
            contributions[k] = round(contributions[k], 2)

    return contributions


def explain_ranking(
    features: dict[str, float],
    dim_scores: dict[str, float],
    final_score: float,
) -> dict[str, Any]:
    contributions = compute_feature_contributions(features, dim_scores)

    strengths: list[str] = []
    weaknesses: list[str] = []

    tech = dim_scores.get("technical_match", 0.0)
    if tech > 0.6:
        strengths.append("Strong technical alignment with job requirements")
    elif tech < 0.2:
        weaknesses.append("Weak technical keyword match")

    beh = dim_scores.get("behavioral", 0.0)
    if beh > 0.7:
        strengths.append("High platform engagement and credibility")
    elif beh < 0.3:
        weaknesses.append("Low platform activity signals")

    prestige = features.get("company_prestige", 0.0)
    if prestige >= 1.0:
        strengths.append("Experience at top-tier technology companies")
    elif prestige >= 0.7:
        strengths.append("Strong company background")

    edu = features.get("education_level", 0.0)
    if edu >= 0.8:
        strengths.append("Advanced degree (PhD level)")
    elif edu >= 0.5:
        strengths.append("Graduate degree")

    risk = features.get("risk_score", 0.0)
    if risk > 0.3:
        weaknesses.append("Profile risk flags detected")

    certs = features.get("certifications", 0.0)
    if certs > 0:
        strengths.append(f"{int(certs)} professional certification(s)")

    growth = features.get("growth_rate", 0.5)
    if growth > 0.7:
        strengths.append("Strong career progression trajectory")

    ret = dim_scores.get("retention", 0.0)
    if ret > 0.7:
        strengths.append("High retention probability")
    elif ret < 0.3:
        weaknesses.append("Potential retention risk")

    return {
        "final_score": round(final_score, 2),
        "dimension_scores": {k: round(float(v), 4) for k, v in dim_scores.items() if v is not None},
        "feature_contributions": contributions,
        "top_positive": sorted(
            [(k, v) for k, v in contributions.items() if v > 0],
            key=lambda x: -x[1],
        )[:5],
        "top_negative": sorted(
            [(k, v) for k, v in contributions.items() if v < 0],
            key=lambda x: x[1],
        )[:3],
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
