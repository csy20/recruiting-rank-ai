import logging
import os
from datetime import date, datetime
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    BEHAVIORAL_WEIGHTS,
    DIMENSION_WEIGHTS,
    ENSEMBLE_WEIGHTS,
    FAIRNESS_CONFIG,
    JD_KEYWORDS,
    MODEL_CONFIG,
    MODEL_PATH,
    REFERENCE_DATE,
    SCORING,
)

logger = logging.getLogger("rank.scoring")


def _compute_technical_match(
    features: dict[str, float],
    jd_weights: dict[str, float] | None = None,
) -> float:
    t = SCORING["technical_match"]
    jd_dims = {}
    for dim_name, dim_cfg in JD_KEYWORDS.items():
        jd_dims[f"jd_match_{dim_name}"] = dim_cfg["weight"]
    if jd_weights:
        for dim in jd_dims:
            jd_key = dim.replace("jd_match_", "")
            if jd_key in jd_weights:
                jd_dims[dim] = jd_weights[jd_key] * 5.0

    weighted_sum = 0.0
    total_weight = 0.0
    for dim, w in jd_dims.items():
        weighted_sum += features.get(dim, 0.0) * w
        total_weight += w
    core_tech_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    retrieval_boost = features.get("retrieval_depth", 0.0)
    ai_boost = features.get("ai_depth", 0.0)
    eval_boost = features.get("eval_depth", 0.0)
    keyword_diversity = features.get("keyword_diversity", 0.0)

    depth_weighted = (
        t["retrieval_subweight"] * min(retrieval_boost * t["retrieval_boost_factor"], 1.0)
        + t["ai_subweight"] * min(ai_boost * t["ai_boost_factor"], 1.0)
        + t["eval_subweight"] * min(eval_boost * t["eval_boost_factor"], 1.0)
        + t["diversity_subweight"] * keyword_diversity
    )

    tech_match = (
        t["core_tech_weight"] * core_tech_score
        + t["depth_weight"] * depth_weighted
        + t["keyword_diversity_weight"] * keyword_diversity
    )
    tech_match = min(tech_match, 1.0)

    sem_sim = features.get("semantic_similarity", 0.0)
    tech_match = 0.90 * tech_match + 0.10 * sem_sim
    return min(tech_match, 1.0)


def _compute_semantic_match(features: dict[str, float]) -> float:
    s = SCORING["semantic_match"]
    is_consulting = features.get("entirely_consulting", 0.0)
    consulting_with_ml = features.get("consulting_with_ml", 0.0)
    consulting_penalty = is_consulting * s["consulting_penalty"]
    if consulting_with_ml:
        consulting_penalty *= 1.0 - s["consulting_ml_credit"]

    retrieval_depth = features.get("retrieval_depth", 0.0)
    ai_depth = features.get("ai_depth", 0.0)
    eval_depth = features.get("eval_depth", 0.0)

    semantic_transfer = max(retrieval_depth, ai_depth, eval_depth) * s["transferable_weight"]
    exp_years = features.get("years_experience", 0.0)
    center = s["ideal_exp_center"]
    spread = center * 2.0
    exp_band = max(0.0, 1.0 - abs(exp_years - center) / spread)

    return max(
        0.0, min(1.0, semantic_transfer + s["exp_band_weight"] * exp_band - consulting_penalty)
    )


def _compute_career_quality(features: dict[str, float]) -> float:
    c = SCORING["career_quality"]
    progression = features.get("career_progression", 0.5)
    seniority = features.get("career_seniority", 0.5)
    has_product = features.get("has_product_exp", 0.0)
    is_consulting = features.get("entirely_consulting", 0.0)
    prestige = features.get("company_prestige", 0.0)
    growth_rate = features.get("growth_rate", 0.5)

    career_quality = (
        c["progression_weight"] * progression
        + c["seniority_weight"] * seniority
        + c["product_weight"] * has_product
        + c["company_prestige_weight"] * prestige
        + c["skill_growth_weight"] * growth_rate
        - c["consulting_penalty"] * is_consulting
    )
    return max(0.0, min(1.0, career_quality))


def _compute_behavioral(features: dict[str, float]) -> float:
    behavioral_score = 0.0
    total_bw = 0.0
    for key, weight in BEHAVIORAL_WEIGHTS.items():
        val = features.get(f"beh_{key}", 0.0)
        if val is None:
            val = 0.0
        behavioral_score += float(val) * weight
        total_bw += weight
    behavioral_score = behavioral_score / total_bw if total_bw > 0 else 0.0
    return max(0.0, min(1.0, behavioral_score))


def _compute_retention(features: dict[str, float]) -> float:
    r = SCORING["retention"]
    retention_overall = features.get("ret_overall", 0.5)
    notice_score = features.get("ret_notice_score", 0.5)
    tenure_score = features.get("ret_tenure_score", 0.5)
    retention_score = (
        r["tenure_weight"] * retention_overall
        + r["notice_weight"] * notice_score
        + r["stability_weight"] * tenure_score
    )
    return max(0.0, min(1.0, retention_score))


def _compute_risk_adjustment(features: dict[str, float]) -> float:
    if features.get("high_confidence_honeypot", 0.0) > 0.5:
        return 0.05
    risk = SCORING["risk"]
    risk_score = features.get("risk_score", 0.0)
    anti_patterns = features.get("anti_pattern_count", 0)
    is_honeypot = features.get("is_honeypot", 0.0)

    risk_penalty = (
        risk["risk_score_penalty"] * risk_score
        + min(anti_patterns * risk["anti_pattern_penalty"], risk["max_anti_pattern_penalty"])
        + is_honeypot * risk["honeypot_penalty"]
    )
    risk_penalty = min(risk_penalty, 1.0)
    return 1.0 - risk_penalty


def compute_dimension_scores(
    features: dict[str, float],
    jd_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    return {
        "technical_match": _compute_technical_match(features, jd_weights),
        "semantic_match": _compute_semantic_match(features),
        "career_quality": _compute_career_quality(features),
        "behavioral": _compute_behavioral(features),
        "retention": _compute_retention(features),
        "risk_adjustment": _compute_risk_adjustment(features),
        "jd_semantic_similarity": features.get("semantic_similarity", 0.0),
    }


def compute_availability_multiplier(
    signals: dict[str, Any],
    reference_date: date | None = None,
) -> float:
    ref = reference_date or REFERENCE_DATE
    last_active = signals.get("last_active_date", "") or ""
    recency_score = 1.0
    if last_active:
        try:
            la_date = datetime.strptime(str(last_active)[:10], "%Y-%m-%d").date()
            days_since = (ref - la_date).days
            if days_since > 180:
                recency_score = 0.45
            elif days_since > 90:
                recency_score = 0.70
            elif days_since > 45:
                recency_score = 0.85
        except (ValueError, TypeError):
            recency_score = 0.45

    rrr = float(signals.get("recruiter_response_rate", 0) or 0)
    responsiveness_score = 0.5 + 0.5 * min(max(rrr, 0.0), 1.0)

    openness_score = 1.0 if signals.get("open_to_work_flag", False) else 0.65

    notice_raw = signals.get("notice_period_days", 30)
    try:
        notice = float(notice_raw) if notice_raw is not None else 30.0
    except (ValueError, TypeError):
        notice = 30.0
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.85
    elif notice <= 90:
        notice_score = 0.70
    else:
        notice_score = 0.50

    multiplier = (
        0.40 * recency_score
        + 0.25 * responsiveness_score
        + 0.20 * openness_score
        + 0.15 * notice_score
    )
    return max(multiplier, 0.35)


def compute_final_score(
    dim_scores: dict[str, float],
    ml_score: float | None = None,
) -> float:
    score = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        val = dim_scores.get(dim, 0.0)
        if val is None:
            val = 0.0
        score += weight * float(val)

    risk_adj = float(dim_scores.get("risk_adjustment", 1.0))
    score = score * risk_adj
    score = max(0.0, min(100.0, score * 100.0))

    if ml_score is not None:
        ew = ENSEMBLE_WEIGHTS
        score = ew["rule_based"] * score + ew["ml_prediction"] * ml_score * 100.0

    return score


def rank_candidates(
    candidate_ids: list[str],
    features_list: list[dict[str, float]],
    jd_weights: dict[str, float] | None = None,
    ml_model: Any | None = None,
    signals_list: list[dict[str, Any]] | None = None,
) -> list[tuple[str, float, int, dict[str, float]]]:
    results: list[tuple[str, float, dict[str, float]]] = []

    for i, (cid, feats) in enumerate(zip(candidate_ids, features_list, strict=True)):
        dim_scores = compute_dimension_scores(feats, jd_weights)
        final_score = compute_final_score(dim_scores)
        if signals_list and i < len(signals_list):
            avail_mult = compute_availability_multiplier(signals_list[i])
            final_score *= avail_mult
        results.append((cid, final_score, dim_scores))

    if ml_model is not None:
        feature_vectors = [_features_to_vector(f) for f in features_list]
        ml_predictions = ml_model.predict(np.array(feature_vectors))
        for i, (cid, _, dim_scores) in enumerate(results):
            ml_score = float(ml_predictions[i]) if i < len(ml_predictions) else None
            if ml_score is not None:
                ml_score = max(0.0, min(1.0, ml_score))
            ensemble_score = compute_final_score(dim_scores, ml_score)
            results[i] = (cid, ensemble_score, dim_scores)

    results.sort(key=lambda x: (-x[1], x[0]))

    ranked: list[tuple[str, float, int, dict[str, float]]] = []
    for rank_pos, (cid, score, dims) in enumerate(results, start=1):
        ranked.append((cid, score, rank_pos, dims))
    return ranked


def _features_to_vector(features: dict[str, float]) -> list[float]:
    keys = [
        "years_experience",
        "num_skills",
        "expert_skills",
        "num_career_entries",
        "num_companies",
        "ai_depth",
        "retrieval_depth",
        "eval_depth",
        "career_progression",
        "career_seniority",
        "avg_tenure_months",
        "skill_total_months",
        "text_length",
        "location_score",
        "company_prestige",
        "keyword_diversity",
        "growth_rate",
        "education_level",
        "education_field",
        "certifications",
        "has_product_exp",
        "entirely_consulting",
    ]
    vec = []
    for key in keys:
        val = features.get(key, 0.0)
        if val is None:
            val = 0.0
        vec.append(float(val))
    return vec


def _build_hireability_target(signals: dict) -> float:
    """Behavioral outcome composite as training target (not circular)."""
    oar = signals.get("offer_acceptance_rate", -1)
    icr = signals.get("interview_completion_rate", 0.5)
    saved = min(signals.get("saved_by_recruiters_30d", 0) / 10.0, 1.0)
    rrr = signals.get("recruiter_response_rate", 0.5)
    active = 1.0 if signals.get("open_to_work_flag", False) else 0.4
    oar_score = oar if oar >= 0 else 0.5
    return 0.30 * oar_score + 0.25 * icr + 0.20 * saved + 0.15 * rrr + 0.10 * active


def train_model(
    features_list: list[dict[str, float]],
    scores: list[float] | None = None,
    signals_list: list[dict] | None = None,
    model_path: str | None = None,
) -> Any:
    if not features_list:
        logger.warning("No features for model training")
        return None

    X = np.array([_features_to_vector(f) for f in features_list])

    if scores is not None and len(scores) == len(features_list):
        y = np.array(scores)
    elif signals_list is not None and len(signals_list) == len(features_list):
        logger.info("Building hireability targets from behavioral signals")
        y = np.array([_build_hireability_target(s) for s in signals_list])
    else:
        logger.warning("No training targets available (provide scores or signals_list)")
        return None

    mc = MODEL_CONFIG
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=mc["test_size"], random_state=mc["random_state"]
    )

    model = RandomForestRegressor(
        n_estimators=mc["n_estimators"],
        max_depth=mc["max_depth"],
        random_state=mc["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("Model trained - MSE: %.4f, R2: %.4f", mse, r2)

    save_path = model_path or MODEL_PATH
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(model, save_path)
        logger.info("Model saved to %s", save_path)

    return model


def load_model(model_path: str) -> Any:
    if not os.path.exists(model_path):
        logger.warning("Model not found at %s", model_path)
        return None
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        return None


def calibrate_scores(
    ranked: list[tuple[str, float, int, dict[str, float]]],
) -> list[tuple[str, float, int, dict[str, float]]]:
    if not ranked:
        return ranked
    scores = [r[1] for r in ranked]
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 0.01:
        return ranked

    calibrated: list[tuple[str, float, int, dict[str, float]]] = []
    for cid, score, rank_pos, dims in ranked:
        normalized = (score - min_s) / (max_s - min_s)
        calibrated_score = normalized * 100.0
        calibrated.append((cid, calibrated_score, rank_pos, dims))
    return calibrated


def audit_fairness(
    ranked: list[tuple[str, float, int, dict[str, float]]],
    features_list: list[dict[str, float]],
    candidate_ids: list[str],
) -> dict[str, Any]:
    fc = FAIRNESS_CONFIG
    if not fc["audit_enabled"]:
        return {"audit_enabled": False}

    id_to_feats = dict(zip(candidate_ids, features_list, strict=True))
    groups: dict[str, list[str]] = {
        "consulting": [],
        "non_consulting": [],
        "high_prestige": [],
        "low_prestige": [],
        "high_location": [],
        "low_location": [],
    }

    for cid, _score, _rank, _dims in ranked:
        feats = id_to_feats.get(cid, {}) or {}
        if feats.get("entirely_consulting", 0.0) > 0.5:
            groups["consulting"].append(cid)
        else:
            groups["non_consulting"].append(cid)
        if feats.get("company_prestige", 0.0) >= 0.7:
            groups["high_prestige"].append(cid)
        else:
            groups["low_prestige"].append(cid)
        if feats.get("location_score", 0.0) >= 0.7:
            groups["high_location"].append(cid)
        else:
            groups["low_location"].append(cid)

    dims_to_audit = fc["dimensions_to_audit"]
    id_to_dims: dict[str, dict[str, float]] = {}
    for cid, _score, _rank, dims in ranked:
        id_to_dims[cid] = dims

    disparities: dict[str, Any] = {}
    for group_name, (advantaged_key, disadvantaged_key) in [
        ("consulting", ("non_consulting", "consulting")),
        ("prestige", ("high_prestige", "low_prestige")),
        ("location", ("high_location", "low_location")),
    ]:
        group_disparities = {}
        for dim in dims_to_audit:
            adv_scores = [
                id_to_dims[cid].get(dim, 0.0) for cid in groups[advantaged_key] if cid in id_to_dims
            ]
            disadv_scores = [
                id_to_dims[cid].get(dim, 0.0)
                for cid in groups[disadvantaged_key]
                if cid in id_to_dims
            ]
            adv_mean = float(np.mean(adv_scores)) if adv_scores else 0.0
            disadv_mean = float(np.mean(disadv_scores)) if disadv_scores else 0.0
            disparity = adv_mean - disadv_mean if adv_mean > 0 else 0.0
            group_disparities[dim] = {
                "advantaged_mean": round(adv_mean, 4),
                "disadvantaged_mean": round(disadv_mean, 4),
                "disparity": round(disparity, 4),
                "flagged": disparity > fc["disparity_threshold"],
            }
        disparities[group_name] = group_disparities

    return {
        "audit_enabled": True,
        "disparity_threshold": fc["disparity_threshold"],
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "disparities": disparities,
    }


def _get_weakest_jd_dimension(
    dim_scores: dict[str, float],
    features: dict[str, float],
) -> str:
    jd_dims = {
        "embeddings": "embedding experience",
        "vector_db": "vector DB experience",
        "ranking": "ranking systems experience",
        "ml_production": "ML production experience",
        "nlp_ir": "NLP/IR experience",
        "llm": "LLM experience",
        "python": "Python depth",
        "distributed_systems": "distributed systems",
        "data_engineering": "data engineering",
    }
    min_dim = None
    min_val = float("inf")
    for dim, label in jd_dims.items():
        val = features.get(f"jd_match_{dim}", 0.0)
        if val < min_val:
            min_val = val
            min_dim = label
    return min_dim or "technical alignment"


def generate_reasoning(
    cid: str,
    score: float,
    rank: int,
    dim_scores: dict[str, float],
    features: dict[str, float],
    candidate: dict[str, Any] | None = None,
    with_explanation: bool = False,
) -> str:
    profile = (candidate or {}).get("profile") or {}
    signals = (candidate or {}).get("redrob_signals") or {}
    career = (candidate or {}).get("career_history") or []
    first_role = career[0] if career else {}
    current_title = profile.get("current_title", "") or ""
    current_company = profile.get("current_company", "") or ""
    headline = profile.get("headline", "") or ""
    summary = (profile.get("summary", "") or "")[:200]

    tech = dim_scores.get("technical_match", 0.0)
    beh = dim_scores.get("behavioral", 0.0)
    career_q = dim_scores.get("career_quality", 0.0)
    sem = dim_scores.get("semantic_match", 0.0)
    ret = dim_scores.get("retention", 0.0)
    exp_years = features.get("years_experience", 0)
    retrieval = features.get("retrieval_depth", 0.0)
    ai_d = features.get("ai_depth", 0.0)
    prestige = features.get("company_prestige", 0.0)
    edu_level = features.get("education_level", 0.0)
    entirely_consulting = features.get("entirely_consulting", 0.0)
    keyword_div = features.get("keyword_diversity", 0.0)
    is_honeypot = features.get("is_honeypot", 0.0) > 0.5

    rrr = signals.get("recruiter_response_rate", 0)
    o2w = signals.get("open_to_work_flag", False)

    weakness_dim = _get_weakest_jd_dimension(dim_scores, features)

    def _strength_tech():
        if retrieval > 0.3:
            return "Strong retrieval/vector search background"
        if ai_d > 0.3:
            return "Solid ML/AI engineering depth"
        if prestige >= 1.0 and tech > 0.5:
            return f"Top-tier {current_company or ''} ML background with JD-relevant skills".strip()
        if tech > 0.5:
            return "Good technical alignment with this JD's ML/retrieval stack"
        if tech > 0.25:
            return "Adjacent technical skills with some JD keyword overlap"
        if keyword_div > 0.3:
            return "Partial technical match — covers some JD dimensions"
        return "Weak but non-zero technical signal"

    def _strength_beh():
        if beh > 0.7 and rrr and rrr > 0:
            return f"Strong platform engagement (response rate {rrr:.0%}, active)"
        if beh > 0.7:
            return "Highly engaged on platform with strong signal profile"
        if beh > 0.5:
            return "Moderate platform activity with decent engagement metrics"
        if beh > 0.3:
            return "Limited platform presence but still reachable"
        return "Low platform activity — may be slow to respond"

    def _strength_career():
        parts_s = []
        if prestige >= 1.0:
            parts_s.append("top-tier company background")
        elif prestige >= 0.7:
            parts_s.append("strong company experience")
        if career_q > 0.6 and entirely_consulting < 0.5:
            parts_s.append("solid career progression in relevant domains")
        if exp_years and 3 <= exp_years <= 12:
            parts_s.append(f"{exp_years:.0f}yr exp in range")
        edu_labels = {0.8: "PhD-level", 0.5: "Masters-level", 0.2: "Bachelors-level"}
        for thr, label in sorted(edu_labels.items(), reverse=True):
            if edu_level >= thr:
                parts_s.append(f"{label} education")
                break
        return "; ".join(parts_s) if parts_s else ""

    def _concern():
        parts_c = []
        if is_honeypot:
            parts_c.append("HONEYPOT — risk flags triggered")
            return "; ".join(parts_c)
        if entirely_consulting > 0.5:
            parts_c.append("entirely consulting background, lacks product ML exposure")
        elif keyword_div < 0.2:
            parts_c.append(f"narrow technical scope — weakest in {weakness_dim}")
        elif tech < 0.25:
            parts_c.append(f"weak overall tech match — particularly in {weakness_dim}")
        else:
            lowest_score_dim = min(
                [
                    (k, v)
                    for k, v in dim_scores.items()
                    if k
                    in (
                        "technical_match",
                        "behavioral",
                        "career_quality",
                        "retention",
                        "semantic_match",
                    )
                ],
                key=lambda x: x[1],
            )
            dim_label = lowest_score_dim[0].replace("_", " ")
            parts_c.append(f"relative weakness in {dim_label} ({lowest_score_dim[1]:.2f})")
        if beh < 0.3:
            parts_c.append("low platform engagement — response may be delayed")
        if ret < 0.3:
            parts_c.append("retention risk from short tenures or long notice period")
        return "; ".join(parts_c) if parts_c else ""

    concern_text = _concern()
    strength_tech = _strength_tech()
    strength_beh = _strength_beh()
    strength_career = _strength_career()

    if is_honeypot:
        return f"HONEYPOT — {concern_text}"

    if rank <= 10:
        text = f"{strength_tech}. {strength_beh}."
        if strength_career:
            text += f" {strength_career}."
        text += f" Caveat: {concern_text}" if concern_text else ""
        return text

    if rank <= 50:
        text = f"{strength_tech}. Concern: {concern_text}."
        if strength_beh:
            text += f" On the plus side: {strength_beh}."
        if strength_career:
            text += f" {strength_career}."
        return text

    text = f"Key concern: {concern_text}."
    if strength_tech:
        text += f" Upside: {strength_tech}."
    if strength_beh:
        text += f" {strength_beh}."
    return text
