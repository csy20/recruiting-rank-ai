#!/usr/bin/env python3
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import API_HOST, API_PORT, API_CONFIG
from features.extractor import extract_all_features
from scoring.jd_parser import parse_jd, get_jd_dimension_weights
from scoring.explainer import explain_ranking
from scoring.ranker import (
    rank_candidates,
    compute_dimension_scores,
    generate_reasoning,
    calibrate_scores,
    load_model,
    audit_fairness,
)
from rank import _get_candidate_text, _compute_tfidf_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("serve")

app = FastAPI(
    title="Recruiting Rank AI",
    version="2.0.0",
    description="Intelligent candidate ranking API with deep JD understanding and multi-dimension scoring",
)

_ml_model = None


class RankRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    jd_text: Optional[str] = None
    top_k: int = 100
    include_dimensions: bool = True
    include_reasoning: bool = True
    include_contributions: bool = False


class RankResponse(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_ml_model is not None,
        version="2.0.0",
    )


@app.post("/rank", response_model=RankResponse)
def rank_endpoint(req: RankRequest):
    t0 = time.time()

    if not req.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")
    max_candidates = API_CONFIG.get("max_candidates_per_request", 5000)
    if len(req.candidates) > max_candidates:
        raise HTTPException(
            status_code=400,
            detail=f"Too many candidates (max {max_candidates})",
        )

    ids: List[str] = []
    all_features: List[Dict[str, float]] = []
    for c in req.candidates:
        cid = c.get("candidate_id") or f"cand_{len(ids)}"
        ids.append(cid)
        all_features.append(extract_all_features(c))

    jd_weights = None
    if req.jd_text:
        jd_profile = parse_jd(req.jd_text)
        jd_weights = get_jd_dimension_weights(jd_profile)
        logger.info(
            "JD parsed: %d dimensions, exp range: %s",
            len(jd_weights),
            jd_profile.get("experience_years"),
        )
        candidate_texts = [_get_candidate_text(c) for c in req.candidates]
        tfidf_scores = _compute_tfidf_features(
            req.candidates, candidate_texts, req.jd_text
        )
        for i, sim in enumerate(tfidf_scores):
            if i < len(all_features):
                all_features[i]["tfidf_jd_similarity"] = sim

    ranked = rank_candidates(
        ids, all_features, jd_weights=jd_weights, ml_model=_ml_model
    )
    ranked = calibrate_scores(ranked)

    id_to_feats = dict(zip(ids, all_features))
    top_k = min(req.top_k, len(ranked))
    results: List[Dict[str, Any]] = []
    for cid, score, rank_pos, dims in ranked[:top_k]:
        entry: Dict[str, Any] = {
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(score, 4),
        }
        if req.include_dimensions:
            entry["dimensions"] = {
                k: round(float(v), 4) for k, v in dims.items() if v is not None
            }
        if req.include_reasoning:
            feats = id_to_feats.get(cid, {}) or {}
            entry["reasoning"] = generate_reasoning(cid, score, rank_pos, dims, feats)
        if req.include_contributions:
            feats = id_to_feats.get(cid, {}) or {}
            explanation = explain_ranking(feats, dims, score)
            entry["feature_contributions"] = explanation["feature_contributions"]
            entry["strengths"] = explanation["strengths"]
            entry["weaknesses"] = explanation["weaknesses"]
        results.append(entry)

    elapsed = time.time() - t0
    return RankResponse(
        results=results,
        metadata={
            "total_candidates": len(req.candidates),
            "top_k": top_k,
            "elapsed_seconds": round(elapsed, 2),
            "jd_parsed": req.jd_text is not None,
            "model_used": _ml_model is not None,
        },
    )


@app.post("/score")
def score_single(candidate: Dict[str, Any], jd_text: Optional[str] = None):
    feats = extract_all_features(candidate)
    dims = compute_dimension_scores(feats)
    jd_weights = None
    if jd_text:
        jd_profile = parse_jd(jd_text)
        jd_weights = get_jd_dimension_weights(jd_profile)
        dims = compute_dimension_scores(feats, jd_weights=jd_weights)
    return {
        "candidate_id": candidate.get("candidate_id", "unknown"),
        "dimensions": {k: round(float(v), 4) for k, v in dims.items() if v is not None},
    }


@app.post("/audit")
def fairness(candidates: List[Dict[str, Any]]):
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")
    ids: List[str] = []
    all_features: List[Dict[str, float]] = []
    for c in candidates:
        cid = c.get("candidate_id") or f"cand_{len(ids)}"
        ids.append(cid)
        all_features.append(extract_all_features(c))
    ranked = rank_candidates(ids, all_features)
    ranked = calibrate_scores(ranked)
    audit = audit_fairness(ranked, all_features, ids)
    return audit


def main():
    global _ml_model
    model_path = os.environ.get("RR_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        _ml_model = load_model(model_path)
        logger.info("ML model loaded from %s", model_path)

    host = os.environ.get("RR_API_HOST", API_HOST)
    port = int(os.environ.get("RR_API_PORT", str(API_PORT)))
    logger.info("Starting Recruiting Rank AI API on %s:%d", host, port)
    logger.info("Endpoints: /health, /rank, /score, /audit")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
