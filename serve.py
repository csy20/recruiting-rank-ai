#!/usr/bin/env python3
import collections
import logging
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from config import API_CONFIG, API_HOST, API_PORT
from features.extractor import extract_all_features
from rank import _compute_semantic_features, _get_candidate_text
from scoring.explainer import explain_ranking
from scoring.jd_parser import get_jd_dimension_weights, parse_jd
from scoring.ranker import (
    audit_fairness,
    calibrate_scores,
    compute_dimension_scores,
    generate_reasoning,
    load_model,
    rank_candidates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("serve")

app = FastAPI(
    title="Recruiting Rank AI",
    version="2.0.0",
    description="Intelligent candidate ranking API with deep JD understanding "
    "and multi-dimension scoring",
)

_ml_model = None

_MAX_JD_CHARS = 50_000
_request_timestamps: collections.deque = collections.deque()


def _check_rate_limit():
    rate_limit = API_CONFIG.get("rate_limit_per_minute", 60)
    now = time.time()
    while _request_timestamps and _request_timestamps[0] < now - 60:
        _request_timestamps.popleft()
    if len(_request_timestamps) >= rate_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _request_timestamps.append(now)


class RankRequest(BaseModel):
    candidates: list[dict[str, Any]]
    jd_text: str | None = None
    top_k: int = 100
    include_dimensions: bool = True
    include_reasoning: bool = True
    include_contributions: bool = False

    @field_validator("jd_text")
    @classmethod
    def validate_jd_length(cls, v):
        if v is not None and len(v) > _MAX_JD_CHARS:
            raise ValueError(f"JD text too long ({len(v)} chars, max {_MAX_JD_CHARS})")
        return v


class RankResponse(BaseModel):
    results: list[dict[str, Any]]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@app.get("/health", response_model=HealthResponse)
def health():
    _check_rate_limit()
    return HealthResponse(
        status="ok",
        model_loaded=_ml_model is not None,
        version="2.0.0",
    )


@app.post("/rank", response_model=RankResponse)
def rank_endpoint(req: RankRequest):
    _check_rate_limit()
    t0 = time.time()

    if not req.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")
    max_candidates = API_CONFIG.get("max_candidates_per_request", 5000)
    if len(req.candidates) > max_candidates:
        raise HTTPException(
            status_code=400,
            detail=f"Too many candidates (max {max_candidates})",
        )

    ids: list[str] = []
    all_features: list[dict[str, float]] = []
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
        semantic_scores = _compute_semantic_features(req.candidates, candidate_texts, req.jd_text)
        for i, sim in enumerate(semantic_scores):
            if i < len(all_features):
                all_features[i]["semantic_similarity"] = sim

    ranked = rank_candidates(ids, all_features, jd_weights=jd_weights, ml_model=_ml_model)
    ranked = calibrate_scores(ranked)

    id_to_feats = dict(zip(ids, all_features, strict=True))
    top_k = min(req.top_k, len(ranked))
    results: list[dict[str, Any]] = []
    for cid, score, rank_pos, dims in ranked[:top_k]:
        entry: dict[str, Any] = {
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(score, 4),
        }
        if req.include_dimensions:
            entry["dimensions"] = {k: round(float(v), 4) for k, v in dims.items() if v is not None}
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
def score_single(candidate: dict[str, Any], jd_text: str | None = None):
    _check_rate_limit()
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
def fairness(candidates: list[dict[str, Any]]):
    _check_rate_limit()
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")
    ids: list[str] = []
    all_features: list[dict[str, float]] = []
    for c in candidates:
        cid = c.get("candidate_id") or f"cand_{len(ids)}"
        ids.append(cid)
        all_features.append(extract_all_features(c))
    ranked = rank_candidates(ids, all_features)
    ranked = calibrate_scores(ranked)
    audit = audit_fairness(ranked, all_features, ids)
    return audit


def main(host: str | None = None, port: int | None = None, model_path: str | None = None):
    global _ml_model
    model_path = model_path or os.environ.get("RR_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        _ml_model = load_model(model_path)
        logger.info("ML model loaded from %s", model_path)

    host = host or os.environ.get("RR_API_HOST", API_HOST)
    port = port or int(os.environ.get("RR_API_PORT", str(API_PORT)))
    logger.info("Starting Recruiting Rank AI API on %s:%d", host, port)
    logger.info("Endpoints: /health, /rank, /score, /audit")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
