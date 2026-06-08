#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    BASE_DIR,
    OUTPUT_PATH,
    TFIDF_ENABLED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    JD_KEYWORDS,
    MODEL_CONFIG,
    ENSEMBLE_WEIGHTS,
    FAIRNESS_CONFIG,
)
from features.extractor import load_candidates, extract_all_features
from scoring.jd_parser import (
    parse_jd,
    get_jd_dimension_weights,
    get_jd_experience_score,
)
from scoring.explainer import explain_ranking, compute_feature_contributions
from scoring.ranker import (
    compute_dimension_scores,
    compute_final_score,
    rank_candidates,
    generate_reasoning,
    train_model,
    load_model,
    calibrate_scores,
    audit_fairness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")


def _build_jd_text() -> str:
    terms: set = set()
    for dim in JD_KEYWORDS.values():
        for term in dim.get("terms", []):
            terms.add(term)
    return " ".join(sorted(terms))


def _load_jd_text(jd_path: str) -> Optional[str]:
    if not jd_path:
        return None
    if not os.path.exists(jd_path):
        logger.warning(
            "JD file %s not found, falling back to keyword-derived JD", jd_path
        )
        return None
    try:
        with open(jd_path) as f:
            text = f.read().strip()
        if text:
            logger.info("Loaded JD text from %s (%d chars)", jd_path, len(text))
            return text
    except (IOError, OSError) as e:
        logger.warning("Failed to read JD file %s: %s", jd_path, e)
    return None


def _get_candidate_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile") or {}
    texts: List[str] = []
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""
    if headline:
        texts.append(headline)
    if summary:
        texts.append(summary)
    for role in candidate.get("career_history") or []:
        if isinstance(role, dict):
            desc = role.get("description") or ""
            title = role.get("title") or ""
            if desc:
                texts.append(desc)
            if title:
                texts.append(title)
    for skill in candidate.get("skills") or []:
        if isinstance(skill, dict):
            name = skill.get("name") or ""
            if name:
                texts.append(name)
    return " ".join(texts)


def _compute_tfidf_features(
    candidates: List[Dict[str, Any]],
    all_texts: List[str],
    jd_text: str,
) -> List[float]:
    logger.info("Computing TF-IDF features (%d candidates)...", len(all_texts))
    if not all_texts:
        return []
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
    )
    corpus = all_texts + [jd_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vec = tfidf_matrix[-1:]
    candidate_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(candidate_vecs, jd_vec).flatten()
    if len(similarities) > 0:
        logger.info(
            "TF-IDF similarity range: [%.4f, %.4f]",
            float(similarities.min()),
            float(similarities.max()),
        )
    return [float(s) for s in similarities]


def _extract_single(
    args: Tuple[Dict[str, Any], int, int],
) -> Tuple[str, Dict[str, float]]:
    candidate, idx, total = args
    if idx % 10000 == 0 and idx > 0:
        logger.info("  processing %d/%d...", idx, total)
    cid = candidate.get("candidate_id") or str(idx)
    return cid, extract_all_features(candidate)


def _extract_all(
    candidates: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, float]]]:
    n = len(candidates)
    if n == 0:
        return []

    if n < 500:
        results: List[Tuple[str, Dict[str, float]]] = []
        for i, c in enumerate(candidates):
            if i % 10000 == 0 and i > 0:
                logger.info("  processing %d/%d...", i, n)
            cid = c.get("candidate_id") or str(i)
            results.append((cid, extract_all_features(c)))
        return results

    logger.info("Using multiprocessing (%d workers)...", min(8, os.cpu_count() or 1))
    args = [(c, i, n) for i, c in enumerate(candidates)]
    results_map: Dict[int, Tuple[str, Dict[str, float]]] = {}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        futures = {executor.submit(_extract_single, a): i for i, a in enumerate(args)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_map[idx] = future.result()
            except Exception as e:
                logger.error("Worker failed at index %d: %s", idx, e)
                cid = candidates[idx].get("candidate_id") or str(idx)
                results_map[idx] = (cid, {})
    return [results_map[i] for i in range(n)]


def precompute(
    path: str,
    out_dir: str,
    jd_path: Optional[str] = None,
    train: bool = False,
):
    t0 = time.time()
    logger.info("Loading candidates from %s...", path)
    candidates = load_candidates(path)
    if not candidates:
        logger.warning("No candidates loaded!")
        return
    logger.info("Loaded %d candidates in %.1fs", len(candidates), time.time() - t0)

    ids: List[str] = []
    all_features: List[Dict[str, float]] = []

    logger.info("Extracting features...")
    extracted = _extract_all(candidates)
    for cid, feats in extracted:
        ids.append(cid)
        all_features.append(feats)

    jd_text = _load_jd_text(jd_path) if jd_path else _build_jd_text()
    if TFIDF_ENABLED and jd_text:
        candidate_texts = [_get_candidate_text(c) for c in candidates]
        tfidf_scores = _compute_tfidf_features(candidates, candidate_texts, jd_text)
        for i, sim in enumerate(tfidf_scores):
            if i < len(all_features):
                all_features[i]["tfidf_jd_similarity"] = sim

    if not all_features:
        logger.warning("No features extracted!")
        return

    feature_keys = list(all_features[0].keys())
    feature_matrix = np.zeros((len(all_features), len(feature_keys)), dtype=np.float32)
    for i, feats in enumerate(all_features):
        for j, key in enumerate(feature_keys):
            val = feats.get(key, 0.0)
            if val is None:
                val = 0.0
            feature_matrix[i, j] = float(val)

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "features.npz")
    np.savez_compressed(npz_path, features=feature_matrix, keys=feature_keys)
    logger.info("Saved features to %s", npz_path)

    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "feature_index"])
        for i, cid in enumerate(ids):
            w.writerow([cid, i])
    logger.info("Saved metadata to %s", meta_path)

    if train:
        logger.info("Training ML model on rule-based scores...")
        train_model(all_features, model_path=None)

    logger.info("Precomputation done in %.1fs total", time.time() - t0)


def rank(
    candidates_path: str,
    output_path: str,
    precomputed_dir: Optional[str] = None,
    jd_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_json: bool = False,
):
    t0 = time.time()
    logger.info("Loading candidates from %s...", candidates_path)
    candidates = load_candidates(candidates_path)
    if not candidates:
        logger.warning("No candidates loaded!")
        return
    logger.info("Loaded %d candidates in %.1fs", len(candidates), time.time() - t0)

    jd_profile = None
    jd_text_loaded = _load_jd_text(jd_path) if jd_path else None
    if jd_text_loaded:
        jd_profile = parse_jd(jd_text_loaded)
        jd_weights = get_jd_dimension_weights(jd_profile)
        logger.info("JD parsed: %d dimensions analyzed", len(jd_weights))
    else:
        jd_weights = None

    all_features: List[Dict[str, float]] = []
    if precomputed_dir and os.path.exists(
        os.path.join(precomputed_dir, "features.npz")
    ):
        logger.info("Loading precomputed features...")
        npz = np.load(os.path.join(precomputed_dir, "features.npz"))
        feature_matrix = npz["features"]
        feature_keys = list(npz["keys"])

        meta_path = os.path.join(precomputed_dir, "metadata.csv")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                reader = csv.DictReader(f)
                id_to_idx = {
                    row["candidate_id"]: int(row["feature_index"]) for row in reader
                }
            for c in candidates:
                cid = c.get("candidate_id")
                if cid in id_to_idx:
                    idx = id_to_idx[cid]
                    feats = {
                        feature_keys[j]: float(feature_matrix[idx, j])
                        for j in range(len(feature_keys))
                    }
                    all_features.append(feats)
                else:
                    all_features.append(extract_all_features(c))
        else:
            for c in candidates:
                all_features.append(extract_all_features(c))
        logger.info("Loaded features for %d candidates", len(all_features))
    else:
        logger.info("Extracting features on the fly...")
        extracted = _extract_all(candidates)
        all_features = [feats for _, feats in extracted]
        logger.info("Extracted features for %d candidates", len(all_features))

    if not all_features:
        logger.warning("No features available for ranking!")
        return

    if TFIDF_ENABLED:
        ref_text = jd_text_loaded or _build_jd_text()
        if ref_text and "tfidf_jd_similarity" not in all_features[0]:
            candidate_texts = [_get_candidate_text(c) for c in candidates]
            tfidf_scores = _compute_tfidf_features(
                candidates, candidate_texts, ref_text
            )
            for i, sim in enumerate(tfidf_scores):
                if i < len(all_features):
                    all_features[i]["tfidf_jd_similarity"] = sim

    ml_model = (
        load_model(model_path) if model_path and os.path.exists(model_path) else None
    )
    if model_path and ml_model is None:
        logger.info("No pre-trained model found, training from rule-based scores...")
        ml_model = train_model(all_features, model_path=model_path)

    logger.info("Ranking candidates...")
    ids = [c.get("candidate_id") or str(i) for i, c in enumerate(candidates)]
    ranked = rank_candidates(
        ids, all_features, jd_weights=jd_weights, ml_model=ml_model
    )

    logger.info("Calibrating scores...")
    ranked = calibrate_scores(ranked)

    logger.info("Generating reasoning...")
    id_to_feats: Dict[str, Dict[str, float]] = dict(zip(ids, all_features))
    results: List[Tuple[str, int, float, str, Dict[str, float]]] = []
    for cid, score, rank_pos, dims in ranked[:100]:
        feats = id_to_feats.get(cid, {}) or {}
        reasoning = generate_reasoning(cid, score, rank_pos, dims, feats)
        results.append((cid, rank_pos, round(score, 2), reasoning, dims))

    if output_json:
        json_output = []
        for cid, rank_pos, score, reasoning, dims in results:
            feats = id_to_feats.get(cid, {}) or {}
            explanation = explain_ranking(feats, dims, score)
            entry = {
                "candidate_id": cid,
                "rank": rank_pos,
                "score": score,
                "reasoning": reasoning,
                "dimensions": {
                    k: round(float(v), 4) for k, v in dims.items() if v is not None
                },
                "feature_contributions": explanation["feature_contributions"],
                "strengths": explanation["strengths"],
                "weaknesses": explanation["weaknesses"],
            }
            json_output.append(entry)

        json_path = (
            output_path.replace(".csv", ".json")
            if output_path.endswith(".csv")
            else (output_path + ".json")
        )
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        logger.info("Written JSON to %s", json_path)

        audit = audit_fairness(ranked[:100], all_features, ids)
        if audit.get("audit_enabled"):
            audit_path = (
                output_path.replace(".csv", "_fairness_audit.json")
                if output_path.endswith(".csv")
                else (output_path + "_fairness_audit.json")
            )
            with open(audit_path, "w") as f:
                json.dump(audit, f, indent=2)
            flagged = []
            for group, dims in audit.get("disparities", {}).items():
                for dim, info in dims.items():
                    if info.get("flagged"):
                        flagged.append(f"{group}/{dim}")
            if flagged:
                logger.warning("Fairness flags: %s", ", ".join(flagged))
            else:
                logger.info(
                    "Fairness audit passed — no significant disparities detected"
                )

    logger.info("Writing top-100 to %s...", output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank_pos, score, reasoning, _dims in results:
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    logger.info("Done in %.1fs", time.time() - t0)


def serve(host: str, port: int, model_path: Optional[str] = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        logger.error("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="Recruiting Rank AI", version="2.0.0")

    ml_model = load_model(model_path) if model_path else None

    class RankRequest(BaseModel):
        candidates: List[Dict[str, Any]]
        jd_text: Optional[str] = None
        top_k: int = 100

    class ScoreResponse(BaseModel):
        candidate_id: str
        score: float
        dimensions: Dict[str, float]
        reasoning: str

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": ml_model is not None}

    @app.post("/rank")
    def rank_endpoint(req: RankRequest):
        if not req.candidates:
            raise HTTPException(status_code=400, detail="No candidates provided")
        if len(req.candidates) > 5000:
            raise HTTPException(status_code=400, detail="Too many candidates")

        t0 = time.time()
        ids: List[str] = []
        all_features: List[Dict[str, float]] = []
        for c in req.candidates:
            cid = c.get("candidate_id") or str(len(ids))
            ids.append(cid)
            all_features.append(extract_all_features(c))

        jd_weights = None
        if req.jd_text:
            jd_profile = parse_jd(req.jd_text)
            jd_weights = get_jd_dimension_weights(jd_profile)
            candidate_texts = [_get_candidate_text(c) for c in req.candidates]
            tfidf_scores = _compute_tfidf_features(
                req.candidates, candidate_texts, req.jd_text
            )
            for i, sim in enumerate(tfidf_scores):
                if i < len(all_features):
                    all_features[i]["tfidf_jd_similarity"] = sim

        ranked = rank_candidates(
            ids, all_features, jd_weights=jd_weights, ml_model=ml_model
        )
        ranked = calibrate_scores(ranked)

        id_to_feats = dict(zip(ids, all_features))
        top_k = min(req.top_k, len(ranked))
        results = []
        for cid, score, rank_pos, dims in ranked[:top_k]:
            feats = id_to_feats.get(cid, {}) or {}
            reasoning = generate_reasoning(cid, score, rank_pos, dims, feats)
            results.append(
                {
                    "candidate_id": cid,
                    "rank": rank_pos,
                    "score": round(score, 4),
                    "dimensions": {k: round(float(v), 4) for k, v in dims.items()},
                    "reasoning": reasoning,
                }
            )

        elapsed = time.time() - t0
        return {
            "results": results,
            "metadata": {
                "total_candidates": len(req.candidates),
                "top_k": top_k,
                "elapsed_seconds": round(elapsed, 2),
            },
        }

    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Recruiting Rank AI — Candidate Ranking System"
    )
    parser.add_argument("--candidates", help="Path to candidates.jsonl")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Output CSV path")
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Precompute features (run before ranking on large datasets)",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Directory with precomputed features (data/ by default)",
    )
    parser.add_argument(
        "--jd",
        default=None,
        help="Path to JD description text file for TF-IDF similarity scoring",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train ML model from rule-based scores during precompute",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to pre-trained ML model (.pkl)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON results + fairness audit alongside CSV",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server for real-time ranking",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    args = parser.parse_args()

    if args.serve:
        serve(args.host, args.port, args.model)
        return

    if not args.candidates:
        parser.error("--candidates is required (or use --serve for API mode)")

    if args.precompute:
        precompute(
            args.candidates,
            args.features_dir or "data",
            jd_path=args.jd,
            train=args.train,
        )
    else:
        rank(
            args.candidates,
            args.out,
            args.features_dir or "data",
            jd_path=args.jd,
            model_path=args.model,
            output_json=args.json,
        )


if __name__ == "__main__":
    main()
