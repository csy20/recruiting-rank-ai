#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    JD_KEYWORDS,
    OUTPUT_PATH,
    SENTENCE_TRANSFORMER_MODEL,
)
from features.extractor import _get_combined_text, extract_all_features, load_candidates
from scoring.explainer import explain_ranking
from scoring.jd_parser import (
    get_jd_dimension_weights,
    parse_jd,
)
from scoring.ranker import (
    audit_fairness,
    calibrate_scores,
    generate_reasoning,
    load_model,
    rank_candidates,
    train_model,
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


def _load_jd_text(jd_path: str) -> str | None:
    if not jd_path:
        return None
    if not os.path.exists(jd_path):
        logger.warning("JD file %s not found, falling back to keyword-derived JD", jd_path)
        return None
    try:
        with open(jd_path) as f:
            text = f.read().strip()
        if text:
            logger.info("Loaded JD text from %s (%d chars)", jd_path, len(text))
            return text
    except OSError as e:
        logger.warning("Failed to read JD file %s: %s", jd_path, e)
    return None


def _get_candidate_text(candidate: dict[str, Any]) -> str:
    return _get_combined_text(candidate)


def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model: %s", SENTENCE_TRANSFORMER_MODEL)
    return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)


def _precompute_embeddings(
    candidates: list[dict[str, Any]],
    out_dir: str,
) -> np.ndarray:
    texts = [_get_candidate_text(c) for c in candidates]
    logger.info("Encoding %d candidates with sentence-transformers...", len(texts))
    t0 = time.time()
    model = _get_sentence_transformer()
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    logger.info("Encoded %d embeddings in %.1fs, shape=%s", len(embs), time.time() - t0, embs.shape)

    emb_path = os.path.join(out_dir, "candidate_embeddings.npy")
    np.save(emb_path, embs)
    logger.info("Saved embeddings to %s", emb_path)

    emb_ids_path = os.path.join(out_dir, "embedding_ids.csv")
    with open(emb_ids_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "embedding_index"])
        for i, c in enumerate(candidates):
            w.writerow([c.get("candidate_id") or str(i), i])
    logger.info("Saved embedding IDs to %s (%d entries)", emb_ids_path, len(candidates))
    return embs


def _build_bm25_index(candidates: list[dict[str, Any]]) -> Any:
    from rank_bm25 import BM25Okapi

    texts = [_get_candidate_text(c) for c in candidates]
    tokenized = [t.lower().split() for t in texts]
    logger.info("Building BM25 index on %d candidates...", len(tokenized))
    t0 = time.time()
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index built in %.1fs", time.time() - t0)
    return bm25


def _bm25_prefilter(
    bm25: Any,
    candidates: list[dict[str, Any]],
    query: str,
    top_k: int = 5000,
) -> tuple[list[int], list[dict[str, Any]]]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_candidates = [candidates[i] for i in top_indices]
    logger.info("BM25 prefilter: %d -> %d candidates", len(candidates), len(top_candidates))
    return list(top_indices), top_candidates


def _dense_rerank(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    indices: list[int],
    top_k: int = 2000,
) -> list[int]:
    selected = candidate_embs[indices]
    sims = cosine_similarity(query_emb.reshape(1, -1), selected).flatten()
    reranked_order = np.argsort(sims)[::-1][:top_k]
    return [indices[i] for i in reranked_order]


def _compute_semantic_features(
    candidates: list[dict[str, Any]],
    all_texts: list[str],
    jd_text: str,
    precomputed_dir: str | None = None,
    orig_indices: list[int] | None = None,
) -> list[float]:
    logger.info("Computing semantic similarity features...")

    if precomputed_dir and orig_indices is not None:
        emb_path = os.path.join(precomputed_dir, "candidate_embeddings.npy")
        if os.path.exists(emb_path):
            candidate_embs = np.load(emb_path)
            logger.info("Loaded precomputed embeddings: %s", candidate_embs.shape)
            model = _get_sentence_transformer()
            jd_emb = model.encode([jd_text], show_progress_bar=False, normalize_embeddings=True)
            selected = candidate_embs[orig_indices]
            sims = cosine_similarity(jd_emb, selected).flatten()
            return [float(s) for s in sims]

    model = _get_sentence_transformer()
    jd_emb = model.encode([jd_text], show_progress_bar=False, normalize_embeddings=True)
    candidate_embs = model.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
    similarities = cosine_similarity(candidate_embs, jd_emb).flatten()
    return [float(s) for s in similarities]


def _extract_single(
    args: tuple[dict[str, Any], int, int],
) -> tuple[str, dict[str, float]]:
    candidate, idx, total = args
    if idx % 500 == 0 and idx > 0:
        logger.info("  processing %d/%d...", idx, total)
    cid = candidate.get("candidate_id") or str(idx)
    return cid, extract_all_features(candidate)


def _extract_all(
    candidates: list[dict[str, Any]],
) -> list[tuple[str, dict[str, float]]]:
    n = len(candidates)
    if n == 0:
        return []

    if n < 500:
        results: list[tuple[str, dict[str, float]]] = []
        for i, c in enumerate(candidates):
            if i % 100 == 0 and i > 0:
                logger.info("  processing %d/%d...", i, n)
            cid = c.get("candidate_id") or str(i)
            results.append((cid, extract_all_features(c)))
        return results

    logger.info("Using multiprocessing (%d workers)...", min(8, os.cpu_count() or 1))
    args = [(c, i, n) for i, c in enumerate(candidates)]
    results_map: dict[int, tuple[str, dict[str, float]]] = {}
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


def _load_embeddings(precomputed_dir: str) -> tuple[np.ndarray | None, dict[str, int] | None]:
    emb_path = os.path.join(precomputed_dir, "candidate_embeddings.npy")
    id_path = os.path.join(precomputed_dir, "embedding_ids.csv")
    if os.path.exists(emb_path) and os.path.exists(id_path):
        embs = np.load(emb_path)
        id_to_idx: dict[str, int] = {}
        with open(id_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_to_idx[row["candidate_id"]] = int(row["embedding_index"])
        return embs, id_to_idx
    return None, None


def precompute(
    path: str,
    out_dir: str,
    jd_path: str | None = None,
    train: bool = False,
):
    t0 = time.time()
    logger.info("Loading candidates from %s...", path)
    candidates = load_candidates(path)
    if not candidates:
        logger.warning("No candidates loaded!")
        return
    logger.info("Loaded %d candidates in %.1fs", len(candidates), time.time() - t0)

    os.makedirs(out_dir, exist_ok=True)

    ids: list[str] = []
    all_features: list[dict[str, float]] = []

    logger.info("Extracting features...")
    extracted = _extract_all(candidates)
    for cid, feats in extracted:
        ids.append(cid)
        all_features.append(feats)

    _precompute_embeddings(candidates, out_dir)

    jd_text = _load_jd_text(jd_path) if jd_path else _build_jd_text()
    if jd_text:
        candidate_texts = [_get_candidate_text(c) for c in candidates]
        semantic_scores = _compute_semantic_features(candidates, candidate_texts, jd_text)
        for i, sim in enumerate(semantic_scores):
            if i < len(all_features):
                all_features[i]["semantic_similarity"] = sim

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
        logger.info("Training ML model on behavioral signals...")
        signals_list = [c.get("redrob_signals", {}) for c in candidates]
        train_model(all_features, signals_list=signals_list, model_path=None)

    logger.info("Precomputation done in %.1fs total", time.time() - t0)


def rank(
    candidates_path: str,
    output_path: str,
    precomputed_dir: str | None = None,
    jd_path: str | None = None,
    model_path: str | None = None,
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

    precomputed = precomputed_dir and os.path.exists(os.path.join(precomputed_dir, "features.npz"))
    all_features: list[dict[str, float]] = []
    # Track which original candidate indices made it through the pipeline
    used_indices: list[int] = []

    if precomputed:
        logger.info("Loading precomputed features...")
        npz = np.load(os.path.join(precomputed_dir, "features.npz"))
        feature_matrix = npz["features"]
        feature_keys = list(npz["keys"])

        meta_path = os.path.join(precomputed_dir, "metadata.csv")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                reader = csv.DictReader(f)
                id_to_idx = {row["candidate_id"]: int(row["feature_index"]) for row in reader}

            # Two-stage retrieval: BM25 -> dense -> full scoring
            ref_text = jd_text_loaded or _build_jd_text()
            bm25 = _build_bm25_index(candidates)
            bm25_indices, bm25_cands = _bm25_prefilter(bm25, candidates, ref_text, top_k=5000)

            emb, emb_id_map = _load_embeddings(precomputed_dir)
            if emb is not None:
                model = _get_sentence_transformer()
                jd_emb = model.encode(
                    [ref_text], show_progress_bar=False, normalize_embeddings=True
                )
                dense_indices = _dense_rerank(jd_emb, emb, bm25_indices, top_k=2000)
            else:
                dense_indices = bm25_indices[:2000]

            used_indices = dense_indices

            for orig_idx in dense_indices:
                c = candidates[orig_idx]
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
            logger.info(
                "Two-stage retrieval selected %d candidates (BM25->%d, dense->%d)",
                len(all_features),
                len(bm25_indices),
                len(dense_indices),
            )
        else:
            logger.info("No metadata csv, extracting features for all candidates...")
            extracted = _extract_all(candidates)
            all_features = [feats for _, feats in extracted]
            used_indices = list(range(len(candidates)))
    else:
        logger.info("Extracting features on the fly...")
        extracted = _extract_all(candidates)
        all_features = [feats for _, feats in extracted]
        used_indices = list(range(len(candidates)))

    if not all_features:
        logger.warning("No features available for ranking!")
        return

    ref_text = jd_text_loaded or _build_jd_text()
    if ref_text and "semantic_similarity" not in all_features[0]:
        used_candidates = [candidates[i] for i in used_indices]
        candidate_texts = [_get_candidate_text(c) for c in used_candidates]
        emb_dir = precomputed_dir if precomputed else None
        semantic_scores = _compute_semantic_features(
            used_candidates,
            candidate_texts,
            ref_text,
            precomputed_dir=emb_dir,
            orig_indices=used_indices,
        )
        for i, sim in enumerate(semantic_scores):
            if i < len(all_features):
                all_features[i]["semantic_similarity"] = sim

    ml_model = load_model(model_path) if model_path and os.path.exists(model_path) else None
    if model_path and ml_model is None:
        logger.info("No pre-trained model found, training from behavioral signals...")
        sigs = [c.get("redrob_signals", {}) for c in candidates]
        ml_model = train_model(all_features, signals_list=sigs, model_path=model_path)

    logger.info("Ranking candidates...")
    used_candidates = [candidates[i] for i in used_indices]
    ids = [c.get("candidate_id") or str(i) for i, c in enumerate(used_candidates)]
    sigs = [c.get("redrob_signals", {}) for c in used_candidates]
    ranked = rank_candidates(
        ids,
        all_features,
        jd_weights=jd_weights,
        ml_model=ml_model,
        signals_list=sigs,
    )

    logger.info("Calibrating scores...")
    ranked = calibrate_scores(ranked)

    logger.info("Generating reasoning...")
    id_to_feats: dict[str, dict[str, float]] = dict(zip(ids, all_features, strict=True))
    id_to_candidate: dict[str, dict[str, Any]] = {
        c.get("candidate_id") or str(i): c for i, c in enumerate(used_candidates)
    }
    results: list[tuple[str, int, float, str, dict[str, float]]] = []
    for cid, score, rank_pos, dims in ranked[:100]:
        feats = id_to_feats.get(cid, {}) or {}
        cand = id_to_candidate.get(cid, {})
        reasoning = generate_reasoning(cid, score, rank_pos, dims, feats, candidate=cand)
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
                "dimensions": {k: round(float(v), 4) for k, v in dims.items() if v is not None},
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
                logger.info("Fairness audit passed — no significant disparities detected")

    logger.info("Writing top-100 to %s...", output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank_pos, score, reasoning, _dims in results:
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)


def serve(host: str, port: int, model_path: str | None = None):
    import serve as serve_module

    serve_module.main(host=host, port=port, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="Recruiting Rank AI — Candidate Ranking System")
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
        help="Path to JD description text file for similarity scoring",
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
