#!/usr/bin/env python3
"""
Recruiting Rank AI — Candidate Ranking System

Usage:
    python rank.py --candidates <candidates.jsonl> --out <submission.csv>

Pre-computation (optional):
    python rank.py --precompute --candidates <candidates.jsonl>
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional

import numpy as np

from features.extractor import load_candidates, extract_all_features
from scoring.ranker import (
    compute_dimension_scores,
    compute_final_score,
    rank_candidates,
    generate_reasoning,
)


def precompute(path: str, out_dir: str):
    t0 = time.time()
    print(f"Loading candidates from {path}...")
    candidates = load_candidates(path)
    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    ids = []
    all_features = []
    for i, c in enumerate(candidates):
        if i % 10000 == 0 and i > 0:
            print(f"  processing {i}/{len(candidates)}...")
        ids.append(c["candidate_id"])
        all_features.append(extract_all_features(c))

    feature_keys = list(all_features[0].keys())
    feature_matrix = np.zeros((len(all_features), len(feature_keys)), dtype=np.float32)
    for i, feats in enumerate(all_features):
        for j, key in enumerate(feature_keys):
            feature_matrix[i, j] = feats.get(key, 0.0)

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "features.npz")
    np.savez_compressed(npz_path, features=feature_matrix, keys=feature_keys)
    print(f"Saved features to {npz_path}")

    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "feature_index"])
        for i, cid in enumerate(ids):
            w.writerow([cid, i])
    print(f"Saved metadata to {meta_path}")
    print(f"Precomputation done in {time.time() - t0:.1f}s total")


def rank(
    candidates_path: str,
    output_path: str,
    precomputed_dir: Optional[str] = None,
):
    t0 = time.time()
    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    if precomputed_dir and os.path.exists(
        os.path.join(precomputed_dir, "features.npz")
    ):
        print("Loading precomputed features...")
        npz = np.load(os.path.join(precomputed_dir, "features.npz"))
        feature_matrix = npz["features"]
        feature_keys = list(npz["keys"])
        with open(os.path.join(precomputed_dir, "metadata.csv")) as f:
            reader = csv.DictReader(f)
            id_to_idx = {
                row["candidate_id"]: int(row["feature_index"]) for row in reader
            }
        all_features = []
        for c in candidates:
            cid = c["candidate_id"]
            idx = id_to_idx[cid]
            feats = {
                feature_keys[j]: float(feature_matrix[idx, j])
                for j in range(len(feature_keys))
            }
            all_features.append(feats)
        print(f"Loaded features for {len(all_features)} candidates")
    else:
        print("Extracting features on the fly...")
        all_features = []
        for i, c in enumerate(candidates):
            if i % 10000 == 0 and i > 0:
                print(f"  extracting {i}/{len(candidates)}...")
            all_features.append(extract_all_features(c))
        print(f"Extracted features for {len(all_features)} candidates")

    print("Ranking candidates...")
    ids = [c["candidate_id"] for c in candidates]
    ranked = rank_candidates(ids, all_features)

    print("Generating reasoning...")
    results = []
    for cid, score, rank_pos, dims in ranked[:100]:
        idx = ids.index(cid)
        reasoning = generate_reasoning(cid, score, rank_pos, dims, all_features[idx])
        results.append((cid, rank_pos, round(score, 2), reasoning))

    print(f"Writing top-100 to {output_path}...")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank_pos, score, reasoning in results:
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    print(f"Done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Recruiting Rank AI — Candidate Ranking System"
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
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
    args = parser.parse_args()

    if args.precompute:
        precompute(args.candidates, args.features_dir or "data")
    else:
        rank(args.candidates, args.out, args.features_dir or "data")


if __name__ == "__main__":
    main()
