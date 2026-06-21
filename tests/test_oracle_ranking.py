import numpy as np

from features.extractor import extract_all_features
from scoring.ranker import calibrate_scores, rank_candidates


class TestRankingInvariants:
    def test_oracle_prefers_jd_relevant_over_irrelevant(self, sample_candidate):
        jd_relevant = dict(sample_candidate)
        jd_relevant["candidate_id"] = "JD_MATCH"
        jd_relevant["profile"] = dict(jd_relevant["profile"])
        jd_relevant["profile"]["summary"] = (
            "Expert in embeddings, vector search, FAISS, Pinecone, "
            "dense retrieval, BM25, ranking systems, NDCG, MRR, "
            "semantic search, transformers, BERT, fine-tuning, LoRA, RAG. "
            "Built production retrieval pipelines at scale with Python and PyTorch."
        )
        jd_relevant["skills"] = [
            {"name": "Python", "proficiency": "expert", "duration_months": 72},
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 48},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 36},
            {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 24},
            {"name": "Docker", "proficiency": "intermediate", "duration_months": 18},
        ]

        irrelevant = dict(sample_candidate)
        irrelevant["candidate_id"] = "IRRELEVANT"
        irrelevant["profile"] = dict(irrelevant["profile"])
        irrelevant["profile"]["summary"] = (
            "Marketing manager with experience in content creation "
            "and social media strategy. Managed brand campaigns."
        )
        irrelevant["skills"] = [
            {"name": "Marketing", "proficiency": "expert", "duration_months": 60},
            {"name": "Content", "proficiency": "expert", "duration_months": 36},
        ]

        feats_relevant = extract_all_features(jd_relevant)
        feats_irrelevant = extract_all_features(irrelevant)

        for k in ("semantic_similarity",):
            feats_relevant[k] = 0.9
            feats_irrelevant[k] = 0.1

        ranked = rank_candidates(
            ["JD_MATCH", "IRRELEVANT"],
            [feats_relevant, feats_irrelevant],
        )

        assert ranked[0][0] == "JD_MATCH", (
            f"Expected JD_MATCH first, got {ranked[0][0]} "
            f"(scores: {ranked[0][1]:.2f} vs {ranked[1][1]:.2f})"
        )
        assert ranked[0][1] > ranked[1][1] + 5.0, (
            f"JD-relevant candidate should score significantly higher: "
            f"{ranked[0][1]:.2f} vs {ranked[1][1]:.2f}"
        )

    def test_calibration_preserves_order(self, sample_features):
        feats_a = dict(sample_features)
        feats_b = dict(sample_features)
        for k in ("jd_match_embeddings", "jd_match_vector_db", "ai_depth", "retrieval_depth"):
            feats_b[k] = feats_a[k] * 0.3

        ranked = rank_candidates(["A", "B"], [feats_a, feats_b])
        calibrated = calibrate_scores(ranked)
        assert calibrated[0][0] == "A"
        assert calibrated[1][0] == "B"
        assert calibrated[0][1] >= calibrated[1][1]

    def test_scores_non_increasing(self, sample_features):
        feats = [sample_features for _ in range(5)]
        with np.nditer(np.array([0.9, 0.1, 0.5, 0.3, 0.7])) as it:
            for i, sim in enumerate(it):
                f = dict(feats[i])
                f["semantic_similarity"] = float(sim)
                feats[i] = f

        ranked = rank_candidates([f"C_{i}" for i in range(5)], feats)
        calibrated = calibrate_scores(ranked)
        scores = [r[1] for r in calibrated]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_honeypot_ranked_low(self, sample_candidate, sample_features):
        clean = dict(sample_features)
        honeypot = dict(sample_features)
        honeypot["is_honeypot"] = 1.0
        honeypot["risk_score"] = 1.0
        honeypot["high_confidence_honeypot"] = 1.0

        ranked = rank_candidates(["CLEAN", "HONEYPOT"], [clean, honeypot])
        clean_rank = next(r for r in ranked if r[0] == "CLEAN")
        hp_rank = next(r for r in ranked if r[0] == "HONEYPOT")
        assert clean_rank[1] > hp_rank[1], (
            f"Honeypot ({hp_rank[1]:.2f}) should score below clean ({clean_rank[1]:.2f})"
        )

    def test_disqualifiers_cap_scores(self, sample_features):
        disqualified = dict(sample_features)
        disqualified["disq_consulting_only"] = 1.0

        ranked = rank_candidates(["NORMAL", "DISQ"], [sample_features, disqualified])
        normal_score = ranked[0][1]
        disq_score = ranked[1][1]
        assert disq_score <= 55.0, f"consulting_only disqualified candidate scored {disq_score:.2f}"
        assert normal_score > disq_score, "Disqualified candidate outranks normal candidate"

    def test_repeated_rank_is_deterministic(self, sample_features):
        feats = [dict(sample_features) for _ in range(5)]
        for i, f in enumerate(feats):
            f["years_experience"] = float(10 - i)

        result_a = rank_candidates([f"C_{i}" for i in range(5)], feats)
        result_b = rank_candidates([f"C_{i}" for i in range(5)], feats)

        for ra, rb in zip(result_a, result_b, strict=True):
            assert ra[0] == rb[0], f"Determinism broken: {ra[0]} vs {rb[0]}"
            assert abs(ra[1] - rb[1]) < 1e-6, f"Score mismatch: {ra[1]:.6f} vs {rb[1]:.6f}"

    def test_ml_scores_bounded(self, sample_features):
        from scoring.ranker import train_model

        mod_feats = []
        for i in range(10):
            f = dict(sample_features)
            f["years_experience"] = float(i + 1)
            mod_feats.append(f)
        model = train_model(mod_feats, scores=[0.9 - i * 0.05 for i in range(10)])
        assert model is not None

        ranked = rank_candidates([f"C_{i}" for i in range(10)], mod_feats, ml_model=model)
        for cid, score, _rank_pos, _dims in ranked:
            assert 0 <= score <= 100, f"Score {score:.2f} out of range for {cid}"

    def test_oracle_known_ordering(self, sample_features):
        candidates = []
        for i in range(5):
            f = dict(sample_features)
            multiplier = 1.0 - i * 0.15
            for k in (
                "jd_match_embeddings",
                "jd_match_vector_db",
                "jd_match_ranking",
                "ai_depth",
                "retrieval_depth",
                "keyword_diversity",
            ):
                f[k] = f.get(k, 0.5) * multiplier
            candidates.append(f)

        ranked = rank_candidates([f"BEST_{i}" for i in range(5)], candidates)
        for j in range(len(ranked) - 1):
            r1, r2 = ranked[j], ranked[j + 1]
            assert r1[1] >= r2[1], (
                f"Order violation at position {j}: {r1[0]} ({r1[1]:.2f}) < {r2[0]} ({r2[1]:.2f})"
            )

    def test_concept_boost_improves_semantic_match(self, sample_candidate):
        from scoring.skill_graph import compute_concept_boost

        jd_text = "building RAG pipeline with vector search and embedding models"
        cand_match = "experience with RAG, FAISS, and sentence-transformers"
        cand_no_match = "marketing content creation and social media strategy"
        boost_match = compute_concept_boost(jd_text, cand_match)
        boost_no = compute_concept_boost(jd_text, cand_no_match)
        assert boost_match >= boost_no, (
            f"Relevant candidate ({boost_match:.4f}) should have "
            f"higher concept boost than irrelevant ({boost_no:.4f})"
        )
