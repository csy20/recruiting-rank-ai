import json

from features.extractor import extract_all_features, load_candidates
from scoring.jd_parser import get_jd_dimension_weights, parse_jd
from scoring.ranker import calibrate_scores, generate_reasoning, rank_candidates


class TestFullPipeline:
    def test_jsonl_to_ranked_csv(self, tmp_path, sample_candidate):
        jsonl_path = tmp_path / "candidates.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(3):
                c = dict(sample_candidate)
                c["candidate_id"] = f"CAND_{i:04d}"
                f.write(json.dumps(c) + "\n")

        candidates = load_candidates(str(jsonl_path))
        assert len(candidates) == 3

        features_list = [extract_all_features(c) for c in candidates]
        ids = [c.get("candidate_id") for c in candidates]

        ranked = rank_candidates(ids, features_list)
        calibrated = calibrate_scores(ranked)

        assert len(calibrated) == 3
        for cid, score, rank_pos, _dims in calibrated:
            assert 0 <= score <= 100
            assert 1 <= rank_pos <= 3
            assert cid.startswith("CAND_")

    def test_with_jd_parsing(self, sample_candidate):
        jd_text = """
        Requirements: Python, PyTorch, embeddings, vector databases, production ML.
        Preferred: Kubernetes, AWS.
        """
        jd_profile = parse_jd(jd_text)
        jd_weights = get_jd_dimension_weights(jd_profile)
        assert len(jd_weights) > 0

        features = extract_all_features(sample_candidate)
        features["tfidf_jd_similarity"] = 0.5

        ranked = rank_candidates(
            [sample_candidate["candidate_id"]], [features], jd_weights=jd_weights
        )
        calibrated = calibrate_scores(ranked)
        assert len(calibrated) == 1
        cid, score, rank_pos, dims = calibrated[0]
        assert score > 0

    def test_reasoning_generation(self, sample_candidate):
        features = extract_all_features(sample_candidate)
        dims = {
            "technical_match": 0.8,
            "semantic_match": 0.6,
            "career_quality": 0.7,
            "behavioral": 0.7,
            "retention": 0.6,
            "risk_adjustment": 0.95,
        }
        reasoning = generate_reasoning(sample_candidate["candidate_id"], 85.0, 1, dims, features)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_all_dimensions_produced(self, sample_candidate):
        features = extract_all_features(sample_candidate)
        expected_keys = {
            "years_experience",
            "num_skills",
            "ai_depth",
            "retrieval_depth",
            "keyword_diversity",
            "company_prestige",
            "education_level",
            "risk_score",
            "is_honeypot",
            "skill_match_score",
        }
        assert expected_keys.issubset(features.keys())
