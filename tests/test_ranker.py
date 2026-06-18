import os

from scoring.jd_parser import (
    get_jd_dimension_weights,
    get_jd_experience_score,
    parse_jd,
)
from scoring.ranker import (
    _features_to_vector,
    audit_fairness,
    calibrate_scores,
    compute_dimension_scores,
    compute_final_score,
    generate_reasoning,
    load_model,
    rank_candidates,
    train_model,
)

SAMPLE_FEATURES = {
    "years_experience": 8.0,
    "num_skills": 10,
    "expert_skills": 5,
    "num_career_entries": 3,
    "num_companies": 2,
    "jd_match_embeddings": 0.8,
    "jd_match_vector_db": 0.6,
    "jd_match_ranking": 0.7,
    "jd_match_ml_production": 0.5,
    "jd_match_nlp_ir": 0.4,
    "jd_match_python": 1.0,
    "jd_match_llm": 0.3,
    "jd_match_distributed_systems": 0.2,
    "jd_match_data_engineering": 0.1,
    "ai_depth": 0.7,
    "retrieval_depth": 0.6,
    "eval_depth": 0.5,
    "keyword_diversity": 0.6,
    "career_progression": 0.8,
    "career_seniority": 0.7,
    "growth_rate": 0.6,
    "has_product_exp": 1.0,
    "entirely_consulting": 0.0,
    "consulting_with_ml": 0.0,
    "company_prestige": 1.0,
    "beh_recruiter_response_rate": 0.8,
    "beh_interview_completion_rate": 0.9,
    "beh_open_to_work_flag": 1.0,
    "beh_saved_by_recruiters_30d": 0.5,
    "beh_search_appearance_30d": 0.4,
    "beh_github_activity_score": 0.7,
    "beh_recent_activity": 0.9,
    "beh_profile_completeness": 0.8,
    "beh_offer_acceptance_rate": 0.7,
    "beh_avg_response_time_hours": 0.6,
    "beh_verified_contact": 1.0,
    "beh_connection_density": 0.5,
    "ret_overall": 0.8,
    "ret_notice_score": 0.7,
    "ret_tenure_score": 0.8,
    "ret_job_hop_penalty": 0.0,
    "risk_score": 0.0,
    "anti_pattern_count": 0,
    "is_honeypot": 0.0,
    "education_level": 0.7,
    "education_field": 1.0,
    "certifications": 1.0,
    "text_length": 500,
    "location_score": 1.0,
    "tfidf_jd_similarity": 0.0,
}


class TestComputeDimensionScores:
    def test_all_dimensions_present(self):
        dims = compute_dimension_scores(SAMPLE_FEATURES)
        for d in [
            "technical_match",
            "semantic_match",
            "career_quality",
            "behavioral",
            "retention",
            "risk_adjustment",
        ]:
            assert d in dims
            assert 0 <= dims[d] <= 1.0

    def test_high_risk_lowers_score(self):
        high_risk = SAMPLE_FEATURES.copy()
        high_risk["risk_score"] = 0.8
        high_risk["is_honeypot"] = 1.0
        assert (
            compute_dimension_scores(high_risk)["risk_adjustment"]
            < compute_dimension_scores(SAMPLE_FEATURES)["risk_adjustment"]
        )


class TestComputeFinalScore:
    def test_basic(self):
        dims = compute_dimension_scores(SAMPLE_FEATURES)
        score = compute_final_score(dims)
        assert 0 <= score <= 100

    def test_with_ml_score(self):
        dims = compute_dimension_scores(SAMPLE_FEATURES)
        score = compute_final_score(dims, ml_score=0.8)
        assert 0 <= score <= 100


class TestRankCandidates:
    def test_ranking_order(self):
        high = SAMPLE_FEATURES.copy()
        low = SAMPLE_FEATURES.copy()
        for k in low:
            if k.startswith("jd_match_"):
                low[k] = 0.0
        low["ai_depth"] = 0.0
        low["retrieval_depth"] = 0.0

        ranked = rank_candidates(["high", "low"], [high, low])
        assert ranked[0][0] == "high"
        assert ranked[1][0] == "low"
        assert ranked[0][1] >= ranked[1][1]

    def test_with_jd_weights(self):
        jd_weights = {"embeddings": 0.5, "vector_db": 0.5}
        ranked = rank_candidates(["a"], [SAMPLE_FEATURES], jd_weights=jd_weights)
        assert len(ranked) == 1


class TestCalibrateScores:
    def test_calibration(self):
        data = [("a", 50.0, 1, {}), ("b", 25.0, 2, {})]
        calibrated = calibrate_scores(data)
        assert calibrated[0][1] == 100.0
        assert calibrated[1][1] == 0.0

    def test_empty(self):
        assert calibrate_scores([]) == []


class TestGenerateReasoning:
    def test_basic(self):
        dims = compute_dimension_scores(SAMPLE_FEATURES)
        score = compute_final_score(dims)
        reasoning = generate_reasoning("CAND_001", score, 1, dims, SAMPLE_FEATURES)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_honeypot_flag(self):
        honeypot_feats = SAMPLE_FEATURES.copy()
        honeypot_feats["is_honeypot"] = 1.0
        dims = compute_dimension_scores(honeypot_feats)
        reasoning = generate_reasoning("CAND_H", 10.0, 100, dims, honeypot_feats)
        assert "HONEYPOT" in reasoning


class TestFeaturesToVector:
    def test_vector_consistency(self):
        vec1 = _features_to_vector(SAMPLE_FEATURES)
        vec2 = _features_to_vector(SAMPLE_FEATURES)
        assert vec1 == vec2
        assert len(vec1) == 22

    def test_empty_features(self):
        vec = _features_to_vector({})
        assert len(vec) == 22
        assert all(v == 0.0 for v in vec)


class TestJDParser:
    def test_parse_jd_basic(self):
        jd_text = """
        About: We need a Senior ML Engineer.
        Requirements: 5+ years Python, TensorFlow, NLP experience.
        Preferred: Kubernetes, AWS, PhD in CS.
        """
        profile = parse_jd(jd_text)
        assert "keywords" in profile
        assert "dimension_analysis" in profile
        assert "experience_years" in profile
        assert profile["text_length"] > 0

    def test_get_dimension_weights(self):
        jd_text = "Requirements: Python, embeddings, vector databases."
        profile = parse_jd(jd_text)
        weights = get_jd_dimension_weights(profile)
        assert len(weights) > 0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_experience_score(self):
        profile = {"experience_years": (3.0, 7.0)}
        assert get_jd_experience_score(5.0, profile) == 1.0
        assert get_jd_experience_score(10.0, profile) < 1.0


class TestAuditFairness:
    def test_enabled_with_empty(self):
        result = audit_fairness([], [], [])
        assert result["audit_enabled"] is True
        for group in [
            "consulting",
            "non_consulting",
            "high_prestige",
            "low_prestige",
            "high_location",
            "low_location",
        ]:
            assert result["group_sizes"][group] == 0

    def test_basic_audit(self, sample_features):
        features_list = [sample_features, sample_features]
        ids = ["a", "b"]
        dims = {
            "technical_match": 0.8,
            "semantic_match": 0.6,
            "career_quality": 0.7,
            "behavioral": 0.7,
            "retention": 0.6,
        }
        ranked = [("a", 90.0, 1, dims), ("b", 70.0, 2, dims)]
        result = audit_fairness(ranked, features_list, ids)
        assert result["audit_enabled"] is True
        assert "disparities" in result
        assert "group_sizes" in result

    def test_empty_group_handling(self, sample_features):
        consulting_feats = sample_features.copy()
        consulting_feats["entirely_consulting"] = 1.0
        features_list = [consulting_feats]
        ids = ["c"]
        dims = {
            "technical_match": 0.5,
            "semantic_match": 0.5,
            "career_quality": 0.5,
            "behavioral": 0.5,
            "retention": 0.5,
        }
        ranked = [("c", 50.0, 1, dims)]
        result = audit_fairness(ranked, features_list, ids)
        assert result["audit_enabled"] is True


class TestTrainModel:
    def test_empty_features(self):
        model = train_model([])
        assert model is None

    def test_training_with_scores(self, sample_features):
        model = train_model(
            [sample_features, sample_features],
            scores=[0.8, 0.6],
        )
        assert model is not None
        feature_vec = _features_to_vector(sample_features)
        pred = model.predict([feature_vec])
        assert 0 <= float(pred[0]) <= 1.0


class TestLoadModel:
    def test_nonexistent_path(self):
        model = load_model("/tmp/nonexistent_model.pkl")
        assert model is None

    def test_save_and_load_cycle(self, tmp_path, sample_features):
        from scoring.ranker import train_model

        model_path = str(tmp_path / "test_model.pkl")
        original = train_model(
            [sample_features, sample_features], scores=[0.8, 0.6], model_path=model_path
        )
        assert original is not None
        assert os.path.exists(model_path)
        loaded = load_model(model_path)
        assert loaded is not None
        feature_vec = _features_to_vector(sample_features)
        pred = loaded.predict([feature_vec])
        assert 0 <= float(pred[0]) <= 1.0
