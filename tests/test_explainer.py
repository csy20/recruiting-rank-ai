from scoring.explainer import compute_feature_contributions, explain_ranking

SAMPLE_FEATURES = {
    "years_experience": 8.0,
    "ai_depth": 0.7,
    "retrieval_depth": 0.6,
    "keyword_diversity": 0.6,
    "company_prestige": 1.0,
    "education_level": 0.7,
    "certifications": 1.0,
    "risk_score": 0.0,
    "anti_pattern_count": 0,
    "tfidf_jd_similarity": 0.0,
    "growth_rate": 0.6,
}

SAMPLE_DIMS = {
    "technical_match": 0.8,
    "semantic_match": 0.6,
    "career_quality": 0.7,
    "behavioral": 0.7,
    "retention": 0.6,
    "risk_adjustment": 0.95,
    "jd_semantic_similarity": 0.5,
}


class TestComputeFeatureContributions:
    def test_basic(self):
        contribs = compute_feature_contributions(SAMPLE_FEATURES, SAMPLE_DIMS)
        assert len(contribs) > 0
        assert "technical_match" in contribs

    def test_all_positive(self):
        contribs = compute_feature_contributions(SAMPLE_FEATURES, SAMPLE_DIMS)
        assert all(isinstance(v, float) for v in contribs.values())

    def test_high_risk_penalty(self):
        high_risk = SAMPLE_FEATURES.copy()
        high_risk["risk_score"] = 0.8
        high_risk["anti_pattern_count"] = 5
        contribs = compute_feature_contributions(high_risk, SAMPLE_DIMS)
        assert contribs["risk_penalty_detail"] < 0


class TestExplainRanking:
    def test_basic(self):
        result = explain_ranking(SAMPLE_FEATURES, SAMPLE_DIMS, 75.0)
        assert "strengths" in result
        assert "weaknesses" in result
        assert "feature_contributions" in result
        assert result["final_score"] == 75.0

    def test_strengths_high_tech(self):
        result = explain_ranking(SAMPLE_FEATURES, SAMPLE_DIMS, 75.0)
        assert any("technical" in s.lower() for s in result["strengths"])

    def test_empty_features(self):
        result = explain_ranking({}, {}, 0.0)
        assert isinstance(result["strengths"], list)
