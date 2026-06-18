from features.extractor import (
    _analyze_career_history,
    _compute_behavioral_score,
    _compute_growth_rate,
    _compute_jd_technical_match,
    _compute_keyword_diversity,
    _compute_location_score,
    _compute_retention_score,
    _compute_skill_categories,
    _detect_anti_patterns,
    _detect_education,
    _detect_honeypot,
    _extract_ai_depth,
    _get_combined_text,
    extract_all_features,
    load_candidates,
)

SAMPLE_CANDIDATE = {
    "candidate_id": "CAND_0001",
    "profile": {
        "headline": "Senior ML Engineer at Google",
        "summary": "Experienced in building production ML systems with Python and "
        "PyTorch. Expertise in NLP, embeddings, and vector search. "
        "PhD in Computer Science.",
        "current_company": "google",
        "current_title": "Senior ML Engineer",
        "years_of_experience": 8,
        "location": "Bangalore, India",
        "country": "India",
    },
    "career_history": [
        {
            "company": "Google",
            "title": "Senior ML Engineer",
            "description": "Built embedding pipelines and ranking systems at scale",
            "industry": "technology",
            "duration_months": 36,
            "is_current": True,
            "start_date": "2023-01-01",
        },
        {
            "company": "Amazon",
            "title": "ML Engineer",
            "description": "Worked on product recommendation systems",
            "industry": "technology",
            "duration_months": 24,
            "start_date": "2021-01-01",
            "end_date": "2022-12-31",
        },
    ],
    "skills": [
        {"name": "Python", "proficiency": "expert", "duration_months": 60},
        {"name": "PyTorch", "proficiency": "expert", "duration_months": 36},
        {"name": "AWS", "proficiency": "intermediate", "duration_months": 24},
        {"name": "Kubernetes", "proficiency": "intermediate", "duration_months": 12},
    ],
    "redrob_signals": {
        "recruiter_response_rate": 0.8,
        "interview_completion_rate": 0.9,
        "open_to_work_flag": True,
        "saved_by_recruiters_30d": 15,
        "search_appearance_30d": 30,
        "github_activity_score": 85,
        "last_active_date": "2026-05-15",
        "profile_completeness_score": 95,
        "notice_period_days": 30,
    },
}


class TestLoadCandidates:
    def test_jsonl(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"id": "1"}\n{"id": "2"}\n')
        result = load_candidates(str(p))
        assert len(result) == 2

    def test_json_array(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('[{"id": "1"}, {"id": "2"}]')
        result = load_candidates(str(p))
        assert len(result) == 2

    def test_limit(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"id": "1"}\n{"id": "2"}\n{"id": "3"}\n')
        result = load_candidates(str(p), limit=2)
        assert len(result) == 2

    def test_empty_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"id": "1"}\n\n{"id": "2"}\n')
        result = load_candidates(str(p))
        assert len(result) == 2

    def test_malformed_line_skipped(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"id": "1"}\nnot json\n{"id": "2"}\n')
        result = load_candidates(str(p))
        assert len(result) == 2


class TestGetCombinedText:
    def test_combines_fields(self):
        text = _get_combined_text(SAMPLE_CANDIDATE)
        assert "Senior" in text
        assert "embedding" in text
        assert "Python" in text

    def test_empty_profile(self):
        result = _get_combined_text({"profile": {}, "career_history": [], "skills": []})
        assert result == ""


class TestAnalyzeCareerHistory:
    def test_basic(self):
        career = _analyze_career_history(SAMPLE_CANDIDATE)
        assert career["distinct_companies"] == 2
        assert career["role_count"] == 2
        assert career["has_product_exp"] is True
        assert career["max_company_prestige"] >= 1.0

    def test_empty(self):
        career = _analyze_career_history({"profile": {}, "career_history": []})
        assert career["role_count"] == 0
        assert career["avg_tenure_months"] == 0.0
        assert career["career_progression"] == 0.5

    def test_consulting(self):
        cand = {
            "profile": {"current_company": "tcs"},
            "career_history": [
                {
                    "company": "tcs",
                    "title": "dev",
                    "industry": "services",
                    "duration_months": 24,
                }
            ],
        }
        career = _analyze_career_history(cand)
        assert career["entirely_consulting"] is True


class TestJDMatch:
    def test_basic(self):
        text = "I use embedding and vector databases with faiss"
        scores = _compute_jd_technical_match(text)
        assert scores.get("embeddings", 0) > 0
        assert scores.get("vector_db", 0) > 0

    def test_no_match(self):
        scores = _compute_jd_technical_match("")
        assert all(v == 0.0 for v in scores.values())


class TestAntiPatterns:
    def test_detects(self):
        assert _detect_anti_patterns("I am a prompt engineering expert") == 1

    def test_clean(self):
        assert _detect_anti_patterns("Python ML Engineer") == 0


class TestAIDepth:
    def test_basic(self):
        cand = {
            "profile": {"current_title": "ML Engineer"},
            "career_history": [],
            "skills": [],
        }
        text = "machine learning deep learning nlp transformer"
        assert _extract_ai_depth(text, cand) > 0

    def test_no_ai(self):
        cand = {
            "profile": {"current_title": "Marketing"},
            "career_history": [],
            "skills": [],
        }
        assert _extract_ai_depth("I like marketing", cand) == 0.0


class TestBehavioralScore:
    def test_full_score(self):
        signals = {
            "recruiter_response_rate": 1.0,
            "interview_completion_rate": 1.0,
            "open_to_work_flag": True,
            "saved_by_recruiters_30d": 100,
            "search_appearance_30d": 100,
            "github_activity_score": 100,
            "last_active_date": "2026-06-01",
            "profile_completeness_score": 100,
        }
        result = _compute_behavioral_score(signals)
        assert result["recruiter_response_rate"] == 1.0

    def test_empty(self):
        result = _compute_behavioral_score({})
        assert result["github_activity_score"] == 0.0

    def test_none(self):
        result = _compute_behavioral_score(None)
        assert result["github_activity_score"] == 0.0


class TestRetentionScore:
    def test_stable(self):
        cand = {
            "profile": {},
            "career_history": [{"duration_months": 48}, {"duration_months": 36}],
            "redrob_signals": {"notice_period_days": 30},
        }
        career = _analyze_career_history(cand)
        result = _compute_retention_score(cand, career)
        assert result["overall"] > 0.5


class TestHoneypot:
    def test_clean(self):
        result = _detect_honeypot(SAMPLE_CANDIDATE)
        assert result["is_honeypot"] is False

    def test_keyword_stuffer_no_ml_bg(self):
        cand = {
            "profile": {
                "summary": "I love AI and machine learning",
                "current_title": "Marketing Manager",
                "years_of_experience": 2,
            },
            "career_history": [],
            "skills": [
                {"proficiency": "expert", "duration_months": 1},
                {"proficiency": "expert", "duration_months": 1},
                {"proficiency": "expert", "duration_months": 1},
                {"proficiency": "expert", "duration_months": 1},
                {"proficiency": "expert", "duration_months": 1},
                {"proficiency": "expert", "duration_months": 1},
            ],
        }
        result = _detect_honeypot(cand)
        assert result["is_honeypot"] is True


class TestLocationScore:
    def test_preferred_city(self):
        cand = {
            "profile": {"location": "Bangalore, India", "country": "India"},
            "redrob_signals": {},
        }
        assert _compute_location_score(cand) == 1.0

    def test_outside_india(self):
        cand = {
            "profile": {"location": "New York, USA", "country": "USA"},
            "redrob_signals": {},
        }
        assert _compute_location_score(cand) == 0.75


class TestEducation:
    def test_detect_phd(self):
        cand = {
            "profile": {"summary": "PhD in Computer Science"},
            "career_history": [],
            "skills": [],
        }
        result = _detect_education(cand)
        assert result["education_level"] >= 0.8
        assert result["education_field"] >= 0.7

    def test_detect_bachelors(self):
        cand = {"profile": {"summary": "B.Tech"}, "career_history": [], "skills": []}
        result = _detect_education(cand)
        assert result["education_level"] >= 0.2


class TestSkillCategories:
    def test_categorization(self):
        skills = ["python", "aws", "docker", "kubernetes", "tensorflow"]
        result = _compute_skill_categories(skills)
        assert result.get("skill_cat_cloud_infra", 0) >= 2
        assert result.get("skill_cat_backend", 0) >= 1


class TestKeywordDiversity:
    def test_basic(self):
        text = "embeddings vector database ranking python"
        diversity = _compute_keyword_diversity(text)
        assert diversity > 0


class TestGrowthRate:
    def test_basic(self):
        career = {
            "total_experience_months": 60,
            "career_seniority": 0.8,
            "role_count": 3,
        }
        rate = _compute_growth_rate(career)
        assert rate > 0

    def test_empty(self):
        assert _compute_growth_rate({}) == 0.5


class TestExtractAllFeatures:
    def test_basic(self):
        features = extract_all_features(SAMPLE_CANDIDATE)
        assert features["years_experience"] == 8
        assert features["num_skills"] == 4
        assert features["company_prestige"] >= 1.0
        assert features["education_level"] >= 0.8
        assert features["education_field"] >= 0.7
        assert "skill_cat_cloud_infra" in features
        assert "keyword_diversity" in features
        assert "growth_rate" in features

    def test_all_keys_present(self):
        features = extract_all_features(SAMPLE_CANDIDATE)
        assert "jd_match_embeddings" in features
        assert "ai_depth" in features
        assert "retrieval_depth" in features
        assert "beh_recruiter_response_rate" in features
        assert "ret_overall" in features
        assert "beh_offer_acceptance_rate" in features
        assert "beh_verified_contact" in features
        assert "beh_connection_density" in features

    def test_empty_candidate(self):
        features = extract_all_features({})
        assert isinstance(features, dict)
        assert features.get("years_experience", 0) == 0
