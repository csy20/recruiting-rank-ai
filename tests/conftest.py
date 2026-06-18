import pytest


@pytest.fixture
def sample_candidate():
    return {
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


@pytest.fixture
def sample_features():
    return {
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
        "risk_score": 0.0,
        "anti_pattern_count": 0,
        "is_honeypot": 0.0,
        "education_level": 0.7,
        "education_field": 1.0,
        "certifications": 1.0,
        "text_length": 500,
        "location_score": 1.0,
        "semantic_similarity": 0.0,
    }


@pytest.fixture
def sample_dim_scores():
    return {
        "technical_match": 0.8,
        "semantic_match": 0.6,
        "career_quality": 0.7,
        "behavioral": 0.7,
        "retention": 0.6,
        "risk_adjustment": 0.95,
        "jd_semantic_similarity": 0.5,
    }
