import os
from datetime import date

from config.data import (
    CERTIFICATION_PATTERNS,
    COMPANY_TIERS,
    CONSULTING_COMPANIES,
    CONSULTING_WITH_ML_BOOKS,
    EDUCATION_FIELDS,
    EDUCATION_PATTERNS,
    JD_ANTI_PATTERNS,
    NON_TECH_TITLES,
    PREFERRED_LOCATIONS,
    SENIORITY_KEYWORDS,
    SKILL_CATEGORIES,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES_PATH = os.environ.get(
    "RR_CANDIDATES_PATH",
    os.path.join(BASE_DIR, "data", "candidates.jsonl"),
)
OUTPUT_PATH = os.environ.get("RR_OUTPUT_PATH", os.path.join(BASE_DIR, "submission.csv"))
FEATURES_PATH = os.environ.get("RR_FEATURES_PATH", os.path.join(BASE_DIR, "data", "features.npz"))
METADATA_PATH = os.environ.get("RR_METADATA_PATH", os.path.join(BASE_DIR, "data", "metadata.csv"))
MODEL_PATH = os.environ.get("RR_MODEL_PATH", os.path.join(BASE_DIR, "data", "ranker_model.pkl"))
JD_PATH = os.environ.get("RR_JD_PATH", "")
API_HOST = os.environ.get("RR_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("RR_API_PORT", "8000"))

REFERENCE_DATE = date(2026, 6, 1)

JD_KEYWORDS = {
    "embeddings": {
        "terms": [
            "embedding",
            "sentence-transformer",
            "sentence transformer",
            "bge",
            "e5",
            "openai embedding",
        ],
        "weight": 3.0,
    },
    "vector_db": {
        "terms": [
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "faiss",
            "opensearch",
            "elasticsearch",
            "vector database",
            "hybrid search",
            "vector search",
            "annoy",
            "hnsw",
            "vector store",
        ],
        "weight": 3.0,
    },
    "ranking": {
        "terms": [
            "ranking",
            "ranker",
            "learning to rank",
            "ltr",
            "ndcg",
            "mrr",
            "map",
            "retrieval",
            "relevance",
            "evaluation framework",
            "a/b test",
            "ab test",
            "offline evaluation",
            "search relevance",
            "recommendation",
            "discovery feed",
            "click-through",
        ],
        "weight": 3.0,
    },
    "python": {"terms": ["python"], "weight": 1.5},
    "ml_production": {
        "terms": [
            "production",
            "deploy",
            "deployment",
            "pipeline",
            "serving",
            "inference",
            "model deployment",
            "ml system",
            "feature store",
            "feature engineering",
            "a/b test",
            "ab test",
        ],
        "weight": 2.5,
    },
    "llm": {
        "terms": [
            "llm",
            "fine-tune",
            "fine tuning",
            "lora",
            "qlora",
            "peft",
            "rag",
            "retrieval augmented generation",
            "prompt",
        ],
        "weight": 1.5,
    },
    "nlp_ir": {
        "terms": [
            "nlp",
            "natural language",
            "information retrieval",
            "text classification",
            "ner",
            "named entity",
            "text mining",
            "semantic",
            "transformers",
            "hugging face",
        ],
        "weight": 2.0,
    },
    "distributed_systems": {
        "terms": [
            "distributed",
            "kafka",
            "spark",
            "flink",
            "beam",
            "scal",
            "high throughput",
            "low latency",
        ],
        "weight": 1.0,
    },
    "data_engineering": {
        "terms": [
            "etl",
            "data pipeline",
            "data warehouse",
            "airflow",
            "data engineering",
            "data infrastructure",
            "feature store",
        ],
        "weight": 0.5,
    },
}

SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"

BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.20,
    "interview_completion_rate": 0.12,
    "open_to_work_flag": 0.10,
    "saved_by_recruiters_30d": 0.08,
    "search_appearance_30d": 0.08,
    "github_activity_score": 0.10,
    "recent_activity": 0.05,
    "reachability": 0.08,
    "profile_completeness": 0.05,
    "offer_acceptance_rate": 0.05,
    "avg_response_time_hours": 0.04,
    "verified_contact": 0.04,
    "connection_density": 0.04,
}

DIMENSION_WEIGHTS = {
    "technical_match": 0.35,
    "semantic_match": 0.20,
    "career_quality": 0.10,
    "behavioral": 0.15,
    "retention": 0.10,
    "risk_adjustment": 0.0,
    "jd_semantic_similarity": 0.10,
}

SCORING = {
    "technical_match": {
        "core_tech_weight": 0.50,
        "depth_weight": 0.35,
        "keyword_diversity_weight": 0.15,
        "retrieval_subweight": 0.40,
        "ai_subweight": 0.35,
        "eval_subweight": 0.15,
        "diversity_subweight": 0.10,
        "retrieval_boost_factor": 1.5,
        "ai_boost_factor": 1.3,
        "eval_boost_factor": 1.5,
    },
    "semantic_match": {
        "transferable_weight": 0.50,
        "exp_band_weight": 0.20,
        "skill_breadth_weight": 0.15,
        "industry_relevance_weight": 0.15,
        "consulting_penalty": 0.30,
        "consulting_ml_credit": 0.5,
        "ideal_exp_min": 3.0,
        "ideal_exp_max": 12.0,
        "ideal_exp_center": 6.0,
    },
    "career_quality": {
        "progression_weight": 0.25,
        "seniority_weight": 0.20,
        "product_weight": 0.20,
        "company_prestige_weight": 0.15,
        "tenure_stability_weight": 0.0,
        "skill_growth_weight": 0.0,
        "production_signal_weight": 0.20,
        "consulting_penalty": 0.10,
    },
    "retention": {
        "tenure_weight": 0.50,
        "notice_weight": 0.25,
        "stability_weight": 0.25,
        "ideal_tenure_min": 2.0,
        "ideal_tenure_max": 5.0,
        "short_stint_months": 12,
    },
    "risk": {
        "risk_score_penalty": 0.40,
        "anti_pattern_penalty": 0.12,
        "honeypot_penalty": 0.80,
        "max_anti_pattern_penalty": 0.25,
    },
}

REDROB_BEHAVIORAL_SIGNALS = {
    "profile_completeness_score": {"range": (0, 100), "type": "float"},
    "recruiter_response_rate": {"range": (0, 1), "type": "float"},
    "interview_completion_rate": {"range": (0, 1), "type": "float"},
    "open_to_work_flag": {"range": (0, 1), "type": "bool"},
    "saved_by_recruiters_30d": {"range": (0, None), "type": "int"},
    "search_appearance_30d": {"range": (0, None), "type": "int"},
    "github_activity_score": {"range": (-1, 100), "type": "float"},
    "notice_period_days": {"range": (0, 180), "type": "int"},
    "willing_to_relocate": {"range": (0, 1), "type": "bool"},
    "offer_acceptance_rate": {"range": (-1, 1), "type": "float"},
    "avg_response_time_hours": {"range": (0, None), "type": "float"},
    "verified_email": {"range": (0, 1), "type": "bool"},
    "verified_phone": {"range": (0, 1), "type": "bool"},
    "linkedin_connected": {"range": (0, 1), "type": "bool"},
    "applications_submitted_30d": {"range": (0, None), "type": "int"},
    "profile_views_received_30d": {"range": (0, None), "type": "int"},
    "connection_count": {"range": (0, None), "type": "int"},
    "endorsements_received": {"range": (0, None), "type": "int"},
}

HONEYPOT_THRESHOLD = 0.6
INFLATED_DURATION_MONTHS = 60
INFLATED_SKILL_EXPERT_THRESHOLD = 6
INFLATED_SKILL_COUNT_THRESHOLD = 6
INFLATED_LOW_DURATION_THRESHOLD = 3
SHORT_STINT_MONTHS = 12

MODEL_CONFIG = {
    "enabled": True,
    "type": "random_forest",
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "test_size": 0.2,
}

ENSEMBLE_WEIGHTS = {
    "rule_based": 0.60,
    "ml_prediction": 0.25,
    "semantic_similarity": 0.15,
}

FAIRNESS_CONFIG = {
    "audit_enabled": True,
    "disparity_threshold": 0.15,
    "dimensions_to_audit": [
        "technical_match",
        "semantic_match",
        "career_quality",
        "behavioral",
        "retention",
    ],
}

API_CONFIG = {
    "max_candidates_per_request": 5000,
    "timeout_seconds": 30,
    "rate_limit_per_minute": 60,
}


def _deep_update(base: dict, overrides: dict):
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _load_yaml_overrides():
    yaml_path = os.environ.get("RR_CONFIG_PATH", os.path.join(BASE_DIR, "config.yaml"))
    if not os.path.exists(yaml_path):
        return
    try:
        import yaml

        with open(yaml_path) as f:
            overrides = yaml.safe_load(f)
        if not overrides:
            return

        global REFERENCE_DATE, SENTENCE_TRANSFORMER_MODEL
        global MODEL_CONFIG, ENSEMBLE_WEIGHTS, FAIRNESS_CONFIG, API_CONFIG
        global DIMENSION_WEIGHTS, SCORING, BEHAVIORAL_WEIGHTS

        if "reference_date" in overrides:
            from datetime import datetime as _dt

            REFERENCE_DATE = _dt.strptime(overrides["reference_date"], "%Y-%m-%d").date()
        if "sentence_transformer" in overrides:
            SENTENCE_TRANSFORMER_MODEL = str(overrides["sentence_transformer"])
        if "model" in overrides:
            MODEL_CONFIG.update(overrides["model"])
        if "ensemble_weights" in overrides:
            ENSEMBLE_WEIGHTS.update(overrides["ensemble_weights"])
        if "fairness" in overrides:
            FAIRNESS_CONFIG.update(overrides["fairness"])
        if "api" in overrides:
            API_CONFIG.update(overrides["api"])
        if "dimension_weights" in overrides:
            DIMENSION_WEIGHTS.update(overrides["dimension_weights"])
        if "scoring" in overrides:
            _deep_update(SCORING, overrides["scoring"])
        if "behavioral_weights" in overrides:
            BEHAVIORAL_WEIGHTS.update(overrides["behavioral_weights"])

    except ImportError:
        pass
    except Exception as e:
        import logging

        logging.getLogger("rank.config").warning("Failed to load YAML config: %s", e)


_load_yaml_overrides()
