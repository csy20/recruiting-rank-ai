import os
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATES_PATH = os.environ.get(
    "RR_CANDIDATES_PATH",
    os.path.join(
        os.path.dirname(BASE_DIR),
        "Downloads",
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "candidates.jsonl",
    ),
)
OUTPUT_PATH = os.environ.get("RR_OUTPUT_PATH", os.path.join(BASE_DIR, "submission.csv"))
FEATURES_PATH = os.environ.get(
    "RR_FEATURES_PATH", os.path.join(BASE_DIR, "data", "features.npz")
)
METADATA_PATH = os.environ.get(
    "RR_METADATA_PATH", os.path.join(BASE_DIR, "data", "metadata.csv")
)
MODEL_PATH = os.environ.get(
    "RR_MODEL_PATH", os.path.join(BASE_DIR, "data", "ranker_model.pkl")
)
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

JD_ANTI_PATTERNS = [
    "chatgpt",
    "prompt engineering",
    "langchain tutorial",
    "content creation",
    "marketing manager",
]

TFIDF_ENABLED = True
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)

CONSULTING_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "l&t infotech",
    "mindtree",
    "ltimindtree",
    "mphasis",
    "hexaware",
    "cyient",
    "persistent",
}

CONSULTING_WITH_ML_BOOKS = {
    "accenture",
    "ibm",
    "cognizant",
    "tcs",
    "infosys",
}

SENIORITY_KEYWORDS = [
    "senior",
    "staff",
    "principal",
    "lead",
    "head",
    "chief",
    "manager",
    "director",
    "architect",
    "fellow",
]

BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.20,
    "interview_completion_rate": 0.12,
    "open_to_work_flag": 0.10,
    "saved_by_recruiters_30d": 0.08,
    "search_appearance_30d": 0.08,
    "github_activity_score": 0.10,
    "recent_activity": 0.10,
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
    "risk_adjustment": -0.05,
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
        "tenure_stability_weight": 0.10,
        "skill_growth_weight": 0.10,
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

COMPANY_TIERS = {
    "tier1": {
        "companies": {
            "google",
            "meta",
            "microsoft",
            "amazon",
            "apple",
            "netflix",
            "openai",
            "deepmind",
            "anthropic",
            "nvidia",
            "ibm research",
            "microsoft research",
        },
        "score": 1.0,
    },
    "tier2": {
        "companies": {
            "uber",
            "lyft",
            "airbnb",
            "twitter",
            "linkedin",
            "salesforce",
            "oracle",
            "adobe",
            "intel",
            "ibm",
            "databricks",
            "snowflake",
            "palantir",
            "stripe",
            "square",
            "shopify",
            "spotify",
            "slack",
            "mongodb",
            "elastic",
        },
        "score": 0.85,
    },
    "tier3": {
        "companies": {
            "flipkart",
            "ola",
            "paytm",
            "swiggy",
            "zomato",
            "razorpay",
            "phonepe",
            "cred",
            "freshworks",
            "chargebee",
            "postman",
            "hasura",
            "zoho",
            "dream11",
            "unacademy",
            "byju",
            "meesho",
            "sharechat",
            "dunzo",
            "groww",
            "upstox",
        },
        "score": 0.70,
    },
    "tier4": {
        "companies": {
            "tcs",
            "infosys",
            "wipro",
            "hcl",
            "tech mahindra",
            "ltimindtree",
            "mindtree",
            "mphasis",
            "hexaware",
        },
        "score": 0.50,
    },
}

SKILL_CATEGORIES = {
    "cloud_infra": {
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
        "terraform",
        "cloudformation",
        "jenkins",
        "ci/cd",
        "ansible",
        "puppet",
        "chef",
        "helm",
        "istio",
    },
    "ml_ai": {
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "hugging face",
        "transformers",
        "neural network",
        "llm",
        "rag",
        "langchain",
        "lora",
        "qlora",
        "peft",
        "mlops",
        "model deployment",
        "model serving",
    },
    "data_engineering": {
        "spark",
        "kafka",
        "airflow",
        "hadoop",
        "hive",
        "presto",
        "trino",
        "dbt",
        "looker",
        "tableau",
        "power bi",
        "etl",
        "data pipeline",
        "data warehouse",
        "snowflake",
        "redshift",
        "bigquery",
        "databricks",
    },
    "backend": {
        "python",
        "java",
        "scala",
        "go",
        "rust",
        "c++",
        "node.js",
        "typescript",
        "django",
        "flask",
        "fastapi",
        "spring boot",
        "microservices",
        "rest api",
        "graphql",
        "grpc",
    },
    "databases": {
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "cassandra",
        "redis",
        "elasticsearch",
        "dynamodb",
        "cosmosdb",
        "couchbase",
    },
    "vector_search": {
        "faiss",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "annoy",
        "hnsw",
        "vector search",
        "embedding",
        "sentence-transformer",
        "dense retrieval",
        "sparse retrieval",
        "hybrid search",
        "semantic search",
    },
    "devops_cicd": {
        "jenkins",
        "github actions",
        "gitlab ci",
        "circleci",
        "travis ci",
        "argocd",
        "flux",
        "gitops",
        "helm",
        "kustomize",
    },
}

EDUCATION_PATTERNS = {
    "phd": {"keywords": ["phd", "doctorate", "doctor of philosophy"], "score": 1.0},
    "masters": {
        "keywords": [
            "master",
            "msc",
            "m.sc",
            "m.tech",
            "m.e",
            "post graduate",
            "postgraduate",
        ],
        "score": 0.7,
    },
    "bachelors": {
        "keywords": ["bachelor", "bsc", "b.sc", "b.tech", "b.e", "undergraduate"],
        "score": 0.4,
    },
    "diploma": {
        "keywords": ["diploma", "certificate program", "professional course"],
        "score": 0.2,
    },
}

EDUCATION_FIELDS = {
    "cs": {
        "keywords": [
            "computer science",
            "computer engineering",
            "software engineering",
            "information technology",
            "data science",
            "artificial intelligence",
            "machine learning",
            "computational",
        ],
        "score": 1.0,
    },
    "engineering": {
        "keywords": [
            "engineering",
            "electrical",
            "electronics",
            "mechanical",
            "civil",
            "aerospace",
            "mathematics",
            "physics",
            "statistics",
        ],
        "score": 0.7,
    },
    "business": {
        "keywords": [
            "business administration",
            "mba",
            "finance",
            "economics",
            "commerce",
            "management",
            "marketing",
        ],
        "score": 0.4,
    },
    "sciences": {
        "keywords": [
            "biology",
            "chemistry",
            "biotechnology",
            "life sciences",
            "environmental",
            "neuroscience",
            "cognitive science",
        ],
        "score": 0.3,
    },
}

CERTIFICATION_PATTERNS = [
    "aws certified",
    "gcp certified",
    "azure certified",
    "pmp",
    "certified scrum master",
    "csm",
    "cissp",
    "tensorflow certificate",
    "tensorflow developer certificate",
    "databricks certified",
    "snowflake certified",
    "kafka certified",
    "kubernetes certification",
    "cka",
    "ckad",
    "terraform certified",
    "hashicorp certified",
    "comptia",
    "aws solutions architect",
    "aws developer",
    "google professional",
    "oracle certified",
    "red hat certified",
    "cisco certified",
]

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

PREFERRED_LOCATIONS = {
    "india_tier1": {
        "cities": {
            "bangalore",
            "bengaluru",
            "hyderabad",
            "pune",
            "mumbai",
            "chennai",
            "delhi",
            "gurgaon",
            "gurugram",
            "noida",
        },
        "score": 1.0,
    },
    "india_tier2": {
        "cities": {
            "kolkata",
            "ahmedabad",
            "jaipur",
            "lucknow",
            "chandigarh",
            "indore",
            "bhopal",
            "coimbatore",
            "kochi",
            "nagpur",
        },
        "score": 0.80,
    },
    "global_hub": {
        "cities": {
            "san francisco",
            "new york",
            "london",
            "seattle",
            "berlin",
            "singapore",
            "toronto",
            "zurich",
            "amsterdam",
            "sydney",
        },
        "score": 0.75,
    },
    "global_other": {
        "score": 0.40,
    },
}

NON_TECH_TITLES = [
    "marketing",
    "hr",
    "sales",
    "accountant",
    "content",
    "writer",
    "designer",
    "support",
    "executive",
]

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
    "tfidf_semantic": 0.15,
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

        global REFERENCE_DATE, TFIDF_ENABLED, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
        global MODEL_CONFIG, ENSEMBLE_WEIGHTS, FAIRNESS_CONFIG, API_CONFIG
        global DIMENSION_WEIGHTS, SCORING, BEHAVIORAL_WEIGHTS

        if "reference_date" in overrides:
            from datetime import datetime as _dt

            REFERENCE_DATE = _dt.strptime(
                overrides["reference_date"], "%Y-%m-%d"
            ).date()
        if "tfidf" in overrides:
            t = overrides["tfidf"]
            if "enabled" in t:
                TFIDF_ENABLED = bool(t["enabled"])
            if "max_features" in t:
                TFIDF_MAX_FEATURES = int(t["max_features"])
            if "ngram_range" in t:
                TFIDF_NGRAM_RANGE = tuple(t["ngram_range"])
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


def _deep_update(base: dict, overrides: dict):
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


_load_yaml_overrides()
