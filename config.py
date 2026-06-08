import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATES_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "Downloads",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
    "candidates.jsonl",
)

OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.npz")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")

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
    "python": {
        "terms": ["python"],
        "weight": 1.5,
    },
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

BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.25,
    "interview_completion_rate": 0.15,
    "open_to_work_flag": 0.15,
    "saved_by_recruiters_30d": 0.10,
    "search_appearance_30d": 0.10,
    "github_activity_score": 0.10,
    "recent_activity": 0.10,
    "profile_completeness": 0.05,
}

DIMENSION_WEIGHTS = {
    "technical_match": 0.40,
    "semantic_match": 0.20,
    "career_quality": 0.10,
    "behavioral": 0.15,
    "retention": 0.10,
    "risk_adjustment": -0.05,
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
