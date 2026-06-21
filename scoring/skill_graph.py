SKILL_GROUPS: dict[str, set[str]] = {
    "deep_learning_frameworks": {
        "pytorch",
        "tensorflow",
        "keras",
        "jax",
        "mxnet",
        "caffe",
        "chainer",
        "theano",
        "torch",
    },
    "ml_frameworks": {
        "scikit-learn",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "prophet",
    },
    "nlp_tools": {
        "transformers",
        "hugging face",
        "huggingface",
        "spacy",
        "nltk",
        "stanford nlp",
        "coreference",
        "tokenizer",
        "bert",
        "gpt",
        "llm",
    },
    "cloud_providers": {
        "aws",
        "gcp",
        "azure",
        "oracle cloud",
        "ibm cloud",
        "digitalocean",
        "linode",
    },
    "container_orchestration": {
        "docker",
        "kubernetes",
        "k8s",
        "openshift",
        "nomad",
        "docker compose",
        "docker-compose",
    },
    "infrastructure_as_code": {
        "terraform",
        "pulumi",
        "cloudformation",
        "ansible",
        "chef",
        "puppet",
        "saltstack",
    },
    "ci_cd": {
        "jenkins",
        "github actions",
        "gitlab ci",
        "circleci",
        "travis ci",
        "teamcity",
        "bamboo",
        "argo",
    },
    "stream_processing": {
        "kafka",
        "flink",
        "spark streaming",
        "storm",
        "pulsar",
        "rabbitmq",
        "pubsub",
        "kinesis",
    },
    "data_warehouses": {
        "snowflake",
        "redshift",
        "bigquery",
        "clickhouse",
        "druid",
        "pinot",
    },
    "vector_databases": {
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "chroma",
        "vespa",
        "vald",
    },
    "sql_databases": {
        "postgresql",
        "postgres",
        "mysql",
        "sqlite",
        "mariadb",
        "oracle db",
        "sql server",
        "cockroachdb",
    },
    "nosql_databases": {
        "mongodb",
        "cassandra",
        "dynamodb",
        "redis",
        "couchbase",
        "couchdb",
        "neo4j",
        "arangodb",
    },
    "python_ecosystem": {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "dash",
        "flask",
        "fastapi",
        "django",
        "celery",
    },
    "java_ecosystem": {
        "java",
        "scala",
        "kotlin",
        "spring",
        "spring boot",
        "hadoop",
        "spark",
        "hive",
    },
    "monitoring_observability": {
        "prometheus",
        "grafana",
        "datadog",
        "new relic",
        "sentry",
        "elk",
        "elastic stack",
        "jaeger",
    },
}

SKILL_SYNONYMS: dict[str, list[str]] = {
    "python": ["python3", "cpython"],
    "tensorflow": ["tf", "tf2", "tensorflow 2"],
    "pytorch": ["torch", "py-torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "aws": ["amazon web services", "aws cloud"],
    "gcp": ["google cloud", "google cloud platform"],
    "azure": ["microsoft azure", "ms azure"],
    "kubernetes": ["k8s", "kube"],
    "elasticsearch": ["es", "elastic search"],
    "postgresql": ["postgres", "psql"],
    "javascript": ["js", "ecmascript", "node.js", "nodejs"],
}


def build_skill_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}

    for _group_name, skills in SKILL_GROUPS.items():
        skills_list = list(skills)
        for i, skill in enumerate(skills_list):
            if skill not in graph:
                graph[skill] = set()
            for j, other in enumerate(skills_list):
                if i != j:
                    graph[skill].add(other)

    for canonical, synonyms in SKILL_SYNONYMS.items():
        if canonical not in graph:
            graph[canonical] = set()
        for syn in synonyms:
            graph[syn] = {canonical}
            graph[canonical].add(syn)

    return graph


SKILL_GRAPH = build_skill_graph()


def get_related_skills(skill_name: str, max_distance: int = 1) -> dict[str, float]:
    skill_lower = skill_name.lower().strip()
    if skill_lower in SKILL_GRAPH:
        direct = SKILL_GRAPH[skill_lower]
        result: dict[str, float] = {s: 1.0 for s in direct}
        if max_distance >= 2:
            for related in direct:
                if related in SKILL_GRAPH:
                    for second_hop in SKILL_GRAPH[related]:
                        if second_hop != skill_lower and second_hop not in result:
                            result[second_hop] = 0.5
        return result
    return {}


def compute_skill_match(
    candidate_skills: list[str],
    required_skills: list[str],
) -> tuple[float, list[str], list[str]]:
    if not required_skills:
        return 1.0, [], []

    candidate_lower = set(s.lower().strip() for s in candidate_skills if s)
    required_lower = [s.lower().strip() for s in required_skills if s]

    if not required_lower:
        return 1.0, [], []

    exact_matches: list[str] = []
    transferable_matches: list[str] = []

    for req in required_lower:
        if req in candidate_lower:
            exact_matches.append(req)
            continue

        for cand_skill in candidate_lower:
            related = get_related_skills(req, max_distance=1)
            if cand_skill in related:
                transferable_matches.append(f"{req}~{cand_skill}")
                break

    exact_score = len(exact_matches) / len(required_lower)
    transfer_score = len(transferable_matches) / len(required_lower) * 0.6
    total_score = min(exact_score + transfer_score, 1.0)

    return total_score, exact_matches, transferable_matches


def get_skill_category(skill_name: str) -> str | None:
    skill_lower = skill_name.lower().strip()
    for category, skills in SKILL_GROUPS.items():
        if skill_lower in skills:
            return category
    for canonical, synonyms in SKILL_SYNONYMS.items():
        all_forms = [canonical] + [s.lower().strip() for s in synonyms]
        if skill_lower in all_forms:
            for cat, skills in SKILL_GROUPS.items():
                if canonical in skills:
                    return cat
        for syn in synonyms:
            if skill_lower == syn.lower().strip():
                for cat, skills in SKILL_GROUPS.items():
                    if canonical in skills:
                        return cat
    return None


CONCEPT_GRAPH: dict[str, set[str]] = {
    "rag_pipeline": {
        "langchain",
        "llamaindex",
        "haystack",
        "rag",
        "retrieval augmented generation",
        "augmented generation",
        "semantic kernel",
        "vector store",
        "document retrieval",
        "context retrieval",
        "knowledge base",
        "corpus",
        "indexing pipeline",
        "document chunking",
        "text splitting",
        "embedding pipeline",
        "retrieval qa",
        "qa system",
        "question answering",
        "document search",
        "internal search",
        "enterprise search",
        "knowledge retrieval",
        "context augmentation",
        "knowledge enhanced",
        "retrieval system",
    },
    "vector_database": {
        "pinecone",
        "weaviate",
        "milvus",
        "qdrant",
        "faiss",
        "chroma",
        "vespa",
        "vald",
        "annoy",
        "hnswlib",
        "pgvector",
        "vector search",
        "similarity search",
        "nearest neighbor",
        "approximate nearest neighbor",
        "ann",
        "vector index",
        "embedding index",
        "vector store",
        "hybrid search",
        "dense retrieval",
        "vector database",
        "vector db",
        "vector similarity",
        "semantic search",
        "embedding search",
        "neural search",
    },
    "lexical_retrieval": {
        "bm25",
        "elasticsearch",
        "opensearch",
        "solr",
        "lucene",
        "tf-idf",
        "tfidf",
        "inverted index",
        "sparse retrieval",
        "keyword search",
        "full-text search",
        "text search",
        "indexing",
        "search engine",
        "document index",
        "term frequency",
        "inverse document frequency",
        "okapi",
        "bm25f",
        "elastic search",
    },
    "semantic_search": {
        "dense retrieval",
        "dpr",
        "colbert",
        "ann",
        "hnsw",
        "semantic search",
        "embedding search",
        "neural search",
        "vector search",
        "similarity search",
        "retrieval system",
        "document search",
        "internal docs search",
        "knowledge base search",
        "enterprise search",
        "hybrid search",
        "neural retrieval",
        "dense passage",
        "late interaction",
        "bi-encoder",
        "cross-encoder",
    },
    "ranking_ir": {
        "learning to rank",
        "ltr",
        "ndcg",
        "mrr",
        "map",
        "ranking",
        "ranker",
        "relevance",
        "search relevance",
        "result ranking",
        "candidate retrieval",
        "two-stage retrieval",
        "retrieval ranking",
        "reranking",
        "cross-encoder",
        "pointwise",
        "pairwise",
        "listwise",
        "ranking model",
        "relevance model",
        "retrieval metric",
        "ranking metric",
        "information retrieval",
    },
    "embedding_models": {
        "sentence-transformer",
        "sentence transformer",
        "bge",
        "e5",
        "openai embedding",
        "text-embedding",
        "ada",
        "instructor",
        "bert embedding",
        "word2vec",
        "glove",
        "fasttext",
        "embedding model",
        "dense embedding",
        "semantic embedding",
        "text embedding",
        "embedding api",
        "embedding service",
        "embedding as a service",
    },
    "llm_frameworks": {
        "llm",
        "large language model",
        "gpt",
        "gpt-4",
        "gpt-3.5",
        "gpt-4o",
        "gpt-4o-mini",
        "claude",
        "llama",
        "llama2",
        "llama3",
        "mistral",
        "mixtral",
        "gemini",
        "gemma",
        "phi",
        "falcon",
        "mpt",
        "dolly",
        "openai",
        "anthropic",
        "cohere",
        "ai21",
        "huggingface",
        "hugging face",
        "transformer model",
        "pretrained model",
        "foundation model",
    },
    "fine_tuning": {
        "fine-tune",
        "fine tuning",
        "lora",
        "qlora",
        "peft",
        "adapter",
        "prompt tuning",
        "soft prompt",
        "prefix tuning",
        "instruction tuning",
        "sft",
        "rlhf",
        "dpo",
        "model alignment",
        "supervised fine-tuning",
        "parameter efficient",
        "model adaptation",
        "transfer learning",
    },
    "evaluation_metrics": {
        "ndcg",
        "mrr",
        "map",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "auc",
        "roc",
        "evaluation",
        "benchmark",
        "a/b test",
        "ab test",
        "offline eval",
        "online eval",
        "relevance",
        "ranking metric",
        "cross-validation",
        "holdout",
        "test set",
        "validation",
        "ground truth",
        "judgment list",
        "relevance judgment",
        "human eval",
        "evaluation framework",
        "metric",
    },
    "production_ml": {
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
        "mlops",
        "model registry",
        "model versioning",
        "a/b test",
        "ab test",
        "canary",
        "rollout",
        "gradual rollout",
        "online",
        "low latency",
        "high throughput",
        "real-time",
        "model monitoring",
        "model serving",
        "prediction service",
        "inference pipeline",
        "training pipeline",
        "data pipeline",
        "ci/cd for ml",
    },
    "nlp_tasks": {
        "nlp",
        "natural language",
        "text classification",
        "ner",
        "named entity",
        "text mining",
        "semantic",
        "transformers",
        "tokenization",
        "parsing",
        "dependency parsing",
        "coreference",
        "summarization",
        "text generation",
        "sentiment",
        "intent detection",
        "slot filling",
        "language model",
        "sequence labeling",
        "text understanding",
        "natural language understanding",
        "nlu",
        "natural language generation",
        "nlg",
        "machine translation",
        "sequence to sequence",
    },
    "data_pipeline": {
        "etl",
        "data pipeline",
        "data warehouse",
        "airflow",
        "spark",
        "kafka",
        "flink",
        "beam",
        "data engineering",
        "data infrastructure",
        "data lake",
        "data catalog",
        "data quality",
        "data governance",
        "data processing",
        "stream processing",
        "batch processing",
        "data ingestion",
        "data integration",
        "data transformation",
    },
    "cloud_infra": {
        "aws",
        "gcp",
        "azure",
        "cloud",
        "kubernetes",
        "k8s",
        "docker",
        "terraform",
        "cloudformation",
        "ecs",
        "eks",
        "gke",
        "aks",
        "lambda",
        "serverless",
        "sagemaker",
        "vertex ai",
        "mlflow",
        "ec2",
        "s3",
        "gcs",
        "cloud run",
        "cloud function",
        "container",
        "orchestration",
    },
    "information_extraction": {
        "information extraction",
        "ie",
        "relation extraction",
        "entity extraction",
        "entity linking",
        "knowledge graph",
        "triple extraction",
        "ontology",
        "schema matching",
        "data extraction",
        "web scraping",
        "crawling",
        "document parsing",
        "pdf parsing",
        "html parsing",
        "structure extraction",
        "information retrieval",
    },
    "recommendation_systems": {
        "recommendation",
        "recommender",
        "collaborative filtering",
        "content-based",
        "matrix factorization",
        "fm",
        "ffm",
        "wide and deep",
        "deepfm",
        "two-tower",
        "candidate generation",
        "personalization",
        "discovery feed",
        "feed ranking",
        "click-through",
        "ctr prediction",
        "user embedding",
        "item embedding",
        "recommendation system",
        "recsys",
        "retrieval ranking",
    },
    "search_systems": {
        "search",
        "search engine",
        "site search",
        "web search",
        "query understanding",
        "query expansion",
        "query rewriting",
        "spell correction",
        "suggestion",
        "autocomplete",
        "typeahead",
        "faceted search",
        "filtering",
        "search ranking",
        "search relevance",
        "search quality",
        "query intent",
        "query classification",
        "semantic search",
    },
    "ml_platform": {
        "ml platform",
        "ml infrastructure",
        "feature platform",
        "model platform",
        "ml pipeline",
        "ml serving",
        "model training",
        "model evaluation",
        "experiment tracking",
        "hyperparameter tuning",
        "model optimization",
        "distributed training",
        "model parallelism",
        "data parallelism",
        "gpu training",
        "tpu training",
        "pytorch",
        "tensorflow",
        "jax",
        "keras",
    },
    "agent_systems": {
        "agent",
        "autonomous agent",
        "tool use",
        "function calling",
        "reasoning",
        "chain of thought",
        "planning",
        "react",
        "reflexion",
        "toolformer",
        "agent framework",
        "multi-agent",
        "orchestration",
        "workflow",
        "task decomposition",
        "tool retrieval",
    },
    "evaluation_quality": {
        "eval set",
        "golden set",
        "test collection",
        "human annotation",
        "judgment",
        "relevance label",
        "rating",
        "grading",
        "quality assurance",
        "regression testing",
        "evaluation pipeline",
        "automated evaluation",
        "llm as judge",
    },
}


CONCEPT_MAP: dict[str, str] = {}
for _concept, _terms in CONCEPT_GRAPH.items():
    for _term in _terms:
        CONCEPT_MAP[_term] = _concept


def compute_concept_boost(jd_text: str, candidate_text: str) -> float:
    jd_lower = jd_text.lower()
    cand_lower = candidate_text.lower()

    jd_concepts: set[str] = set()
    cand_concepts: set[str] = set()

    for term, concept in CONCEPT_MAP.items():
        if term in jd_lower:
            jd_concepts.add(concept)
        if term in cand_lower:
            cand_concepts.add(concept)

    if not jd_concepts:
        return 0.0

    overlap = len(jd_concepts & cand_concepts)
    return min(overlap * 0.015, 0.15)


def compute_skill_breadth(skills: list[str]) -> dict[str, float]:
    categories: dict[str, set[str]] = {}
    uncovered: int = 0

    for skill in skills:
        skill_lower = skill.lower().strip()
        cat = get_skill_category(skill_lower)
        if cat:
            if cat not in categories:
                categories[cat] = set()
            categories[cat].add(skill_lower)
        else:
            uncovered += 1

    result: dict[str, float] = {}
    for cat, cat_skills in categories.items():
        result[f"cat_{cat}"] = float(len(cat_skills))
    result["uncovered_skills"] = float(uncovered)
    result["category_count"] = float(len(categories))
    return result
