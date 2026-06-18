
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
