# Recruiting Rank AI

Intelligent candidate discovery and multi-dimensional ranking system. Built for the Redrob Hackathon, it evaluates candidates across 7 scoring dimensions, uses a skill knowledge graph for transferable skill detection, applies ML ensemble predictions, and includes built-in fairness auditing.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Scoring Methodology](#scoring-methodology)
- [Skill Knowledge Graph](#skill-knowledge-graph)
- [ML Ensemble Model](#ml-ensemble-model)
- [Fairness Auditing](#fairness-auditing)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Development](#development)
- [Testing](#testing)

---

## Problem Statement

Recruiters face three core challenges when screening technical candidates at scale:

1. **Surface-level keyword matching** — Simple string matching misses transferable skills. A candidate with PyTorch experience is likely proficient in TensorFlow, but naive matchers miss this connection.
2. **Cold start for new JDs** — Each job description requires re-tuning scoring weights. A system must dynamically adapt to arbitrary JD text without manual configuration.
3. **Fairness blind spots** — Ranking systems can inadvertently encode bias against consulting backgrounds, non-premier companies, or non-hub locations. These disparities must be measured and surfaced.

This system addresses all three through a combination of skill knowledge graphs, dynamic JD parsing, multi-dimension scoring with configurable weights, and automated fairness auditing.

---

## System Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Candidates   │───▶│  Feature Engine  │───▶│  7-Dimension    │
│  (JSON/JSONL) │    │  (extractor.py)  │    │  Scorer         │
└──────────────┘    │  60+ features    │    │  (ranker.py)    │
                    └──────────────────┘    └────────┬────────┘
┌──────────────┐    ┌──────────────────┐             │
│   JD Text    │───▶│   JD Parser      │             │
│  (raw text)  │    │  (jd_parser.py)  │             ▼
└──────────────┘    │  sections,       │    ┌─────────────────┐
                    │  weights, terms  │    │   Calibrator    │
                    └──────────────────┘    │  0-100 scale    │
                                            └────────┬────────┘
┌──────────────┐    ┌──────────────────┐             │
│ Skill Graph  │───▶│   Explainer      │             ▼
│ 150+ skills  │    │  (explainer.py)  │    ┌─────────────────┐
│ 15 groups    │    │  contributions,  │    │  Ranked Output  │
└──────────────┘    │  strengths/wk    │    │  CSV / JSON     │
                    └──────────────────┘    └─────────────────┘

┌──────────────┐    ┌──────────────────┐
│ ML Model     │───▶│  Fairness Audit  │
│ RandomForest │    │  3 group axes    │
└──────────────┘    │  disparity check │
                    └──────────────────┘
```

### Data Flow

1. **Ingestion** — Candidates loaded from JSON/JSONL. Each record contains profile, career history, skills, and behavioral signals.
2. **JD Parsing** — Raw JD text is split into sections (About, Requirements, Preferred), terms are extracted and categorized, dimension weights are computed dynamically.
3. **Feature Extraction** — 60+ features computed per candidate: technical keyword matches, AI/retrieval depth, career progression metrics, behavioral scores, education/certification detection, location scoring, honeypot/risk detection.
4. **TF-IDF Semantic Matching** — If JD provided, cosine similarity between JD and each candidate's combined profile text via TF-IDF vectorization.
5. **7-Dimension Scoring** — Each candidate scored across technical_match, semantic_match, career_quality, behavioral, retention, risk_adjustment, and jd_semantic_similarity.
6. **Calibration** — Raw scores min-max scaled to 0-100 across the candidate pool.
7. **Explainability** — Feature contribution analysis with strengths/weaknesses for each candidate.
8. **Fairness Audit** — Disparity measurement across consulting, prestige, and location axes.

---

## Scoring Methodology

### The 7 Dimensions

| Dimension | Weight | What it measures | Key signals used |
|-----------|--------|------------------|------------------|
| **Technical Match** | 35% | Keyword coverage against JD terms; depth in AI, retrieval, evaluation; keyword diversity | `jd_match_*` scores, `ai_depth`, `retrieval_depth`, `eval_depth`, `keyword_diversity` |
| **Semantic Match** | 20% | Transferable skills via skill graph; experience band fit; skill breadth; consulting penalty/credit | `compute_skill_match()`, `compute_skill_breadth()`, experience_band score, industry relevance |
| **Career Quality** | 10% | Career progression trajectory, seniority, product vs consulting, company prestige, tenure stability | `career_progression`, `career_seniority`, `has_product_exp`, `company_prestige`, `growth_rate` |
| **Behavioral** | 15% | Recruiter engagement, platform activity, profile completeness, verification signals | 12 weighted signals from `redrob_signals` (response rate, github activity, saved count, etc.) |
| **Retention** | 10% | Average tenure, notice period, job-hopping patterns, stability | `avg_tenure_months`, `notice_period_days`, `short_stints` count |
| **Risk Adjustment** | Multiplicative | Honeypot detection, keyword stuffing, anti-patterns, inflated expertise claims | `risk_score`, `is_honeypot`, `anti_pattern_count` — applied as final penalty multiplier |
| **JD Semantic Similarity** | 10% | TF-IDF cosine similarity between JD text and candidate combined profile | `tfidf_jd_similarity` |

### How the Final Score is Computed

```python
# 1. Each dimension is scored 0.0–1.0
dim_scores = compute_dimension_scores(features, jd_weights)

# 2. Weighted sum of dimensions (risk adjustment weight is 0.0)
score = sum(dim_scores[dim] * DIMENSION_WEIGHTS[dim] for dim in dimensions)

# 3. Multiplicative risk penalty applied once
score = score * dim_scores["risk_adjustment"]

# 4. Optional ML ensemble blend
if ml_score:
    ensemble = 0.60 * score + 0.25 * ml_score + 0.15 * tfidf_similarity

# 5. Min-max calibration to 0–100 across candidate pool
```

---

## Skill Knowledge Graph

The skill graph (`scoring/skill_graph.py`) is a manually curated knowledge base of 150+ skills organized into 15 groups with synonym mappings.

### Skill Groups

| Group | Example skills |
|-------|---------------|
| Programming Languages | Python, Java, Scala, Go, Rust, C++ |
| ML/DL Frameworks | PyTorch, TensorFlow, Keras, JAX, scikit-learn |
| NLP & LLMs | Hugging Face, Transformers, spaCy, LangChain |
| Vector Search & Embeddings | FAISS, Pinecone, Weaviate, Milvus, Qdrant |
| Data Engineering | Spark, Kafka, Airflow, Hadoop, Hive |
| Cloud Platforms | AWS, GCP, Azure |
| Container & Orchestration | Docker, Kubernetes, Helm, Istio |
| Databases | PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch |
| MLOps & Production | MLflow, Kubeflow, TensorRT, ONNX |
| CI/CD & DevOps | Jenkins, GitHub Actions, GitLab CI, ArgoCD |
| Monitoring & Observability | Prometheus, Grafana, Datadog |
| Testing & Quality | pytest, unittest, Selenium, Jest |
| Web Frameworks | FastAPI, Flask, Django, Spring Boot |
| Big Data & Streaming | Flink, Beam, Storm, Pulsar |
| Infrastructure as Code | Terraform, CloudFormation, Pulumi, Ansible |

### Transferable Skill Detection

When a candidate lists "PyTorch" but not "TensorFlow", the graph identifies them as related (distance = 1) and gives partial credit. This surfaces candidates who may lack exact keywords but possess equivalent expertise.

```python
# Example: PyTorch experience gives partial credit for TensorFlow
get_related_skills("pytorch")
# Returns: {"tensorflow": 1.0, "keras": 0.5, "jax": 0.5, ...}
```

---

## ML Ensemble Model

### Training

A `RandomForestRegressor` learns from rule-based scoring output:

```
Rule-based scores → training targets
Candidate features → training features (22-dimensional vector)
```

### Prediction

Final score is an ensemble blend:

| Component | Weight | Source |
|-----------|--------|--------|
| Rule-based score | 60% | `compute_final_score()` |
| ML prediction | 25% | RandomForest regressor |
| TF-IDF similarity | 15% | Cosine similarity between JD and profile |

### When to use

```bash
# Train during precompute
python rank.py --precompute --candidates ./candidates.jsonl --train

# Use pre-trained model
python rank.py --candidates ./candidates.jsonl --model ./data/ranker_model.pkl
```

---

## Fairness Auditing

The system audits three protected group axes:

| Axis | Group A | Group B | Rationale |
|------|---------|---------|-----------|
| **Consulting** | Candidates from consulting firms (TCS, Infosys, Wipro, etc.) | Non-consulting background | Consulting candidates often have broader but shallower experience |
| **Company Prestige** | Tier 1–2 companies (Google, Meta, Microsoft, etc.) | Tier 3–4 companies | Prestige bias is a known issue in automated screening |
| **Location** | Indian tech hubs (Bangalore, Hyderabad, Pune, etc.) | Global/other locations | Geographic bias can disadvantage remote or non-hub candidates |

For each axis, it measures mean score disparity across 5 dimensions (technical_match, semantic_match, career_quality, behavioral, retention). Disparities exceeding the configurable threshold (default 0.15) are flagged.

```bash
# Run audit via API
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"candidates": [...]}'
```

---

## API Reference

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "model_loaded": true, "version": "2.0.0"}
```

### `POST /rank` — Rank candidates against JD

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [
      {
        "candidate_id": "CAND_0001",
        "profile": {"headline": "Senior ML Engineer", "summary": "..."},
        "career_history": [...],
        "skills": [{"name": "Python", "proficiency": "expert"}],
        "redrob_signals": {...}
      }
    ],
    "jd_text": "Looking for ML Engineer with embedding experience",
    "top_k": 10,
    "include_dimensions": true,
    "include_reasoning": true,
    "include_contributions": false
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `candidates` | `List[Dict]` | (required) | Array of candidate objects |
| `jd_text` | `string` | `null` | Job description text (max 50,000 chars) |
| `top_k` | `int` | `100` | Number of top results to return |
| `include_dimensions` | `bool` | `true` | Include per-dimension scores |
| `include_reasoning` | `bool` | `true` | Include natural language reasoning |
| `include_contributions` | `bool` | `false` | Include feature-level contribution breakdown |

### `POST /score` — Score a single candidate

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"profile": {"headline": "ML Engineer"}, "jd_text": "Looking for ML Engineer"}'
```

### `POST /audit` — Run fairness audit

```bash
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '[{...candidate...}, {...candidate...}]'
```

### Rate Limiting

All endpoints enforce a configurable rate limit (default: 60 requests/minute). When exceeded, returns HTTP 429.

---

## CLI Usage

```bash
python rank.py [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--candidates PATH` | Path to candidates file (JSON/JSONL) | `data/candidates.jsonl` |
| `--out PATH` | Output CSV path | `submission.csv` |
| `--precompute` | Precompute features for faster ranking | `false` |
| `--features-dir DIR` | Directory for precomputed features | `data/` |
| `--jd PATH` | Path to JD description text file | (none) |
| `--train` | Train ML model during precompute | `false` |
| `--model PATH` | Path to pre-trained ML model (.pkl) | (none) |
| `--json` | Output JSON results + fairness audit | `false` |
| `--serve` | Start FastAPI server | `false` |
| `--host HOST` | API server host | `0.0.0.0` |
| `--port PORT` | API server port | `8000` |
| `--limit N` | Max candidates to process | (all) |

---

## Configuration

Settings are loaded from Python (`config/__init__.py`) with optional overrides from `config.yaml` and environment variables.

### YAML Overrides (`config.yaml`)

```yaml
dimension_weights:
  technical_match: 0.40
  behavioral: 0.10

scoring:
  technical_match:
    core_tech_weight: 0.55
    depth_weight: 0.30

fairness:
  disparity_threshold: 0.10

model:
  n_estimators: 200
  max_depth: 15

api:
  max_candidates_per_request: 5000
  rate_limit_per_minute: 60
```

### Environment Variables

| Variable | Overrides | Default |
|----------|-----------|---------|
| `RR_CANDIDATES_PATH` | Candidate data file path | `data/candidates.jsonl` |
| `RR_OUTPUT_PATH` | Output CSV path | `submission.csv` |
| `RR_MODEL_PATH` | Pre-trained model path | `data/ranker_model.pkl` |
| `RR_API_HOST` | API server host | `0.0.0.0` |
| `RR_API_PORT` | API server port | `8000` |
| `RR_CONFIG_PATH` | YAML config file path | `config.yaml` |

---

## Project Structure

```
├── rank.py                 CLI entry point (precompute, rank, serve, train)
├── serve.py                Standalone FastAPI server with rate limiting
├── config/                 Python configuration package
│   ├── __init__.py         Dynamic settings, weights, YAML override loader
│   └── data.py             Static data (company tiers, skill categories, patterns)
├── config.yaml             External YAML configuration overrides
├── models.py               Pydantic data validation models
├── pyproject.toml          Project metadata, build config, ruff/pytest settings
├── .pre-commit-config.yaml Pre-commit hooks (ruff, trailing whitespace, etc.)
├── Makefile                Task runner (install, test, lint, docker-build)
├── Dockerfile              Multi-stage Docker build
├── docker-compose.yml      Service orchestration
│
├── features/
│   └── extractor.py        60+ feature extractors with skill graph integration
│
├── scoring/
│   ├── ranker.py           7-dimension scoring + ML ensemble + calibration + audit
│   ├── jd_parser.py        Dynamic JD understanding (section splitting, term extraction)
│   ├── skill_graph.py      Knowledge graph with 15 groups, 150+ skills, synonyms
│   └── explainer.py        Feature contribution analysis, strengths/weaknesses
│
├── utils/
│   └── nlp_utils.py        Word-boundary regex matching, TF-IDF helpers
│
├── tests/
│   ├── test_extractor.py       34 tests — feature extraction pipeline
│   ├── test_nlp_utils.py       18 tests — NLP utility functions
│   ├── test_ranker.py          23 tests — scoring, ranking, calibration, JD parser
│   ├── test_skill_graph.py     12 tests — graph building, skill matching
│   ├── test_explainer.py        6 tests — feature contributions, reasoning
│   ├── test_serve.py           12 tests — API endpoints, validation
│   ├── test_models.py          20 tests — Pydantic model validation
│   ├── test_integration.py      4 tests — full pipeline integration
│   ├── test_oracle_ranking.py   9 tests — ranking invariants & oracle tests
│   ├── test_rank_embeddings.py  4 tests — embedding loading validation
│   └── test_rank_encoding.py    1 test  — encoding chunking logic
│
├── data/                   Precomputed features and model cache
├── .github/workflows/      CI pipeline (ruff lint + pytest + docker build)
└── requirements.txt        Python dependencies
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install pyyaml pytest ruff

# 2. Run tests to verify setup
python3 -m pytest tests/ -v

# 3. Rank candidates from a file
python rank.py --candidates ./data/candidates.jsonl

# 4. With a job description
python rank.py --candidates ./data/candidates.jsonl --jd ./job_description.txt

# 5. Precompute features (faster subsequent runs)
python rank.py --precompute --candidates ./data/candidates.jsonl

# 6. JSON output with fairness audit
python rank.py --candidates ./data/candidates.jsonl --jd ./job_description.txt --json

# 7. Train and use ML model
python rank.py --precompute --candidates ./data/candidates.jsonl --train
python rank.py --candidates ./data/candidates.jsonl --model ./data/ranker_model.pkl

# 8. Start API server
python rank.py --serve --port 8000
# or standalone
python serve.py
```

---

## Docker

```bash
# Build
make docker-build
# or
docker build -t recruiting-rank-ai .

# Run API server
make docker-run
# or
docker run -p 8000:8000 recruiting-rank-ai

# With docker-compose
make docker-compose-up
```

---

## Development

### Setup

```bash
# Install dev dependencies
make install

# Set up pre-commit hooks
pre-commit install

# Format code
make format

# Lint + auto-fix
make lint-fix
```

### Linting

The project uses [ruff](https://docs.astral.sh/ruff/) with rulesets for:
- **E/W**: pycodestyle errors and warnings
- **F**: pyflakes (logic errors, unused imports)
- **I**: isort (import ordering)
- **N**: pep8-naming (variable/function naming conventions)
- **UP**: pyupgrade (modern Python syntax)
- **B**: flake8-bugbear (common bugs)

```bash
make lint      # check only
make lint-fix  # auto-fix
make format    # format code
```

---

## Testing

### Test Suite

13 test files with 140+ tests covering all components:

| File | Tests | Coverage |
|------|-------|----------|
| `test_extractor.py` | 34 | Feature extraction (all 30+ extractors, edge cases) |
| `test_ranker.py` | 23 | Scoring, ranking, calibration, JD parser, audit |
| `test_nlp_utils.py` | 18 | NLP utilities (pattern matching, phrase extraction) |
| `test_skill_graph.py` | 12 | Graph building, BFS, skill matching, concept boost |
| `test_serve.py` | 12 | API endpoints (health, rank, score, audit) |
| `test_models.py` | 20 | Pydantic model validation, field parsing, defaults |
| `test_oracle_ranking.py` | 9 | Ranking invariants (monotonicity, determinism, caps) |
| `test_explainer.py` | 6 | Feature contributions, reasoning generation |
| `test_integration.py` | 4 | Full pipeline: JSONL → ranked CSV |
| `test_rank_embeddings.py` | 4 | Embedding loading, validation, error handling |
| `test_rank_encoding.py` | 1 | Encoding chunking for long texts |

```bash
# Full test suite
python3 -m pytest tests/ -v

# With coverage report
python3 -m pytest tests/ --cov=. --cov-report=term-missing

# Individual modules
python3 -m pytest tests/test_extractor.py -v
python3 -m pytest tests/test_ranker.py -v
python3 -m pytest tests/test_skill_graph.py -v
python3 -m pytest tests/test_explainer.py -v
python3 -m pytest tests/test_serve.py -v
python3 -m pytest tests/test_nlp_utils.py -v
python3 -m pytest tests/test_models.py -v
python3 -m pytest tests/test_integration.py -v
python3 -m pytest tests/test_oracle_ranking.py -v
python3 -m pytest tests/test_rank_embeddings.py -v
```
