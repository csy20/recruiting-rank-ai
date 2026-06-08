# Recruiting Rank AI

Intelligent candidate discovery and ranking system for the Redrob Hackathon. Uses multi-dimension scoring, ML ensemble models, skill graph transfer learning, TF-IDF semantic matching, and fairness auditing to rank candidates against job descriptions.

## Architecture

```
┌────────────┐   ┌──────────────┐   ┌───────────────┐
│ Candidates │──▶│  Extractor   │──▶│    Ranker     │
│  (JSONL)   │   │ (60+ feats)  │   │ (7-dim score) │
└────────────┘   └──────────────┘   └───────┬───────┘
                                            │
┌────────────┐   ┌──────────────┐          ▼
│ JD Text    │──▶│  JD Parser   │──▶┌───────────────┐
│  (.txt)    │   │ (sections,   │   │   Calibrate   │
└────────────┘   │  weights)    │   │ (min-max 0-100)│
                 └──────────────┘   └───────┬───────┘
                                            │
┌────────────┐   ┌──────────────┐          ▼
│ Skill Graph│──▶│  Explainer   │──▶┌───────────────┐
│(15 groups) │   │ (contribs,   │   │   Ranked      │
└────────────┘   │ strengths)   │   │   Output      │
                 └──────────────┘   └───────────────┘

┌────────────┐   ┌──────────────┐
│ ML Model   │──▶│  Fairness    │
│(RandomForest│   │   Audit     │
└────────────┘   └──────────────┘
```

## Scoring Dimensions

| Dimension | Default Weight | What it measures |
|-----------|---------------|------------------|
| Technical Match | 35% | Embeddings, vector DB, ranking, ML production, Python, LLMs, distributed systems |
| Semantic Match | 20% | Transferable skills, equivalent technologies, experience band |
| JD Semantic Similarity | 10% | TF-IDF cosine similarity between JD and candidate profile |
| Career Quality | 10% | Career progression, seniority trajectory, product vs consulting, company prestige |
| Behavioral | 15% | Recruiter response rate, interview completion, platform activity, GitHub signals |
| Retention | 10% | Tenure stability, notice period, job-hopping patterns |
| Risk Adjustment | -5% | Honeypot detection, keyword stuffing, timeline anomalies |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
python3 -m pytest tests/ -v

# Precompute features (recommended for large datasets)
python rank.py --precompute --candidates ./candidates.jsonl

# Rank candidates
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# With JD text and JSON output
python rank.py --candidates ./candidates.jsonl --jd ./job_description.txt --json

# Train ML model during precompute
python rank.py --precompute --candidates ./candidates.jsonl --train

# Start API server
python rank.py --serve --port 8000
```

## API

Start standalone server:
```bash
python serve.py
# or via rank.py
python rank.py --serve
```

### `/health` — Health check
```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"version":"2.0.0"}
```

### `/rank` — Rank candidates
```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [{"candidate_id": "1", "profile": {"headline": "ML Engineer"}}],
    "jd_text": "Looking for ML Engineer with embedding experience",
    "top_k": 10,
    "include_dimensions": true,
    "include_reasoning": true,
    "include_contributions": true
  }'
```

### `/score` — Score single candidate
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "candidate": {"profile": {"headline": "ML Engineer"}},
    "jd_text": "Looking for ML Engineer"
  }'
```

### `/audit` — Fairness audit
```bash
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"candidates": [...]}'
```

## CLI Usage

```
python rank.py [options]

Options:
  --candidates PATH    Path to candidates.jsonl
  --out PATH           Output CSV path (default: ./submission.csv)
  --precompute         Precompute features for faster ranking
  --features-dir DIR   Directory for precomputed features (default: data/)
  --jd PATH            Path to JD description text file
  --train              Train ML model during precompute
  --model PATH         Path to pre-trained ML model (.pkl)
  --json               Output JSON results + fairness audit
  --serve              Start FastAPI server
  --host HOST          API server host (default: 0.0.0.0)
  --port PORT          API server port (default: 8000)
```

## Configuration

Edit `config.yaml` to override any setting (no code changes needed):

```yaml
dimension_weights:
  technical_match: 0.35
  behavioral: 0.15

fairness:
  disparity_threshold: 0.15

model:
  n_estimators: 100
  max_depth: 10

api:
  max_candidates_per_request: 5000
```

Environment variables also supported via `RR_*` prefix (e.g. `RR_MODEL_PATH`, `RR_API_PORT`).

## Project Structure

```
├── rank.py              CLI entry point (precompute, rank, serve, train)
├── serve.py             Standalone FastAPI server
├── config.py            Python configuration (loads YAML overrides)
├── config.yaml          External YAML configuration
├── models.py            Pydantic data validation models
├── Makefile             Task runner (install, test, docker-build)
├── Dockerfile           Multi-stage Docker build
├── docker-compose.yml   Service orchestration
├── features/
│   └── extractor.py     60+ feature extractors with skill graph integration
├── scoring/
│   ├── ranker.py        7-dimension scoring + ML ensemble + calibration + audit
│   ├── jd_parser.py     Dynamic JD understanding (section splitting, term extraction)
│   ├── skill_graph.py   Knowledge graph with 15 groups, 150+ skills, synonym mapping
│   └── explainer.py     Feature contribution analysis, strengths/weaknesses
├── utils/
│   └── nlp_utils.py     Word-boundary regex matching, TF-IDF helpers
├── tests/
│   ├── test_extractor.py   34 tests
│   ├── test_nlp_utils.py   18 tests
│   └── test_ranker.py      23 tests (incl. JD parser)
├── data/                Precomputed features and model cache
└── .github/workflows/   CI pipeline
```

## Docker

```bash
# Build
make docker-build
# or
docker build -t recruiting-rank-ai .

# Run
make docker-run
# or
docker run -p 8000:8000 recruiting-rank-ai

# Compose
make docker-compose-up
# or
docker-compose up --build
```

## Testing

```bash
# All tests
python3 -m pytest tests/ -v

# Individual test files
python3 -m pytest tests/test_extractor.py -v
python3 -m pytest tests/test_ranker.py -v
python3 -m pytest tests/test_nlp_utils.py -v
```

## Fairness Auditing

The system audits three protected group axes:
- **Consulting vs non-consulting** background
- **High vs low company prestige** score
- **High vs low location** score

For each axis, it measures mean score disparity across 5 dimensions. Disparities exceeding the configurable threshold (default 0.15) are flagged. Results are written to a `_fairness_audit.json` file.

## ML Model

A RandomForest regressor learns from rule-based scoring to provide ML-enhanced predictions. Ensemble weights: 60% rule-based, 25% ML prediction, 15% TF-IDF semantic.

```bash
# Train during precompute
python rank.py --precompute --candidates ./candidates.jsonl --train

# Use pre-trained model
python rank.py --candidates ./candidates.jsonl --model ./data/ranker_model.pkl
```
