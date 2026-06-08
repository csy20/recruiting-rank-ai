# Recruiting Rank AI

Intelligent Candidate Discovery & Ranking System for the Redrob Hackathon.

## Approach

6-dimension evaluation framework based on the JD requirements:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Technical Match | 40% | Embeddings, vector DB, ranking, evaluation, production ML |
| Semantic Match | 20% | Transferable skills, equivalent technologies, experience band |
| Career Quality | 10% | Progression, product vs consulting, seniority trajectory |
| Behavioral | 15% | Recruiter response rate, interview completion, platform activity |
| Retention | 10% | Tenure stability, notice period, job-hopping patterns |
| Risk Adjustment | -5% | Honeypot detection, keyword stuffing, timeline anomalies |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Precompute features** (recommended for faster ranking):
```bash
python rank.py --precompute --candidates ./candidates.jsonl --features-dir ./data
```

**Rank candidates and produce submission:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --features-dir ./data
```

**Single-step (precompute + rank):**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## File Structure

```
├── rank.py              # Entry point
├── config.py             # Weights, keywords, anti-patterns
├── features/
│   └── extractor.py      # Feature extraction (50+ features per candidate)
├── scoring/
│   └── ranker.py         # 6-dimension scoring + final ranking
├── utils/
│   └── nlp_utils.py      # Tokenization, keyword matching utilities
├── data/                 # Precomputed features cache
└── requirements.txt
```

## Design Decisions

- **No external API calls**: Runs fully offline on CPU
- **No GPU needed**: Lightweight rule-based + keyword scoring
- **Honeypot detection**: Checks for impossible timelines, skill inflation, experience mismatches
- **Anti keyword-stuffer**: Penalizes AI buzzwords without supporting career history
- **Behavioral multipliers**: Uses recruiter response rate, interview completion, GitHub activity as confidence modifiers
