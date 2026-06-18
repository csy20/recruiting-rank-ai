from fastapi.testclient import TestClient

from serve import app

client = TestClient(app)

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
        },
    ],
    "skills": [
        {"name": "Python", "proficiency": "expert", "duration_months": 60},
        {"name": "PyTorch", "proficiency": "expert", "duration_months": 36},
    ],
    "redrob_signals": {
        "recruiter_response_rate": 0.8,
        "github_activity_score": 85,
    },
}


class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data


class TestRankEndpoint:
    def test_empty_candidates_returns_400(self):
        resp = client.post("/rank", json={"candidates": []})
        assert resp.status_code == 400

    def test_single_candidate(self):
        resp = client.post("/rank", json={"candidates": [SAMPLE_CANDIDATE]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["candidate_id"] == "CAND_0001"
        assert "score" in data["results"][0]
        assert "metadata" in data

    def test_with_jd_text(self, monkeypatch):
        monkeypatch.setattr(
            "serve._compute_semantic_features",
            lambda c, t, j: [0.5] * len(c),
        )
        payload = {
            "candidates": [SAMPLE_CANDIDATE],
            "jd_text": "Requirements: Python, PyTorch, embeddings, "
            "vector databases, production ML.",
        }
        resp = client.post("/rank", json=payload)
        assert resp.status_code == 200
        assert resp.json()["metadata"]["jd_parsed"] is True

    def test_top_k_limits_results(self):
        candidates = [{**SAMPLE_CANDIDATE, "candidate_id": f"CAND_{i:04d}"} for i in range(10)]
        resp = client.post("/rank", json={"candidates": candidates, "top_k": 3})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 3

    def test_too_many_candidates(self):
        candidates = [{} for _ in range(5001)]
        resp = client.post("/rank", json={"candidates": candidates})
        assert resp.status_code == 400

    def test_include_dimensions(self):
        resp = client.post(
            "/rank",
            json={"candidates": [SAMPLE_CANDIDATE], "include_dimensions": True},
        )
        assert "dimensions" in resp.json()["results"][0]

    def test_exclude_reasoning(self):
        resp = client.post(
            "/rank",
            json={"candidates": [SAMPLE_CANDIDATE], "include_reasoning": False},
        )
        assert "reasoning" not in resp.json()["results"][0]


class TestScoreEndpoint:
    def test_score_single(self):
        resp = client.post("/score", json=SAMPLE_CANDIDATE)
        assert resp.status_code == 200
        data = resp.json()
        assert "dimensions" in data
        assert data["candidate_id"] == "CAND_0001"


class TestAuditEndpoint:
    def test_audit_empty_returns_400(self):
        resp = client.post("/audit", json=[])
        assert resp.status_code == 400

    def test_audit_basic(self):
        resp = client.post("/audit", json=[SAMPLE_CANDIDATE])
        assert resp.status_code == 200
        data = resp.json()
        assert "audit_enabled" in data
