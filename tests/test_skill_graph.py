from scoring.skill_graph import (
    build_skill_graph,
    compute_skill_breadth,
    compute_skill_match,
    get_related_skills,
    get_skill_category,
)


class TestBuildSkillGraph:
    def test_graph_has_nodes(self):
        graph = build_skill_graph()
        assert len(graph) > 0

    def test_synonyms_linked(self):
        graph = build_skill_graph()
        assert "python3" in graph
        assert "python" in graph["python3"]


class TestGetRelatedSkills:
    def test_direct_skills(self):
        related = get_related_skills("pytorch", max_distance=1)
        assert "tensorflow" in related
        assert related["tensorflow"] == 1.0

    def test_unknown_skill(self):
        assert get_related_skills("nonexistent_skill_xyz") == {}

    def test_case_insensitive(self):
        r1 = get_related_skills("pytorch")
        r2 = get_related_skills("PyTorch")
        assert r1 == r2


class TestComputeSkillMatch:
    def test_exact_match(self):
        score, exact, transfer = compute_skill_match(["python", "pytorch"], ["python", "pytorch"])
        assert score == 1.0
        assert len(exact) == 2

    def test_partial_match(self):
        score, exact, transfer = compute_skill_match(["python"], ["python", "pytorch"])
        assert score > 0
        assert len(exact) == 1

    def test_empty_required(self):
        score, exact, transfer = compute_skill_match(["python"], [])
        assert score == 1.0

    def test_no_match(self):
        score, exact, transfer = compute_skill_match(["cooking"], ["python", "pytorch"])
        assert score == 0.0


class TestGetSkillCategory:
    def test_known_skill(self):
        assert get_skill_category("pytorch") == "deep_learning_frameworks"

    def test_synonym_resolution(self):
        assert get_skill_category("sklearn") == "ml_frameworks"

    def test_unknown_skill(self):
        assert get_skill_category("zzz_invalid_skill") is None


class TestComputeSkillBreadth:
    def test_basic(self):
        result = compute_skill_breadth(["python", "aws", "pytorch"])
        assert result["category_count"] >= 2
        assert result["uncovered_skills"] == 0
