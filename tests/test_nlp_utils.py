from utils.nlp_utils import (
    compile_keyword_patterns,
    compute_jd_candidate_overlap,
    count_keyword_matches,
    count_ngram_matches,
    count_pattern_matches,
    extract_key_phrases,
    extract_technical_terms,
    extract_years,
    has_production_indicators,
    tokenize,
)


class TestTokenize:
    def test_empty(self):
        assert tokenize("") == set()

    def test_none(self):
        assert tokenize(None) == set()

    def test_basic(self):
        tokens = tokenize("Python Machine Learning")
        assert "python" in tokens
        assert "machine" in tokens
        assert "learning" in tokens


class TestCountKeywordMatches:
    def test_empty_text(self):
        assert count_keyword_matches("", ["test"]) == 0

    def test_no_match(self):
        assert count_keyword_matches("hello world", ["foo", "bar"]) == 0

    def test_partial_match(self):
        assert count_keyword_matches("python machine learning", ["python"]) == 1

    def test_word_boundary(self):
        assert count_keyword_matches("embedded systems", ["embedding"]) == 0

    def test_case_insensitive(self):
        assert count_keyword_matches("PYTHON", ["python"]) == 1


class TestCountNgramMatches:
    def test_empty(self):
        assert count_ngram_matches("", ["test"]) == 0.0

    def test_ratio(self):
        result = count_ngram_matches("python java", ["python", "ruby"])
        assert result == 0.5

    def test_empty_keywords(self):
        assert count_ngram_matches("hello", []) == 0.0


class TestExtractYears:
    def test_empty(self):
        assert extract_years("") == 0.0

    def test_simple(self):
        assert extract_years("5 years") == 5.0

    def test_decimal(self):
        assert extract_years("3.5 years") == 3.5


class TestHasProductionIndicators:
    def test_empty(self):
        assert not has_production_indicators("")

    def test_production(self):
        assert has_production_indicators("deployed to production")

    def test_no_match(self):
        assert not has_production_indicators("hello world")


class TestCompiledPatterns:
    def test_compile_and_match(self):
        patterns = compile_keyword_patterns(["python", "machine learning"])
        assert count_pattern_matches("I love Python", patterns) == 1
        assert count_pattern_matches("Machine Learning is great", patterns) == 1

    def test_word_boundary(self):
        patterns = compile_keyword_patterns(["python"])
        assert count_pattern_matches("pythonic", patterns) == 0
        assert count_pattern_matches("python", patterns) == 1


class TestExtractKeyPhrases:
    def test_empty(self):
        assert extract_key_phrases("") == []

    def test_basic(self):
        phrases = extract_key_phrases("python machine learning deep learning")
        terms = [p[0] for p in phrases]
        assert "python" in terms
        assert "machine" in terms
        assert "learning" in terms


class TestExtractTechnicalTerms:
    def test_basic(self):
        terms = extract_technical_terms("python and tensorflow and kubernetes")
        assert "python" in terms
        assert "tensorflow" in terms
        assert "kubernetes" in terms

    def test_empty(self):
        assert extract_technical_terms("") == {}


class TestJdCandidateOverlap:
    def test_basic(self):
        jd = {"python": 0.5, "tensorflow": 0.5}
        cand = {"python": 0.8, "pytorch": 0.2}
        overlap = compute_jd_candidate_overlap(jd, cand)
        assert overlap == 0.5

    def test_empty_jd(self):
        assert compute_jd_candidate_overlap({}, {"python": 1.0}) == 0.0
