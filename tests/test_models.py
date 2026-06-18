from models import (
    Candidate,
    CareerEntry,
    DimensionScores,
    Profile,
    RankRequest,
    RedrobSignals,
    Skill,
)


class TestSkill:
    def test_defaults(self):
        s = Skill()
        assert s.name == ""
        assert s.proficiency == "beginner"
        assert s.duration_months == 0

    def test_duration_coerced(self):
        s = Skill(name="Python", duration_months=None)
        assert s.duration_months == 0

    def test_duration_negative_clamped(self):
        s = Skill(name="Python", duration_months=-5)
        assert s.duration_months == 0

    def test_proficiency_normalized(self):
        s = Skill(name="Python", proficiency=None)
        assert s.proficiency == "beginner"

    def test_proficiency_invalid_fallback(self):
        s = Skill(name="Python", proficiency="unknown")
        assert s.proficiency == "beginner"


class TestCareerEntry:
    def test_defaults(self):
        e = CareerEntry()
        assert e.company == ""
        assert e.duration_months == 0

    def test_duration_negative_clamped(self):
        e = CareerEntry(company="Google", duration_months=-1)
        assert e.duration_months == 0

    def test_parsed_start_none(self):
        e = CareerEntry()
        assert e.parsed_start() is None

    def test_parsed_start_valid(self):
        e = CareerEntry(start_date="2020-01-15")
        d = e.parsed_start()
        assert d is not None
        assert d.year == 2020
        assert d.month == 1

    def test_parsed_start_invalid(self):
        e = CareerEntry(start_date="not-a-date")
        assert e.parsed_start() is None

    def test_parsed_end_none(self):
        e = CareerEntry()
        assert e.parsed_end() is None


class TestProfile:
    def test_defaults(self):
        p = Profile()
        assert p.headline == ""
        assert p.years_of_experience == 0.0

    def test_years_coerced(self):
        p = Profile(years_of_experience=None)
        assert p.years_of_experience == 0.0

    def test_years_negative_clamped(self):
        p = Profile(years_of_experience=-5.0)
        assert p.years_of_experience == 0.0

    def test_years_invalid_fallback(self):
        p = Profile(years_of_experience="invalid")
        assert p.years_of_experience == 0.0


class TestRedrobSignals:
    def test_defaults(self):
        r = RedrobSignals()
        assert r.recruiter_response_rate == 0.0
        assert r.notice_period_days == 30
        assert r.github_activity_score == -1.0

    def test_rate_clamped_above(self):
        r = RedrobSignals(recruiter_response_rate=1.5)
        assert r.recruiter_response_rate == 1.0

    def test_rate_clamped_below(self):
        r = RedrobSignals(recruiter_response_rate=-0.5)
        assert r.recruiter_response_rate == 0.0

    def test_rate_none(self):
        r = RedrobSignals(recruiter_response_rate=None)
        assert r.recruiter_response_rate == 0.0

    def test_github_none(self):
        r = RedrobSignals(github_activity_score=None)
        assert r.github_activity_score == -1.0

    def test_connection_count_none(self):
        r = RedrobSignals(connection_count=None)
        assert r.connection_count == 0

    def test_connection_count_negative(self):
        r = RedrobSignals(connection_count=-5)
        assert r.connection_count == 0


class TestCandidate:
    def test_defaults(self):
        c = Candidate()
        assert c.candidate_id == ""
        assert c.profile is not None
        assert c.career_history == []
        assert c.skills == []
        assert c.redrob_signals is not None

    def test_none_id(self):
        c = Candidate(candidate_id=None)
        assert c.candidate_id == ""

    def test_with_data(self):
        c = Candidate(
            candidate_id="CAND_001",
            profile={"headline": "Engineer"},
            career_history=[{"company": "Google"}],
            skills=[{"name": "Python"}],
            redrob_signals={"github_activity_score": 90},
        )
        assert c.candidate_id == "CAND_001"
        assert c.profile.headline == "Engineer"
        assert len(c.career_history) == 1
        assert c.career_history[0].company == "Google"
        assert c.skills[0].name == "Python"
        assert c.redrob_signals.github_activity_score == 90.0


class TestDimensionScores:
    def test_defaults(self):
        d = DimensionScores()
        assert d.technical_match == 0.0
        assert d.risk_adjustment == 1.0
        assert d.jd_semantic_similarity is None


class TestRankRequest:
    def test_valid(self):
        req = RankRequest(candidates=[{"id": "1"}], top_k=50)
        assert len(req.candidates) == 1
        assert req.top_k == 50

    def test_top_k_default(self):
        req = RankRequest(candidates=[{"id": "1"}])
        assert req.top_k == 100
