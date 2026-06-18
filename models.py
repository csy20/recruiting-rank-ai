from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Skill(BaseModel):
    name: str = ""
    proficiency: str | None = "beginner"
    duration_months: int | None = 0

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_duration(cls, v):
        if v is None:
            return 0
        return max(0, int(v))

    @field_validator("proficiency", mode="before")
    @classmethod
    def normalize_proficiency(cls, v):
        if v is None:
            return "beginner"
        v = str(v).lower()
        if v in ("expert", "advanced", "intermediate", "beginner"):
            return v
        return "beginner"


class CareerEntry(BaseModel):
    company: str = ""
    title: str = ""
    description: str = ""
    industry: str = ""
    duration_months: int | None = 0
    is_current: bool | None = False
    start_date: str | None = ""
    end_date: str | None = ""

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_duration(cls, v):
        if v is None:
            return 0
        return max(0, int(v))

    def parsed_start(self) -> date | None:
        if not self.start_date:
            return None
        try:
            return datetime.strptime(str(self.start_date)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    def parsed_end(self) -> date | None:
        if not self.end_date:
            return None
        try:
            return datetime.strptime(str(self.end_date)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None


class Profile(BaseModel):
    headline: str | None = ""
    summary: str | None = ""
    current_company: str | None = ""
    current_title: str | None = ""
    years_of_experience: float | None = 0.0
    location: str | None = ""
    country: str | None = ""

    @field_validator("years_of_experience", mode="before")
    @classmethod
    def coerce_years(cls, v):
        if v is None:
            return 0.0
        try:
            return max(0.0, float(v))
        except (ValueError, TypeError):
            return 0.0


class RedrobSignals(BaseModel):
    recruiter_response_rate: float | None = 0.0
    interview_completion_rate: float | None = 0.0
    open_to_work_flag: bool | None = False
    saved_by_recruiters_30d: int | None = 0
    search_appearance_30d: int | None = 0
    github_activity_score: float | None = -1.0
    notice_period_days: int | None = 30
    willing_to_relocate: bool | None = False
    offer_acceptance_rate: float | None = 0.5
    avg_response_time_hours: float | None = 24.0
    verified_email: bool | None = False
    verified_phone: bool | None = False
    linkedin_connected: bool | None = False
    applications_submitted_30d: int | None = 0
    profile_views_received_30d: int | None = 0
    connection_count: int | None = 0
    endorsements_received: int | None = 0
    last_active_date: str | None = ""
    profile_completeness_score: float | None = 0.0

    @field_validator(
        "recruiter_response_rate", "interview_completion_rate", mode="before"
    )
    @classmethod
    def clamp_rate(cls, v):
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("github_activity_score", mode="before")
    @classmethod
    def coerce_github(cls, v):
        if v is None:
            return -1.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return -1.0

    @field_validator(
        "connection_count",
        "applications_submitted_30d",
        "profile_views_received_30d",
        mode="before",
    )
    @classmethod
    def coerce_count(cls, v):
        if v is None:
            return 0
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return 0


class Candidate(BaseModel):
    candidate_id: str = ""
    profile: Profile | None = None
    career_history: list[CareerEntry] | None = None
    skills: list[Skill] | None = None
    redrob_signals: RedrobSignals | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.setdefault("profile", {})
            data.setdefault("career_history", [])
            data.setdefault("skills", [])
            data.setdefault("redrob_signals", {})
            if "candidate_id" not in data:
                data["candidate_id"] = ""
        return data

    @field_validator("candidate_id", mode="before")
    @classmethod
    def coerce_id(cls, v):
        if v is None:
            return ""
        return str(v)


class DimensionScores(BaseModel):
    technical_match: float = 0.0
    semantic_match: float = 0.0
    career_quality: float = 0.0
    behavioral: float = 0.0
    retention: float = 0.0
    risk_adjustment: float = 1.0
    jd_semantic_similarity: float | None = None


class RankedResult(BaseModel):
    candidate_id: str
    rank: int
    score: float
    dimensions: DimensionScores
    reasoning: str = ""
    feature_contributions: dict[str, float] | None = None
    skill_matches: list[str] | None = None


class JDAnalysis(BaseModel):
    experience_range: tuple = (0.0, 20.0)
    dimension_weights: dict[str, float] = {}
    required_terms: dict[str, float] = {}
    preferred_terms: dict[str, float] = {}
    technical_terms: dict[str, float] = {}
    sections: dict[str, str] = {}


class RankRequest(BaseModel):
    candidates: list[dict[str, Any]]
    jd_text: str | None = None
    top_k: int = Field(default=100, ge=1, le=5000)
    include_dimensions: bool = True
    include_reasoning: bool = True
    include_contributions: bool = False


class RankResponse(BaseModel):
    results: list[dict[str, Any]]
    metadata: dict[str, Any]


class AuditEntry(BaseModel):
    group: str
    dimension: str
    advantaged_mean: float
    disadvantaged_mean: float
    disparity: float
    flagged: bool


class FairnessReport(BaseModel):
    audit_enabled: bool
    disparity_threshold: float
    group_sizes: dict[str, int]
    disparities: dict[str, dict[str, AuditEntry]]
    passed: bool
