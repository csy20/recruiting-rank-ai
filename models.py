from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class Skill(BaseModel):
    name: str = ""
    proficiency: Optional[str] = "beginner"
    duration_months: Optional[int] = 0

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
    duration_months: Optional[int] = 0
    is_current: Optional[bool] = False
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_duration(cls, v):
        if v is None:
            return 0
        return max(0, int(v))

    def parsed_start(self) -> Optional[date]:
        if not self.start_date:
            return None
        try:
            return datetime.strptime(str(self.start_date)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    def parsed_end(self) -> Optional[date]:
        if not self.end_date:
            return None
        try:
            return datetime.strptime(str(self.end_date)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None


class Profile(BaseModel):
    headline: Optional[str] = ""
    summary: Optional[str] = ""
    current_company: Optional[str] = ""
    current_title: Optional[str] = ""
    years_of_experience: Optional[float] = 0.0
    location: Optional[str] = ""
    country: Optional[str] = ""

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
    recruiter_response_rate: Optional[float] = 0.0
    interview_completion_rate: Optional[float] = 0.0
    open_to_work_flag: Optional[bool] = False
    saved_by_recruiters_30d: Optional[int] = 0
    search_appearance_30d: Optional[int] = 0
    github_activity_score: Optional[float] = -1.0
    notice_period_days: Optional[int] = 30
    willing_to_relocate: Optional[bool] = False
    offer_acceptance_rate: Optional[float] = 0.5
    avg_response_time_hours: Optional[float] = 24.0
    verified_email: Optional[bool] = False
    verified_phone: Optional[bool] = False
    linkedin_connected: Optional[bool] = False
    applications_submitted_30d: Optional[int] = 0
    profile_views_received_30d: Optional[int] = 0
    connection_count: Optional[int] = 0
    endorsements_received: Optional[int] = 0
    last_active_date: Optional[str] = ""
    profile_completeness_score: Optional[float] = 0.0

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
    profile: Optional[Profile] = None
    career_history: Optional[List[CareerEntry]] = None
    skills: Optional[List[Skill]] = None
    redrob_signals: Optional[RedrobSignals] = None

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
    jd_semantic_similarity: Optional[float] = None


class RankedResult(BaseModel):
    candidate_id: str
    rank: int
    score: float
    dimensions: DimensionScores
    reasoning: str = ""
    feature_contributions: Optional[Dict[str, float]] = None
    skill_matches: Optional[List[str]] = None


class JDAnalysis(BaseModel):
    experience_range: tuple = (0.0, 20.0)
    dimension_weights: Dict[str, float] = {}
    required_terms: Dict[str, float] = {}
    preferred_terms: Dict[str, float] = {}
    technical_terms: Dict[str, float] = {}
    sections: Dict[str, str] = {}


class RankRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    jd_text: Optional[str] = None
    top_k: int = Field(default=100, ge=1, le=5000)
    include_dimensions: bool = True
    include_reasoning: bool = True
    include_contributions: bool = False


class RankResponse(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


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
    group_sizes: Dict[str, int]
    disparities: Dict[str, Dict[str, AuditEntry]]
    passed: bool
