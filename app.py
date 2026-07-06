from enum import Enum
import random
import time
from typing import List, Optional

import streamlit as st
from google import genai
from pydantic import BaseModel, Field, ValidationError


PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]


class QuestionPriority(str, Enum):
    BLOCKING = "blocking"
    HIGH_IMPACT = "high_impact"
    HELPFUL = "helpful"
    ADMINISTRATIVE = "administrative"


class ReadinessStatus(str, Enum):
    NOT_READY = "not_ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_WITH_ASSUMPTIONS = "ready_with_assumptions"
    READY_TO_SOURCE = "ready_to_source"


class ApprovalStatus(str, Enum):
    DRAFT = "draft"
    PENDING_RECRUITER_REVIEW = "pending_recruiter_review"
    APPROVED = "approved"
    REJECTED_FOR_REVISION = "rejected_for_revision"


class ClarificationQuestion(BaseModel):
    question: str
    priority: QuestionPriority
    reason: str
    affected_area: str
    related_fields: List[str] = Field(default_factory=list)


class SourceEvidence(BaseModel):
    evidence_id: str
    field_name: str
    value: str
    source_type: str
    supporting_text: str


class SourcingBrief(BaseModel):
    role_title: Optional[str] = None
    team_name: Optional[str] = None
    seniority_levels: List[str] = Field(default_factory=list)
    level_flexibility: Optional[str] = None
    key_responsibilities: List[str] = Field(default_factory=list)

    locations: List[str] = Field(default_factory=list)
    location_flexibility: Optional[str] = None
    work_arrangement: Optional[str] = None
    employment_type: Optional[str] = None
    number_of_openings: Optional[int] = None
    target_hire_date: Optional[str] = None

    visa_sponsorship: Optional[bool] = None
    relocation_available: Optional[bool] = None
    compensation_range: Optional[str] = None

    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    must_have_qualifications: List[str] = Field(default_factory=list)
    nice_to_have_qualifications: List[str] = Field(default_factory=list)

    non_negotiable_requirements: List[str] = Field(default_factory=list)
    flexible_requirements: List[str] = Field(default_factory=list)
    acceptable_alternatives: List[str] = Field(default_factory=list)
    disqualifying_conditions: List[str] = Field(default_factory=list)
    sourcing_caveats: List[str] = Field(default_factory=list)

    target_companies: List[str] = Field(default_factory=list)
    target_job_titles: List[str] = Field(default_factory=list)
    adjacent_profile_types: List[str] = Field(default_factory=list)

    candidate_screening_questions: List[str] = Field(default_factory=list)
    expected_screening_evidence: List[str] = Field(default_factory=list)

    missing_information: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    clarification_questions: List[ClarificationQuestion] = Field(default_factory=list)
    source_evidence: List[SourceEvidence] = Field(default_factory=list)

    brief_version: int = 1
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT


class ReadinessDimension(BaseModel):
    dimension_name: str
    score: int = Field(ge=0)
    max_score: int = Field(gt=0)
    completed_checks: List[str] = Field(default_factory=list)
    unresolved_checks: List[str] = Field(default_factory=list)
    rationale: str


class ReadinessAssessment(BaseModel):
    total_score: int = Field(ge=0, le=100)
    readiness_status: ReadinessStatus
    dimensions: List[ReadinessDimension] = Field(default_factory=list)
    blocking_gaps: List[str] = Field(default_factory=list)
    high_impact_gaps: List[str] = Field(default_factory=list)
    helpful_gaps: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    recruiter_message: str
    can_begin_sourcing: bool = False
    requires_recruiter_approval: bool = True


class AgentAction(str, Enum):
    REQUEST_BLOCKING_CLARIFICATION = "request_blocking_clarification"
    REQUEST_HIGH_IMPACT_CLARIFICATION = "request_high_impact_clarification"
    REQUEST_RECRUITER_APPROVAL = "request_recruiter_approval"
    APPROVED_FOR_SOURCING = "approved_for_sourcing"
    REVISE_ROLE_BRIEF = "revise_role_brief"


class AgentRoutingDecision(BaseModel):
    recommended_action: AgentAction
    reason: str
    next_owner: str
    required_inputs: List[str] = Field(default_factory=list)
    can_proceed_without_additional_llm_call: bool


class SourcingLaunchPlan(BaseModel):
    role_title: str
    approval_status: ApprovalStatus
    difficulty_band: str
    prototype_shortlist_estimate: str
    difficulty_factors: List[str] = Field(default_factory=list)
    recommended_initial_search_focus: List[str] = Field(default_factory=list)
    remaining_assumptions: List[str] = Field(default_factory=list)
    calibration_note: str
    launch_rationale: str


SOURCING_READINESS_INSTRUCTIONS = """
You are S.O.N.A.R.'s Role Intake Analyzer.

Convert unstructured recruiting information into a structured,
evidence-backed SourcingBrief.

Use only the provided job description, recruiter intake notes, and
clarification answers. Do not invent requirements, policies, locations,
compensation, timelines, hiring preferences, target companies, or
constraints. Treat anything not provided as unknown.

Create source_evidence for important extracted facts. Distinguish
must-have, preferred, non-negotiable, flexible, alternative, and
disqualifying requirements. Classify skills separately from
qualifications. Capture missing information, ambiguities, contradictions,
and no more than five recruiter-facing clarification questions.

Gemini is responsible only for structured extraction, normalization, gap
detection, contradiction detection, and clarification planning. Do not
calculate a readiness score. Do not decide whether sourcing can begin.
Do not approve the role. Keep approval_status as draft unless explicit
recruiter approval is included in the input. Do not use protected
personal characteristics or proxies as sourcing or screening criteria.
"""


READINESS_WEIGHTS = {
    "role_definition": 20,
    "candidate_eligibility": 25,
    "search_parameters": 20,
    "assessment_alignment": 20,
    "hiring_logistics": 15,
}

READINESS_THRESHOLDS = {
    "ready_to_source": 85,
    "ready_with_assumptions": 70,
    "needs_clarification": 50,
}

BLOCKING_GAP_RULES = {
    "role_title": "The role title or role family has not been confirmed.",
    "seniority_levels": "The required seniority level has not been confirmed.",
    "key_responsibilities": "The core responsibilities are not sufficiently defined.",
    "must_have_skills": "The mandatory skills are not sufficiently defined.",
    "locations": "The permitted hiring location has not been confirmed.",
    "work_arrangement": "The remote, hybrid, or on-site expectation has not been confirmed.",
}

HIGH_IMPACT_GAP_RULES = {
    "level_flexibility": "It is unclear whether candidates at adjacent seniority levels may be considered.",
    "non_negotiable_requirements": "The hiring manager's non-negotiable requirements have not been explicitly documented.",
    "acceptable_alternatives": "Acceptable substitutes for skills, technologies, or experience have not been clarified.",
    "visa_sponsorship": "Visa sponsorship availability has not been confirmed.",
    "compensation_range": "The approved compensation range has not been provided.",
    "number_of_openings": "The number of openings has not been confirmed.",
    "target_hire_date": "The target hiring date has not been confirmed.",
}

HELPFUL_GAP_RULES = {
    "team_name": "The hiring team or organizational context has not been provided.",
    "target_job_titles": "Target and adjacent job titles have not yet been defined.",
    "target_companies": "Example source companies or company profiles have not been defined.",
    "candidate_screening_questions": "Candidate screening questions have not yet been documented.",
    "expected_screening_evidence": "The evidence candidates should provide during screening has not been documented.",
}

TEMPORARY_ERROR_MARKERS = (
    "429",
    "503",
    "unavailable",
    "resource_exhausted",
    "high demand",
    "overloaded",
    "temporarily unavailable",
    "deadline exceeded",
)


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def is_temporary_model_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(marker in error_text for marker in TEMPORARY_ERROR_MARKERS)


def get_gemini_client() -> genai.Client:
    # Security handling for API keys: the key is read from Streamlit secrets at
    # runtime and is never stored in source code, logs, notebook state, or UI.
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError as error:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml "
            "or Streamlit Community Cloud secrets."
        ) from error
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty.")
    return genai.Client(api_key=api_key)


def analyze_sourcing_brief(
    client: genai.Client,
    job_description: str,
    intake_context: str = "",
    clarification_answers: str = "",
    max_retries_per_model: int = 2,
) -> tuple[SourcingBrief, str]:
    # Role Intake Agent: Gemini turns recruiter text into a validated
    # SourcingBrief. It does not score readiness or approve sourcing.
    if not job_description.strip():
        raise ValueError("A job description is required.")

    combined_input = f"""
JOB DESCRIPTION
---------------
{job_description.strip()}

RECRUITER NOTES OR INTAKE TRANSCRIPT
------------------------------------
{intake_context.strip() or "No intake context provided."}

CLARIFICATION ANSWERS
---------------------
{clarification_answers.strip() or "No clarification answers provided."}
"""

    last_error = None
    attempted_models = []

    for model_name in FALLBACK_MODELS:
        attempted_models.append(model_name)
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=combined_input,
                    config={
                        "system_instruction": SOURCING_READINESS_INSTRUCTIONS,
                        "response_mime_type": "application/json",
                        "response_schema": SourcingBrief,
                        "temperature": 0.1,
                    },
                )
                response_text = (response.text or "").strip()
                if not response_text:
                    raise RuntimeError("The model returned an empty response.")
                brief = SourcingBrief.model_validate_json(response_text)
                brief.approval_status = ApprovalStatus.DRAFT
                return brief, model_name
            except ValidationError as error:
                raise RuntimeError(
                    "Gemini returned output that did not match the SourcingBrief schema."
                ) from error
            except Exception as error:
                last_error = error
                if not is_temporary_model_error(error):
                    raise RuntimeError(
                        f"Sourcing brief analysis failed with {model_name}: {error}"
                    ) from error
                if attempt < max_retries_per_model:
                    time.sleep(2**attempt + random.uniform(0.0, 0.5))

    attempted = ", ".join(attempted_models)
    raise RuntimeError(
        "S.O.N.A.R. could not reach an available Gemini model after "
        f"trying: {attempted}. Last error: {last_error}"
    )


def unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def calculate_sourcing_readiness(brief: SourcingBrief) -> ReadinessAssessment:
    # Readiness Tool: deterministic Python rules calculate the score, gaps,
    # readiness status, and can_begin_sourcing flag. Gemini never performs this
    # scoring step.
    def completed(value) -> bool:
        return not is_missing(value)

    def build_dimension(dimension_name: str, checks: dict) -> ReadinessDimension:
        completed_checks = [
            name for name, (is_complete, _) in checks.items() if is_complete
        ]
        unresolved_checks = [
            name for name, (is_complete, _) in checks.items() if not is_complete
        ]
        score = sum(points for is_complete, points in checks.values() if is_complete)
        max_score = sum(points for _, points in checks.values())
        return ReadinessDimension(
            dimension_name=dimension_name,
            score=score,
            max_score=max_score,
            completed_checks=completed_checks,
            unresolved_checks=unresolved_checks,
            rationale=f"{len(completed_checks)} of {len(checks)} readiness checks completed.",
        )

    blocking_gaps = [
        message
        for field_name, message in BLOCKING_GAP_RULES.items()
        if is_missing(getattr(brief, field_name))
    ]
    high_impact_gaps = [
        message
        for field_name, message in HIGH_IMPACT_GAP_RULES.items()
        if is_missing(getattr(brief, field_name))
    ]
    helpful_gaps = [
        message
        for field_name, message in HELPFUL_GAP_RULES.items()
        if is_missing(getattr(brief, field_name))
    ]

    blocking_gaps.extend(
        f"Unresolved contradiction: {item}" for item in brief.contradictions
    )
    high_impact_gaps.extend(f"Unresolved ambiguity: {item}" for item in brief.ambiguities)

    role_definition = build_dimension(
        "Role Definition",
        {
            "Role title confirmed": (completed(brief.role_title), 5),
            "Seniority confirmed": (completed(brief.seniority_levels), 5),
            "Core responsibilities defined": (completed(brief.key_responsibilities), 7),
            "Team context provided": (completed(brief.team_name), 3),
        },
    )

    candidate_eligibility = build_dimension(
        "Candidate Eligibility",
        {
            "Must-have skills defined": (completed(brief.must_have_skills), 7),
            "Must-have qualifications defined": (
                completed(brief.must_have_qualifications),
                5,
            ),
            "Non-negotiables documented": (
                completed(brief.non_negotiable_requirements),
                5,
            ),
            "Level flexibility clarified": (completed(brief.level_flexibility), 4),
            "Acceptable alternatives documented": (
                completed(brief.acceptable_alternatives),
                4,
            ),
        },
    )

    sourcing_direction_available = any(
        [
            completed(brief.target_companies),
            completed(brief.target_job_titles),
            completed(brief.adjacent_profile_types),
        ]
    )
    search_parameters = build_dimension(
        "Search Parameters",
        {
            "Hiring locations confirmed": (completed(brief.locations), 6),
            "Work arrangement confirmed": (completed(brief.work_arrangement), 5),
            "Employment type confirmed": (completed(brief.employment_type), 3),
            "Target titles defined": (completed(brief.target_job_titles), 3),
            "Sourcing direction provided": (sourcing_direction_available, 3),
        },
    )

    assessment_alignment = build_dimension(
        "Assessment Alignment",
        {
            "Screening questions documented": (
                completed(brief.candidate_screening_questions),
                7,
            ),
            "Expected screening evidence documented": (
                completed(brief.expected_screening_evidence),
                7,
            ),
            "Core responsibilities support assessment": (
                completed(brief.key_responsibilities),
                6,
            ),
        },
    )

    hiring_logistics = build_dimension(
        "Hiring Logistics",
        {
            "Number of openings confirmed": (completed(brief.number_of_openings), 3),
            "Target hiring date confirmed": (completed(brief.target_hire_date), 3),
            "Visa sponsorship confirmed": (completed(brief.visa_sponsorship), 3),
            "Compensation range confirmed": (completed(brief.compensation_range), 3),
            "Location flexibility clarified": (
                completed(brief.location_flexibility),
                3,
            ),
        },
    )

    dimensions = [
        role_definition,
        candidate_eligibility,
        search_parameters,
        assessment_alignment,
        hiring_logistics,
    ]
    total_score = sum(dimension.score for dimension in dimensions)
    blocking_gaps = unique(blocking_gaps)
    high_impact_gaps = unique(high_impact_gaps)
    helpful_gaps = unique(helpful_gaps)

    if blocking_gaps:
        readiness_status = ReadinessStatus.NOT_READY
        can_begin_sourcing = False
    elif total_score >= READINESS_THRESHOLDS["ready_to_source"]:
        readiness_status = ReadinessStatus.READY_TO_SOURCE
        can_begin_sourcing = True
    elif total_score >= READINESS_THRESHOLDS["ready_with_assumptions"]:
        readiness_status = ReadinessStatus.READY_WITH_ASSUMPTIONS
        can_begin_sourcing = True
    elif total_score >= READINESS_THRESHOLDS["needs_clarification"]:
        readiness_status = ReadinessStatus.NEEDS_CLARIFICATION
        can_begin_sourcing = False
    else:
        readiness_status = ReadinessStatus.NOT_READY
        can_begin_sourcing = False

    strengths = [
        f"{dimension.dimension_name} is well defined."
        for dimension in dimensions
        if dimension.score / dimension.max_score >= 0.75
    ]

    if blocking_gaps:
        recruiter_message = (
            f"This role is not ready to source. {len(blocking_gaps)} blocking "
            "decision(s) must be resolved with the hiring manager."
        )
    elif readiness_status == ReadinessStatus.READY_TO_SOURCE:
        recruiter_message = "The role is sufficiently defined for recruiter review and approval."
    elif readiness_status == ReadinessStatus.READY_WITH_ASSUMPTIONS:
        recruiter_message = (
            "Sourcing may begin if the remaining assumptions are documented "
            "and accepted by the recruiter."
        )
    elif readiness_status == ReadinessStatus.NEEDS_CLARIFICATION:
        recruiter_message = (
            "The role is partially defined but requires further clarification "
            "before sourcing begins."
        )
    else:
        recruiter_message = (
            "The role requires substantial clarification before a consistent "
            "sourcing search can be launched."
        )

    return ReadinessAssessment(
        total_score=total_score,
        readiness_status=readiness_status,
        dimensions=dimensions,
        blocking_gaps=blocking_gaps,
        high_impact_gaps=high_impact_gaps,
        helpful_gaps=helpful_gaps,
        strengths=strengths,
        recruiter_message=recruiter_message,
        can_begin_sourcing=can_begin_sourcing,
        requires_recruiter_approval=True,
    )


def route_next_action(
    brief: SourcingBrief, readiness: ReadinessAssessment
) -> AgentRoutingDecision:
    # Approval Router: deterministic workflow routing. It may request approval,
    # but it cannot grant approval without the recruiter's explicit action.
    if readiness.blocking_gaps:
        return AgentRoutingDecision(
            recommended_action=AgentAction.REQUEST_BLOCKING_CLARIFICATION,
            reason="The role has blocking gaps that could prevent a consistent sourcing search.",
            next_owner="Recruiter or hiring manager",
            required_inputs=readiness.blocking_gaps,
            can_proceed_without_additional_llm_call=False,
        )
    if not readiness.can_begin_sourcing and readiness.high_impact_gaps:
        return AgentRoutingDecision(
            recommended_action=AgentAction.REQUEST_HIGH_IMPACT_CLARIFICATION,
            reason="High-impact gaps could materially affect the candidate pool or search strategy.",
            next_owner="Recruiter or hiring manager",
            required_inputs=readiness.high_impact_gaps,
            can_proceed_without_additional_llm_call=False,
        )
    if brief.approval_status == ApprovalStatus.APPROVED:
        return AgentRoutingDecision(
            recommended_action=AgentAction.APPROVED_FOR_SOURCING,
            reason="The role is ready and has been approved by the recruiter.",
            next_owner="Recruiting team",
            required_inputs=[],
            can_proceed_without_additional_llm_call=True,
        )
    if readiness.can_begin_sourcing:
        return AgentRoutingDecision(
            recommended_action=AgentAction.REQUEST_RECRUITER_APPROVAL,
            reason="The role is sufficiently defined, but human approval is required before launch.",
            next_owner="Recruiter",
            required_inputs=["Recruiter approval", "Acceptance of remaining assumptions"],
            can_proceed_without_additional_llm_call=True,
        )
    return AgentRoutingDecision(
        recommended_action=AgentAction.REVISE_ROLE_BRIEF,
        reason="The role is not ready and requires further revision before sourcing can begin.",
        next_owner="Recruiter",
        required_inputs=["Updated job description", "Additional hiring-manager context"],
        can_proceed_without_additional_llm_call=False,
    )


def apply_recruiter_approval(
    brief: SourcingBrief,
    readiness: ReadinessAssessment,
    recruiter_approves: bool,
    approval_note: str = "",
) -> SourcingBrief:
    # Human-in-the-loop approval: the app requires explicit recruiter approval.
    # Even then, approval is blocked if deterministic readiness says the role
    # cannot begin sourcing.
    updated_brief = brief.model_copy(deep=True)
    if recruiter_approves and readiness.can_begin_sourcing:
        updated_brief.approval_status = ApprovalStatus.APPROVED
        if approval_note.strip():
            updated_brief.sourcing_caveats.append(
                f"Recruiter approval note: {approval_note.strip()}"
            )
    elif recruiter_approves and not readiness.can_begin_sourcing:
        updated_brief.approval_status = ApprovalStatus.REJECTED_FOR_REVISION
        updated_brief.sourcing_caveats.append(
            "Recruiter attempted approval, but the role is not ready according to the readiness tool."
        )
    else:
        updated_brief.approval_status = ApprovalStatus.REJECTED_FOR_REVISION
        if approval_note.strip():
            updated_brief.sourcing_caveats.append(
                f"Recruiter revision note: {approval_note.strip()}"
            )
    return updated_brief


def estimate_sourcing_launch_plan(
    brief: SourcingBrief, readiness: ReadinessAssessment
) -> SourcingLaunchPlan:
    # Launch Readiness Tool: deterministic rule-based planning after approval.
    # The shortlist estimate is a prototype planning estimate, not a statistical
    # forecast or candidate-sourcing engine.
    if brief.approval_status != ApprovalStatus.APPROVED:
        raise RuntimeError("A launch plan can only be created after recruiter approval.")

    difficulty_points = 0
    difficulty_factors = []
    seniority_text = " ".join(brief.seniority_levels).lower()
    must_have_skills_text = " ".join(brief.must_have_skills).lower()

    if "senior" in seniority_text:
        difficulty_points += 2
        difficulty_factors.append(
            "Senior-level roles usually require a narrower candidate pool."
        )
    if "kafka" in must_have_skills_text:
        difficulty_points += 2
        difficulty_factors.append(
            "Kafka is a mandatory skill, which narrows the sourcing pool."
        )
    if brief.locations and len(brief.locations) <= 2:
        difficulty_points += 1
        difficulty_factors.append(
            "The search is limited to a small number of approved locations."
        )
    if brief.compensation_range:
        difficulty_factors.append(
            "Compensation is confirmed, which reduces launch uncertainty."
        )
    else:
        difficulty_points += 2
        difficulty_factors.append(
            "Compensation is unconfirmed, which increases launch uncertainty."
        )
    if brief.visa_sponsorship is True:
        difficulty_points -= 1
        difficulty_factors.append(
            "Visa sponsorship is available, which may expand the candidate pool."
        )
    if readiness.high_impact_gaps:
        difficulty_points += 1
        difficulty_factors.append(
            "Some high-impact ambiguity remains and should be monitored."
        )

    if difficulty_points <= 2:
        difficulty_band = "moderate"
        prototype_shortlist_estimate = "5 to 7 business days"
    elif difficulty_points <= 4:
        difficulty_band = "moderate-high"
        prototype_shortlist_estimate = "7 to 10 business days"
    else:
        difficulty_band = "high"
        prototype_shortlist_estimate = "10 to 15 business days"

    recommended_focus = []
    if brief.target_job_titles:
        recommended_focus.extend(
            f"Search for candidates with title: {title}"
            for title in brief.target_job_titles[:5]
        )
    if brief.target_companies:
        recommended_focus.append(
            "Prioritize high-scale technology companies such as "
            + ", ".join(brief.target_companies[:4])
            + "."
        )
    if brief.must_have_skills:
        recommended_focus.append(
            "Screen early for must-have skills: "
            + ", ".join(brief.must_have_skills)
            + "."
        )

    remaining_assumptions = readiness.high_impact_gaps + readiness.helpful_gaps
    calibration_note = (
        "This is a prototype rule-based estimate. It is not a trained "
        "forecasting model. In production, thresholds and time bands should "
        "be calibrated using company-specific recruiting data."
    )
    launch_rationale = (
        "The role has passed the readiness check and has recruiter approval. "
        "The plan uses transparent business rules based on seniority, mandatory "
        "skills, location constraints, compensation clarity, sponsorship "
        "availability, and unresolved assumptions."
    )
    return SourcingLaunchPlan(
        role_title=brief.role_title or "Unconfirmed role",
        approval_status=brief.approval_status,
        difficulty_band=difficulty_band,
        prototype_shortlist_estimate=prototype_shortlist_estimate,
        difficulty_factors=difficulty_factors,
        recommended_initial_search_focus=recommended_focus,
        remaining_assumptions=remaining_assumptions,
        calibration_note=calibration_note,
        launch_rationale=launch_rationale,
    )


def format_bool(value: Optional[bool]) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def render_list(items: List[str], empty_message: str = "None") -> None:
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.caption(empty_message)


st.set_page_config(page_title="S.O.N.A.R.", layout="wide")

st.title("S.O.N.A.R.")
st.caption("Sourcing Operations & Navigation via Agentic Recruitment")
st.write(
    "A recruiter-controlled Sourcing Readiness Agent that turns a job "
    "description, intake notes, and clarification answers into an "
    "evidence-backed sourcing brief, deterministic readiness score, approval "
    "route, and rule-based launch plan."
)

with st.form("sonar_inputs"):
    job_description = st.text_area("Job description", height=260)
    intake_context = st.text_area("Recruiter intake notes", height=220)
    clarification_answers = st.text_area(
        "Optional clarification answers", height=140
    )
    analyze = st.form_submit_button("Analyze Role", type="primary")

if analyze:
    try:
        with st.spinner("Running Role Intake Agent and deterministic tools..."):
            client = get_gemini_client()
            brief, model_used = analyze_sourcing_brief(
                client=client,
                job_description=job_description,
                intake_context=intake_context,
                clarification_answers=clarification_answers,
            )
            readiness = calculate_sourcing_readiness(brief)
            routing = route_next_action(brief, readiness)
            st.session_state["brief"] = brief
            st.session_state["readiness"] = readiness
            st.session_state["routing"] = routing
            st.session_state["model_used"] = model_used
    except Exception as error:
        st.error(str(error))

brief = st.session_state.get("brief")
readiness = st.session_state.get("readiness")
routing = st.session_state.get("routing")
model_used = st.session_state.get("model_used")

if brief and readiness and routing:
    st.divider()
    st.subheader("Readiness Diagnosis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Readiness score", f"{readiness.total_score}/100")
    col2.metric("Readiness status", readiness.readiness_status.value)
    col3.metric("Can begin sourcing", str(readiness.can_begin_sourcing))
    st.write(f"Recruiter message: {readiness.recruiter_message}")
    st.caption(f"Gemini structured extraction model: {model_used}")

    with st.expander("Dimension scores", expanded=True):
        for dimension in readiness.dimensions:
            st.write(
                f"**{dimension.dimension_name}:** "
                f"{dimension.score}/{dimension.max_score}"
            )

    left, middle, right = st.columns(3)
    with left:
        st.subheader("Blocking gaps")
        render_list(readiness.blocking_gaps)
    with middle:
        st.subheader("High-impact gaps")
        render_list(readiness.high_impact_gaps)
    with right:
        st.subheader("Helpful gaps")
        render_list(readiness.helpful_gaps)

    st.subheader("Clarification Questions")
    if brief.clarification_questions:
        for question in brief.clarification_questions:
            st.write(f"**[{question.priority.value}] {question.question}**")
            st.caption(f"{question.reason} Affected area: {question.affected_area}")
    else:
        st.caption("No clarification questions generated.")

    st.subheader("Structured Sourcing Brief")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Role summary:** {brief.role_title or 'Not confirmed'}")
        st.write("**Must-have skills**")
        render_list(brief.must_have_skills)
        st.write("**Locations**")
        render_list(brief.locations)
        st.write(f"**Compensation:** {brief.compensation_range or 'Unknown'}")
        st.write(f"**Visa sponsorship:** {format_bool(brief.visa_sponsorship)}")
        st.write(f"**Evidence count:** {len(brief.source_evidence)}")
    with col_b:
        st.write("**Key responsibilities**")
        render_list(brief.key_responsibilities)
        st.write("**Acceptable alternatives**")
        render_list(brief.acceptable_alternatives)
        st.write("**Screening evidence**")
        render_list(brief.expected_screening_evidence)

    st.subheader("Approval Router")
    st.write(f"**Routing action:** {routing.recommended_action.value}")
    st.write(f"**Next owner:** {routing.next_owner}")
    st.write(routing.reason)
    if routing.required_inputs:
        st.write("**Required inputs**")
        render_list(routing.required_inputs)

    st.subheader("Human Approval Gate")
    recruiter_approves = st.checkbox(
        "Recruiter approves this role for sourcing launch",
        disabled=not readiness.can_begin_sourcing,
    )
    approval_note = st.text_input("Approval or revision note")

    approved_brief = apply_recruiter_approval(
        brief=brief,
        readiness=readiness,
        recruiter_approves=recruiter_approves,
        approval_note=approval_note,
    )
    approval_routing = route_next_action(approved_brief, readiness)
    st.write(f"**Approval routing action:** {approval_routing.recommended_action.value}")
    st.write(f"**Approval status:** {approved_brief.approval_status.value}")

    if approved_brief.approval_status == ApprovalStatus.APPROVED:
        launch_plan = estimate_sourcing_launch_plan(approved_brief, readiness)
        st.subheader("Rule-Based Launch Readiness Plan")
        col_x, col_y = st.columns(2)
        col_x.metric("Launch difficulty band", launch_plan.difficulty_band)
        col_y.metric(
            "Prototype shortlist estimate",
            launch_plan.prototype_shortlist_estimate,
        )
        st.write("**Difficulty factors**")
        render_list(launch_plan.difficulty_factors)
        st.write("**Recommended initial search focus**")
        render_list(launch_plan.recommended_initial_search_focus)
        st.write("**Remaining assumptions**")
        render_list(launch_plan.remaining_assumptions)
        st.write(f"**Calibration note:** {launch_plan.calibration_note}")
        st.caption(launch_plan.launch_rationale)
