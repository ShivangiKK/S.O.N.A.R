# S.O.N.A.R.

**Sourcing Operations & Navigation via Agentic Recruitment**

Track: **Agents for Business**

S.O.N.A.R. is a recruiter-controlled Sourcing Readiness Agent. It accepts a job description, recruiter intake notes, and optional clarification answers. Gemini creates a structured, evidence-backed sourcing brief, then deterministic Python tools calculate sourcing readiness, classify gaps, route the next action, require recruiter approval, and create a rule-based launch readiness plan.

## Problem

Recruiting teams often begin sourcing before a role is fully defined. Job descriptions and intake notes may leave critical details unclear, including must-have skills, acceptable alternatives, location constraints, compensation, sponsorship, target hiring date, and screening evidence.

Those gaps create recruiter rework, inconsistent sourcing, repeated hiring-manager clarification, and slower time to shortlist.

## Solution

S.O.N.A.R. helps recruiters decide whether a role is ready to source before candidate search begins.

The app:

1. accepts the job description, recruiter notes, and optional clarification answers;
2. uses Gemini only as a Role Intake Agent for structured extraction into a Pydantic `SourcingBrief`;
3. uses deterministic Python tools to score readiness and classify gaps;
4. routes the next action based on readiness and approval state;
5. requires human recruiter approval before launch planning;
6. creates a transparent, rule-based sourcing launch plan after approval.

This is not candidate sourcing, resume matching, outreach automation, authentication, or a recruiting database.

## Architecture

```text
Recruiter inputs
JD + intake notes + clarification answers
        |
        v
Role Intake Agent
Gemini + Pydantic structured output
        |
        v
SourcingBrief
Evidence-backed structured state
        |
        v
Readiness Tool
Deterministic scoring and gap classification
        |
        v
Approval Router
Deterministic next-action routing
        |
        v
Human Recruiter Approval
Required approval gate
        |
        v
Launch Readiness Tool
Rule-based prototype launch plan
```

## Why This Is Agentic

S.O.N.A.R. is not a one-shot LLM summary. The workflow separates agent reasoning, state, tools, routing, and human approval:

- **Role Intake Agent:** Gemini extracts normalized, evidence-backed role requirements into a validated schema.
- **Structured state:** `SourcingBrief` acts as the workflow memory passed to downstream tools.
- **Deterministic tools:** Python code, not Gemini, calculates readiness, classifies gaps, routes actions, and estimates launch difficulty.
- **Human-in-the-loop control:** The app cannot approve sourcing automatically. A recruiter must explicitly approve, and approval is blocked if readiness rules say sourcing cannot begin.
- **Transparent launch planning:** The shortlist estimate is labeled as a prototype rule-based estimate, not a statistical forecast.

## Hackathon Concepts Applied

- **Agent / multi-agent workflow:** Role Intake Agent, Readiness Tool, Approval Router, Human Approval Gate, and Launch Readiness Tool each have separate responsibilities.
- **Agent skills / tools:** Gemini extraction is combined with deterministic business-rule tools.
- **Security features:** API keys are read from Streamlit secrets and never hard-coded.
- **Deployability:** The app is a self-contained Streamlit MVP with a minimal dependency file.
- **Evaluation-ready design:** The structured schema, evidence count, gap lists, deterministic scores, and routing action make the workflow inspectable and testable.

## Setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` intentionally contains only:

```text
streamlit
google-genai==2.10.0
pydantic==2.12.5
```

## Streamlit Secrets

Do not hard-code your Gemini API key.

For local development, create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

For Streamlit Community Cloud, add the same key in the app's Secrets settings:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

The app reads the key with:

```python
st.secrets["GEMINI_API_KEY"]
```

The Colab notebook used Colab Secrets. The Streamlit MVP does not import or use `google.colab.userdata`.

## Run Locally

From this directory:

```bash
streamlit run app.py
```

Use `sample_inputs/senior_data_engineer_demo.txt` as a demo case. Paste the sections into the corresponding Streamlit text areas.

## Outputs

The app displays:

- readiness score;
- readiness status;
- whether sourcing can begin;
- recruiter message;
- blocking, high-impact, and helpful gaps;
- clarification questions;
- role summary;
- must-have skills;
- locations;
- compensation;
- visa sponsorship;
- evidence count;
- approval routing action;
- launch difficulty band;
- prototype shortlist estimate;
- calibration note.

## Limitations

- The prototype does not source candidates.
- The prototype does not match resumes.
- The prototype does not include authentication.
- The prototype does not store data in a database.
- Gemini extraction quality depends on the clarity of the provided inputs.
- The launch estimate is rule-based and not a trained statistical forecast.
- The readiness rules are transparent but should be calibrated before production use.

## Future Work

- Add evaluation sets for field-level extraction precision, recall, and F1.
- Calibrate readiness thresholds with historical recruiting outcomes.
- Add regression checks to ensure later clarification passes preserve earlier useful brief state.
- Add production monitoring for unsupported claims and protected-trait sourcing criteria.
- Add optional integrations with applicant tracking systems after the readiness and approval workflow is validated.
- Extend downstream modules for sourcing, screening, or matching only after recruiter approval and governance requirements are defined.
