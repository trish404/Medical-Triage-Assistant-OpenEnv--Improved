"""
Medical Triage Assistant — OpenEnv Server  v3.0
================================================
FastAPI server exposing all 10 clinical tasks via the OpenEnv HTTP API.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from medical_triage_env import (
    MedicalTriageEnv, TaskType, TriageAction, create_environment, TASK_DIFFICULTY,
)

app = FastAPI(
    title="Medical Triage Assistant — OpenEnv",
    description=(
        "10-task emergency department triage environment. "
        "Tasks cover ESI assignment, patient intake, queue management, "
        "medication safety, discharge planning, mass casualty START triage, "
        "sepsis screening (qSOFA), bed allocation, SBAR shift handoff, "
        "and consent/capacity assessment."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active environment instances keyed by task name
_envs: Dict[str, MedicalTriageEnv] = {}

TASK_METADATA = {
    t.value: {
        "difficulty": TASK_DIFFICULTY[t],
        "description": {
            "esi_assignment":     "Assign Emergency Severity Index (1-5) using ESI v4 decision tree",
            "intake_interview":   "Conduct structured OLDCARTS patient intake interview",
            "queue_management":   "Prioritise ED waiting room with real-time deterioration events",
            "medication_check":   "Screen proposed medications for allergy cross-reactivity and drug interactions",
            "discharge_planning": "Build personalised discharge instructions, warnings, and follow-up timing",
            "mass_casualty":      "Apply START MCI triage protocol (RED/YELLOW/GREEN/BLACK) across multiple victims",
            "sepsis_screening":   "Screen ward patients using qSOFA criteria and initiate sepsis bundles",
            "bed_allocation":     "Allocate typed ED beds to patients under inventory constraints",
            "shift_handoff":      "Complete SBAR structured nurse-to-nurse handoff for critical patients",
            "consent_assessment": "Assess capacity and navigate consent pathways across 3 clinical scenarios",
        }[t.value],
    }
    for t in TaskType
}


# ── Request/Response Models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: int = 42

class StepRequest(BaseModel):
    action_type:       str = "text"
    content:           str
    target_patient_id: Optional[str] = None


# ── Root / Health ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Medical Triage Assistant OpenEnv",
        "version": "3.0.0",
        "tasks": list(TASK_METADATA.keys()),
        "total_tasks": len(TASK_METADATA),
        "unique_differentiators": [
            "10 tasks vs competitors' 3-5",
            "Multi-step sequential decisions (not single-step classification)",
            "Real clinical protocols: ESI, START MCI, qSOFA, SBAR, Capacity Assessment",
            "Dynamic deterioration events in queue management",
            "Sophisticated graders: Kendall-tau, F1, partial credit, penalty scaling",
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok", "active_envs": list(_envs.keys())}

@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_METADATA}

@app.get("/tasks/{task_name}")
def task_info(task_name: str):
    if task_name not in TASK_METADATA:
        raise HTTPException(404, f"Unknown task '{task_name}'")
    return TASK_METADATA[task_name]


# ── OpenEnv API ────────────────────────────────────────────────────────────────

@app.post("/tasks/{task_name}/reset")
def reset_task(task_name: str, request: ResetRequest = ResetRequest()):
    if task_name not in TASK_METADATA:
        raise HTTPException(404, f"Unknown task '{task_name}'")
    env = create_environment(task_name, seed=request.seed)
    _envs[task_name] = env
    obs = env.reset()
    return {
        "task": task_name,
        "seed": request.seed,
        "observation": obs.model_dump(),
        "message": obs.message,
        "available_actions": obs.available_actions,
        "max_steps": env.max_steps,
    }


@app.post("/tasks/{task_name}/step")
def step_task(task_name: str, request: StepRequest):
    if task_name not in _envs:
        raise HTTPException(
            400,
            f"Task '{task_name}' not initialised. Call /tasks/{task_name}/reset first.",
        )
    env = _envs[task_name]
    action = TriageAction(
        action_type=request.action_type,
        content=request.content,
        target_patient_id=request.target_patient_id,
    )
    try:
        result = env.step(action)
    except Exception as e:
        raise HTTPException(500, f"Environment error: {e}")

    return {
        "observation":       result.observation.model_dump(),
        "message":           result.observation.message,
        "reward":            result.reward,
        "done":              result.done,
        "info":              result.info,
        "step_number":       env.step_count,
        "max_steps":         env.max_steps,
        "available_actions": result.observation.available_actions,
    }


# ── Convenience: /reset and /step (legacy single-task API) ────────────────────

@app.post("/reset")
def reset_default(request: ResetRequest = ResetRequest()):
    """Reset the first task (esi_assignment) for legacy compatibility."""
    return reset_task("esi_assignment", request)

@app.post("/step")
def step_default(request: StepRequest):
    """Step the first task (esi_assignment) for legacy compatibility."""
    return step_task("esi_assignment", request)


# ── Leaderboard / Scoring Summary ────────────────────────────────────────────

@app.get("/tasks/{task_name}/grading_info")
def grading_info(task_name: str):
    grading_notes = {
        "esi_assignment":     "Exact: 1.0 | Off-by-1: 0.5 | Off-by-2: 0.2 | Larger: 0.0",
        "intake_interview":   "70% field completeness + 30% efficiency (penalise excess steps)",
        "queue_management":   "60% Kendall-τ order + 20% critical patients first + 20% deterioration response",
        "medication_check":   "30% allergy recall + 10% allergy precision + 30% interaction recall + 10% interaction precision + 20% base — 0.15 per wrong approval",
        "discharge_planning": "50% instruction completeness + 30% warning coverage + 20% timing accuracy",
        "mass_casualty":      "Fraction correct tags — 0.15 per RED↔BLACK critical swap",
        "sepsis_screening":   "60% F1 (flag precision/recall) + 40% bundle completeness — average across flagged patients",
        "bed_allocation":     "Type correctness — 0.10 per over-allocated bed + 0.10 bonus for high-priority patients assigned first",
        "shift_handoff":      "Average mandatory field coverage across patients + 10% optional bonus",
        "consent_assessment": "50% step coverage + 50% correct consent path — 0.30 per dangerous shortcut — averaged across 3 scenarios",
    }
    if task_name not in grading_notes:
        raise HTTPException(404, f"Unknown task '{task_name}'")
    return {"task": task_name, "grading": grading_notes[task_name]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


# ── Dashboard UI (served at /web for HuggingFace Spaces) ─────────────────────
from fastapi.responses import HTMLResponse
from pathlib import Path

@app.get("/web", response_class=HTMLResponse)
@app.get("/web/", response_class=HTMLResponse)
def dashboard():
    """Serve the interactive dashboard UI."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Dashboard not found — ensure dashboard.html is present.</h1>", 404)