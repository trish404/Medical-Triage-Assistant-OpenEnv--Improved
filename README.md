---
title: Medical Triage Assistant OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# 🏥 Medical Triage Assistant — OpenEnv v3.0

> **10-task emergency department environment with real clinical protocols, multi-step sequential decisions, and sophisticated grading functions.**


## Task Roster

| # | Task | Difficulty | Protocol | Max Steps |
|---|------|-----------|---------|-----------|
| 1 | `esi_assignment` | Easy | Emergency Severity Index v4 | 5 |
| 2 | `intake_interview` | Medium | OLDCARTS structured intake | 15 |
| 3 | `queue_management` | Hard | Priority queue + deterioration events | 20 |
| 4 | `medication_check` | Medium | Allergy cross-reactivity + interactions | 12 |
| 5 | `discharge_planning` | Medium | Instructions / warnings / timing | 18 |
| 6 | `mass_casualty` | **Hard** | START MCI triage (RED/YELLOW/GREEN/BLACK) | 18 |
| 7 | `sepsis_screening` | **Hard** | qSOFA criteria + full bundle ordering | 22 |
| 8 | `bed_allocation` | Medium | Resource-constraint bed matching | 15 |
| 9 | `shift_handoff` | Medium | SBAR nurse-to-nurse handoff | 42 |
| 10 | `consent_assessment` | Medium | Capacity evaluation + consent pathway | 36 |

---

## Baseline Benchmark Results

Evaluated on seed 42. Scores are in range (0, 1).

| Task | Llama 3.3 70B (Groq) | Llama 3.1 8B (Groq) |
|------|---------------------|---------------------|
| esi_assignment | **0.999** | **0.999** |
| intake_interview | **0.910** | **0.910** |
| queue_management | 0.460 | **0.800** |
| medication_check | **0.700** | 0.600 |
| discharge_planning | **0.400** | 0.180 |
| mass_casualty | **0.999** | 0.833 |
| sepsis_screening | **0.999** | 0.857 |
| bed_allocation | **0.999** | 0.842 |
| shift_handoff | **0.999** | 0.667 |
| consent_assessment | **0.642** | 0.242 |
| **AGGREGATE** | **0.811** | **0.693** |

---

## Grading Functions

| Task | Method |
|------|--------|
| `esi_assignment` | Diff-based: exact=1.0, off-by-1=0.5, off-by-2=0.2 |
| `intake_interview` | 70% field completeness + 30% efficiency |
| `queue_management` | 60% Kendall-τ + 20% critical-first + 20% deterioration response |
| `medication_check` | 30% allergy recall + 10% precision + 30% interaction recall + 10% precision − 0.15 per wrong approval |
| `discharge_planning` | 50% instruction coverage + 30% warning coverage + 20% timing accuracy |
| `mass_casualty` | Fraction correct − 0.15 per RED↔BLACK critical swap |
| `sepsis_screening` | 60% F1 (flag precision/recall) + 40% bundle completeness |
| `bed_allocation` | Type correctness − 0.10 per over-allocated bed + priority bonus |
| `shift_handoff` | Average mandatory SBAR field coverage + 10% optional bonus |
| `consent_assessment` | 50% step coverage + 50% correct consent path − 0.30 per dangerous shortcut |

---

## Clinical Protocols Implemented

- **ESI v4** — Emergency Severity Index (ACEP/ENA standard). 5-level triage based on life threat, vital signs, and anticipated resource needs.
- **START MCI** — Simple Triage and Rapid Treatment for mass casualty incidents. 5-step decision tree producing RED/YELLOW/GREEN/BLACK tags.
- **qSOFA** — quick Sequential Organ Failure Assessment for sepsis screening. Score based on RR, GCS, and SBP.
- **SBAR** — Situation-Background-Assessment-Recommendation nurse-to-nurse handoff protocol.
- **OLDCARTS** — Structured symptom intake: Onset, Location, Duration, Character, Aggravating, Relieving, Timing, Severity.
- **Capacity Assessment** — 4-criterion legal standard for informed consent with 6 distinct consent pathways.

---

## API Usage

```python
import requests

# Reset a task
r = requests.post("https://trishie-medical-triage-openenv-v2.hf.space/tasks/sepsis_screening/reset",
                  json={"seed": 42})
obs = r.json()
print(obs["message"])

# Take a step
r = requests.post("https://trishie-medical-triage-openenv-v2.hf.space/tasks/sepsis_screening/step",
                  json={"action_type": "text", "content": "screen:SS001"})
result = r.json()
print(result["reward"], result["done"])
```

---

## Running Inference

```bash
# Install dependencies
uv sync

# Run with Groq (free)
API_BASE_URL=https://api.groq.com/openai/v1 MODEL_NAME=llama-3.3-70b-versatile HF_TOKEN=your_groq_key ENV_BASE_URL=http://localhost:7860 uv run python inference.py
```

---

## Project Structure

```
├── medical_triage_env.py   # All 10 task environments + graders
├── inference.py            # LLM agent with chain-of-thought + state tracking
├── main.py                 # FastAPI OpenEnv server
├── dashboard.html          # Interactive web UI served at /web
├── openenv.yaml            # Environment manifest
├── pyproject.toml          # Dependencies
└── Dockerfile              # Container definition
```

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/web` | GET | Interactive dashboard UI |
| `/docs` | GET | Auto-generated API docs |
| `/health` | GET | Health check |
| `/tasks` | GET | List all 10 tasks |
| `/tasks/{task}/reset` | POST | Reset a task episode |
| `/tasks/{task}/step` | POST | Take an action |
| `/tasks/{task}/grading_info` | GET | Grading breakdown |

---

*Built for the OpenEnv Hackathon — benchmarking AI agents on high-stakes clinical decision-making.*
