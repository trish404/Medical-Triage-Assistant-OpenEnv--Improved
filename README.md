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
| 6 | `mass_casualty` | Hard | START MCI triage (RED/YELLOW/GREEN/BLACK) | 14 |
| 7 | `sepsis_screening` | Hard | qSOFA criteria + full bundle ordering | 16 |
| 8 | `bed_allocation` | Medium | Resource-constraint bed matching | 15 |
| 9 | `shift_handoff` | Medium | SBAR nurse-to-nurse handoff protocol | 22 |
| 10 | `consent_assessment` | Medium | Capacity evaluation + consent pathway | 12 |

---

## Baseline Benchmark

| Task | Claude Sonnet | GPT-4o | Llama 3.3 70B | Random |
|------|--------------|--------|---------------|--------|
| esi_assignment | 0.92 | 0.88 | 0.80 | 0.20 |
| intake_interview | 0.87 | 0.84 | 0.73 | 0.10 |
| queue_management | 0.78 | 0.72 | 0.61 | 0.15 |
| medication_check | 0.84 | 0.79 | 0.68 | 0.05 |
| discharge_planning | 0.81 | 0.76 | 0.65 | 0.12 |
| mass_casualty | 0.83 | 0.75 | 0.62 | 0.25 |
| sepsis_screening | 0.79 | 0.71 | 0.58 | 0.08 |
| bed_allocation | 0.86 | 0.80 | 0.69 | 0.18 |
| shift_handoff | 0.82 | 0.74 | 0.60 | 0.05 |
| consent_assessment | 0.88 | 0.81 | 0.69 | 0.10 |
| **OVERALL** | **0.840** | **0.780** | **0.665** | **0.128** |

---

## Grading Functions

| Task | Method |
|------|--------|
| esi_assignment | Diff-based: exact=1.0, off-by-1=0.5, off-by-2=0.2 |
| intake_interview | 70% field completeness + 30% efficiency |
| queue_management | 60% Kendall-tau + 20% critical-first + 20% deterioration response |
| medication_check | 30% allergy recall + 10% precision + 30% interaction recall + 10% precision - 0.15 per wrong approval |
| discharge_planning | 50% instruction + 30% warning + 20% timing accuracy |
| mass_casualty | Fraction correct - 0.15 per RED/BLACK swap |
| sepsis_screening | 60% F1 + 40% bundle completeness |
| bed_allocation | Type correctness - 0.10 per over-alloc + 0.10 priority bonus |
| shift_handoff | Avg mandatory field coverage + 10% optional bonus |
| consent_assessment | 50% steps + 50% correct path - 0.30 per dangerous shortcut |

---

## Quick Start

```bash
uv sync
uv run server
# then in another terminal:
ANTHROPIC_API_KEY=your_key python inference.py
```

---

## Clinical Protocols

- **ESI v4** — Emergency Severity Index (ACEP/ENA standard)
- **START** — Simple Triage and Rapid Treatment for mass casualty incidents
- **qSOFA** — quick SOFA for sepsis screening
- **SBAR** — Situation-Background-Assessment-Recommendation
- **OLDCARTS** — structured symptom intake
- **Capacity Assessment** — 4-criterion legal standard for informed consent
