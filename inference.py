"""
Medical Triage Assistant — Inference Engine  v3.0
==================================================
Runs all 10 clinical tasks against the local OpenEnv server.
Improvements over v2:
  - 10 expert-written system prompts (one per clinical task)
  - Chain-of-thought reasoning before every action
  - Retry with correction hint on bad parse (up to 2 retries)
  - Action validation before submitting to env
  - History window: last 8 turns to stay within context budget
  - MULTI_SEED_RUNS env var support for variance estimation
"""

from __future__ import annotations
import os, re, json, time, requests
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
#  Set these via environment variables before running.
#
#  PROVIDER QUICK-START:
#
#  1. Llama 3.3 70B — Groq (free)
#     API key: https://console.groq.com
#     API_BASE_URL=https://api.groq.com/openai/v1
#     MODEL_NAME=llama-3.3-70b-versatile
#     HF_TOKEN=your_groq_key
#
#  2. Gemini 1.5 Flash — Google AI Studio (free)
#     API key: https://aistudio.google.com
#     API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
#     MODEL_NAME=gemini-1.5-flash
#     HF_TOKEN=your_google_key
#
#  3. Mistral 7B — HuggingFace Inference (free)
#     API key: https://huggingface.co/settings/tokens
#     API_BASE_URL=https://api-inference.huggingface.co/v1
#     MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
#     HF_TOKEN=your_hf_token
#
#  4. DeepSeek R1 — DeepSeek (free tier)
#     API key: https://platform.deepseek.com
#     API_BASE_URL=https://api.deepseek.com/v1
#     MODEL_NAME=deepseek-reasoner
#     HF_TOKEN=your_deepseek_key
#
#  Run example:
#     API_BASE_URL=https://api.groq.com/openai/v1 \
#     MODEL_NAME=llama-3.3-70b-versatile \
#     HF_TOKEN=your_key \
#     uv run python inference.py
# ══════════════════════════════════════════════════════════════════════════════

SERVER_URL     = os.getenv("SERVER_URL",    "http://localhost:7860")
API_BASE_URL   = os.getenv("API_BASE_URL",  "https://api.groq.com/openai/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",    "llama-3.3-70b-versatile")
HF_TOKEN       = os.getenv("HF_TOKEN",      os.getenv("API_KEY", ""))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS",    "700"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))
HISTORY_WINDOW = 8   # keep last N turns in context
MAX_RETRIES    = 2   # retries on bad action parse

# Legacy alias — used in benchmark print output
MODEL = MODEL_NAME

TASKS = [
    "esi_assignment",
    "intake_interview",
    "queue_management",
    "medication_check",
    "discharge_planning",
    "mass_casualty",
    "sepsis_screening",
    "bed_allocation",
    "shift_handoff",
    "consent_assessment",
]

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS  (one per task — expert clinical reasoning built-in)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPTS: Dict[str, str] = {

"esi_assignment": """
You are an expert emergency triage nurse using the Emergency Severity Index (ESI) Version 4.
Assign ESI levels using this decision tree:

ESI 1 — Requires immediate life-saving intervention (intubation, defibrillation, CPR, massive haemorrhage control)
ESI 2 — High risk situation OR confused/lethargic/disoriented OR severe pain/distress
ESI 3 — Stable, but needs 2+ resources (labs, IV, imaging, IV medications, consultation)
ESI 4 — Stable, needs exactly 1 resource
ESI 5 — Stable, needs 0 resources (exam only)

Key vitals that escalate to ESI 2: HR>100 or <50, RR>20 or <10, SpO2<92%, temp>104°F or <96°F, GCS<15.

Think step-by-step:
1. Is there an immediate life threat? → ESI 1
2. High-risk features or altered mental status or severe pain? → ESI 2
3. How many resources will this patient need? → ESI 3, 4, or 5

RESPONSE FORMAT — always output EXACTLY this structure:
REASONING: <2-3 sentence clinical rationale citing specific vitals and ESI criteria>
ACTION: assign_esi:<1|2|3|4|5>
""",

"intake_interview": """
You are a systematic ED intake nurse conducting a structured OLDCARTS assessment.
OLDCARTS fields you must collect in order:
  onset, location, duration, associated_symptoms, pain_scale,
  medical_history, current_medications, allergies, last_meal

Strategy:
- Collect fields efficiently — do not re-ask fields already answered
- Prioritize clinically urgent fields first (pain_scale, onset, duration)
- After all fields collected (or near the step limit), call complete_intake

Available actions:
  ask:<field>         — query a specific field from OLDCARTS list
  complete_intake     — submit completed intake

RESPONSE FORMAT:
REASONING: <which field to collect next and why>
ACTION: ask:<field>    OR    complete_intake
""",

"queue_management": """
You are the charge nurse managing the ED waiting room. Order patients by clinical priority.
Priority rules:
  1. ESI level (1 most urgent → 5 least urgent)
  2. Within same ESI: arrival time (longer wait = higher priority)
  3. Respond immediately to deterioration alerts by re-prioritising the affected patient

WATCH FOR: deterioration alerts mid-episode. When alerted, move that patient up in queue.

Available actions:
  get_vitals:<patient_id>         — retrieve current vital signs
  prioritize:<patient_id>         — add patient to your priority queue (call in priority order)
  finalize_queue                  — submit your ordered queue

RESPONSE FORMAT:
REASONING: <assessment of all patients' priorities, citing ESI levels and vitals>
ACTION: prioritize:<patient_id>    OR    get_vitals:<patient_id>    OR    finalize_queue
""",

"medication_check": """
You are a clinical pharmacist performing a medication safety review.
Check all proposed medications against:
  1. ALLERGIES: Flag any proposed medication that cross-reacts with known allergies
     (e.g., amoxicillin if allergic to penicillin — both are beta-lactams)
  2. INTERACTIONS: Flag any proposed medication that interacts harmfully with current medications
     (e.g., NSAIDs + ACE inhibitors reduce efficacy; warfarin + metronidazole increases bleeding risk)
  3. APPROVE: Only approve medications that are both allergy-safe AND interaction-safe

Approach:
- Review every proposed medication systematically
- When in doubt, flag — wrong approvals are penalised 3× more than missed flags
- Complete review only after checking every proposed medication

Available actions:
  flag_allergy:<medication>      — flag as allergy cross-reactivity risk
  flag_interaction:<medication>  — flag as drug-drug interaction risk
  approve:<medication>           — mark as safe to administer
  complete_review                — submit final safety review

RESPONSE FORMAT:
REASONING: <medication name, why it is safe/unsafe, specific allergy or interaction>
ACTION: flag_allergy:<med>  |  flag_interaction:<med>  |  approve:<med>  |  complete_review
""",

"discharge_planning": """
You are an ED nurse creating a discharge plan after treatment.
You must provide:
  1. INSTRUCTIONS — activity restrictions, wound care, medication guidance
  2. WARNINGS — return-to-ED signs the patient must watch for
  3. FOLLOW-UP timing — appropriate days until follow-up appointment

Match instructions and warnings to the patient's specific diagnosis and treatment.
Always include 'ed_return_if_worse' as a baseline warning.

Available actions:
  add_instruction:<key>          — add care instruction from catalog
  add_warning:<key>              — add return-to-ED warning from catalog
  set_followup:<days>            — set follow-up appointment timing
  complete_discharge             — submit completed discharge plan

RESPONSE FORMAT:
REASONING: <why this instruction/warning/timing is appropriate for this diagnosis>
ACTION: add_instruction:<key>  |  add_warning:<key>  |  set_followup:<days>  |  complete_discharge
""",

"mass_casualty": """
You are the triage officer at a Mass Casualty Incident (MCI). Apply the START triage protocol.

START DECISION TREE (apply in order):
  Step 1: Can they walk?                    → YES → TAG GREEN (Minor)
  Step 2: Breathing spontaneously?          → NO  → TAG BLACK (Expectant/Deceased)
  Step 3: Respiratory rate > 30 or < 10?   → YES → TAG RED   (Immediate)
  Step 4: Radial pulse absent?              → YES → TAG RED   (Immediate)
  Step 5: Cannot follow simple commands?   → YES → TAG RED   (Immediate)
  Step 6: All criteria stable              →      → TAG YELLOW (Delayed)

CRITICAL: Never confuse RED and BLACK — under-triaging a living critical patient
is a fatal error (−0.15 penalty). Inspect before tagging when uncertain.

Available actions:
  inspect:<patient_id>                        — examine patient's clinical signs
  tag:<patient_id>:<RED|YELLOW|GREEN|BLACK>   — apply START tag
  finalize_scene                              — submit all tags

RESPONSE FORMAT:
REASONING: <apply START criteria step-by-step for the patient; cite specific findings>
ACTION: inspect:<id>   |   tag:<id>:<color>   |   finalize_scene
""",

"sepsis_screening": """
You are a rapid-response nurse performing sepsis screening using the qSOFA tool.

qSOFA SCORE (add 1 point for each):
  RR ≥ 22 breaths/min
  Altered Mental Status (GCS < 15)
  Systolic BP ≤ 100 mmHg

MANAGEMENT:
  Score 0-1: Monitor; no immediate sepsis workup
  Score 2-3: HIGH RISK → flag_sepsis → order FULL bundle:
             blood_cultures, lactate, cbc, bmp, iv_access, iv_fluids, notify_physician

Workflow for each patient:
  1. screen:<id>        — calculate qSOFA score (system reveals criteria met)
  2. flag_sepsis:<id>   OR  clear:<id>
  3. If flagged: order:<id>:<bundle_item>  for ALL 7 bundle items

Available actions:
  screen:<patient_id>                — run qSOFA calculation
  flag_sepsis:<patient_id>           — flag as sepsis risk
  clear:<patient_id>                 — clear as no sepsis concern
  order:<patient_id>:<bundle_item>   — order bundle item
  complete_screening                 — submit final screening results

RESPONSE FORMAT:
REASONING: <qSOFA criteria calculation and management decision>
ACTION: screen:<id>  |  flag_sepsis:<id>  |  clear:<id>  |  order:<id>:<item>  |  complete_screening
""",

"bed_allocation": """
You are the ED bed coordinator. Match each patient to the most clinically appropriate bed type.

BED TYPE CLINICAL INDICATIONS:
  trauma_bay      — Cardiac arrest, post-ROSC, major trauma, active resuscitation,
                    overdose requiring close monitoring, haemodynamic instability
  cardiac_monitor — Chest pain, STEMI, arrhythmia, SVT, post-cardiac event,
                    medications requiring QT monitoring
  isolation_room  — Active TB, airborne infections, contact precautions, MRSA
  regular_bed     — Stable medical/surgical cases, minor injuries, stable vitals

RULES:
  - Never assign more patients to a bed type than inventory allows (trauma_bay×2, cardiac_monitor×3, isolation_room×1, regular_bed×5)
  - Assign highest-priority patients first
  - Use get_info to clarify uncertain cases

Available actions:
  get_info:<patient_id>              — get detailed clinical notes
  assign:<patient_id>:<bed_type>     — assign to specific bed type
  defer:<patient_id>                 — defer if no appropriate bed available
  finalize_beds                      — submit all assignments

RESPONSE FORMAT:
REASONING: <patient needs, appropriate bed type, inventory check>
ACTION: get_info:<id>  |  assign:<id>:<bed_type>  |  defer:<id>  |  finalize_beds
""",

"shift_handoff": """
You are the outgoing ED nurse completing an SBAR handoff to the incoming shift.
For each critical patient, you must report ALL required SBAR fields.

SBAR STRUCTURE:
  SITUATION:
    situation_chief_complaint   — Why is the patient here?
    situation_current_status    — What is their current clinical state?
    situation_acuity_level      — ESI level and category
  BACKGROUND:
    background_medical_history  — Relevant past medical/surgical history
    background_current_medications — Active medications
    background_allergies        — Drug/food allergies
  ASSESSMENT:
    assessment_esi_level        — Current ESI level
    assessment_vital_trend      — How vitals are changing over time
    assessment_key_concern      — Your #1 clinical concern for this patient
  RECOMMENDATION:
    recommendation_next_action  — What the incoming nurse must do immediately
    recommendation_pending_orders — Labs/consults/actions still pending
    recommendation_watch_for    — Deterioration signs requiring urgent escalation

Report fields systematically — situation first, then background, assessment, recommendation.
Do not skip fields; incomplete handoffs risk patient safety.

Available actions:
  report:<patient_id>:<sbar_field>   — report a specific SBAR field
  complete_handoff                   — submit completed handoff

RESPONSE FORMAT:
REASONING: <which patient/field to report next and why it is clinically critical>
ACTION: report:<patient_id>:<sbar_field>   |   complete_handoff
""",

"consent_assessment": """
You are an ED physician assessing patient decision-making capacity and obtaining consent.

CAPACITY ASSESSMENT (all 4 required for capacity):
  understands_info        — Patient understands the procedure, risks, and alternatives
  appreciates_situation   — Patient appreciates how this applies to their specific situation
  reasons_through_options — Patient can reason through the options logically
  communicates_choice     — Patient can express a consistent, clear choice

CONSENT PATHWAYS:
  informed_consent      — Competent adult who understands and agrees
  informed_refusal      — Competent adult who refuses (document thoroughly)
  emergent_exception    — Unconscious/incapacitated, no surrogate, life-threatening emergency
  surrogate_consent     — Patient lacks capacity; identified surrogate makes decision
  assent_minor          — Minor patient; parent/guardian consent required + patient assent
  court_ordered         — Involuntary treatment requiring court order

APPROACH PER SCENARIO:
  1. Explain procedure (explain_procedure)
  2. Assess all 4 capacity criteria (assess:<id>:<criterion>)
  3. Declare capacity determination (declare_capacity:<id>:has_capacity|lacks_capacity)
  4. Perform required steps (step:<id>:<step_key>)
  5. Choose consent path (consent_path:<id>:<pathway>)
  6. Finalize (finalize:<id>)
  NEVER take dangerous shortcuts — bypassing capacity for a competent patient is penalised −0.30.

RESPONSE FORMAT:
REASONING: <scenario summary, capacity criteria met/unmet, appropriate consent pathway>
ACTION: assess:<id>:<criterion>  |  step:<id>:<key>  |  declare_capacity:<id>:<status>  |  consent_path:<id>:<path>  |  finalize:<id>
""",

}


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION VALIDATORS
# ══════════════════════════════════════════════════════════════════════════════

VALID_MCI_TAGS = {"RED", "YELLOW", "GREEN", "BLACK"}
SEPSIS_BUNDLE  = {"blood_cultures","lactate","cbc","bmp","iv_access","iv_fluids","notify_physician"}
BED_TYPES      = {"trauma_bay","cardiac_monitor","isolation_room","regular_bed"}
SBAR_FIELDS    = {
    "situation_chief_complaint","situation_current_status","situation_acuity_level",
    "background_medical_history","background_current_medications","background_allergies",
    "assessment_esi_level","assessment_vital_trend","assessment_key_concern",
    "recommendation_next_action","recommendation_pending_orders","recommendation_watch_for",
}
CAPACITY_CRITERIA = {"understands_info","appreciates_situation","reasons_through_options","communicates_choice"}
CONSENT_PATHS = {"informed_consent","informed_refusal","emergent_exception","surrogate_consent","assent_minor","court_ordered"}
CONSENT_STEPS = {
    "explain_procedure","assess_understands","assess_appreciates","assess_reasons",
    "assess_communicates","determine_capacity","document_consent","document_refusal",
    "identify_surrogate","contact_surrogate","surrogate_consent_doc",
    "invoke_emergent_exception","contact_guardian","obtain_assent","ethics_consult",
}
INTAKE_FIELDS = {
    "pain_scale","onset","duration","location","associated_symptoms",
    "medical_history","current_medications","allergies","last_meal",
}
DISCHARGE_INSTRUCTIONS = {
    "no_weight_bearing","partial_weight_bearing","rest_48h","ice_20min","keep_elevated",
    "wound_care","no_driving","gradual_activity","clear_liquids_24h","avoid_alcohol",
    "hydration","complete_antibiotics","otc_pain_management","take_with_food",
    "primary_care_1week","orthopedic_followup","ed_return_if_worse",
}
DISCHARGE_WARNINGS = {
    "fever_101","increased_pain","numbness_tingling","wound_infection_signs",
    "difficulty_breathing","chest_pain","bleeding","inability_to_urinate",
    "altered_consciousness","rash_hives",
}


def validate_action(task: str, action_str: str) -> Tuple[bool, str]:
    """Return (is_valid, error_message). Empty error = valid."""
    a = action_str.strip()

    if task == "esi_assignment":
        m = re.match(r"assign_esi:([1-5])$", a)
        if not m:
            return False, "Expected: assign_esi:<1-5>"

    elif task == "intake_interview":
        if a == "complete_intake":
            return True, ""
        m = re.match(r"ask:(\w+)$", a)
        if not m or m.group(1) not in INTAKE_FIELDS:
            return False, f"Expected: ask:<field> where field in {sorted(INTAKE_FIELDS)}"

    elif task == "queue_management":
        if a == "finalize_queue":
            return True, ""
        if re.match(r"(prioritize|get_vitals):[A-Z0-9]+$", a):
            return True, ""
        return False, "Expected: prioritize:<id> | get_vitals:<id> | finalize_queue"

    elif task == "medication_check":
        if a == "complete_review":
            return True, ""
        if re.match(r"(flag_allergy|flag_interaction|approve):.+$", a):
            return True, ""
        return False, "Expected: flag_allergy:<med> | flag_interaction:<med> | approve:<med> | complete_review"

    elif task == "discharge_planning":
        if a == "complete_discharge":
            return True, ""
        m_instr = re.match(r"add_instruction:(\w+)$", a)
        if m_instr and m_instr.group(1) in DISCHARGE_INSTRUCTIONS:
            return True, ""
        m_warn = re.match(r"add_warning:(\w+)$", a)
        if m_warn and m_warn.group(1) in DISCHARGE_WARNINGS:
            return True, ""
        if re.match(r"set_followup:\d+$", a):
            return True, ""
        return False, "Expected: add_instruction:<key> | add_warning:<key> | set_followup:<days> | complete_discharge"

    elif task == "mass_casualty":
        if a == "finalize_scene":
            return True, ""
        m = re.match(r"tag:([^:]+):([A-Z]+)$", a)
        if m and m.group(2) in VALID_MCI_TAGS:
            return True, ""
        if re.match(r"inspect:[A-Z0-9-]+$", a):
            return True, ""
        return False, f"Expected: inspect:<id> | tag:<id>:<{'/'.join(VALID_MCI_TAGS)}> | finalize_scene"

    elif task == "sepsis_screening":
        if a == "complete_screening":
            return True, ""
        if re.match(r"(screen|flag_sepsis|clear):[A-Z0-9]+$", a):
            return True, ""
        m = re.match(r"order:([^:]+):(\w+)$", a)
        if m and m.group(2) in SEPSIS_BUNDLE:
            return True, ""
        return False, f"Expected: screen:<id> | flag_sepsis:<id> | clear:<id> | order:<id>:<bundle_item> | complete_screening"

    elif task == "bed_allocation":
        if a == "finalize_beds":
            return True, ""
        if re.match(r"(get_info|defer):[A-Z0-9]+$", a):
            return True, ""
        m = re.match(r"assign:([^:]+):(\w+)$", a)
        if m and m.group(2) in BED_TYPES:
            return True, ""
        return False, f"Expected: get_info:<id> | assign:<id>:<bed_type> | defer:<id> | finalize_beds"

    elif task == "shift_handoff":
        if a == "complete_handoff":
            return True, ""
        m = re.match(r"report:([^:]+):(.+)$", a)
        if m and m.group(2) in SBAR_FIELDS:
            return True, ""
        return False, f"Expected: report:<patient_id>:<sbar_field> | complete_handoff"

    elif task == "consent_assessment":
        if re.match(r"finalize:[A-Z0-9]+$", a):
            return True, ""
        m = re.match(r"assess:([^:]+):(\w+)$", a)
        if m and m.group(2) in CAPACITY_CRITERIA:
            return True, ""
        m = re.match(r"step:([^:]+):(\w+)$", a)
        if m and m.group(2) in CONSENT_STEPS:
            return True, ""
        m = re.match(r"declare_capacity:([^:]+):(has_capacity|lacks_capacity)$", a)
        if m:
            return True, ""
        m = re.match(r"consent_path:([^:]+):(\w+)$", a)
        if m and m.group(2) in CONSENT_PATHS:
            return True, ""
        return False, "Expected: assess:<id>:<criterion> | step:<id>:<key> | declare_capacity:<id>:<status> | consent_path:<id>:<path> | finalize:<id>"

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
#  LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """
    Call any OpenAI-compatible API and return raw text response.
    Supports: Groq, Google Gemini, HuggingFace Inference, DeepSeek.
    Configured via API_BASE_URL, MODEL_NAME, HF_TOKEN env vars.
    """
    # Build message list with system prompt prepended
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": full_messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_action(raw: str) -> Optional[str]:
    """Extract the action from REASONING/ACTION block."""
    # Prefer explicit ACTION: line
    m = re.search(r"ACTION:\s*(.+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: look for known action prefixes anywhere in text
    patterns = [
        r"\b(assign_esi:\d)\b",
        r"\b(ask:\w+)\b",
        r"\b(complete_\w+)\b",
        r"\b(finalize\w*)\b",
        r"\b(prioritize:[A-Z0-9]+)\b",
        r"\b(get_vitals:[A-Z0-9]+)\b",
        r"\b(flag_allergy:\S+)\b",
        r"\b(flag_interaction:\S+)\b",
        r"\b(approve:\S+)\b",
        r"\b(add_instruction:\w+)\b",
        r"\b(add_warning:\w+)\b",
        r"\b(set_followup:\d+)\b",
        r"\b(inspect:[A-Z0-9-]+)\b",
        r"\b(tag:[A-Z0-9-]+:[A-Z]+)\b",
        r"\b(screen:[A-Z0-9]+)\b",
        r"\b(flag_sepsis:[A-Z0-9]+)\b",
        r"\b(clear:[A-Z0-9]+)\b",
        r"\b(order:[A-Z0-9]+:\w+)\b",
        r"\b(get_info:[A-Z0-9]+)\b",
        r"\b(assign:[A-Z0-9]+:\w+)\b",
        r"\b(defer:[A-Z0-9]+)\b",
        r"\b(report:[A-Z0-9]+:\w+)\b",
        r"\b(assess:[A-Z0-9]+:\w+)\b",
        r"\b(step:[A-Z0-9]+:\w+)\b",
        r"\b(declare_capacity:[A-Z0-9]+:\w+)\b",
        r"\b(consent_path:[A-Z0-9]+:\w+)\b",
        r"\b(finalize:[A-Z0-9]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(task: str, seed: int = 42) -> float:
    """Run one full episode. Returns final reward."""
    system_prompt = SYSTEM_PROMPTS[task]
    messages: List[Dict[str, str]] = []

    # Reset
    resp = requests.post(f"{SERVER_URL}/tasks/{task}/reset",
                         json={"seed": seed}, timeout=15)
    resp.raise_for_status()
    obs = resp.json()
    initial_message = obs.get("message", str(obs))

    messages.append({"role": "user", "content": initial_message})

    print(f"[START] task={task}", flush=True)

    final_reward = 0.0
    done = False
    step_num = 0

    while not done:
        # Trim history window
        context = messages[-HISTORY_WINDOW * 2:]

        # LLM → raw action
        raw = call_llm(system_prompt, context)
        action_str = parse_action(raw)

        # Retry loop on bad parse or invalid action
        for attempt in range(MAX_RETRIES + 1):
            if action_str is None:
                correction = (
                    "Your previous response did not contain a parseable ACTION. "
                    "You MUST end with:\nACTION: <action_string>\n"
                    "Try again."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": correction})
                raw = call_llm(system_prompt, messages[-HISTORY_WINDOW * 2:])
                action_str = parse_action(raw)
                continue

            valid, err = validate_action(task, action_str)
            if not valid:
                correction = (
                    f"Invalid action '{action_str}': {err}\n"
                    "Correct the action format and try again.\nACTION: ..."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": correction})
                raw = call_llm(system_prompt, messages[-HISTORY_WINDOW * 2:])
                action_str = parse_action(raw)
            else:
                break  # valid action

        if action_str is None:
            print(f"  [WARN] Could not parse action after {MAX_RETRIES} retries. Skipping step.", flush=True)
            break

        messages.append({"role": "assistant", "content": raw})

        # Step environment
        step_resp = requests.post(
            f"{SERVER_URL}/tasks/{task}/step",
            json={"action_type": "text", "content": action_str},
            timeout=15,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward   = result.get("reward", 0.0)
        done     = result.get("done", False)
        obs_msg  = result.get("observation", {}).get("message", str(result))
        step_num += 1

        print(f"[STEP] task={task} step={step_num} action={action_str} reward={reward:.4f} done={done}", flush=True)

        if done:
            final_reward = reward

        messages.append({"role": "user", "content": obs_msg})

    print(f"[END] task={task} score={final_reward:.4f} steps={step_num}", flush=True)
    return final_reward


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark() -> Dict[str, Any]:
    seeds_env = os.getenv("MULTI_SEED_RUNS", "42")
    seeds = [int(s.strip()) for s in seeds_env.split(",")]

    results: Dict[str, List[float]] = {task: [] for task in TASKS}

    print(f"\n{'='*60}")
    print(f"  Medical Triage Benchmark  —  Model: {MODEL}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}\n")

    for seed in seeds:
        print(f"── Seed {seed} ──────────────────────────────")
        for task in TASKS:
            try:
                t0 = time.time()
                reward = run_episode(task, seed)
                elapsed = time.time() - t0
                results[task].append(reward)
                print(f"  {task:<25} reward={reward:.3f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  {task:<25} ERROR: {e}")
                results[task].append(0.0)

    print(f"\n{'='*60}")
    print(f"  FINAL BENCHMARK RESULTS")
    print(f"{'='*60}")
    total_scores = []
    for task in TASKS:
        scores = results[task]
        avg = sum(scores) / max(len(scores), 1)
        total_scores.append(avg)
        bar = "█" * int(avg * 20)
        print(f"  {task:<25} {avg:.3f}  {bar}")
    overall = sum(total_scores) / max(len(total_scores), 1)
    print(f"\n  OVERALL AVERAGE:          {overall:.3f}")
    print(f"{'='*60}\n")

    return {"per_task": {t: sum(v)/max(len(v),1) for t, v in results.items()},
            "overall": overall, "seeds": seeds}


if __name__ == "__main__":
    run_benchmark()
