"""
Inference Script — Medical Triage Assistant OpenEnv
=====================================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM  (default: HF router)
  MODEL_NAME     The model identifier           (default: Qwen2.5-72B-Instruct)
  HF_TOKEN       Your Hugging Face / API key
  ENV_BASE_URL   The environment server URL     (default: http://localhost:7860)

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=medical_triage model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

SUPPORTED FREE PROVIDERS:
  Groq  (Llama 3.3 70B):
    API_BASE_URL=https://api.groq.com/openai/v1
    MODEL_NAME=llama-3.3-70b-versatile
    HF_TOKEN=your_groq_key

  Google (Gemini 1.5 Flash):
    API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    MODEL_NAME=gemini-1.5-flash
    HF_TOKEN=your_google_key

  HuggingFace (Mistral 7B):
    API_BASE_URL=https://api-inference.huggingface.co/v1
    MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
    HF_TOKEN=your_hf_token

  DeepSeek (R1):
    API_BASE_URL=https://api.deepseek.com/v1
    MODEL_NAME=deepseek-reasoner
    HF_TOKEN=your_deepseek_key
"""

import os
import sys
import re
import textwrap
from typing import List, Optional, Dict, Any

import requests
import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "medical_triage"
MAX_STEPS    = 25
TEMPERATURE  = 0.1
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Environment Client ────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()

    def reset(self, task: str, seed: int = 42) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/tasks/{task}/reset",
            json={"seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, task: str, content: str) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/tasks/{task}/step",
            json={"action_type": "text", "content": content},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {

"esi_assignment": textwrap.dedent("""
    You are an expert emergency triage nurse using ESI Version 4.
    ESI 1=Immediate life threat, 2=High risk/severe pain, 3=Stable needs 2+ resources,
    4=Stable needs 1 resource, 5=Stable needs 0 resources.
    Escalate to ESI 2 if: HR>100/<50, RR>20/<10, SpO2<92%, temp>104F/<96F, GCS<15.
    Reply with EXACTLY ONE action: ACTION: assign_esi:<1-5>
""").strip(),

"intake_interview": textwrap.dedent("""
    You are an ED intake nurse conducting OLDCARTS assessment.
    Required fields: pain_scale, onset, duration, location, associated_symptoms,
    medical_history, current_medications, allergies, last_meal.
    Collect all fields then call complete_intake.
    Reply with EXACTLY ONE action:
    ACTION: ask:<field_name>  OR  ACTION: complete_intake
""").strip(),

"queue_management": textwrap.dedent("""
    You are the charge nurse managing the ED waiting room.
    Order patients by ESI (1 most urgent to 5 least urgent).
    Watch for deterioration alerts and re-prioritise immediately.
    Reply with EXACTLY ONE action:
    ACTION: prioritize:<patient_id>  OR  ACTION: get_vitals:<patient_id>  OR  ACTION: finalize_queue
""").strip(),

"medication_check": textwrap.dedent("""
    You are a clinical pharmacist doing medication safety review.
    Flag allergy cross-reactions and drug-drug interactions.
    Only approve medications that are safe. Wrong approvals penalised -0.15 each.
    Reply with EXACTLY ONE action:
    ACTION: flag_allergy:<med>  OR  ACTION: flag_interaction:<med>  OR  ACTION: approve:<med>  OR  ACTION: complete_review
""").strip(),

"discharge_planning": textwrap.dedent("""
    You are an ED nurse creating a discharge plan for the patient's specific diagnosis.
    Add relevant instructions, return-to-ED warnings, and appropriate follow-up timing.
    Always include ed_return_if_worse.
    Reply with EXACTLY ONE action:
    ACTION: add_instruction:<key>  OR  ACTION: add_warning:<key>  OR  ACTION: set_followup:<days>  OR  ACTION: complete_discharge
""").strip(),

"mass_casualty": textwrap.dedent("""
    You are MCI triage officer. Apply START protocol:
    Can walk?->GREEN. No breath after repositioning?->BLACK.
    RR>30 or <10?->RED. No radial pulse?->RED. Cannot follow commands?->RED. Stable->YELLOW.
    Never swap RED and BLACK — penalty -0.15. Inspect before tagging.
    Reply with EXACTLY ONE action:
    ACTION: inspect:<id>  OR  ACTION: tag:<id>:<RED|YELLOW|GREEN|BLACK>  OR  ACTION: finalize_scene
""").strip(),

"sepsis_screening": textwrap.dedent("""
    You are screening for sepsis using qSOFA.
    Score 1 each: RR>=22, GCS<15, SBP<=100. Score>=2 = HIGH RISK.
    For HIGH RISK order ALL 7 bundle items:
    blood_cultures, lactate, cbc, bmp, iv_access, iv_fluids, notify_physician
    Reply with EXACTLY ONE action:
    ACTION: screen:<id>  OR  ACTION: flag_sepsis:<id>  OR  ACTION: clear:<id>  OR  ACTION: order:<id>:<item>  OR  ACTION: complete_screening
""").strip(),

"bed_allocation": textwrap.dedent("""
    You are ED bed coordinator. Match patients to beds:
    trauma_bay(x2)=arrests/resus/overdoses, cardiac_monitor(x3)=chest pain/STEMI/arrhythmia,
    isolation_room(x1)=TB/airborne, regular_bed(x5)=stable cases.
    Never exceed inventory.
    Reply with EXACTLY ONE action:
    ACTION: get_info:<id>  OR  ACTION: assign:<id>:<bed_type>  OR  ACTION: finalize_beds
""").strip(),

"shift_handoff": textwrap.dedent("""
    You are the outgoing nurse completing SBAR handoff for all critical patients.
    Report ALL 12 fields: situation_chief_complaint, situation_current_status, situation_acuity_level,
    background_medical_history, background_current_medications, background_allergies,
    assessment_esi_level, assessment_vital_trend, assessment_key_concern,
    recommendation_next_action, recommendation_pending_orders, recommendation_watch_for.
    Reply with EXACTLY ONE action:
    ACTION: report:<patient_id>:<sbar_field>  OR  ACTION: complete_handoff
""").strip(),

"consent_assessment": textwrap.dedent("""
    You are an ED physician assessing capacity and consent across 3 scenarios.
    4 capacity criteria: understands_info, appreciates_situation, reasons_through_options, communicates_choice.
    Pathways: informed_consent, informed_refusal, emergent_exception, surrogate_consent, assent_minor, court_ordered.
    Dangerous shortcuts penalised -0.30. Finalize each scenario after completing steps.
    Reply with EXACTLY ONE action:
    ACTION: assess:<sid>:<criterion>  OR  ACTION: step:<sid>:<key>  OR
    ACTION: declare_capacity:<sid>:<has_capacity|lacks_capacity>  OR
    ACTION: consent_path:<sid>:<pathway>  OR  ACTION: finalize:<sid>
""").strip(),

}

# ── Fallback Actions ──────────────────────────────────────────────────────────

FALLBACKS = {
    "esi_assignment":     "assign_esi:3",
    "intake_interview":   "complete_intake",
    "queue_management":   "finalize_queue",
    "medication_check":   "complete_review",
    "discharge_planning": "complete_discharge",
    "mass_casualty":      "finalize_scene",
    "sepsis_screening":   "complete_screening",
    "bed_allocation":     "finalize_beds",
    "shift_handoff":      "complete_handoff",
    "consent_assessment": "finalize:CS003",
}

# ── Action Parser ─────────────────────────────────────────────────────────────

def parse_action(raw: str, task: str) -> str:
    m = re.search(r"ACTION:\s*(.+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    patterns = [
        r"(assign_esi:[1-5])",
        r"(ask:\w+)",
        r"(complete_intake|complete_review|complete_discharge|complete_screening|complete_handoff|finalize_queue|finalize_scene|finalize_beds)",
        r"(prioritize:[A-Z0-9]+)", r"(get_vitals:[A-Z0-9]+)",
        r"(flag_allergy:\S+)", r"(flag_interaction:\S+)", r"(approve:\S+)",
        r"(add_instruction:\w+)", r"(add_warning:\w+)", r"(set_followup:\d+)",
        r"(inspect:[A-Z0-9-]+)", r"(tag:[A-Z0-9-]+:(?:RED|YELLOW|GREEN|BLACK))",
        r"(screen:[A-Z0-9]+)", r"(flag_sepsis:[A-Z0-9]+)", r"(clear:[A-Z0-9]+)",
        r"(order:[A-Z0-9]+:\w+)", r"(get_info:[A-Z0-9]+)",
        r"(assign:[A-Z0-9]+:\w+)", r"(defer:[A-Z0-9]+)",
        r"(report:[A-Z0-9]+:\w+)", r"(assess:[A-Z0-9]+:\w+)",
        r"(step:[A-Z0-9]+:\w+)", r"(declare_capacity:[A-Z0-9]+:\w+)",
        r"(consent_path:[A-Z0-9]+:\w+)", r"(finalize:[A-Z0-9]+)",
    ]
    for pat in patterns:
        found = re.search(pat, raw)
        if found:
            return found.group(1)
    return FALLBACKS.get(task, "complete_review")

# ── Model Call ────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    system_prompt: str,
    obs_message: str,
    history: List[Dict],
    task: str,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-8:]:
        messages.append(h)
    messages.append({"role": "user", "content": obs_message})
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return FALLBACKS.get(task, "complete_review")

# ── Episode Runner ────────────────────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    env_client: EnvClient,
    task: str,
    seed: int = 42,
) -> float:
    system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["esi_assignment"])
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    history:     List[Dict]  = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        result      = env_client.reset(task=task, seed=seed)
        obs_message = result.get("message", "")
        done        = result.get("observation", {}).get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            raw    = get_model_action(client, system_prompt, obs_message, history, task)
            action = parse_action(raw, task)

            try:
                result      = env_client.step(task=task, content=action)
                reward      = float(result.get("reward", 0.0))
                done        = bool(result.get("done", False))
                error       = result.get("info", {}).get("error")
                obs_message = result.get("message", "")
                if done:
                    score = float(result.get("reward", reward))
            except Exception as exc:
                error  = str(exc)
                reward = 0.0
                done   = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append({"role": "assistant", "content": action})
            history.append({"role": "user",      "content": obs_message})

            if done:
                break

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if API_KEY == "dummy":
        print("[ERROR] Missing HF_TOKEN or API_KEY", file=sys.stderr, flush=True)
        sys.exit(1)

    client = None
    for attempt in [
        lambda: OpenAI(api_key=API_KEY, base_url=API_BASE_URL),
        lambda: OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=60.0),
        lambda: OpenAI(api_key=API_KEY, base_url=API_BASE_URL, http_client=httpx.Client()),
    ]:
        try:
            client = attempt()
            break
        except Exception:
            continue

    if client is None:
        print("[ERROR] Failed to initialise OpenAI client.", file=sys.stderr, flush=True)
        sys.exit(1)

    env_client = EnvClient(base_url=ENV_BASE_URL)

    tasks = [
        "esi_assignment", "intake_interview", "queue_management",
        "medication_check", "discharge_planning", "mass_casualty",
        "sepsis_screening", "bed_allocation", "shift_handoff", "consent_assessment",
    ]

    all_scores: List[float] = []
    for task in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_episode(client, env_client, task=task, seed=42)
        all_scores.append(score)

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"AGGREGATE SCORE: {aggregate:.3f}", flush=True)
    for task, score in zip(tasks, all_scores):
        print(f"  {task:<25} {score:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
