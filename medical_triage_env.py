"""
Medical Triage Assistant Environment  ─  v3.0
==============================================
10 tasks, real clinical protocols, multi-step sequential decisions.

WHY WE BEAT SMART HOSPITAL
───────────────────────────────────────────────────────────────────────
  Smart Hospital : 3 tasks, SINGLE-STEP classification (pick dept+level)
  Ours           : 10 tasks, MULTI-STEP decisions, real clinical protocols
                   ESI | START MCI | qSOFA | SBAR | Capacity Assessment
───────────────────────────────────────────────────────────────────────

TASK ROSTER
  1. esi_assignment      Easy   Emergency Severity Index (1-5)
  2. intake_interview    Medium OLDCARTS structured intake
  3. queue_management    Hard   Dynamic queue + deterioration events
  4. medication_check    Medium Allergy cross-reactivity + interactions
  5. discharge_planning  Medium Instructions / warnings / timing
  6. mass_casualty       Hard   START MCI triage (RED/YELLOW/GREEN/BLACK)
  7. sepsis_screening    Hard   qSOFA multi-criteria + bundle ordering
  8. bed_allocation      Medium Resource-constraint bed matching
  9. shift_handoff       Medium SBAR nurse-to-nurse handoff protocol
 10. consent_assessment  Medium Capacity evaluation + consent pathway
"""

from __future__ import annotations
import copy, random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

class TaskType(str, Enum):
    ESI_ASSIGNMENT     = "esi_assignment"
    INTAKE_INTERVIEW   = "intake_interview"
    QUEUE_MANAGEMENT   = "queue_management"
    MEDICATION_CHECK   = "medication_check"
    DISCHARGE_PLANNING = "discharge_planning"
    MASS_CASUALTY      = "mass_casualty"
    SEPSIS_SCREENING   = "sepsis_screening"
    BED_ALLOCATION     = "bed_allocation"
    SHIFT_HANDOFF      = "shift_handoff"
    CONSENT_ASSESSMENT = "consent_assessment"

TASK_DIFFICULTY = {
    TaskType.ESI_ASSIGNMENT:     "easy",
    TaskType.INTAKE_INTERVIEW:   "medium",
    TaskType.MEDICATION_CHECK:   "medium",
    TaskType.DISCHARGE_PLANNING: "medium",
    TaskType.BED_ALLOCATION:     "medium",
    TaskType.SHIFT_HANDOFF:      "medium",
    TaskType.CONSENT_ASSESSMENT: "medium",
    TaskType.QUEUE_MANAGEMENT:   "hard",
    TaskType.MASS_CASUALTY:      "hard",
    TaskType.SEPSIS_SCREENING:   "hard",
}

MAX_STEPS = {
    TaskType.ESI_ASSIGNMENT:     5,
    TaskType.INTAKE_INTERVIEW:   15,
    TaskType.QUEUE_MANAGEMENT:   20,
    TaskType.MEDICATION_CHECK:   12,
    TaskType.DISCHARGE_PLANNING: 18,
    TaskType.MASS_CASUALTY:      18,   # 6 patients × (inspect+tag) + finalize = 13 min; buffer for re-inspects
    TaskType.SEPSIS_SCREENING:   22,   # 4 patients × (screen+flag+7 orders) = 20 min
    TaskType.BED_ALLOCATION:     15,
    TaskType.SHIFT_HANDOFF:      42,   # HO001(12)+HO002(7)+HO003(9)+finalize = 29 min; buffer for extras
    TaskType.CONSENT_ASSESSMENT: 36,   # CS001(10)+CS002(6)+CS003(7)+finalize = 24 min; buffer
}


# ══════════════════════════════════════════════════════════════════════════════
#  CORE DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PatientVitals(BaseModel):
    heart_rate:        Optional[int]   = None
    bp_systolic:       Optional[int]   = None
    bp_diastolic:      Optional[int]   = None
    respiratory_rate:  Optional[int]   = None
    oxygen_saturation: Optional[float] = None
    temperature_f:     Optional[float] = None
    pain_scale:        Optional[int]   = None
    gcs:               Optional[int]   = None  # Glasgow Coma Scale 3-15


class Patient(BaseModel):
    patient_id:          str
    age:                 int
    gender:              str
    chief_complaint:     str
    vitals:              PatientVitals
    arrival_minutes_ago: int  = 0
    status:              str  = "waiting"
    correct_esi_level:   Optional[int] = None
    collected_fields:    Dict[str, Any] = Field(default_factory=dict)
    deteriorated:        bool = False


class TriageObservation(BaseModel):
    task_type:            str
    patient:              Optional[Patient]       = None
    queue:                Optional[List[Patient]] = None
    conversation_history: List[Dict[str, str]]    = Field(default_factory=list)
    available_actions:    List[str]               = Field(default_factory=list)
    step_number:          int  = 0
    message:              str  = ""
    done:                 bool = False


class TriageAction(BaseModel):
    action_type:       str
    content:           str
    target_patient_id: Optional[str] = None


class StepResult(BaseModel):
    observation: TriageObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — ESI ASSIGNMENT DATA
# ══════════════════════════════════════════════════════════════════════════════

ESI_CASES: List[Dict[str, Any]] = [
    {
        "patient_id": "P001", "age": 58, "gender": "Male",
        "chief_complaint": "Unresponsive, no breathing, bystander CPR in progress",
        "vitals": PatientVitals(heart_rate=0, respiratory_rate=0, temperature_f=97.2, gcs=3),
        "correct_esi_level": 1,
    },
    {
        "patient_id": "P002", "age": 34, "gender": "Female",
        "chief_complaint": "Severe anaphylaxis after bee sting — throat swelling, cannot swallow",
        "vitals": PatientVitals(heart_rate=128, bp_systolic=72, bp_diastolic=40,
                                respiratory_rate=28, oxygen_saturation=88.0, pain_scale=9, gcs=14),
        "correct_esi_level": 1,
    },
    {
        "patient_id": "P003", "age": 62, "gender": "Male",
        "chief_complaint": "Crushing chest pain radiating to left arm, onset 30 min ago",
        "vitals": PatientVitals(heart_rate=110, bp_systolic=165, bp_diastolic=98,
                                respiratory_rate=22, oxygen_saturation=94.0, pain_scale=9, gcs=15),
        "correct_esi_level": 2,
    },
    {
        "patient_id": "P004", "age": 45, "gender": "Female",
        "chief_complaint": "Sudden worst headache of life, onset 1 hour ago",
        "vitals": PatientVitals(heart_rate=92, bp_systolic=178, bp_diastolic=105,
                                respiratory_rate=18, oxygen_saturation=97.0, pain_scale=10, gcs=15),
        "correct_esi_level": 2,
    },
    {
        "patient_id": "P005", "age": 71, "gender": "Male",
        "chief_complaint": "Right facial droop and arm weakness, onset 45 min ago",
        "vitals": PatientVitals(heart_rate=88, bp_systolic=190, bp_diastolic=110,
                                respiratory_rate=20, oxygen_saturation=96.0, pain_scale=3, gcs=14),
        "correct_esi_level": 2,
    },
    {
        "patient_id": "P006", "age": 28, "gender": "Female",
        "chief_complaint": "Fever 102.8F, severe sore throat, difficulty swallowing 2 days",
        "vitals": PatientVitals(heart_rate=98, bp_systolic=118, bp_diastolic=76,
                                respiratory_rate=18, oxygen_saturation=98.0, temperature_f=102.8,
                                pain_scale=6, gcs=15),
        "correct_esi_level": 3,
    },
    {
        "patient_id": "P007", "age": 52, "gender": "Male",
        "chief_complaint": "Lower back pain after lifting, radiating down right leg",
        "vitals": PatientVitals(heart_rate=82, bp_systolic=130, bp_diastolic=84,
                                respiratory_rate=16, oxygen_saturation=99.0, pain_scale=7, gcs=15),
        "correct_esi_level": 3,
    },
    {
        "patient_id": "P008", "age": 22, "gender": "Female",
        "chief_complaint": "Rolled ankle running, moderate swelling, can bear weight",
        "vitals": PatientVitals(heart_rate=80, bp_systolic=118, bp_diastolic=74,
                                respiratory_rate=16, oxygen_saturation=99.0, pain_scale=4, gcs=15),
        "correct_esi_level": 4,
    },
    {
        "patient_id": "P009", "age": 35, "gender": "Male",
        "chief_complaint": "Minor laceration right palm from kitchen knife, bleeding controlled",
        "vitals": PatientVitals(heart_rate=76, bp_systolic=122, bp_diastolic=80,
                                respiratory_rate=15, oxygen_saturation=99.0, pain_scale=3, gcs=15),
        "correct_esi_level": 4,
    },
    {
        "patient_id": "P010", "age": 48, "gender": "Female",
        "chief_complaint": "Prescription refill for hypertension medication",
        "vitals": PatientVitals(heart_rate=70, bp_systolic=128, bp_diastolic=82,
                                respiratory_rate=14, oxygen_saturation=99.0, pain_scale=0, gcs=15),
        "correct_esi_level": 5,
    },
    {
        "patient_id": "P011", "age": 19, "gender": "Male",
        "chief_complaint": "Mild cold symptoms, runny nose, mild sore throat 3 days",
        "vitals": PatientVitals(heart_rate=72, bp_systolic=115, bp_diastolic=72,
                                respiratory_rate=15, oxygen_saturation=99.0, temperature_f=99.1,
                                pain_scale=2, gcs=15),
        "correct_esi_level": 5,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — INTAKE INTERVIEW DATA
# ══════════════════════════════════════════════════════════════════════════════

INTAKE_REQUIRED_FIELDS = [
    "pain_scale", "onset", "duration", "location", "associated_symptoms",
    "medical_history", "current_medications", "allergies", "last_meal",
]
INTAKE_FIELD_DESCRIPTIONS = {
    "pain_scale":            "Pain level 0-10",
    "onset":                 "When symptoms started",
    "duration":              "How long symptoms have been present",
    "location":              "Where is pain/discomfort located",
    "associated_symptoms":   "Other accompanying symptoms",
    "medical_history":       "Past medical conditions and surgeries",
    "current_medications":   "Current medications",
    "allergies":             "Drug or food allergies",
    "last_meal":             "When patient last ate or drank",
}
PATIENT_INTAKE_ANSWERS: Dict[str, Dict[str, str]] = {
    "P006": {
        "pain_scale": "6/10", "onset": "2 days ago after a meal",
        "duration": "Continuous, gradually worsening",
        "location": "Throat and neck, worse on the right side",
        "associated_symptoms": "Mild fever, fatigue, difficulty swallowing",
        "medical_history": "Seasonal allergies",
        "current_medications": "Cetirizine 10mg daily",
        "allergies": "Penicillin — causes rash",
        "last_meal": "Yesterday evening — liquids only",
    },
    "P007": {
        "pain_scale": "7/10", "onset": "This morning while lifting moving boxes",
        "duration": "Approximately 4 hours",
        "location": "Lower back right side radiating to right leg",
        "associated_symptoms": "Tingling and numbness in right foot",
        "medical_history": "Two prior back episodes, both resolved without surgery",
        "current_medications": "Ibuprofen as needed",
        "allergies": "No known drug allergies",
        "last_meal": "Lunch about 2 hours ago",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — QUEUE MANAGEMENT DATA
# ══════════════════════════════════════════════════════════════════════════════

QUEUE_SCENARIO: List[Dict[str, Any]] = [
    {
        "patient_id": "Q001", "age": 67, "gender": "Male",
        "chief_complaint": "Mild exertional dyspnea, stable at rest",
        "vitals": PatientVitals(heart_rate=88, bp_systolic=140, bp_diastolic=90,
                                respiratory_rate=18, oxygen_saturation=96.0, pain_scale=2),
        "correct_esi_level": 3, "arrival_minutes_ago": 15,
    },
    {
        "patient_id": "Q002", "age": 19, "gender": "Female",
        "chief_complaint": "Wrist pain after bicycle fall, possible fracture",
        "vitals": PatientVitals(heart_rate=82, bp_systolic=112, bp_diastolic=72,
                                respiratory_rate=16, oxygen_saturation=99.0, pain_scale=5),
        "correct_esi_level": 4, "arrival_minutes_ago": 25,
    },
    {
        "patient_id": "Q003", "age": 44, "gender": "Male",
        "chief_complaint": "Right lower quadrant abdominal pain 6 hours, nausea",
        "vitals": PatientVitals(heart_rate=96, bp_systolic=122, bp_diastolic=78,
                                respiratory_rate=17, oxygen_saturation=98.0, pain_scale=6),
        "correct_esi_level": 3, "arrival_minutes_ago": 10,
    },
    {
        "patient_id": "Q004", "age": 31, "gender": "Female",
        "chief_complaint": "Prescription refill for birth control",
        "vitals": PatientVitals(heart_rate=68, bp_systolic=116, bp_diastolic=70,
                                respiratory_rate=14, oxygen_saturation=99.0, pain_scale=0),
        "correct_esi_level": 5, "arrival_minutes_ago": 30,
    },
    {
        "patient_id": "Q005", "age": 78, "gender": "Female",
        "chief_complaint": "Confusion and slurred speech, family noticed 1 hour ago",
        "vitals": PatientVitals(heart_rate=102, bp_systolic=182, bp_diastolic=108,
                                respiratory_rate=20, oxygen_saturation=95.0, gcs=13),
        "correct_esi_level": 2, "arrival_minutes_ago": 5,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 4 — MEDICATION CHECK DATA
# ══════════════════════════════════════════════════════════════════════════════

class MedCheckCase(BaseModel):
    case_id:               str
    age:                   int
    gender:                str
    chief_complaint:       str
    known_allergies:       List[str]
    current_medications:   List[str]
    proposed_medications:  List[str]
    true_allergy_flags:    List[Tuple[str, str]]   # (proposed_med, allergen)
    true_interaction_flags: List[Tuple[str, str]]  # (proposed_med, interacts_with)
    safe_medications:      List[str]


MEDICATION_CASES: List[MedCheckCase] = [
    MedCheckCase(
        case_id="MC001", age=45, gender="Female",
        chief_complaint="Strep throat — needs antibiotic",
        known_allergies=["penicillin"],
        current_medications=["lisinopril", "metformin"],
        proposed_medications=["amoxicillin", "azithromycin", "ibuprofen", "acetaminophen"],
        true_allergy_flags=[("amoxicillin", "penicillin")],
        true_interaction_flags=[("ibuprofen", "lisinopril")],
        safe_medications=["azithromycin", "acetaminophen"],
    ),
    MedCheckCase(
        case_id="MC002", age=68, gender="Male",
        chief_complaint="DVT — on anticoagulation therapy",
        known_allergies=["sulfa"],
        current_medications=["warfarin"],
        proposed_medications=["aspirin", "metronidazole", "ondansetron", "acetaminophen"],
        true_allergy_flags=[],
        true_interaction_flags=[("aspirin", "warfarin"), ("metronidazole", "warfarin")],
        safe_medications=["ondansetron", "acetaminophen"],
    ),
    MedCheckCase(
        case_id="MC003", age=29, gender="Female",
        chief_complaint="UTI — needs antibiotic",
        known_allergies=["sulfa"],
        current_medications=["oral_contraceptive"],
        proposed_medications=["trimethoprim-sulfamethoxazole", "nitrofurantoin", "ibuprofen"],
        true_allergy_flags=[("trimethoprim-sulfamethoxazole", "sulfa")],
        true_interaction_flags=[],
        safe_medications=["nitrofurantoin", "ibuprofen"],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 5 — DISCHARGE PLANNING DATA
# ══════════════════════════════════════════════════════════════════════════════

DISCHARGE_INSTRUCTIONS_CATALOG = {
    "no_weight_bearing":    "No weight bearing until follow-up",
    "partial_weight_bearing": "Partial weight bearing with crutches",
    "rest_48h":             "Rest 48h; avoid strenuous activity",
    "ice_20min":            "Ice 20 min on/off for 48h",
    "keep_elevated":        "Keep limb elevated above heart level",
    "wound_care":           "Keep wound clean/dry; change dressing daily",
    "no_driving":           "Do not drive on opioids or sedating medications",
    "gradual_activity":     "Gradually resume activity over 1-2 weeks",
    "clear_liquids_24h":    "Clear liquids only 24h; advance as tolerated",
    "avoid_alcohol":        "Avoid alcohol while taking antibiotics",
    "hydration":            "Drink 8+ glasses of water per day",
    "complete_antibiotics": "Complete full antibiotic course as prescribed",
    "otc_pain_management":  "Use acetaminophen/ibuprofen as directed for pain",
    "take_with_food":       "Take medication with food to reduce GI upset",
    "primary_care_1week":   "Follow up with primary care physician within 1 week",
    "orthopedic_followup":  "Follow up with orthopedics as scheduled",
    "ed_return_if_worse":   "Return to ED if symptoms worsen",
}
DISCHARGE_WARNING_CATALOG = {
    "fever_101":            "Fever above 101°F",
    "increased_pain":       "Pain significantly worse despite medications",
    "numbness_tingling":    "New or worsening numbness or tingling",
    "wound_infection_signs": "Redness, warmth, or pus at wound site",
    "difficulty_breathing": "Any difficulty breathing",
    "chest_pain":           "Chest pain or palpitations",
    "bleeding":             "Excessive or uncontrolled bleeding",
    "inability_to_urinate": "Cannot urinate or severe abdominal pain",
    "altered_consciousness": "Confusion, dizziness, or loss of consciousness",
    "rash_hives":           "Rash or signs of allergic reaction",
}


class DischargeCase(BaseModel):
    case_id:                str
    age:                    int
    gender:                 str
    diagnosis:              str
    treatment_given:        str
    discharged_medications: List[str]
    required_instructions:  List[str]
    required_warnings:      List[str]
    optimal_followup_days:  int
    followup_tolerance_days: int


DISCHARGE_CASES: List[DischargeCase] = [
    DischargeCase(
        case_id="DC001", age=22, gender="Female",
        diagnosis="Closed distal fibula fracture",
        treatment_given="Posterior splint applied; ketorolac IV given",
        discharged_medications=["acetaminophen 500mg q6h PRN", "ibuprofen 400mg q8h with food"],
        required_instructions=["no_weight_bearing", "keep_elevated", "ice_20min",
                               "orthopedic_followup", "otc_pain_management"],
        required_warnings=["increased_pain", "numbness_tingling", "wound_infection_signs"],
        optimal_followup_days=7, followup_tolerance_days=3,
    ),
    DischargeCase(
        case_id="DC002", age=35, gender="Male",
        diagnosis="Simple right palm laceration — sutured",
        treatment_given="Wound irrigated; 4 nylon sutures placed; tetanus booster given",
        discharged_medications=["cephalexin 500mg QID x7d", "acetaminophen PRN"],
        required_instructions=["wound_care", "complete_antibiotics", "take_with_food",
                               "no_driving", "otc_pain_management"],
        required_warnings=["wound_infection_signs", "fever_101", "bleeding"],
        optimal_followup_days=7, followup_tolerance_days=3,
    ),
    DischargeCase(
        case_id="DC003", age=28, gender="Female",
        diagnosis="Group A Strep pharyngitis (penicillin allergy)",
        treatment_given="IV fluids; azithromycin Z-pack started",
        discharged_medications=["azithromycin 500mg day1 then 250mg days2-5", "ibuprofen PRN"],
        required_instructions=["complete_antibiotics", "rest_48h", "hydration",
                               "avoid_alcohol", "otc_pain_management"],
        required_warnings=["fever_101", "difficulty_breathing", "rash_hives"],
        optimal_followup_days=5, followup_tolerance_days=2,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 6 — MASS CASUALTY TRIAGE (START Protocol)
# ══════════════════════════════════════════════════════════════════════════════
# START Decision Tree:
#   1. Can walk? → GREEN (Minor)
#   2. Breathing after airway repositioning? No → BLACK (Expectant)
#   3. Respiratory rate > 30 or < 10? → RED (Immediate)
#   4. Radial pulse absent? → RED (Immediate)
#   5. Cannot follow simple commands? → RED (Immediate)
#   6. All criteria stable → YELLOW (Delayed)

VALID_MCI_TAGS = {"RED", "YELLOW", "GREEN", "BLACK"}

MCI_SCENARIOS: List[List[Dict[str, Any]]] = [
    [   # Scenario A — commuter bus rollover, 6 victims
        {
            "patient_id": "MCI-A1", "age": 34, "gender": "Male",
            "mechanism": "Walking toward responders; lacerations on arms and forehead",
            "can_walk": True, "spontaneous_breath": True, "respiratory_rate": 16,
            "radial_pulse_present": True, "can_follow_commands": True,
            "correct_tag": "GREEN",
            "clinical_note": "Walking wounded → Minor (GREEN) per START step 1",
        },
        {
            "patient_id": "MCI-A2", "age": 52, "gender": "Female",
            "mechanism": "Pinned under seat; labored rapid breathing",
            "can_walk": False, "spontaneous_breath": True, "respiratory_rate": 34,
            "radial_pulse_present": True, "can_follow_commands": True,
            "correct_tag": "RED",
            "clinical_note": "RR 34 > 30 → Immediate (RED) per START step 3",
        },
        {
            "patient_id": "MCI-A3", "age": 67, "gender": "Male",
            "mechanism": "Head trauma; apneic; no breath after airway repositioning",
            "can_walk": False, "spontaneous_breath": False, "respiratory_rate": None,
            "radial_pulse_present": False, "can_follow_commands": False,
            "correct_tag": "BLACK",
            "clinical_note": "No breath after repositioning → Expectant (BLACK) per START step 2",
        },
        {
            "patient_id": "MCI-A4", "age": 29, "gender": "Female",
            "mechanism": "Ejected from window; unconscious; pale extremities",
            "can_walk": False, "spontaneous_breath": True, "respiratory_rate": 22,
            "radial_pulse_present": False, "can_follow_commands": False,
            "correct_tag": "RED",
            "clinical_note": "No radial pulse → Immediate (RED) per START step 4",
        },
        {
            "patient_id": "MCI-A5", "age": 41, "gender": "Male",
            "mechanism": "Dazed, confused; cannot state name or location when asked",
            "can_walk": False, "spontaneous_breath": True, "respiratory_rate": 20,
            "radial_pulse_present": True, "can_follow_commands": False,
            "correct_tag": "RED",
            "clinical_note": "Cannot follow commands → Immediate (RED) per START step 5",
        },
        {
            "patient_id": "MCI-A6", "age": 23, "gender": "Female",
            "mechanism": "Fractured right femur; screaming in pain; alert",
            "can_walk": False, "spontaneous_breath": True, "respiratory_rate": 18,
            "radial_pulse_present": True, "can_follow_commands": True,
            "correct_tag": "YELLOW",
            "clinical_note": "Breathing, perfused, oriented but cannot walk → Delayed (YELLOW)",
        },
    ]
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 7 — SEPSIS SCREENING (qSOFA + bundle ordering)
# ══════════════════════════════════════════════════════════════════════════════
# qSOFA criteria (each = 1 point):
#   Respiratory Rate >= 22
#   Altered Mental Status (GCS < 15)
#   Systolic BP <= 100
# Score >= 2 → HIGH RISK → initiate sepsis bundle

SEPSIS_BUNDLE_ITEMS = [
    "blood_cultures", "lactate", "cbc", "bmp",
    "iv_access", "iv_fluids", "notify_physician",
]

SEPSIS_WARDS: List[List[Dict[str, Any]]] = [
    [   # Ward A — mixed ED patients
        {
            "patient_id": "SS001", "age": 71, "gender": "Male",
            "chief_complaint": "Confusion, fever, painful urination 2 days",
            "vitals": PatientVitals(heart_rate=108, bp_systolic=94, bp_diastolic=60,
                                    respiratory_rate=26, oxygen_saturation=95.0,
                                    temperature_f=103.2, gcs=13),
            "qsofa_score": 3, "is_sepsis": True,
            "required_orders": SEPSIS_BUNDLE_ITEMS[:],
        },
        {
            "patient_id": "SS002", "age": 45, "gender": "Female",
            "chief_complaint": "Mild tension headache; took ibuprofen at home",
            "vitals": PatientVitals(heart_rate=82, bp_systolic=128, bp_diastolic=84,
                                    respiratory_rate=16, oxygen_saturation=99.0,
                                    temperature_f=99.0, gcs=15),
            "qsofa_score": 0, "is_sepsis": False, "required_orders": [],
        },
        {
            "patient_id": "SS003", "age": 58, "gender": "Male",
            "chief_complaint": "Worsening productive cough, shortness of breath, fever 3 days",
            "vitals": PatientVitals(heart_rate=118, bp_systolic=98, bp_diastolic=62,
                                    respiratory_rate=24, oxygen_saturation=91.0,
                                    temperature_f=104.1, gcs=14),
            "qsofa_score": 3, "is_sepsis": True,
            "required_orders": SEPSIS_BUNDLE_ITEMS[:],
        },
        {
            "patient_id": "SS004", "age": 32, "gender": "Female",
            "chief_complaint": "Sprained wrist after gym accident; pain 5/10",
            "vitals": PatientVitals(heart_rate=78, bp_systolic=116, bp_diastolic=74,
                                    respiratory_rate=15, oxygen_saturation=99.0,
                                    temperature_f=98.6, gcs=15),
            "qsofa_score": 0, "is_sepsis": False, "required_orders": [],
        },
    ]
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 8 — BED ALLOCATION (Resource Constraint Optimization)
# ══════════════════════════════════════════════════════════════════════════════

BED_INVENTORY = {
    "trauma_bay":      2,   # Resuscitations, arrests, major trauma
    "cardiac_monitor": 3,   # Chest pain, arrhythmia, post-STEMI
    "isolation_room":  1,   # Airborne / contact precautions
    "regular_bed":     5,   # General ED cases
}

BED_SCENARIOS: List[List[Dict[str, Any]]] = [
    [
        {
            "patient_id": "BD001", "age": 55, "gender": "Male",
            "chief_complaint": "Cardiac arrest — ROSC achieved in ambulance",
            "required_bed": "trauma_bay", "priority": 1,
            "clinical_note": "Post-ROSC needs trauma bay for full resuscitation support",
        },
        {
            "patient_id": "BD002", "age": 62, "gender": "Female",
            "chief_complaint": "STEMI confirmed on EKG — awaiting cath lab",
            "required_bed": "cardiac_monitor", "priority": 2,
            "clinical_note": "Active STEMI requires continuous cardiac monitoring",
        },
        {
            "patient_id": "BD003", "age": 38, "gender": "Male",
            "chief_complaint": "Active pulmonary TB — confirmed by AFB smear",
            "required_bed": "isolation_room", "priority": 3,
            "clinical_note": "Airborne precautions mandatory; isolation room required",
        },
        {
            "patient_id": "BD004", "age": 28, "gender": "Female",
            "chief_complaint": "Opioid overdose — naloxone administered, now responsive",
            "required_bed": "trauma_bay", "priority": 1,
            "clinical_note": "Risk of re-narcotization; trauma bay for monitoring",
        },
        {
            "patient_id": "BD005", "age": 71, "gender": "Male",
            "chief_complaint": "New-onset palpitations — possible SVT on EKG",
            "required_bed": "cardiac_monitor", "priority": 2,
            "clinical_note": "SVT requires cardiac monitoring for rhythm tracking",
        },
        {
            "patient_id": "BD006", "age": 24, "gender": "Female",
            "chief_complaint": "Uncomplicated UTI — oral antibiotics started",
            "required_bed": "regular_bed", "priority": 4,
            "clinical_note": "Stable UTI; regular ED bed appropriate",
        },
        {
            "patient_id": "BD007", "age": 44, "gender": "Male",
            "chief_complaint": "Soft tissue laceration — sutures required",
            "required_bed": "regular_bed", "priority": 5,
            "clinical_note": "Stable laceration; regular ED bed appropriate",
        },
        {
            "patient_id": "BD008", "age": 33, "gender": "Female",
            "chief_complaint": "Intractable migraine — IV antiemetics ordered",
            "required_bed": "cardiac_monitor", "priority": 3,
            "clinical_note": "IV antiemetics (metoclopramide) require QT monitoring",
        },
    ]
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 9 — SHIFT HANDOFF (SBAR Protocol)
# ══════════════════════════════════════════════════════════════════════════════
# SBAR = Situation | Background | Assessment | Recommendation
# Agent reports each field for each critical patient, then calls complete_handoff

SBAR_COMPONENTS = {
    "situation":      ["chief_complaint", "current_status", "acuity_level"],
    "background":     ["medical_history", "current_medications", "allergies"],
    "assessment":     ["esi_level", "vital_trend", "key_concern"],
    "recommendation": ["next_action", "pending_orders", "watch_for"],
}
SBAR_ALL_FIELDS = [
    f"{comp}_{field}"
    for comp, fields in SBAR_COMPONENTS.items()
    for field in fields
]  # 12 fields total

HANDOFF_SCENARIOS: List[List[Dict[str, Any]]] = [
    [
        {
            "patient_id": "HO001", "age": 62, "gender": "Male",
            "required_fields": SBAR_ALL_FIELDS,  # All 12 fields critical for this patient
            "optional_fields": [],
            "field_values": {
                "situation_chief_complaint": "Crushing chest pain radiating to left arm; STEMI confirmed on EKG",
                "situation_current_status": "Awaiting cardiology call-back; aspirin and nitroglycerin given",
                "situation_acuity_level": "ESI 2 — Emergent",
                "background_medical_history": "Hypertension, hyperlipidemia, 20-pack-year smoking history",
                "background_current_medications": "Lisinopril 10mg, atorvastatin 40mg daily",
                "background_allergies": "No known drug allergies",
                "assessment_esi_level": "ESI 2",
                "assessment_vital_trend": "HR trending up to 118; BP 168/96 then 154/90 post-nitro",
                "assessment_key_concern": "High arrhythmia risk pre-cath; watch for hemodynamic instability",
                "recommendation_next_action": "Continuous cardiac monitor; 12-lead EKG every 30 minutes",
                "recommendation_pending_orders": "Serial troponin q3h; cardiology consult pending",
                "recommendation_watch_for": "Cardiogenic shock signs: SBP<90, diaphoresis, altered mental status",
            },
        },
        {
            "patient_id": "HO002", "age": 28, "gender": "Female",
            "required_fields": [
                "situation_chief_complaint", "situation_current_status", "situation_acuity_level",
                "background_allergies", "assessment_key_concern",
                "recommendation_next_action", "recommendation_watch_for",
            ],
            "optional_fields": [
                "background_medical_history", "assessment_vital_trend",
                "recommendation_pending_orders",
            ],
            "field_values": {
                "situation_chief_complaint": "Anaphylaxis post bee sting; epinephrine 0.3mg IM given",
                "situation_current_status": "Improving; hives resolving; airway patent and stable",
                "situation_acuity_level": "ESI 2 (downgraded from ESI 1 after stabilization)",
                "background_medical_history": "No prior anaphylaxis; no chronic conditions",
                "background_allergies": "Bee venom — severe anaphylaxis (new diagnosis today)",
                "assessment_vital_trend": "BP 72/40 → 108/68; HR 128 → 94 post-treatment",
                "assessment_key_concern": "Biphasic anaphylaxis risk within 4-8h of initial reaction",
                "recommendation_next_action": "Minimum 4h observation post-epinephrine; diphenhydramine and methylprednisolone given",
                "recommendation_pending_orders": "Allergy/immunology referral on discharge; epi-pen prescription",
                "recommendation_watch_for": "Return of throat swelling, stridor, or hypotension — epi at bedside",
            },
        },
        {
            "patient_id": "HO003", "age": 44, "gender": "Male",
            "required_fields": [
                "situation_chief_complaint", "situation_current_status", "situation_acuity_level",
                "background_medical_history", "assessment_vital_trend", "assessment_key_concern",
                "recommendation_next_action", "recommendation_pending_orders", "recommendation_watch_for",
            ],
            "optional_fields": ["background_current_medications", "background_allergies"],
            "field_values": {
                "situation_chief_complaint": "RLQ abdominal pain 8h with nausea; CT ordered for appendicitis workup",
                "situation_current_status": "Pain 8/10; CT abdomen/pelvis completed; awaiting radiology read",
                "situation_acuity_level": "ESI 3 escalated to ESI 2 after clinical deterioration",
                "background_medical_history": "No prior abdominal surgeries; no chronic GI conditions",
                "background_current_medications": "No regular medications",
                "background_allergies": "No known drug allergies",
                "assessment_vital_trend": "HR 96→112; Temp 98.6°F→101.2°F; rebound tenderness now present",
                "assessment_key_concern": "Clinical picture consistent with perforated appendicitis; urgent surgical consult",
                "recommendation_next_action": "NPO status; IV fluids running; surgical consult ordered STAT",
                "recommendation_pending_orders": "CT result expected within 15min; surgery attending paged",
                "recommendation_watch_for": "Peritoneal signs, guarding, rigidity; hemodynamic instability",
            },
        },
    ]
]


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 10 — CONSENT ASSESSMENT (Capacity + Consent Pathway)
# ══════════════════════════════════════════════════════════════════════════════
# 4 Capacity criteria (ALL required for capacity):
#   1. understands_info        — Patient understands the information provided
#   2. appreciates_situation   — Patient appreciates how it applies to them
#   3. reasons_through_options — Patient can reason through the options
#   4. communicates_choice     — Patient can communicate a consistent choice
#
# Consent pathways:
#   informed_consent    — Competent adult agrees
#   informed_refusal    — Competent adult refuses (must document thoroughly)
#   emergent_exception  — Unconscious/incapacitated; no surrogate; life-threatening
#   surrogate_consent   — Patient lacks capacity; surrogate decides
#   assent_minor        — Minor patient; parent/guardian consent required
#   court_ordered       — Involuntary treatment via court order

CAPACITY_CRITERIA = [
    "understands_info", "appreciates_situation",
    "reasons_through_options", "communicates_choice",
]

CONSENT_STEPS_CATALOG = {
    "explain_procedure":       "Explain procedure, risks, benefits, and alternatives",
    "assess_understands":      "Verify patient understands the information provided",
    "assess_appreciates":      "Verify patient appreciates how situation applies to them",
    "assess_reasons":          "Verify patient can reason through options logically",
    "assess_communicates":     "Verify patient can express a consistent choice",
    "determine_capacity":      "Make formal capacity determination (has/lacks capacity)",
    "document_consent":        "Document signed informed consent in chart",
    "document_refusal":        "Document informed refusal; patient signs AMA form",
    "identify_surrogate":      "Identify legal surrogate decision-maker",
    "contact_surrogate":       "Contact and consult surrogate decision-maker",
    "surrogate_consent_doc":   "Document surrogate consent in chart",
    "invoke_emergent_exception": "Invoke emergent exception (implied consent); document",
    "contact_guardian":        "Contact parent or legal guardian for minor",
    "obtain_assent":           "Obtain minor's assent (age-appropriate agreement)",
    "ethics_consult":          "Request ethics committee consultation",
}

DANGEROUS_CONSENT_SHORTCUTS = {
    "CS001": ["invoke_emergent_exception"],  # Patient IS competent — can't bypass
    "CS002": ["document_consent", "document_refusal"],  # Patient IS unconscious — can't get consent
    "CS003": ["document_consent"],  # Minor — parent consent required first
}

CONSENT_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": "CS001", "age": 38, "gender": "Male",
        "procedure_needed": "Appendectomy for acute appendicitis",
        "clinical_urgency": "urgent",
        "presentation": (
            "Alert and oriented x4. Patient understands he has appendicitis and needs surgery. "
            "States he wants the operation. Can explain risks back to nurse. No signs of delirium."
        ),
        "capacity_criteria": {
            "understands_info": True, "appreciates_situation": True,
            "reasons_through_options": True, "communicates_choice": True,
        },
        "has_surrogate": True, "is_minor": False,
        "correct_consent_path": "informed_consent",
        "correct_capacity": True,
        "required_steps": [
            "explain_procedure", "determine_capacity", "document_consent",
        ],
    },
    {
        "scenario_id": "CS002", "age": 55, "gender": "Female",
        "procedure_needed": "Emergency intubation for respiratory failure",
        "clinical_urgency": "emergent",
        "presentation": (
            "Unresponsive; GCS 6. No family present. No advance directive on file. "
            "Rapidly declining O2 sat. No time to locate surrogate. Life-threatening emergency."
        ),
        "capacity_criteria": {
            "understands_info": False, "appreciates_situation": False,
            "reasons_through_options": False, "communicates_choice": False,
        },
        "has_surrogate": False, "is_minor": False,
        "correct_consent_path": "emergent_exception",
        "correct_capacity": False,
        "required_steps": [
            "determine_capacity", "invoke_emergent_exception",
        ],
    },
    {
        "scenario_id": "CS003", "age": 15, "gender": "Male",
        "procedure_needed": "Blood transfusion for traumatic hemorrhage",
        "clinical_urgency": "urgent",
        "presentation": (
            "15-year-old male; motor vehicle accident. Conscious and frightened. "
            "Parents en route — estimated 20 min away. Hemoglobin 6.2; clinically indicated transfusion."
        ),
        "capacity_criteria": {
            "understands_info": True, "appreciates_situation": True,
            "reasons_through_options": False, "communicates_choice": True,
        },
        "has_surrogate": True, "is_minor": True,
        "correct_consent_path": "assent_minor",
        "correct_capacity": False,  # Minor cannot give legal consent regardless
        "required_steps": [
            "explain_procedure", "obtain_assent", "contact_guardian",
            "contact_surrogate", "surrogate_consent_doc",
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  GRADING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def grade_esi(predicted: int, correct: int) -> float:
    diff = abs(predicted - correct)
    return {0: 1.0, 1: 0.5, 2: 0.2}.get(diff, 0.0)


def grade_intake(collected: Dict[str, Any], steps_taken: int) -> float:
    required = set(INTAKE_REQUIRED_FIELDS)
    covered = required & set(collected.keys())
    completeness = len(covered) / len(required)
    if completeness == 0.0:
        return 0.0  # no fields collected — efficiency bonus must not apply
    efficiency = max(0.0, 1.0 - max(0, steps_taken - len(required)) * 0.05)
    return 0.7 * completeness + 0.3 * efficiency


def grade_queue(agent_order: List[str], correct_esis: Dict[str, int],
                deteriorated_ids: set, deterioration_responses: set) -> float:
    import math
    all_ids = list(correct_esis.keys())
    correct_order = sorted(all_ids, key=lambda pid: correct_esis[pid])

    if len(agent_order) < 2:
        tau = 0.0
    else:
        n = len(agent_order)
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if agent_order[i] not in correct_order or agent_order[j] not in correct_order:
                    continue
                agent_rel = agent_order.index(agent_order[i]) < agent_order.index(agent_order[j])
                correct_rel = correct_order.index(agent_order[i]) < correct_order.index(agent_order[j])
                if agent_rel == correct_rel:
                    concordant += 1
                else:
                    discordant += 1
        total = concordant + discordant
        tau = (concordant - discordant) / total if total > 0 else 0.0

    critical_ids = [pid for pid, esi in correct_esis.items() if esi <= 2]
    critical_first = sum(1 for cid in critical_ids
                         if agent_order and agent_order.index(cid) < len(critical_ids)
                         if cid in agent_order) / max(len(critical_ids), 1)

    det_response = len(deterioration_responses & deteriorated_ids) / max(len(deteriorated_ids), 1)

    return 0.60 * ((tau + 1) / 2) + 0.20 * critical_first + 0.20 * det_response


def grade_medication(flagged_allergies: set, flagged_interactions: set,
                     approved_meds: set, case: MedCheckCase,
                     wrong_approvals: int) -> float:
    true_allergy_meds = {med for med, _ in case.true_allergy_flags}
    true_interaction_meds = {med for med, _ in case.true_interaction_flags}

    allergy_recall = (len(flagged_allergies & true_allergy_meds) /
                      max(len(true_allergy_meds), 1))
    allergy_precision = (len(flagged_allergies & true_allergy_meds) /
                         max(len(flagged_allergies), 1)) if flagged_allergies else 1.0

    interaction_recall = (len(flagged_interactions & true_interaction_meds) /
                          max(len(true_interaction_meds), 1))
    interaction_precision = (len(flagged_interactions & true_interaction_meds) /
                             max(len(flagged_interactions), 1)) if flagged_interactions else 1.0

    base = (0.30 * allergy_recall + 0.10 * allergy_precision +
            0.30 * interaction_recall + 0.10 * interaction_precision + 0.20)
    penalty = wrong_approvals * 0.15
    return max(0.0, base - penalty)


def grade_discharge(given_instructions: set, given_warnings: set, given_followup_days: int,
                    case: DischargeCase) -> float:
    req_i = set(case.required_instructions)
    req_w = set(case.required_warnings)
    instruction_score = len(given_instructions & req_i) / max(len(req_i), 1)
    warning_score = len(given_warnings & req_w) / max(len(req_w), 1)
    day_diff = abs(given_followup_days - case.optimal_followup_days)
    timing_score = max(0.0, 1.0 - max(0, day_diff - case.followup_tolerance_days) * 0.2)
    return 0.50 * instruction_score + 0.30 * warning_score + 0.20 * timing_score


def grade_mci(agent_tags: Dict[str, str], scenario: List[Dict]) -> float:
    correct_map = {p["patient_id"]: p["correct_tag"] for p in scenario}
    total = len(correct_map)
    correct_count = sum(1 for pid, tag in agent_tags.items()
                        if correct_map.get(pid) == tag)
    # Critical error: swapping RED and BLACK (over- or under-triage of life-threatening cases)
    critical_errors = sum(
        1 for pid, tag in agent_tags.items()
        if (correct_map.get(pid) == "RED" and tag == "BLACK") or
           (correct_map.get(pid) == "BLACK" and tag == "RED")
    )
    base = correct_count / max(total, 1)
    return max(0.0, base - critical_errors * 0.15)


def grade_sepsis(flagged_ids: set, ordered_items: Dict[str, set],
                 scenario: List[Dict]) -> float:
    true_sepsis = {p["patient_id"] for p in scenario if p["is_sepsis"]}
    true_clear = {p["patient_id"] for p in scenario if not p["is_sepsis"]}

    tp = len(flagged_ids & true_sepsis)
    fp = len(flagged_ids & true_clear)
    fn = len(true_sepsis - flagged_ids)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    bundle_scores = []
    for pid in true_sepsis:
        required = set(SEPSIS_BUNDLE_ITEMS)
        ordered = ordered_items.get(pid, set())
        bundle_scores.append(len(ordered & required) / len(required))
    bundle_avg = sum(bundle_scores) / max(len(bundle_scores), 1)

    return 0.60 * f1 + 0.40 * bundle_avg


def grade_bed_allocation(assignments: Dict[str, str], scenario: List[Dict]) -> float:
    type_correct = sum(
        1 for p in scenario
        if assignments.get(p["patient_id"]) == p["required_bed"]
    ) / max(len(scenario), 1)

    # Penalty for over-allocating beds beyond inventory
    used = {}
    for bed_type in assignments.values():
        used[bed_type] = used.get(bed_type, 0) + 1
    over_alloc = sum(
        max(0, used.get(bt, 0) - cap)
        for bt, cap in BED_INVENTORY.items()
    )
    penalty = over_alloc * 0.10

    # Bonus: highest-priority patients assigned before lower-priority
    priority_map = {p["patient_id"]: p["priority"] for p in scenario}
    assigned_ids = list(assignments.keys())
    priority_bonus = sum(
        1 for pid in assigned_ids[:3]
        if priority_map.get(pid, 99) <= 2
    ) / 3 * 0.10

    return max(0.0, min(1.0, type_correct - penalty + priority_bonus))


def grade_handoff(reported_fields: Dict[str, set], scenario: List[Dict]) -> float:
    scores = []
    for patient in scenario:
        pid = patient["patient_id"]
        required = set(patient["required_fields"])
        optional = set(patient.get("optional_fields", []))
        reported = reported_fields.get(pid, set())
        mandatory_score = len(reported & required) / max(len(required), 1)
        optional_bonus = (len(reported & optional) / max(len(optional), 1)) * 0.10 if optional else 0
        scores.append(min(1.0, mandatory_score + optional_bonus))
    return sum(scores) / max(len(scores), 1)


def grade_consent(
    assessed_criteria: Dict[str, set],
    declared_capacity: Dict[str, Optional[bool]],
    chosen_paths: Dict[str, Optional[str]],
    completed_steps: Dict[str, set],
) -> float:
    scores = []
    for sc in CONSENT_SCENARIOS:
        sid = sc["scenario_id"]
        # Step coverage
        required = set(sc["required_steps"])
        done = completed_steps.get(sid, set())
        step_score = len(done & required) / max(len(required), 1)
        # Path correctness
        path_correct = 1.0 if chosen_paths.get(sid) == sc["correct_consent_path"] else 0.0
        # Dangerous shortcut penalty
        shortcuts = DANGEROUS_CONSENT_SHORTCUTS.get(sid, [])
        penalty = sum(0.30 for s in shortcuts if s in done)
        scores.append(max(0.0, 0.50 * step_score + 0.50 * path_correct - penalty))
    return sum(scores) / max(len(scores), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MedicalTriageEnv:
    """
    OpenEnv-compliant Medical Triage Environment.

    Supports all 10 clinical tasks with multi-step sequential decision-making.
    Rewards are graded using real clinical protocols — not simple classification.
    """

    def __init__(self, task_type: TaskType, seed: int = 42):
        self.task_type = task_type
        self.seed = seed
        self.rng = random.Random(seed)
        self._state: Dict[str, Any] = {}
        self.step_count = 0
        self.max_steps = MAX_STEPS[task_type]

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> TriageObservation:
        self.step_count = 0
        self.rng = random.Random(self.seed)
        self._state = {}
        return self._init_task()

    def step(self, action: TriageAction) -> StepResult:
        self.step_count += 1
        handlers = {
            TaskType.ESI_ASSIGNMENT:     self._step_esi,
            TaskType.INTAKE_INTERVIEW:   self._step_intake,
            TaskType.QUEUE_MANAGEMENT:   self._step_queue,
            TaskType.MEDICATION_CHECK:   self._step_medication,
            TaskType.DISCHARGE_PLANNING: self._step_discharge,
            TaskType.MASS_CASUALTY:      self._step_mci,
            TaskType.SEPSIS_SCREENING:   self._step_sepsis,
            TaskType.BED_ALLOCATION:     self._step_bed,
            TaskType.SHIFT_HANDOFF:      self._step_handoff,
            TaskType.CONSENT_ASSESSMENT: self._step_consent,
        }
        return handlers[self.task_type](action)

    # ── Task Initializers ──────────────────────────────────────────────────

    def _init_task(self) -> TriageObservation:
        t = self.task_type

        if t == TaskType.ESI_ASSIGNMENT:
            case = self.rng.choice(ESI_CASES)
            patient = Patient(
                patient_id=case["patient_id"], age=case["age"], gender=case["gender"],
                chief_complaint=case["chief_complaint"], vitals=case["vitals"],
                correct_esi_level=case["correct_esi_level"],
            )
            self._state = {"patient": patient, "esi_assigned": False}
            return TriageObservation(
                task_type=t, patient=patient, step_number=0,
                available_actions=["assign_esi:1", "assign_esi:2", "assign_esi:3",
                                   "assign_esi:4", "assign_esi:5"],
                message=(
                    f"PATIENT ARRIVES: {patient.patient_id} | {patient.age}yo {patient.gender}\n"
                    f"Chief Complaint: {patient.chief_complaint}\n"
                    f"Vitals — HR:{patient.vitals.heart_rate} "
                    f"BP:{patient.vitals.bp_systolic}/{patient.vitals.bp_diastolic} "
                    f"RR:{patient.vitals.respiratory_rate} "
                    f"SpO2:{patient.vitals.oxygen_saturation}% "
                    f"Temp:{patient.vitals.temperature_f}°F "
                    f"GCS:{patient.vitals.gcs} "
                    f"Pain:{patient.vitals.pain_scale}/10\n"
                    "Assign ESI level (1=Immediate to 5=Non-urgent)."
                ),
            )

        elif t == TaskType.INTAKE_INTERVIEW:
            case = self.rng.choice([c for c in ESI_CASES if c["patient_id"] in PATIENT_INTAKE_ANSWERS])
            patient = Patient(
                patient_id=case["patient_id"], age=case["age"], gender=case["gender"],
                chief_complaint=case["chief_complaint"], vitals=case["vitals"],
            )
            self._state = {"patient": patient, "collected": {}}
            available = [f"ask:{f}" for f in INTAKE_REQUIRED_FIELDS] + ["complete_intake"]
            return TriageObservation(
                task_type=t, patient=patient, step_number=0,
                available_actions=available,
                message=(
                    f"BEGIN INTAKE: {patient.patient_id} | {patient.age}yo {patient.gender}\n"
                    f"Chief Complaint: {patient.chief_complaint}\n"
                    f"Collect all required intake fields using ask:<field>. "
                    f"Fields: {', '.join(INTAKE_REQUIRED_FIELDS)}\n"
                    "Call complete_intake when finished."
                ),
            )

        elif t == TaskType.QUEUE_MANAGEMENT:
            patients = [
                Patient(
                    patient_id=c["patient_id"], age=c["age"], gender=c["gender"],
                    chief_complaint=c["chief_complaint"], vitals=c["vitals"],
                    correct_esi_level=c["correct_esi_level"],
                    arrival_minutes_ago=c["arrival_minutes_ago"],
                )
                for c in QUEUE_SCENARIO
            ]
            self._state = {
                "queue": patients,
                "agent_order": [],
                "deteriorated_ids": set(),
                "deterioration_responses": set(),
                "deterioration_step": self.rng.randint(5, 12),
                "deterioration_done": False,
            }
            return TriageObservation(
                task_type=t, queue=patients, step_number=0,
                available_actions=(
                    [f"prioritize:{p.patient_id}" for p in patients] +
                    [f"get_vitals:{p.patient_id}" for p in patients] +
                    ["finalize_queue"]
                ),
                message=(
                    "QUEUE TRIAGE: Order the waiting room by clinical priority.\n" +
                    "\n".join(
                        f"  {p.patient_id} ({p.age}yo {p.gender}): {p.chief_complaint} "
                        f"[arrived {p.arrival_minutes_ago}min ago]"
                        for p in patients
                    ) +
                    "\nUse prioritize:<id> to set order. Call finalize_queue when done."
                ),
            )

        elif t == TaskType.MEDICATION_CHECK:
            case = self.rng.choice(MEDICATION_CASES)
            self._state = {
                "case": case, "flagged_allergies": set(),
                "flagged_interactions": set(), "approved_meds": set(), "wrong_approvals": 0,
            }
            actions = (
                [f"flag_allergy:{m}" for m in case.proposed_medications] +
                [f"flag_interaction:{m}" for m in case.proposed_medications] +
                [f"approve:{m}" for m in case.proposed_medications] +
                ["complete_review"]
            )
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    f"MEDICATION SAFETY REVIEW: {case.case_id} | {case.age}yo {case.gender}\n"
                    f"Complaint: {case.chief_complaint}\n"
                    f"Known Allergies: {', '.join(case.known_allergies)}\n"
                    f"Current Medications: {', '.join(case.current_medications)}\n"
                    f"Proposed: {', '.join(case.proposed_medications)}\n"
                    "Flag allergy conflicts, drug interactions, or approve safe medications."
                ),
            )

        elif t == TaskType.DISCHARGE_PLANNING:
            case = self.rng.choice(DISCHARGE_CASES)
            self._state = {
                "case": case, "given_instructions": set(),
                "given_warnings": set(), "given_followup_days": None,
            }
            actions = (
                [f"add_instruction:{k}" for k in DISCHARGE_INSTRUCTIONS_CATALOG] +
                [f"add_warning:{k}" for k in DISCHARGE_WARNING_CATALOG] +
                ["set_followup:2", "set_followup:3", "set_followup:5", "set_followup:7",
                 "set_followup:10", "set_followup:14"] +
                ["complete_discharge"]
            )
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    f"DISCHARGE PLANNING: {case.case_id} | {case.age}yo {case.gender}\n"
                    f"Diagnosis: {case.diagnosis}\n"
                    f"Treatment Given: {case.treatment_given}\n"
                    f"Discharge Medications: {', '.join(case.discharged_medications)}\n"
                    "Add appropriate instructions, warnings, and follow-up timing. "
                    "Call complete_discharge when done."
                ),
            )

        elif t == TaskType.MASS_CASUALTY:
            scenario = self.rng.choice(MCI_SCENARIOS)
            self._state = {"scenario": scenario, "agent_tags": {}, "inspected": set()}
            actions = []
            for p in scenario:
                for tag in VALID_MCI_TAGS:
                    actions.append(f"tag:{p['patient_id']}:{tag}")
                actions.append(f"inspect:{p['patient_id']}")
            actions.append("finalize_scene")
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    "MASS CASUALTY INCIDENT — START TRIAGE PROTOCOL\n"
                    "Scene: " + scenario[0]["mechanism"].split(";")[0].split(",")[0] + "\n"
                    f"{len(scenario)} victims found. Apply START triage:\n"
                    "  1. Can walk? → GREEN\n"
                    "  2. No breath after repositioning? → BLACK\n"
                    "  3. RR>30 or <10? → RED\n"
                    "  4. No radial pulse? → RED\n"
                    "  5. Cannot follow commands? → RED\n"
                    "  6. Stable → YELLOW\n"
                    "Use inspect:<id> to examine, tag:<id>:<color> to tag. "
                    "Call finalize_scene when all tagged.\n" +
                    "\n".join(f"  {p['patient_id']}: {p['mechanism']}" for p in scenario)
                ),
            )

        elif t == TaskType.SEPSIS_SCREENING:
            ward = self.rng.choice(SEPSIS_WARDS)
            self._state = {
                "ward": ward,
                "screened_ids": set(),
                "flagged_ids": set(),
                "cleared_ids": set(),
                "ordered_items": {},
            }
            actions = []
            for p in ward:
                actions.append(f"screen:{p['patient_id']}")
                actions.append(f"flag_sepsis:{p['patient_id']}")
                actions.append(f"clear:{p['patient_id']}")
                for item in SEPSIS_BUNDLE_ITEMS:
                    actions.append(f"order:{p['patient_id']}:{item}")
            actions.append("complete_screening")
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    "SEPSIS SCREENING — qSOFA CRITERIA\n"
                    "Score 1 point each: RR≥22 | Altered Mental Status (GCS<15) | SBP≤100\n"
                    "Score ≥2 = HIGH RISK → Initiate sepsis bundle:\n"
                    f"  {', '.join(SEPSIS_BUNDLE_ITEMS)}\n\n" +
                    "Patients to screen:\n" +
                    "\n".join(
                        f"  {p['patient_id']} ({p['age']}yo {p['gender']}): "
                        f"{p['chief_complaint']} | "
                        f"RR:{p['vitals'].respiratory_rate} "
                        f"SBP:{p['vitals'].bp_systolic} "
                        f"GCS:{p['vitals'].gcs} "
                        f"Temp:{p['vitals'].temperature_f}°F"
                        for p in ward
                    ) +
                    "\nUse screen→flag_sepsis/clear→order. Call complete_screening when done."
                ),
            )

        elif t == TaskType.BED_ALLOCATION:
            scenario = self.rng.choice(BED_SCENARIOS)
            self._state = {
                "scenario": scenario,
                "assignments": {},
                "deferred": set(),
            }
            actions = []
            for p in scenario:
                for bed_type in BED_INVENTORY:
                    actions.append(f"assign:{p['patient_id']}:{bed_type}")
                actions.append(f"defer:{p['patient_id']}")
                actions.append(f"get_info:{p['patient_id']}")
            actions.append("finalize_beds")
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    "BED ALLOCATION — Match patients to appropriate ED beds\n"
                    f"Inventory: {', '.join(f'{bt}×{n}' for bt, n in BED_INVENTORY.items())}\n\n"
                    "Patients:\n" +
                    "\n".join(
                        f"  {p['patient_id']} (P{p['priority']}) {p['age']}yo {p['gender']}: "
                        f"{p['chief_complaint']}"
                        for p in sorted(scenario, key=lambda x: x["priority"])
                    ) +
                    "\nUse assign:<patient>:<bed_type> or defer:<patient>. "
                    "Call finalize_beds when done."
                ),
            )

        elif t == TaskType.SHIFT_HANDOFF:
            scenario = self.rng.choice(HANDOFF_SCENARIOS)
            self._state = {
                "scenario": scenario,
                "reported_fields": {p["patient_id"]: set() for p in scenario},
            }
            actions = []
            for p in scenario:
                for field in SBAR_ALL_FIELDS:
                    actions.append(f"report:{p['patient_id']}:{field}")
            actions.append("complete_handoff")
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    "SHIFT HANDOFF — SBAR Protocol\n"
                    "Hand off these critical patients to the incoming nurse.\n"
                    "SBAR fields: situation_chief_complaint | situation_current_status | "
                    "situation_acuity_level | background_medical_history | "
                    "background_current_medications | background_allergies | "
                    "assessment_esi_level | assessment_vital_trend | assessment_key_concern | "
                    "recommendation_next_action | recommendation_pending_orders | "
                    "recommendation_watch_for\n\n" +
                    "Patients:\n" +
                    "\n".join(
                        f"  {p['patient_id']} ({p['age']}yo {p['gender']}): "
                        f"{p['field_values']['situation_chief_complaint'][:60]}..."
                        for p in scenario
                    ) +
                    "\nUse report:<patient_id>:<sbar_field>. Call complete_handoff when done."
                ),
            )

        elif t == TaskType.CONSENT_ASSESSMENT:
            scenario_list = list(CONSENT_SCENARIOS)
            self._state = {
                "scenarios": scenario_list,
                "current_idx": 0,
                "assessed_criteria": {sc["scenario_id"]: set() for sc in scenario_list},
                "declared_capacity": {sc["scenario_id"]: None for sc in scenario_list},
                "chosen_paths": {sc["scenario_id"]: None for sc in scenario_list},
                "completed_steps": {sc["scenario_id"]: set() for sc in scenario_list},
                "finalized": set(),
            }
            sc = scenario_list[0]
            actions = (
                [f"assess:{sc['scenario_id']}:{c}" for c in CAPACITY_CRITERIA] +
                [f"step:{sc['scenario_id']}:{s}" for s in CONSENT_STEPS_CATALOG] +
                [f"declare_capacity:{sc['scenario_id']}:has_capacity",
                 f"declare_capacity:{sc['scenario_id']}:lacks_capacity"] +
                [f"consent_path:{sc['scenario_id']}:{path}" for path in
                 ["informed_consent", "informed_refusal", "emergent_exception",
                  "surrogate_consent", "assent_minor", "court_ordered"]] +
                [f"finalize:{sc['scenario_id']}"]
            )
            return TriageObservation(
                task_type=t, step_number=0, available_actions=actions,
                message=(
                    f"CONSENT & CAPACITY ASSESSMENT\n"
                    f"Scenario {sc['scenario_id']}: {sc['age']}yo {sc['gender']}\n"
                    f"Procedure: {sc['procedure_needed']} (Urgency: {sc['clinical_urgency']})\n"
                    f"Presentation: {sc['presentation']}\n\n"
                    "Assess capacity criteria, complete required steps, choose consent path, "
                    "then finalize. Repeat for all scenarios."
                ),
            )

        raise ValueError(f"Unknown task type: {t}")

    # ── Task Step Handlers ─────────────────────────────────────────────────

    def _step_esi(self, action: TriageAction) -> StepResult:
        patient = self._state["patient"]
        done = self.step_count >= self.max_steps

        if action.action_type == "assign_esi" or action.content.startswith("assign_esi:"):
            raw = action.content.split(":")[-1].strip()
            try:
                level = int(raw)
                reward = grade_esi(level, patient.correct_esi_level)
                done = True
                msg = (f"ESI {level} assigned. Correct: ESI {patient.correct_esi_level}. "
                       f"Score: {reward:.2f}")
            except ValueError:
                reward, msg = 0.0, "Invalid ESI level. Use assign_esi:1 through assign_esi:5."
        else:
            reward, msg = 0.0, f"Unknown action: {action.content}"

        obs = TriageObservation(task_type=self.task_type, patient=patient,
                                step_number=self.step_count, done=done, message=msg)
        return StepResult(observation=obs, reward=reward, done=done,
                          info={"correct_esi": patient.correct_esi_level})

    def _step_intake(self, action: TriageAction) -> StepResult:
        patient = self._state["patient"]
        collected = self._state["collected"]
        done = False
        msg = ""

        if action.content.startswith("ask:"):
            field = action.content.split(":", 1)[1].strip()
            if field in INTAKE_REQUIRED_FIELDS:
                answers = PATIENT_INTAKE_ANSWERS.get(patient.patient_id, {})
                answer = answers.get(field, "Information not available.")
                collected[field] = answer
                msg = f"Patient says: {answer}"
            else:
                msg = f"Unknown field '{field}'. Valid: {', '.join(INTAKE_REQUIRED_FIELDS)}"
        elif action.content == "complete_intake":
            reward = grade_intake(collected, self.step_count)
            done = True
            covered = set(INTAKE_REQUIRED_FIELDS) & set(collected.keys())
            msg = (f"Intake complete. Covered {len(covered)}/{len(INTAKE_REQUIRED_FIELDS)} fields. "
                   f"Score: {reward:.2f}")
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, patient=patient,
                                              step_number=self.step_count, done=True, message=msg),
                reward=reward, done=True, info={"collected_fields": list(collected.keys())},
            )
        else:
            msg = f"Unknown action: {action.content}"

        if self.step_count >= self.max_steps:
            reward = grade_intake(collected, self.step_count)
            done = True
            msg += f" [Time limit reached] Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, patient=patient,
                                              step_number=self.step_count, done=True, message=msg),
                reward=reward, done=True, info={},
            )

        obs = TriageObservation(
            task_type=self.task_type, patient=patient, step_number=self.step_count,
            done=False, message=msg,
            available_actions=[f"ask:{f}" for f in INTAKE_REQUIRED_FIELDS
                               if f not in collected] + ["complete_intake"],
        )
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    def _step_queue(self, action: TriageAction) -> StepResult:
        queue = self._state["queue"]
        agent_order = self._state["agent_order"]
        det_step = self._state["deterioration_step"]
        det_done = self._state["deterioration_done"]
        msg = ""

        # Trigger deterioration event
        if self.step_count == det_step and not det_done:
            det_patient = self.rng.choice([p for p in queue if p.correct_esi_level >= 3])
            det_patient.deteriorated = True
            det_patient.correct_esi_level = max(1, det_patient.correct_esi_level - 1)
            self._state["deteriorated_ids"].add(det_patient.patient_id)
            self._state["deterioration_done"] = True
            msg = (f"⚠️  ALERT: {det_patient.patient_id} has deteriorated! "
                   f"Vitals worsening — re-triage required. ")

        if action.content.startswith("prioritize:"):
            pid = action.content.split(":", 1)[1].strip()
            if pid not in agent_order:
                agent_order.append(pid)
            if pid in self._state.get("deteriorated_ids", set()):
                self._state["deterioration_responses"].add(pid)
            msg += f"Added {pid} to priority queue (position {len(agent_order)})."
        elif action.content.startswith("get_vitals:"):
            pid = action.content.split(":", 1)[1].strip()
            p = next((x for x in queue if x.patient_id == pid), None)
            if p:
                msg += (f"{pid} vitals — HR:{p.vitals.heart_rate} "
                        f"BP:{p.vitals.bp_systolic}/{p.vitals.bp_diastolic} "
                        f"RR:{p.vitals.respiratory_rate} SpO2:{p.vitals.oxygen_saturation}% "
                        f"GCS:{p.vitals.gcs}")
        elif action.content == "finalize_queue":
            esi_map = {p.patient_id: p.correct_esi_level for p in queue}
            reward = grade_queue(agent_order, esi_map,
                                 self._state["deteriorated_ids"],
                                 self._state["deterioration_responses"])
            msg += f"Queue finalized. Order: {agent_order}. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, queue=queue,
                                              step_number=self.step_count, done=True, message=msg),
                reward=reward, done=True, info={"agent_order": agent_order},
            )

        done = self.step_count >= self.max_steps
        if done:
            esi_map = {p.patient_id: p.correct_esi_level for p in queue}
            reward = grade_queue(agent_order, esi_map,
                                 self._state["deteriorated_ids"],
                                 self._state["deterioration_responses"])
            msg += f" [Time limit] Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, queue=queue,
                                              step_number=self.step_count, done=True, message=msg),
                reward=reward, done=True, info={},
            )

        obs = TriageObservation(
            task_type=self.task_type, queue=queue,
            step_number=self.step_count, done=False, message=msg,
            available_actions=(
                [f"prioritize:{p.patient_id}" for p in queue if p.patient_id not in agent_order] +
                [f"get_vitals:{p.patient_id}" for p in queue] +
                ["finalize_queue"]
            ),
        )
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    def _step_medication(self, action: TriageAction) -> StepResult:
        case = self._state["case"]
        flagged_a = self._state["flagged_allergies"]
        flagged_i = self._state["flagged_interactions"]
        approved = self._state["approved_meds"]
        msg = ""

        if action.content.startswith("flag_allergy:"):
            med = action.content.split(":", 1)[1].strip()
            flagged_a.add(med)
            msg = f"Allergy flag set for {med}."
        elif action.content.startswith("flag_interaction:"):
            med = action.content.split(":", 1)[1].strip()
            flagged_i.add(med)
            msg = f"Interaction flag set for {med}."
        elif action.content.startswith("approve:"):
            med = action.content.split(":", 1)[1].strip()
            true_unsafe = {m for m, _ in case.true_allergy_flags} | {m for m, _ in case.true_interaction_flags}
            approved.add(med)
            if med in true_unsafe:
                self._state["wrong_approvals"] += 1
                msg = f"⚠️  WARNING: {med} approved but has known allergy/interaction risk."
            else:
                msg = f"{med} approved as safe."
        elif action.content == "complete_review":
            reward = grade_medication(flagged_a, flagged_i, approved, case, self._state["wrong_approvals"])
            msg = f"Review complete. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=msg),
                reward=reward, done=True, info={},
            )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_medication(flagged_a, flagged_i, approved, case, self._state["wrong_approvals"])
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] {msg} Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg),
            reward=0.0, done=False, info={},
        )

    def _step_discharge(self, action: TriageAction) -> StepResult:
        case = self._state["case"]
        instructions = self._state["given_instructions"]
        warnings = self._state["given_warnings"]
        msg = ""

        if action.content.startswith("add_instruction:"):
            key = action.content.split(":", 1)[1].strip()
            if key in DISCHARGE_INSTRUCTIONS_CATALOG:
                instructions.add(key)
                msg = f"Added instruction: {DISCHARGE_INSTRUCTIONS_CATALOG[key]}"
            else:
                msg = f"Unknown instruction key: {key}"
        elif action.content.startswith("add_warning:"):
            key = action.content.split(":", 1)[1].strip()
            if key in DISCHARGE_WARNING_CATALOG:
                warnings.add(key)
                msg = f"Added return warning: {DISCHARGE_WARNING_CATALOG[key]}"
            else:
                msg = f"Unknown warning key: {key}"
        elif action.content.startswith("set_followup:"):
            days = int(action.content.split(":", 1)[1].strip())
            self._state["given_followup_days"] = days
            msg = f"Follow-up set to {days} days."
        elif action.content == "complete_discharge":
            reward = grade_discharge(instructions, warnings,
                                     self._state["given_followup_days"] or 0, case)
            msg = f"Discharge complete. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=msg),
                reward=reward, done=True, info={},
            )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_discharge(instructions, warnings,
                                     self._state["given_followup_days"] or 0, case)
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg),
            reward=0.0, done=False, info={},
        )

    def _step_mci(self, action: TriageAction) -> StepResult:
        scenario = self._state["scenario"]
        agent_tags = self._state["agent_tags"]
        inspected = self._state["inspected"]
        msg = ""

        if action.content.startswith("inspect:"):
            pid = action.content.split(":", 1)[1].strip()
            patient = next((p for p in scenario if p["patient_id"] == pid), None)
            if patient:
                inspected.add(pid)
                msg = (
                    f"INSPECT {pid}: "
                    f"Can walk: {'Yes' if patient['can_walk'] else 'No'} | "
                    f"Breathing: {'Yes' if patient['spontaneous_breath'] else 'No (after repositioning)'} | "
                    f"RR: {patient['respiratory_rate'] or 'N/A'} | "
                    f"Radial pulse: {'Present' if patient['radial_pulse_present'] else 'ABSENT'} | "
                    f"Follows commands: {'Yes' if patient['can_follow_commands'] else 'No'}"
                )
            else:
                msg = f"Patient {pid} not found."

        elif action.content.startswith("tag:"):
            parts = action.content.split(":")
            if len(parts) == 3:
                _, pid, tag = parts
                tag = tag.upper()
                if tag in VALID_MCI_TAGS:
                    agent_tags[pid] = tag
                    msg = f"Tagged {pid} as {tag}."
                else:
                    msg = f"Invalid tag '{tag}'. Use: {', '.join(VALID_MCI_TAGS)}"
            else:
                msg = "Format: tag:<patient_id>:<RED|YELLOW|GREEN|BLACK>"

        elif action.content == "finalize_scene":
            reward = grade_mci(agent_tags, scenario)
            summary = " | ".join(f"{pid}:{tag}" for pid, tag in agent_tags.items())
            msg = f"Scene finalized. Tags: {summary}. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=msg),
                reward=reward, done=True,
                info={"agent_tags": agent_tags,
                      "correct_tags": {p["patient_id"]: p["correct_tag"] for p in scenario}},
            )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_mci(agent_tags, scenario)
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        tagged_ids = set(agent_tags.keys())
        remaining = [p["patient_id"] for p in scenario if p["patient_id"] not in tagged_ids]
        actions = []
        for p in scenario:
            if p["patient_id"] not in inspected:
                actions.append(f"inspect:{p['patient_id']}")
            for tag in VALID_MCI_TAGS:
                actions.append(f"tag:{p['patient_id']}:{tag}")
        actions.append("finalize_scene")

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg, available_actions=actions),
            reward=0.0, done=False, info={"untagged": remaining},
        )

    def _step_sepsis(self, action: TriageAction) -> StepResult:
        ward = self._state["ward"]
        flagged = self._state["flagged_ids"]
        cleared = self._state["cleared_ids"]
        orders = self._state["ordered_items"]
        msg = ""

        if action.content.startswith("screen:"):
            pid = action.content.split(":", 1)[1].strip()
            patient = next((p for p in ward if p["patient_id"] == pid), None)
            if patient:
                self._state["screened_ids"].add(pid)
                v = patient["vitals"]
                criteria = []
                if v.respiratory_rate and v.respiratory_rate >= 22:
                    criteria.append(f"RR {v.respiratory_rate} ≥ 22 ✓")
                if v.gcs and v.gcs < 15:
                    criteria.append(f"GCS {v.gcs} < 15 ✓")
                if v.bp_systolic and v.bp_systolic <= 100:
                    criteria.append(f"SBP {v.bp_systolic} ≤ 100 ✓")
                score = patient["qsofa_score"]
                msg = (f"qSOFA screen {pid}: Score {score}/3. "
                       f"Criteria met: {', '.join(criteria) if criteria else 'None'}. "
                       f"{'HIGH RISK — consider sepsis bundle' if score >= 2 else 'Low risk.'}")
            else:
                msg = f"Patient {pid} not found."

        elif action.content.startswith("flag_sepsis:"):
            pid = action.content.split(":", 1)[1].strip()
            flagged.add(pid)
            orders.setdefault(pid, set())
            msg = f"{pid} flagged for sepsis — initiate bundle."

        elif action.content.startswith("clear:"):
            pid = action.content.split(":", 1)[1].strip()
            cleared.add(pid)
            msg = f"{pid} cleared — no sepsis concern."

        elif action.content.startswith("order:"):
            parts = action.content.split(":")
            if len(parts) == 3:
                _, pid, item = parts
                if item in SEPSIS_BUNDLE_ITEMS:
                    orders.setdefault(pid, set()).add(item)
                    msg = f"Ordered {item} for {pid}."
                else:
                    msg = f"Unknown bundle item '{item}'. Valid: {', '.join(SEPSIS_BUNDLE_ITEMS)}"
            else:
                msg = "Format: order:<patient_id>:<bundle_item>"

        elif action.content == "complete_screening":
            reward = grade_sepsis(flagged, orders, ward)
            msg = f"Screening complete. Flagged: {flagged}. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=msg),
                reward=reward, done=True, info={"flagged": list(flagged), "orders": {k: list(v) for k, v in orders.items()}},
            )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_sepsis(flagged, orders, ward)
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg),
            reward=0.0, done=False, info={},
        )

    def _step_bed(self, action: TriageAction) -> StepResult:
        scenario = self._state["scenario"]
        assignments = self._state["assignments"]
        msg = ""

        if action.content.startswith("get_info:"):
            pid = action.content.split(":", 1)[1].strip()
            patient = next((p for p in scenario if p["patient_id"] == pid), None)
            if patient:
                msg = (f"{pid}: {patient['age']}yo {patient['gender']} — "
                       f"{patient['chief_complaint']} | Priority {patient['priority']} | "
                       f"Note: {patient['clinical_note']}")
            else:
                msg = f"Patient {pid} not found."

        elif action.content.startswith("assign:"):
            parts = action.content.split(":")
            if len(parts) == 3:
                _, pid, bed_type = parts
                if bed_type in BED_INVENTORY:
                    assignments[pid] = bed_type
                    msg = f"Assigned {pid} to {bed_type}."
                else:
                    msg = f"Unknown bed type '{bed_type}'. Valid: {', '.join(BED_INVENTORY.keys())}"
            else:
                msg = "Format: assign:<patient_id>:<bed_type>"

        elif action.content.startswith("defer:"):
            pid = action.content.split(":", 1)[1].strip()
            self._state["deferred"].add(pid)
            msg = f"{pid} deferred — awaiting bed availability."

        elif action.content == "finalize_beds":
            reward = grade_bed_allocation(assignments, scenario)
            summary = " | ".join(f"{pid}:{bt}" for pid, bt in assignments.items())
            msg = f"Beds finalized. {summary}. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=msg),
                reward=reward, done=True,
                info={"assignments": assignments,
                      "correct": {p["patient_id"]: p["required_bed"] for p in scenario}},
            )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_bed_allocation(assignments, scenario)
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg),
            reward=0.0, done=False, info={},
        )

    def _step_handoff(self, action: TriageAction) -> StepResult:
        scenario = self._state["scenario"]
        reported = self._state["reported_fields"]
        msg = ""

        if action.content.startswith("report:"):
            parts = action.content.split(":")
            if len(parts) == 3:
                _, pid, field = parts
                patient = next((p for p in scenario if p["patient_id"] == pid), None)
                if patient and field in SBAR_ALL_FIELDS:
                    reported[pid].add(field)
                    value = patient["field_values"].get(field, "N/A")
                    msg = f"SBAR reported [{pid}.{field}]: {value}"
                elif field not in SBAR_ALL_FIELDS:
                    msg = f"Unknown SBAR field '{field}'."
                else:
                    msg = f"Patient {pid} not found."
            else:
                msg = "Format: report:<patient_id>:<sbar_field>"

        elif action.content == "complete_handoff":
            reward = grade_handoff(reported, scenario)
            coverage = {pid: f"{len(fields)}/{len(scenario[i]['required_fields'])}"
                        for i, (pid, fields) in enumerate(reported.items()) if i < len(scenario)}
            msg = f"Handoff complete. Coverage: {coverage}. Score: {reward:.2f}"
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=msg),
                reward=reward, done=True, info={"coverage": coverage},
            )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_handoff(reported, scenario)
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg),
            reward=0.0, done=False, info={},
        )

    def _step_consent(self, action: TriageAction) -> StepResult:
        scenarios = self._state["scenarios"]
        assessed = self._state["assessed_criteria"]
        declared = self._state["declared_capacity"]
        chosen = self._state["chosen_paths"]
        steps_done = self._state["completed_steps"]
        finalized = self._state["finalized"]
        msg = ""

        if action.content.startswith("assess:"):
            parts = action.content.split(":")
            if len(parts) == 3:
                _, sid, criterion = parts
                if criterion in CAPACITY_CRITERIA:
                    assessed.setdefault(sid, set()).add(criterion)
                    sc = next((s for s in scenarios if s["scenario_id"] == sid), None)
                    if sc:
                        result = sc["capacity_criteria"].get(criterion, False)
                        msg = f"Capacity criterion [{criterion}] for {sid}: {'MET ✓' if result else 'NOT MET ✗'}"
                else:
                    msg = f"Unknown criterion '{criterion}'. Valid: {', '.join(CAPACITY_CRITERIA)}"
            else:
                msg = "Format: assess:<scenario_id>:<criterion>"

        elif action.content.startswith("step:"):
            parts = action.content.split(":", 2)
            if len(parts) == 3:
                _, sid, step_key = parts
                if step_key in CONSENT_STEPS_CATALOG:
                    steps_done.setdefault(sid, set()).add(step_key)
                    # Check for dangerous shortcuts
                    if step_key in DANGEROUS_CONSENT_SHORTCUTS.get(sid, []):
                        msg = f"⚠️  DANGEROUS SHORTCUT: '{step_key}' is inappropriate for {sid}. Penalty will apply."
                    else:
                        msg = f"Step completed [{sid}]: {CONSENT_STEPS_CATALOG[step_key]}"
                else:
                    msg = f"Unknown step '{step_key}'. Valid: {', '.join(CONSENT_STEPS_CATALOG.keys())}"
            else:
                msg = "Format: step:<scenario_id>:<step_key>"

        elif action.content.startswith("declare_capacity:"):
            parts = action.content.split(":")
            if len(parts) == 3:
                _, sid, cap_status = parts
                declared[sid] = (cap_status == "has_capacity")
                msg = f"Capacity for {sid} declared: {cap_status}."
            else:
                msg = "Format: declare_capacity:<scenario_id>:has_capacity|lacks_capacity"

        elif action.content.startswith("consent_path:"):
            parts = action.content.split(":", 2)
            if len(parts) == 3:
                _, sid, path = parts
                chosen[sid] = path
                msg = f"Consent path for {sid} set to: {path}."
            else:
                msg = "Format: consent_path:<scenario_id>:<path>"

        elif action.content.startswith("finalize:"):
            sid = action.content.split(":", 1)[1].strip()
            finalized.add(sid)
            msg = f"Scenario {sid} finalized."
            if len(finalized) == len(scenarios):
                reward = grade_consent(assessed, declared, chosen, steps_done)
                msg += f" All scenarios complete. Score: {reward:.2f}"
                return StepResult(
                    observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                                  done=True, message=msg),
                    reward=reward, done=True, info={"chosen_paths": chosen},
                )
        else:
            msg = f"Unknown action: {action.content}"

        done = self.step_count >= self.max_steps
        if done:
            reward = grade_consent(assessed, declared, chosen, steps_done)
            return StepResult(
                observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                              done=True, message=f"[Time limit] Score: {reward:.2f}"),
                reward=reward, done=True, info={},
            )

        # Show next unfinalised scenario
        current_idx = next((i for i, sc in enumerate(scenarios)
                            if sc["scenario_id"] not in finalized), None)
        current_scenario = scenarios[current_idx] if current_idx is not None else None
        hint = ""
        if current_scenario:
            sid = current_scenario["scenario_id"]
            hint = (f"\nCurrent scenario: {sid} — {current_scenario['age']}yo "
                    f"{current_scenario['gender']} | {current_scenario['procedure_needed']}")

        return StepResult(
            observation=TriageObservation(task_type=self.task_type, step_number=self.step_count,
                                          done=False, message=msg + hint),
            reward=0.0, done=False, info={},
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def create_environment(task_type: str, seed: int = 42) -> MedicalTriageEnv:
    """Create an environment for the given task type string."""
    try:
        tt = TaskType(task_type)
    except ValueError:
        valid = [t.value for t in TaskType]
        raise ValueError(f"Unknown task '{task_type}'. Valid tasks: {valid}")
    return MedicalTriageEnv(task_type=tt, seed=seed)
