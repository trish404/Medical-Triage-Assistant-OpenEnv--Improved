"""
Comprehensive Test Suite — Medical Triage Assistant OpenEnv v3.0
================================================================
Tests every task, every grader, every action validator, server endpoints,
manifest completeness, and edge-case penalty mechanics.

Run locally (no server needed for env tests):
    python test_comprehensive.py

Run with live server (for HTTP tests):
    SERVER_URL=http://localhost:7860 python test_comprehensive.py

Exit codes:
    0 = all tests passed
    1 = one or more tests failed
"""

from __future__ import annotations
import os, sys, time, traceback, importlib, subprocess
from typing import Any, Callable, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Tiny test framework
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
SKIP = "\033[33m SKIP\033[0m"
BOLD = "\033[1m"
RST  = "\033[0m"

results: List[Dict[str, Any]] = []

def test(name: str, fn: Callable, skip_reason: str = "") -> bool:
    if skip_reason:
        print(f"{SKIP} {name}  ({skip_reason})")
        results.append({"name": name, "status": "skip"})
        return True
    try:
        fn()
        print(f"{PASS} {name}")
        results.append({"name": name, "status": "pass"})
        return True
    except Exception as e:
        tb = traceback.format_exc().strip().split("\n")[-1]
        print(f"{FAIL} {name}\n       {tb}")
        results.append({"name": name, "status": "fail", "error": str(e)})
        return False

def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}  expected={b!r}  got={a!r}"

def assert_in(a, b, msg=""):
    assert a in b, f"{msg}  {a!r} not in {b!r}"

def assert_range(v, lo, hi, msg=""):
    assert lo <= v <= hi, f"{msg}  {v} not in [{lo}, {hi}]"

def section(title: str):
    print(f"\n{BOLD}── {title} {'─'*(55-len(title))}{RST}")

# ──────────────────────────────────────────────────────────────────────────────
#  Import environment (must work without server)
# ──────────────────────────────────────────────────────────────────────────────

section("ENVIRONMENT IMPORT")

env_module = None

def _import_env():
    global env_module
    import medical_triage_env as m
    env_module = m

test("medical_triage_env imports cleanly", _import_env)

if env_module is None:
    print("\nCannot continue without environment module. Fix import errors first.")
    sys.exit(1)

from medical_triage_env import (
    MedicalTriageEnv, TaskType, TriageAction, create_environment,
    grade_esi, grade_intake, grade_queue, grade_medication, grade_discharge,
    grade_mci, grade_sepsis, grade_bed_allocation, grade_handoff, grade_consent,
    ESI_CASES, INTAKE_REQUIRED_FIELDS, QUEUE_SCENARIO, MEDICATION_CASES,
    DISCHARGE_CASES, MCI_SCENARIOS, SEPSIS_WARDS, BED_SCENARIOS,
    HANDOFF_SCENARIOS, CONSENT_SCENARIOS,
    SEPSIS_BUNDLE_ITEMS, BED_INVENTORY, SBAR_ALL_FIELDS,
    CAPACITY_CRITERIA, CONSENT_STEPS_CATALOG, DANGEROUS_CONSENT_SHORTCUTS,
    MAX_STEPS, TASK_DIFFICULTY,
)

ALL_TASKS = [t.value for t in TaskType]
EXPECTED_TASKS = [
    "esi_assignment", "intake_interview", "queue_management",
    "medication_check", "discharge_planning", "mass_casualty",
    "sepsis_screening", "bed_allocation", "shift_handoff", "consent_assessment",
]

# ──────────────────────────────────────────────────────────────────────────────
#  1. TASK REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

section("TASK REGISTRY")

def _task_count():
    assert len(ALL_TASKS) == 10, f"Expected 10 tasks, got {len(ALL_TASKS)}"

def _task_names():
    for name in EXPECTED_TASKS:
        assert name in ALL_TASKS, f"Missing task: {name}"

def _difficulty_coverage():
    difficulties = set(TASK_DIFFICULTY.values())
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties

def _max_steps_defined():
    for t in TaskType:
        assert t in MAX_STEPS, f"MAX_STEPS missing for {t}"
        assert MAX_STEPS[t] >= 5, f"MAX_STEPS[{t}] too small: {MAX_STEPS[t]}"

test("10 tasks registered",         _task_count)
test("all expected task names exist",_task_names)
test("easy/medium/hard all present", _difficulty_coverage)
test("MAX_STEPS defined for all",    _max_steps_defined)

# ──────────────────────────────────────────────────────────────────────────────
#  2. RESET — all 10 tasks
# ──────────────────────────────────────────────────────────────────────────────

section("RESET — ALL 10 TASKS")

for task_str in EXPECTED_TASKS:
    def _make_reset_test(ts):
        def _test():
            env = create_environment(ts, seed=42)
            obs = env.reset()
            assert obs.message, f"Empty message on reset for {ts}"
            assert obs.task_type == ts
            assert obs.step_number == 0
            assert not obs.done
            assert len(obs.available_actions) > 0, f"No available actions for {ts}"
        return _test
    test(f"reset: {task_str}", _make_reset_test(task_str))

# ──────────────────────────────────────────────────────────────────────────────
#  3. STEP — valid action on each task
# ──────────────────────────────────────────────────────────────────────────────

section("STEP — VALID ACTION EACH TASK")

FIRST_ACTIONS = {
    "esi_assignment":     "assign_esi:2",
    "intake_interview":   "ask:pain_scale",
    "queue_management":   "get_vitals:Q001",
    "medication_check":   "flag_allergy:amoxicillin",
    "discharge_planning": "add_instruction:no_weight_bearing",
    "mass_casualty":      "inspect:MCI-A1",
    "sepsis_screening":   "screen:SS001",
    "bed_allocation":     "get_info:BD001",
    "shift_handoff":      "report:HO001:situation_chief_complaint",
    "consent_assessment": "assess:CS001:understands_info",
}

for task_str, action_content in FIRST_ACTIONS.items():
    def _make_step_test(ts, ac):
        def _test():
            env = create_environment(ts, seed=42)
            env.reset()
            result = env.step(TriageAction(action_type="text", content=ac))
            assert result.observation.message, f"Empty message after step in {ts}"
            assert result.observation.step_number == 1
            assert_range(result.reward, 0.0, 1.0, f"Reward out of range for {ts}")
        return _test
    test(f"step: {task_str}", _make_step_test(task_str, action_content))

# ──────────────────────────────────────────────────────────────────────────────
#  4. EPISODE COMPLETION — each task reaches done=True and returns a reward
# ──────────────────────────────────────────────────────────────────────────────

section("EPISODE COMPLETION")

TERMINAL_ACTIONS = {
    "esi_assignment":     ["assign_esi:1"],
    "intake_interview":   ["complete_intake"],
    "queue_management":   ["prioritize:Q005","prioritize:Q001","prioritize:Q003","prioritize:Q002","prioritize:Q004","finalize_queue"],
    "medication_check":   ["complete_review"],
    "discharge_planning": ["set_followup:7","complete_discharge"],
    "mass_casualty":      [
        "tag:MCI-A1:GREEN","tag:MCI-A2:RED","tag:MCI-A3:BLACK",
        "tag:MCI-A4:RED","tag:MCI-A5:RED","tag:MCI-A6:YELLOW","finalize_scene"
    ],
    "sepsis_screening":   ["complete_screening"],
    "bed_allocation":     ["finalize_beds"],
    "shift_handoff":      ["complete_handoff"],
    "consent_assessment": [
        "finalize:CS001","finalize:CS002","finalize:CS003",
    ],
}

for task_str, actions in TERMINAL_ACTIONS.items():
    def _make_complete_test(ts, acts):
        def _test():
            env = create_environment(ts, seed=42)
            env.reset()
            result = None
            for act in acts:
                result = env.step(TriageAction(action_type="text", content=act))
                if result.done:
                    break
            assert result is not None and result.done, f"{ts} never reached done=True"
            assert_range(result.reward, 0.0, 1.0, f"Final reward out of [0,1] for {ts}")
        return _test
    test(f"completes: {task_str}", _make_complete_test(task_str, TERMINAL_ACTIONS[task_str]))

# ──────────────────────────────────────────────────────────────────────────────
#  5. MAX STEPS ENFORCEMENT
# ──────────────────────────────────────────────────────────────────────────────

section("MAX STEPS ENFORCEMENT")

def _max_steps_esi():
    env = create_environment("esi_assignment", seed=42)
    env.reset()
    # Do nothing useful until step limit
    result = None
    for _ in range(MAX_STEPS[TaskType.ESI_ASSIGNMENT] + 5):
        result = env.step(TriageAction(action_type="text", content="unknown_action"))
        if result.done:
            break
    assert result.done, "ESI task should terminate at max_steps"
    assert env.step_count <= MAX_STEPS[TaskType.ESI_ASSIGNMENT] + 1

def _max_steps_sepsis():
    env = create_environment("sepsis_screening", seed=42)
    env.reset()
    result = None
    for _ in range(MAX_STEPS[TaskType.SEPSIS_SCREENING] + 5):
        result = env.step(TriageAction(action_type="text", content="screen:SS001"))
        if result.done:
            break
    assert result.done, "Sepsis task should terminate at max_steps"

test("ESI task terminates at max_steps",    _max_steps_esi)
test("Sepsis task terminates at max_steps", _max_steps_sepsis)

# ──────────────────────────────────────────────────────────────────────────────
#  6. GRADING FUNCTIONS — unit tests
# ──────────────────────────────────────────────────────────────────────────────

section("GRADERS — UNIT TESTS")

# ── ESI ──
def _grade_esi_exact():
    assert_eq(grade_esi(1, 1), 1.0, "exact ESI")
    assert_eq(grade_esi(3, 3), 1.0, "exact ESI")
    assert_eq(grade_esi(5, 5), 1.0, "exact ESI")

def _grade_esi_off_by_one():
    assert_eq(grade_esi(2, 1), 0.5, "off-by-1")
    assert_eq(grade_esi(1, 2), 0.5, "off-by-1")

def _grade_esi_off_by_two():
    assert_eq(grade_esi(3, 1), 0.2, "off-by-2")

def _grade_esi_large_miss():
    assert_eq(grade_esi(5, 1), 0.0, "large miss")
    assert_eq(grade_esi(1, 5), 0.0, "large miss")

test("grade_esi: exact match = 1.0",   _grade_esi_exact)
test("grade_esi: off-by-1 = 0.5",      _grade_esi_off_by_one)
test("grade_esi: off-by-2 = 0.2",      _grade_esi_off_by_two)
test("grade_esi: large miss = 0.0",    _grade_esi_large_miss)

# ── Intake ──
def _grade_intake_full():
    full = {f: "value" for f in INTAKE_REQUIRED_FIELDS}
    score = grade_intake(full, len(full))
    assert score > 0.7, f"Full intake score too low: {score}"

def _grade_intake_empty():
    score = grade_intake({}, 0)
    assert score == 0.0, f"Empty intake should score 0.0, got {score}"

def _grade_intake_partial():
    half = {f: "v" for f in INTAKE_REQUIRED_FIELDS[:4]}
    score = grade_intake(half, 4)
    assert 0.0 < score < 0.8, f"Partial intake score unexpected: {score}"

test("grade_intake: all fields = high score", _grade_intake_full)
test("grade_intake: no fields = 0.0",         _grade_intake_empty)
test("grade_intake: partial fields = mid",    _grade_intake_partial)

# ── Queue ──
def _grade_queue_perfect():
    correct_esis = {"Q005": 2, "Q001": 3, "Q003": 3, "Q002": 4, "Q004": 5}
    correct_order = sorted(correct_esis.keys(), key=lambda x: correct_esis[x])
    score = grade_queue(correct_order, correct_esis, set(), set())
    assert score >= 0.5, f"Perfect queue order scored too low: {score}"

def _grade_queue_reversed():
    correct_esis = {"Q005": 2, "Q001": 3, "Q003": 3, "Q002": 4, "Q004": 5}
    reversed_order = sorted(correct_esis.keys(), key=lambda x: -correct_esis[x])
    score = grade_queue(reversed_order, correct_esis, set(), set())
    # Reversed should score much lower than correct
    correct_order = sorted(correct_esis.keys(), key=lambda x: correct_esis[x])
    good_score = grade_queue(correct_order, correct_esis, set(), set())
    assert score < good_score, "Reversed order should score lower"

def _grade_queue_deterioration():
    correct_esis = {"Q005": 2, "Q001": 3, "Q002": 4, "Q004": 5}
    order = list(correct_esis.keys())
    # Responded to deterioration
    score_with = grade_queue(order, correct_esis, {"Q001"}, {"Q001"})
    # Did not respond
    score_without = grade_queue(order, correct_esis, {"Q001"}, set())
    assert score_with >= score_without, "Responding to deterioration should help score"

test("grade_queue: correct order scores high",    _grade_queue_perfect)
test("grade_queue: reversed order scores lower",  _grade_queue_reversed)
test("grade_queue: deterioration response bonus", _grade_queue_deterioration)

# ── Medication ──
def _grade_medication_perfect():
    case = MEDICATION_CASES[0]  # MC001: amoxicillin/penicillin, ibuprofen/lisinopril
    flagged_a = {"amoxicillin"}
    flagged_i = {"ibuprofen"}
    score = grade_medication(flagged_a, flagged_i, {"azithromycin","acetaminophen"}, case, 0)
    assert score > 0.7, f"Perfect medication check scored too low: {score}"

def _grade_medication_wrong_approval():
    case = MEDICATION_CASES[0]
    score_clean = grade_medication({"amoxicillin"}, {"ibuprofen"}, set(), case, 0)
    score_penalty = grade_medication({"amoxicillin"}, {"ibuprofen"}, set(), case, 2)
    assert score_penalty < score_clean, "Wrong approvals should reduce score"
    assert score_penalty == max(0.0, score_clean - 0.30), "Each wrong approval = -0.15"

def _grade_medication_no_flags():
    case = MEDICATION_CASES[0]
    score = grade_medication(set(), set(), set(), case, 0)
    assert score < 0.5, f"No flags should score low: {score}"

test("grade_medication: perfect review = high",        _grade_medication_perfect)
test("grade_medication: wrong approval penalty -0.15", _grade_medication_wrong_approval)
test("grade_medication: zero flags = low score",       _grade_medication_no_flags)

# ── Discharge ──
def _grade_discharge_perfect():
    case = DISCHARGE_CASES[0]  # DC001: fibula fracture
    score = grade_discharge(
        set(case.required_instructions),
        set(case.required_warnings),
        case.optimal_followup_days,
        case,
    )
    assert score >= 0.95, f"Perfect discharge should score ≥0.95, got {score}"

def _grade_discharge_empty():
    case = DISCHARGE_CASES[0]
    score = grade_discharge(set(), set(), 0, case)
    assert score < 0.3, f"Empty discharge should score low, got {score}"

def _grade_discharge_timing():
    case = DISCHARGE_CASES[0]  # optimal=7, tolerance=3
    good = grade_discharge(set(case.required_instructions), set(case.required_warnings), 7, case)
    bad  = grade_discharge(set(case.required_instructions), set(case.required_warnings), 30, case)
    assert good > bad, "Wrong followup timing should hurt score"

test("grade_discharge: perfect plan = ≥0.95",      _grade_discharge_perfect)
test("grade_discharge: empty plan = low",           _grade_discharge_empty)
test("grade_discharge: timing affects score",       _grade_discharge_timing)

# ── MCI ──
def _grade_mci_perfect():
    scenario = MCI_SCENARIOS[0]
    perfect_tags = {p["patient_id"]: p["correct_tag"] for p in scenario}
    score = grade_mci(perfect_tags, scenario)
    assert_eq(score, 1.0, "Perfect MCI tags")

def _grade_mci_red_black_swap():
    scenario = MCI_SCENARIOS[0]
    tags = {p["patient_id"]: p["correct_tag"] for p in scenario}
    # Swap one RED patient to BLACK (critical error)
    red_patient = next(p["patient_id"] for p in scenario if p["correct_tag"] == "RED")
    tags[red_patient] = "BLACK"
    score = grade_mci(tags, scenario)
    perfect = grade_mci({p["patient_id"]: p["correct_tag"] for p in scenario}, scenario)
    # Total reduction = base miss (1/n) + critical penalty (0.15)
    # With 6 patients: 1/6 + 0.15 ≈ 0.317 — verify both components are present
    total_reduction = perfect - score
    assert total_reduction > 0.15, f"Swap should reduce score by more than 0.15 (got {total_reduction:.3f})"
    assert total_reduction > 0.30, f"Swap should reduce score by >0.30 including base miss (got {total_reduction:.3f})"
    assert score < 1.0, "RED/BLACK swap should not score perfect"

def _grade_mci_all_wrong():
    scenario = MCI_SCENARIOS[0]
    # Assign opposite colour to everyone
    swap = {"RED":"GREEN","GREEN":"RED","YELLOW":"BLACK","BLACK":"YELLOW"}
    tags = {p["patient_id"]: swap[p["correct_tag"]] for p in scenario}
    score = grade_mci(tags, scenario)
    assert score <= 0.0, f"All wrong should score 0 or negative, got {score}"

test("grade_mci: all correct = 1.0",           _grade_mci_perfect)
test("grade_mci: RED↔BLACK swap = -0.15",       _grade_mci_red_black_swap)
test("grade_mci: all wrong = ≤0.0",            _grade_mci_all_wrong)

# ── Sepsis ──
def _grade_sepsis_perfect():
    ward = SEPSIS_WARDS[0]
    true_sepsis = {p["patient_id"] for p in ward if p["is_sepsis"]}
    orders = {pid: set(SEPSIS_BUNDLE_ITEMS) for pid in true_sepsis}
    score = grade_sepsis(true_sepsis, orders, ward)
    assert score >= 0.9, f"Perfect sepsis screening should score ≥0.9, got {score}"

def _grade_sepsis_no_flags():
    ward = SEPSIS_WARDS[0]
    score = grade_sepsis(set(), {}, ward)
    assert score == 0.0, f"No flags = recall 0, should score 0, got {score}"

def _grade_sepsis_false_positives():
    ward = SEPSIS_WARDS[0]
    true_sepsis = {p["patient_id"] for p in ward if p["is_sepsis"]}
    all_ids = {p["patient_id"] for p in ward}
    orders = {pid: set(SEPSIS_BUNDLE_ITEMS) for pid in all_ids}
    perfect_score = grade_sepsis(true_sepsis, orders, ward)
    fp_score = grade_sepsis(all_ids, orders, ward)   # flags everyone
    assert fp_score < perfect_score, "False positives should hurt precision"

test("grade_sepsis: perfect = ≥0.9",           _grade_sepsis_perfect)
test("grade_sepsis: no flags = 0.0",           _grade_sepsis_no_flags)
test("grade_sepsis: false positives hurt",     _grade_sepsis_false_positives)

# ── Bed Allocation ──
def _grade_beds_perfect():
    scenario = BED_SCENARIOS[0]
    perfect = {p["patient_id"]: p["required_bed"] for p in scenario}
    score = grade_bed_allocation(perfect, scenario)
    assert score >= 0.9, f"Perfect bed allocation should score ≥0.9, got {score}"

def _grade_beds_over_allocation():
    scenario = BED_SCENARIOS[0]
    # Assign everyone to trauma_bay (capacity=2) — massive over-allocation
    all_trauma = {p["patient_id"]: "trauma_bay" for p in scenario}
    score = grade_bed_allocation(all_trauma, scenario)
    perfect = {p["patient_id"]: p["required_bed"] for p in scenario}
    perfect_score = grade_bed_allocation(perfect, scenario)
    assert score < perfect_score, "Over-allocation should hurt score"

test("grade_bed_allocation: perfect = ≥0.9",    _grade_beds_perfect)
test("grade_bed_allocation: over-alloc penalty", _grade_beds_over_allocation)

# ── Handoff ──
def _grade_handoff_full():
    scenario = HANDOFF_SCENARIOS[0]
    all_reported = {p["patient_id"]: set(p["required_fields"]) for p in scenario}
    score = grade_handoff(all_reported, scenario)
    assert score >= 0.9, f"Full handoff score too low: {score}"

def _grade_handoff_empty():
    scenario = HANDOFF_SCENARIOS[0]
    score = grade_handoff({}, scenario)
    assert score == 0.0, f"Empty handoff should score 0.0, got {score}"

test("grade_handoff: all fields = ≥0.9",    _grade_handoff_full)
test("grade_handoff: no fields = 0.0",      _grade_handoff_empty)

# ── Consent ──
def _grade_consent_perfect():
    assessed    = {sc["scenario_id"]: set(CAPACITY_CRITERIA) for sc in CONSENT_SCENARIOS}
    declared    = {sc["scenario_id"]: sc["correct_capacity"] for sc in CONSENT_SCENARIOS}
    chosen      = {sc["scenario_id"]: sc["correct_consent_path"] for sc in CONSENT_SCENARIOS}
    steps_done  = {sc["scenario_id"]: set(sc["required_steps"]) for sc in CONSENT_SCENARIOS}
    score = grade_consent(assessed, declared, chosen, steps_done)
    assert score >= 0.9, f"Perfect consent should score ≥0.9, got {score}"

def _grade_consent_dangerous_shortcut():
    # CS001 is a competent adult — invoking emergent_exception is a dangerous shortcut
    assessed   = {sc["scenario_id"]: set(CAPACITY_CRITERIA) for sc in CONSENT_SCENARIOS}
    declared   = {sc["scenario_id"]: sc["correct_capacity"] for sc in CONSENT_SCENARIOS}
    chosen     = {sc["scenario_id"]: sc["correct_consent_path"] for sc in CONSENT_SCENARIOS}
    good_steps = {sc["scenario_id"]: set(sc["required_steps"]) for sc in CONSENT_SCENARIOS}
    bad_steps  = {k: v.copy() for k, v in good_steps.items()}
    # Add the dangerous shortcut for CS001
    bad_steps["CS001"].add("invoke_emergent_exception")
    good_score = grade_consent(assessed, declared, chosen, good_steps)
    bad_score  = grade_consent(assessed, declared, chosen, bad_steps)
    assert bad_score < good_score, "Dangerous shortcut should reduce score"

def _grade_consent_wrong_path():
    assessed   = {sc["scenario_id"]: set(CAPACITY_CRITERIA) for sc in CONSENT_SCENARIOS}
    declared   = {sc["scenario_id"]: sc["correct_capacity"] for sc in CONSENT_SCENARIOS}
    good_chosen = {sc["scenario_id"]: sc["correct_consent_path"] for sc in CONSENT_SCENARIOS}
    bad_chosen  = {sc["scenario_id"]: "court_ordered" for sc in CONSENT_SCENARIOS}
    steps_done  = {sc["scenario_id"]: set(sc["required_steps"]) for sc in CONSENT_SCENARIOS}
    good_score = grade_consent(assessed, declared, good_chosen, steps_done)
    bad_score  = grade_consent(assessed, declared, bad_chosen, steps_done)
    assert bad_score < good_score, "Wrong consent path should lower score"

test("grade_consent: perfect = ≥0.9",              _grade_consent_perfect)
test("grade_consent: dangerous shortcut = penalty", _grade_consent_dangerous_shortcut)
test("grade_consent: wrong path lowers score",      _grade_consent_wrong_path)

# ──────────────────────────────────────────────────────────────────────────────
#  7. CLINICAL DATA INTEGRITY
# ──────────────────────────────────────────────────────────────────────────────

section("CLINICAL DATA INTEGRITY")

def _esi_cases_coverage():
    levels = {c["correct_esi_level"] for c in ESI_CASES}
    for lvl in [1, 2, 3, 4, 5]:
        assert lvl in levels, f"Missing ESI level {lvl} in case library"

def _esi_cases_have_vitals():
    for c in ESI_CASES:
        assert c["vitals"] is not None, f"{c['patient_id']} missing vitals"
        assert c["correct_esi_level"] in range(1, 6)

def _mci_scenario_six_patients():
    for scenario in MCI_SCENARIOS:
        assert len(scenario) == 6, f"MCI scenario should have 6 patients, got {len(scenario)}"

def _mci_has_all_tags():
    for scenario in MCI_SCENARIOS:
        tags_used = {p["correct_tag"] for p in scenario}
        assert tags_used == {"RED","YELLOW","GREEN","BLACK"}, f"MCI missing tags: {tags_used}"

def _sepsis_ward_has_mixed():
    for ward in SEPSIS_WARDS:
        is_sepsis = [p["is_sepsis"] for p in ward]
        assert any(is_sepsis), "Ward must have at least one sepsis patient"
        assert not all(is_sepsis), "Ward must have at least one non-sepsis patient"

def _bed_inventory_consistent():
    total_capacity = sum(BED_INVENTORY.values())
    assert total_capacity >= 8, "Bed inventory should fit all 8 patients in scenario"

def _consent_scenarios_distinct_paths():
    paths = {sc["correct_consent_path"] for sc in CONSENT_SCENARIOS}
    assert len(paths) == 3, f"3 consent scenarios should have 3 distinct paths, got {paths}"

def _sbar_fields_count():
    assert len(SBAR_ALL_FIELDS) == 12, f"Expected 12 SBAR fields, got {len(SBAR_ALL_FIELDS)}"

def _capacity_criteria_count():
    assert len(CAPACITY_CRITERIA) == 4, f"Expected 4 capacity criteria, got {len(CAPACITY_CRITERIA)}"

def _sepsis_bundle_count():
    assert len(SEPSIS_BUNDLE_ITEMS) == 7, f"Expected 7 bundle items, got {len(SEPSIS_BUNDLE_ITEMS)}"

test("ESI cases cover all 5 levels",          _esi_cases_coverage)
test("ESI cases have valid vitals + levels",  _esi_cases_have_vitals)
test("MCI scenario has exactly 6 patients",   _mci_scenario_six_patients)
test("MCI scenario covers all 4 tags",        _mci_has_all_tags)
test("Sepsis ward has mixed pos/neg",         _sepsis_ward_has_mixed)
test("Bed inventory fits all patients",       _bed_inventory_consistent)
test("3 consent scenarios = 3 distinct paths",_consent_scenarios_distinct_paths)
test("SBAR has exactly 12 fields",           _sbar_fields_count)
test("Capacity has exactly 4 criteria",      _capacity_criteria_count)
test("Sepsis bundle has 7 items",            _sepsis_bundle_count)

# ──────────────────────────────────────────────────────────────────────────────
#  8. QUEUE DETERIORATION EVENT
# ──────────────────────────────────────────────────────────────────────────────

section("QUEUE DETERIORATION EVENT")

def _deterioration_fires():
    env = create_environment("queue_management", seed=42)
    env.reset()
    det_step = env._state["deterioration_step"]
    messages = []
    for i in range(det_step + 2):
        result = env.step(TriageAction(action_type="text", content="get_vitals:Q001"))
        messages.append(result.observation.message)
        if result.done:
            break
    # Check that at least one message contained deterioration alert
    alert_msgs = [m for m in messages if "ALERT" in m or "deteriorat" in m.lower()]
    assert len(alert_msgs) > 0, "Deterioration event never fired"

def _deterioration_tracked():
    env = create_environment("queue_management", seed=42)
    env.reset()
    det_step = env._state["deterioration_step"]
    det_patient_id = None
    for i in range(det_step + 2):
        result = env.step(TriageAction(action_type="text", content="get_vitals:Q001"))
        if env._state.get("deteriorated_ids"):
            det_patient_id = list(env._state["deteriorated_ids"])[0]
            break
    assert det_patient_id is not None, "No patient was tracked as deteriorated"
    # Respond to deterioration
    env.step(TriageAction(action_type="text", content=f"prioritize:{det_patient_id}"))
    assert det_patient_id in env._state["deterioration_responses"], \
        "Agent response to deteriorated patient not tracked"

test("deterioration event fires mid-episode",     _deterioration_fires)
test("deterioration response is tracked",         _deterioration_tracked)

# ──────────────────────────────────────────────────────────────────────────────
#  9. ACTION VALIDATORS (from inference.py)
# ──────────────────────────────────────────────────────────────────────────────

section("ACTION VALIDATORS")

try:
    sys.path.insert(0, os.path.dirname(__file__) or ".")
    from inference import validate_action
    HAS_INFERENCE = True
except Exception:
    HAS_INFERENCE = False

VALIDATOR_CASES = [
    # (task, action, should_be_valid)
    ("esi_assignment",     "assign_esi:3",                          True),
    ("esi_assignment",     "assign_esi:6",                          False),
    ("esi_assignment",     "assign_esi:0",                          False),
    ("intake_interview",   "ask:pain_scale",                        True),
    ("intake_interview",   "ask:blood_pressure",                    False),  # not a field
    ("intake_interview",   "complete_intake",                       True),
    ("queue_management",   "prioritize:Q001",                       True),
    ("queue_management",   "finalize_queue",                        True),
    ("medication_check",   "flag_allergy:amoxicillin",              True),
    ("medication_check",   "approve:azithromycin",                  True),
    ("medication_check",   "complete_review",                       True),
    ("discharge_planning", "add_instruction:no_weight_bearing",     True),
    ("discharge_planning", "add_instruction:nonexistent_key",       False),
    ("discharge_planning", "set_followup:7",                        True),
    ("mass_casualty",      "tag:MCI-A1:RED",                        True),
    ("mass_casualty",      "tag:MCI-A1:PURPLE",                     False),
    ("mass_casualty",      "inspect:MCI-A1",                        True),
    ("mass_casualty",      "finalize_scene",                        True),
    ("sepsis_screening",   "screen:SS001",                          True),
    ("sepsis_screening",   "flag_sepsis:SS001",                     True),
    ("sepsis_screening",   "order:SS001:blood_cultures",            True),
    ("sepsis_screening",   "order:SS001:invalid_item",              False),
    ("bed_allocation",     "assign:BD001:trauma_bay",               True),
    ("bed_allocation",     "assign:BD001:penthouse_suite",          False),
    ("bed_allocation",     "finalize_beds",                         True),
    ("shift_handoff",      "report:HO001:situation_chief_complaint",True),
    ("shift_handoff",      "report:HO001:made_up_field",            False),
    ("shift_handoff",      "complete_handoff",                      True),
    ("consent_assessment", "assess:CS001:understands_info",         True),
    ("consent_assessment", "assess:CS001:bad_criterion",            False),
    ("consent_assessment", "consent_path:CS001:informed_consent",   True),
    ("consent_assessment", "consent_path:CS001:bad_path",           False),
    ("consent_assessment", "finalize:CS001",                        True),
]

for task, action, expected_valid in VALIDATOR_CASES:
    def _make_validator_test(ts, ac, ev):
        def _test():
            if not HAS_INFERENCE:
                raise AssertionError("inference.py could not be imported")
            valid, err = validate_action(ts, ac)
            if ev:
                assert valid, f"Expected VALID but got INVALID: {err}"
            else:
                assert not valid, f"Expected INVALID but got VALID for: {ac}"
        return _test
    short_action = action[:35] + "…" if len(action) > 35 else action
    test(
        f"validator: {task[:20]:<20} {short_action:<38} → {'valid' if expected_valid else 'invalid'}",
        _make_validator_test(task, action, expected_valid),
        skip_reason="" if HAS_INFERENCE else "inference.py not importable",
    )

# ──────────────────────────────────────────────────────────────────────────────
#  10. REWARD BOUNDS — all tasks must return rewards in [0, 1]
# ──────────────────────────────────────────────────────────────────────────────

section("REWARD BOUNDS [0.0, 1.0]")

SEEDS = [42, 7, 100]

for task_str in EXPECTED_TASKS:
    def _make_bounds_test(ts):
        def _test():
            for seed in SEEDS:
                env = create_environment(ts, seed=seed)
                env.reset()
                # Run to completion by exhausting steps
                result = None
                actions = list(FIRST_ACTIONS.values()) + ["complete_intake","complete_review",
                           "complete_discharge","finalize_scene","complete_screening",
                           "finalize_beds","complete_handoff","finalize:CS001","finalize:CS002","finalize:CS003"]
                for _ in range(MAX_STEPS[TaskType(ts)] + 5):
                    act = actions[env.step_count % len(actions)]
                    result = env.step(TriageAction(action_type="text", content=act))
                    if result.done:
                        break
                if result and result.done:
                    assert 0.0 <= result.reward <= 1.0, \
                        f"Final reward {result.reward} out of [0,1] for {ts} seed={seed}"
        return _test
    test(f"reward in [0,1]: {task_str}", _make_bounds_test(task_str))

# ──────────────────────────────────────────────────────────────────────────────
#  11. OPENENV MANIFEST
# ──────────────────────────────────────────────────────────────────────────────

section("OPENENV MANIFEST (openenv.yaml)")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

def _yaml_parseable():
    if not HAS_YAML:
        raise AssertionError("PyYAML not installed")
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert data is not None

def _yaml_has_10_tasks():
    if not HAS_YAML:
        raise AssertionError("PyYAML not installed")
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    tasks = data.get("tasks", [])
    assert len(tasks) == 10, f"openenv.yaml has {len(tasks)} tasks, expected 10"

def _yaml_task_names_match():
    if not HAS_YAML:
        raise AssertionError("PyYAML not installed")
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    manifest_names = {t["name"] for t in data.get("tasks", [])}
    for name in EXPECTED_TASKS:
        assert name in manifest_names, f"Task '{name}' missing from openenv.yaml"

def _yaml_has_required_fields():
    if not HAS_YAML:
        raise AssertionError("PyYAML not installed")
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert "name" in data
    assert "version" in data
    assert "tasks" in data
    assert "endpoints" in data

def _yaml_has_ui_endpoint():
    if not HAS_YAML:
        raise AssertionError("PyYAML not installed")
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    endpoints = data.get("endpoints", {})
    assert "ui" in endpoints or "web" in str(endpoints), "openenv.yaml missing /web endpoint"

test("openenv.yaml is valid YAML",        _yaml_parseable,
     skip_reason="" if HAS_YAML else "PyYAML not installed")
test("openenv.yaml has 10 tasks",         _yaml_has_10_tasks,
     skip_reason="" if HAS_YAML else "PyYAML not installed")
test("openenv.yaml task names match code",_yaml_task_names_match,
     skip_reason="" if HAS_YAML else "PyYAML not installed")
test("openenv.yaml has required fields",  _yaml_has_required_fields,
     skip_reason="" if HAS_YAML else "PyYAML not installed")
test("openenv.yaml has /web UI endpoint", _yaml_has_ui_endpoint,
     skip_reason="" if HAS_YAML else "PyYAML not installed")

# ──────────────────────────────────────────────────────────────────────────────
#  12. FILE EXISTENCE
# ──────────────────────────────────────────────────────────────────────────────

section("REQUIRED FILES PRESENT")

REQUIRED_FILES = [
    "medical_triage_env.py",
    "inference.py",
    "main.py",
    "dashboard.html",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "server/app.py",
]

for fname in REQUIRED_FILES:
    def _make_file_test(fn):
        def _test():
            assert os.path.exists(fn), f"Missing required file: {fn}"
        return _test
    test(f"file exists: {fname}", _make_file_test(fname))

# ──────────────────────────────────────────────────────────────────────────────
#  13. SYNTAX CHECK ALL PY FILES
# ──────────────────────────────────────────────────────────────────────────────

section("PYTHON SYNTAX CHECK")

PY_FILES = ["medical_triage_env.py", "inference.py", "main.py", "server/app.py"]

for pyfile in PY_FILES:
    def _make_syntax_test(f):
        def _test():
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", f],
                capture_output=True, text=True
            )
            assert result.returncode == 0, f"Syntax error in {f}:\n{result.stderr}"
        return _test
    test(f"syntax: {pyfile}", _make_syntax_test(pyfile))

# ──────────────────────────────────────────────────────────────────────────────
#  14. HTTP SERVER TESTS (skipped if server not running)
# ──────────────────────────────────────────────────────────────────────────────

section("HTTP SERVER TESTS (live)")

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

try:
    import requests as _req
    _req.get(f"{SERVER_URL}/health", timeout=3)
    SERVER_UP = True
except Exception:
    SERVER_UP = False

skip_msg = "" if SERVER_UP else f"server not reachable at {SERVER_URL}"

def _http_health():
    import requests as req
    r = req.get(f"{SERVER_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"

def _http_tasks_list():
    import requests as req
    r = req.get(f"{SERVER_URL}/tasks", timeout=5)
    assert r.status_code == 200
    data = r.json()
    tasks = data.get("tasks", {})
    assert len(tasks) == 10, f"Expected 10 tasks via HTTP, got {len(tasks)}"

def _http_root():
    import requests as req
    r = req.get(f"{SERVER_URL}/", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data.get("total_tasks") == 10

def _http_dashboard():
    import requests as req
    r = req.get(f"{SERVER_URL}/web", timeout=5)
    assert r.status_code == 200
    assert "Medical Triage" in r.text or "medical" in r.text.lower()

def _http_reset_all_tasks():
    import requests as req
    for task in EXPECTED_TASKS:
        r = req.post(f"{SERVER_URL}/tasks/{task}/reset",
                     json={"seed": 42}, timeout=10)
        assert r.status_code == 200, f"Reset failed for {task}: {r.status_code}"
        data = r.json()
        assert "message" in data, f"No message in reset response for {task}"
        assert data.get("max_steps", 0) > 0

def _http_step_each_task():
    import requests as req
    for task, action in FIRST_ACTIONS.items():
        # Reset first
        req.post(f"{SERVER_URL}/tasks/{task}/reset", json={"seed": 42}, timeout=10)
        # Step
        r = req.post(f"{SERVER_URL}/tasks/{task}/step",
                     json={"action_type": "text", "content": action}, timeout=10)
        assert r.status_code == 200, f"Step failed for {task}: {r.status_code}"
        data = r.json()
        assert "reward" in data
        assert "done" in data
        assert 0.0 <= data["reward"] <= 1.0, f"HTTP reward out of range for {task}"

def _http_grading_info():
    import requests as req
    for task in EXPECTED_TASKS:
        r = req.get(f"{SERVER_URL}/tasks/{task}/grading_info", timeout=5)
        assert r.status_code == 200, f"grading_info failed for {task}"
        data = r.json()
        assert "grading" in data

def _http_404_bad_task():
    import requests as req
    r = req.post(f"{SERVER_URL}/tasks/nonexistent_task/reset",
                 json={"seed": 42}, timeout=5)
    assert r.status_code == 404

def _http_400_step_without_reset():
    import requests as req
    # Use a task name that likely hasn't been reset in this session
    r = req.post(f"{SERVER_URL}/tasks/esi_assignment/step",
                 json={"action_type": "text", "content": "assign_esi:1"}, timeout=5)
    # Could be 200 (if already reset) or 400 — just ensure no 500
    assert r.status_code != 500, "Server should not 500 on step-without-reset"

test("GET /health → ok",             _http_health,          skip_reason=skip_msg)
test("GET / → total_tasks=10",       _http_root,            skip_reason=skip_msg)
test("GET /tasks → 10 tasks",        _http_tasks_list,      skip_reason=skip_msg)
test("GET /web → dashboard HTML",    _http_dashboard,       skip_reason=skip_msg)
test("POST reset: all 10 tasks",     _http_reset_all_tasks, skip_reason=skip_msg)
test("POST step: all 10 tasks",      _http_step_each_task,  skip_reason=skip_msg)
test("GET grading_info: all tasks",  _http_grading_info,    skip_reason=skip_msg)
test("POST reset unknown → 404",     _http_404_bad_task,    skip_reason=skip_msg)
test("POST step no-reset → not 500", _http_400_step_without_reset, skip_reason=skip_msg)

# ──────────────────────────────────────────────────────────────────────────────
#  15. SEED REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────

section("SEED REPRODUCIBILITY")

def _same_seed_same_case():
    for task_str in ["esi_assignment", "mass_casualty", "sepsis_screening"]:
        env1 = create_environment(task_str, seed=42)
        obs1 = env1.reset()
        env2 = create_environment(task_str, seed=42)
        obs2 = env2.reset()
        assert obs1.message == obs2.message, \
            f"Same seed produced different obs for {task_str}"

def _different_seed_may_differ():
    env1 = create_environment("esi_assignment", seed=1)
    env2 = create_environment("esi_assignment", seed=999)
    obs1 = env1.reset()
    obs2 = env2.reset()
    # ESI has 11 cases; different seeds should sometimes pick different ones
    # (Not guaranteed to differ but this covers the path)
    assert obs1.message is not None and obs2.message is not None

test("same seed → identical episode",         _same_seed_same_case)
test("different seed → both produce episodes",_different_seed_may_differ)

# ──────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

section("SUMMARY")

passed = sum(1 for r in results if r["status"] == "pass")
failed = sum(1 for r in results if r["status"] == "fail")
skipped= sum(1 for r in results if r["status"] == "skip")
total  = len(results)

print(f"\n  Total:   {total}")
print(f"  \033[32mPassed:  {passed}\033[0m")
print(f"  \033[31mFailed:  {failed}\033[0m")
print(f"  \033[33mSkipped: {skipped}\033[0m")

if failed:
    print(f"\n{BOLD}FAILED TESTS:{RST}")
    for r in results:
        if r["status"] == "fail":
            print(f"  ✗ {r['name']}")
            if "error" in r:
                print(f"      {r['error'][:120]}")
    print()
    sys.exit(1)
else:
    print(f"\n\033[32m{BOLD}All tests passed!{RST}")
    if skipped:
        print(f"  ({skipped} tests skipped — start the server and re-run for full coverage)")
    sys.exit(0)