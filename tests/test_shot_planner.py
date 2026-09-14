import sys
import os
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.topic_analyzer import TopicAnalyzer
from app.agents.framework_selector import FrameworkSelector
from app.agents.script_writer import ScriptWriter, ReelScript, ScriptSection, SENSATIONAL_PHRASES
from app.agents.shot_planner import ShotPlanner, ShotPlan, Shot, DISALLOWED_VISUAL_CLAIMS


def test_shot_planner():
    print("==================================================")
    print("SHOT PLANNER RIGOROUS TEST SUITE (LOCKED ARCHITECTURE)")
    print("==================================================")

    analyzer = TopicAnalyzer()
    selector = FrameworkSelector()
    writer = ScriptWriter()
    planner = ShotPlanner()

    # -------------------------------------------------------------
    # TEST 1: Default 5-Shot Plan for Junk Food & Exact Script Preservation
    # -------------------------------------------------------------
    print("\n--- TEST 1: Junk Food Demo & Exact Script Preservation ---")
    topic_1 = "What happens if you eat junk food every day?"
    analysis_1 = analyzer.analyze(topic_1)
    selection_1 = selector.select(analysis_1)
    script_1 = writer.write_script(analysis_1, selection_1, target_duration_seconds=45.0)

    plan_1 = planner.plan_shots(
        reel_script=script_1,
        topic_analysis=analysis_1,
        framework_metadata=selection_1,
        target_duration_seconds=45.0,
        shot_count=5
    )

    assert isinstance(plan_1, ShotPlan)
    assert plan_1.topic == topic_1
    assert plan_1.framework_id == 15
    assert plan_1.framework_name == "What Happens If"
    assert plan_1.shot_count == 5, f"Expected exactly 5 shots, got {plan_1.shot_count}"
    assert len(plan_1.shots) == 5, f"Expected 5 shots list, got {len(plan_1.shots)}"
    assert plan_1.validation_status.startswith("VALID"), f"Invalid status: {plan_1.validation_status}"

    # 1. Exact Spoken Narration Reconstruction Test (must equal approved full_script_with_cta)
    planned_narration = " ".join(s.script_text.strip() for s in plan_1.shots if s.script_text.strip())
    norm_planned = " ".join(planned_narration.split())
    norm_expected = " ".join(script_1.full_script_with_cta.split())

    print("\n[EXACT SCRIPT PRESERVATION VERIFICATION]")
    print(f"ReelScript full_script_without_cta:\n  \"{script_1.full_script_without_cta}\"")
    print(f"ReelScript cta:\n  \"{script_1.cta}\"")
    print(f"ReelScript full_script_with_cta:\n  \"{script_1.full_script_with_cta}\"")
    print(f"\nReconstructed Planned Narration:\n  \"{norm_planned}\"")

    assert norm_planned == norm_expected, (
        f"CRITICAL VIOLATION: Spoken narration was altered!\n"
        f"Expected (full_script_with_cta):\n{norm_expected}\n\nPlanned:\n{norm_planned}"
    )
    print(">>> EXACT SCRIPT PRESERVATION (planned spoken narration == approved full_script_with_cta): 100% MATCH PASSED!")

    # 2. CTA Verification:
    # - CTA appears exactly once
    # - CTA exactly matches ReelScript.cta
    # - CTA is assigned to the final shot
    # - no new spoken sentences are introduced
    cta_normalized = " ".join(script_1.cta.split())
    assert plan_1.cta == script_1.cta, f"CTA was altered: expected '{script_1.cta}', got '{plan_1.cta}'"

    # Check CTA occurrence across all shots' script_text
    cta_occurrences = [i for i, s in enumerate(plan_1.shots) if cta_normalized in " ".join(s.script_text.split())]
    assert len(cta_occurrences) == 1, f"CTA should appear exactly once in shot script_text, found in shots: {cta_occurrences}"
    assert cta_occurrences[0] == len(plan_1.shots) - 1, f"CTA must be assigned to final shot (index {len(plan_1.shots) - 1}), found at index {cta_occurrences[0]}"
    assert plan_1.shots[-1].script_text.strip().endswith(script_1.cta.strip()), "Final shot script_text must end with exact ReelScript.cta"

    # Verify that in shots before the final shot, CTA is not present
    for s in plan_1.shots[:-1]:
        assert cta_normalized not in " ".join(s.script_text.split()), f"CTA found prematurely in shot {s.shot_number}"

    # Verify no new spoken sentences are introduced
    final_shot_without_cta = plan_1.shots[-1].script_text.strip()
    if final_shot_without_cta.endswith(script_1.cta.strip()):
        final_shot_without_cta = final_shot_without_cta[:-len(script_1.cta.strip())].strip()
    narration_before_cta = " ".join(s.script_text.strip() for s in plan_1.shots[:-1]) + " " + final_shot_without_cta
    norm_narration_before_cta = " ".join(narration_before_cta.split())
    norm_without_cta = " ".join(script_1.full_script_without_cta.split())
    assert norm_narration_before_cta == norm_without_cta, "Spoken sentences before CTA do not match full_script_without_cta"
    print(f">>> CTA OCCURRENCE & SPOKEN PURITY VERIFICATION: PASSED!")

    # 3. 8-Second Constraint & Total Duration Verification
    print(f"Total Planned Duration: {plan_1.total_duration_seconds}s (target: {plan_1.target_duration_seconds}s)")
    assert plan_1.total_duration_seconds <= 40.0, f"Total planned duration {plan_1.total_duration_seconds}s must be <= 40.0s for 5 clips"
    assert plan_1.total_duration_seconds >= 35.0, f"Total planned duration {plan_1.total_duration_seconds}s should be around 35-40s"

    for s in plan_1.shots:
        assert s.duration_seconds <= 8.0, f"Shot {s.shot_number} duration ({s.duration_seconds}s) exceeds 8.0s clip limit!"
        assert s.duration_seconds >= 6.0, f"Shot {s.shot_number} duration ({s.duration_seconds}s) is below 6.0s"
        assert s.validation_status.startswith("VALID")

    # 4. Stage Order and Shot Sequencing
    stage_orders = [s.script_stage_order for s in plan_1.shots]
    for i in range(len(stage_orders) - 1):
        assert stage_orders[i] <= stage_orders[i + 1], f"Stages out of order: {stage_orders}"
    assert 1 in stage_orders and 2 in stage_orders and 3 in stage_orders

    # 5. Visual Prompt Quality & Separation Checks
    for s in plan_1.shots:
        assert len(s.visual_goal) > 0
        assert len(s.visual_description) > 0
        assert len(s.subject) > 0
        assert len(s.setting) > 0
        assert len(s.camera_direction) > 0
        assert len(s.motion) > 0
        assert len(s.on_screen_text) > 0
        assert len(s.audio_direction) > 0
        assert len(s.video_generation_prompt) > 0
        assert s.video_generation_prompt.startswith("Vertical 9:16")

        # Visual prompt rules:
        # - Never include narration text inside visual prompt
        assert s.script_text not in s.video_generation_prompt, f"Shot {s.shot_number} leaks narration into prompt"
        # - Never tell video model to display subtitles or captions
        assert not re.search(r'(?i)\b(subtitles?|captions?|overlay\s+text)\b', s.video_generation_prompt)
        # - Never put quotation marks around spoken narration
        assert '"' not in s.video_generation_prompt, f"Shot {s.shot_number} contains double quotes: {s.video_generation_prompt}"
        # - No unnecessary cinematic jargon
        assert "35mm" not in s.video_generation_prompt.lower()
        assert "shallow depth of field" not in s.video_generation_prompt.lower()

        combined_text = f"{s.visual_description} {s.video_generation_prompt} {s.on_screen_text}".lower()
        for phrase in DISALLOWED_VISUAL_CLAIMS:
            assert phrase not in combined_text, f"Disallowed medical claim '{phrase}' found in shot {s.shot_number}"

        print(f"  Shot {s.shot_number} ({s.duration_seconds}s) [Stage {s.script_stage_order}: {s.script_stage_name}]:")
        print(f"    - Spoken:  {s.script_text}")
        print(f"    - Caption: {s.on_screen_text}")
        print(f"    - Prompt:  {s.video_generation_prompt[:90]}...")

    print(f"Generation source: {plan_1.generation_source}")
    print(">>> Test 1 (Junk Food 5-Shot Plan & Script Preservation) PASSED!\n")

    # -------------------------------------------------------------
    # REGRESSION TEST: Single Consultation Sentence & Standalone CTA
    # -------------------------------------------------------------
    print("--- REGRESSION TEST: Single Consultation Sentence & Standalone CTA ---")
    reg_sections = [
        ScriptSection(stage_order=1, stage_name="What happens if", text="What happens if you eat junk food every day?"),
        ScriptSection(stage_order=2, stage_name="Explanation", text="Added sugars and unhealthy fats stress your metabolism over time."),
        ScriptSection(stage_order=3, stage_name="Prevention", text="Swap processed snacks for fruit and nuts. If you notice persistent fatigue, schedule a visit with your physician.")
    ]
    reg_cta = "Follow for more science-backed nutrition tips!"
    reg_script = ReelScript(
        topic="Junk Food",
        framework_id=15,
        framework_name="What Happens If",
        target_duration_seconds=45.0,
        hook=reg_sections[0].text,
        sections=reg_sections,
        cta=reg_cta,
        safety_notes=["Rule compliance note"]
    )

    consultation_phrase = "schedule a visit with your physician"
    assert reg_script.full_script_with_cta.count(consultation_phrase) == 1

    for term in ["doctor", "physician", "consult", "schedule a visit"]:
        assert term not in reg_script.cta.lower()

    reg_plan = planner.plan_shots(
        reel_script=reg_script,
        target_duration_seconds=45.0,
        shot_count=5,
        force_fallback=True
    )
    reg_planned_narration = " ".join(s.script_text for s in reg_plan.shots)
    assert reg_planned_narration.count(consultation_phrase) == 1
    assert reg_planned_narration.count(reg_cta) == 1
    assert reg_plan.shots[-1].script_text.endswith(reg_cta)
    assert " ".join(reg_planned_narration.split()) == " ".join(reg_script.full_script_with_cta.split())

    for s in reg_plan.shots:
        assert not re.search(r'\bthe same a\b', s.video_generation_prompt, re.IGNORECASE)
        assert not re.search(r'\bthe action shows\b', s.video_generation_prompt, re.IGNORECASE)
        assert not re.search(r'\bshot of hands of the recurring\b', s.video_generation_prompt, re.IGNORECASE)
        assert len(s.video_generation_prompt.strip()) > 30
        assert s.duration_seconds <= 8.0

    print(">>> REGRESSION TEST PASSED: Consultation sentence and standalone CTA verified!\n")

    # -------------------------------------------------------------
    # REGRESSION TEST: Visual Prompt Rules & Separation
    # -------------------------------------------------------------
    print("--- REGRESSION TEST: Visual Prompt Rules & Separation ---")
    demo_script = writer.write_script(analysis_1, selection_1, target_duration_seconds=45.0, force_fallback=True)
    demo_plan = planner.plan_shots(demo_script, analysis_1, selection_1, target_duration_seconds=45.0, shot_count=5, force_fallback=True)

    assert len(demo_plan.shots) == 5
    assert demo_plan.total_duration_seconds <= 40.0
    assert demo_plan.total_duration_seconds >= 35.0

    for s in demo_plan.shots:
        # Every shot <= 8.0s
        assert s.duration_seconds <= 8.0, f"Shot {s.shot_number} duration {s.duration_seconds}s > 8.0s"
        assert s.video_generation_prompt.startswith("Vertical 9:16")
        assert not s.video_generation_prompt.startswith("Vertical 9:16 Vertical 9:16")

        # Zero narration leakage
        assert s.script_text.strip() not in s.video_generation_prompt
        for sent in re.split(r'(?<=[.!?])\s+', s.script_text):
            clean_sent = sent.strip().strip('"\'')
            if len(clean_sent) >= 12:
                assert clean_sent.lower() not in s.video_generation_prompt.lower(), (
                    f"Shot {s.shot_number} leaks narration sentence '{clean_sent}' into prompt: {s.video_generation_prompt}"
                )

        # No voiceover instructions or dialogue in prompt
        assert "Include a clear, natural spoken voiceover" not in s.video_generation_prompt
        assert "voiceover" not in s.video_generation_prompt.lower()
        assert '"' not in s.video_generation_prompt

        # No forbidden grammar
        assert not re.search(r'\bthe same a\b', s.video_generation_prompt, re.IGNORECASE)
        assert not re.search(r'\bthe action shows\b', s.video_generation_prompt, re.IGNORECASE)
        assert not re.search(r'\bshot of hands of the recurring\b', s.video_generation_prompt, re.IGNORECASE)
        assert not re.search(r'\ba a\b', s.video_generation_prompt, re.IGNORECASE)
        assert not re.search(r'\brecurring a\b', s.video_generation_prompt, re.IGNORECASE)

    # Shot 2 (Blood Sugar / Insulin): Focuses on glucose and pancreas/insulin, NO digestive tract, NO low fiber, NO bloating
    shot_2 = demo_plan.shots[1]
    assert any(w in shot_2.subject.lower() for w in ["without people", "no people", "3d medical", "3d biological", "visualization"])
    assert not any(w in shot_2.video_generation_prompt.lower() for w in ["person", "adult", "presenter", "face", "hands", "doctor", "patient"])
    assert "realistic human movement" not in shot_2.video_generation_prompt.lower()
    assert "human movement" not in shot_2.video_generation_prompt.lower()
    assert "Continuity: The recurring character" not in shot_2.video_generation_prompt
    assert not any(w in shot_2.video_generation_prompt.lower() for w in ["low fiber", "digestion", "bloating", "digestive tract", "slow digestion"]), (
        f"Shot 2 contains low fiber / digestion / bloating: {shot_2.video_generation_prompt}"
    )

    # Shot 3 (Low Fiber / Digestion / Energy): Clean digestive animation without people, NO pancreas, NO insulin, NO blood glucose
    shot_3 = demo_plan.shots[2]
    assert "Continuity: The recurring character" not in shot_3.video_generation_prompt
    assert not any(w in shot_3.video_generation_prompt.lower() for w in ["person", "adult", "presenter", "face", "hands", "doctor", "patient"])
    assert "realistic human movement" not in shot_3.video_generation_prompt.lower()
    assert "human movement" not in shot_3.video_generation_prompt.lower()
    assert not any(w in shot_3.video_generation_prompt.lower() for w in ["pancreas", "insulin", "blood glucose"]), (
        f"Shot 3 contains pancreas / insulin / blood glucose: {shot_3.video_generation_prompt}"
    )

    # Shot 2 and Shot 3 prompts are meaningfully different
    assert shot_2.video_generation_prompt != shot_3.video_generation_prompt, (
        "Shot 2 and Shot 3 must not have the same visual prompt!"
    )

    # Shot 4: Uses proper hand phrasing and never describes hair/clothing when only hands visible
    shot_4 = demo_plan.shots[3]
    assert "close-up of the young adult's hands" in shot_4.video_generation_prompt.lower()
    assert "shot of hands of the recurring" not in shot_4.video_generation_prompt.lower()
    assert not any(w in shot_4.video_generation_prompt.lower() for w in ["short dark hair", "neutral-colored casual shirt", "wearing"]), (
        f"Shot 4 describes hair/clothing when only hands should be visible: {shot_4.video_generation_prompt}"
    )

    # Character consistency across Shots 1 and 5
    for s_idx in [0, 4]:
        s_obj = demo_plan.shots[s_idx]
        assert "young adult with short dark hair, wearing a simple neutral-colored casual shirt, in a bright modern kitchen" in (s_obj.video_generation_prompt + s_obj.subject), (
            f"Shot {s_obj.shot_number} does not use consistent character description: {s_obj.video_generation_prompt}"
        )

    print(">>> REGRESSION TEST PASSED: Visual-only prompt rules and separation verified!\n")

    # -------------------------------------------------------------
    # TEST 2: Arbitrary Health Topic (Leg Cramps at Night)
    # -------------------------------------------------------------
    print("--- TEST 2: Arbitrary Health Topic (Leg Cramps at Night) ---")
    topic_2 = "Why do legs cramp at night?"
    analysis_2 = analyzer.analyze(topic_2)
    selection_2 = selector.select(analysis_2)
    script_2 = writer.write_script(analysis_2, selection_2, target_duration_seconds=45.0, force_fallback=True)

    plan_2 = planner.plan_shots(
        reel_script=script_2,
        topic_analysis=analysis_2,
        framework_metadata=selection_2,
        target_duration_seconds=45.0,
        shot_count=5,
        force_fallback=True
    )

    assert plan_2.shot_count == 5
    assert len(plan_2.shots) == 5
    assert plan_2.total_duration_seconds <= 40.0
    assert all(s.duration_seconds <= 8.0 for s in plan_2.shots)

    norm_planned_2 = " ".join(" ".join(s.script_text for s in plan_2.shots).split())
    norm_orig_2 = " ".join(script_2.full_script_with_cta.split())
    assert norm_planned_2 == norm_orig_2, "Narration mismatch on leg cramps topic"
    print(f"Framework used: #{plan_2.framework_id} {plan_2.framework_name} (Total: {plan_2.total_duration_seconds}s)")
    print(">>> Test 2 (Leg Cramps & Narration Preservation) PASSED!\n")

    # -------------------------------------------------------------
    # TEST 3: Non-Health Arbitrary Topic (Aviation / Turbulence)
    # -------------------------------------------------------------
    print("--- TEST 3: Non-Health Arbitrary Topic (Aviation / Turbulence) ---")
    topic_3 = "How do airplanes stay in the air during extreme turbulence?"
    analysis_3 = analyzer.analyze(topic_3)
    selection_3 = selector.select(analysis_3)
    script_3 = writer.write_script(analysis_3, selection_3, target_duration_seconds=40.0, force_fallback=True)

    plan_3 = planner.plan_shots(
        reel_script=script_3,
        topic_analysis=analysis_3,
        framework_metadata=selection_3,
        target_duration_seconds=40.0,
        shot_count=5,
        force_fallback=True
    )

    assert plan_3.shot_count == 5
    assert plan_3.total_duration_seconds <= 40.0
    assert all(s.duration_seconds <= 8.0 for s in plan_3.shots)
    norm_planned_3 = " ".join(" ".join(s.script_text for s in plan_3.shots).split())
    norm_orig_3 = " ".join(script_3.full_script_with_cta.split())
    assert norm_planned_3 == norm_orig_3, "Narration mismatch on turbulence topic"
    print(f"Framework used: #{plan_3.framework_id} {plan_3.framework_name} (Total: {plan_3.total_duration_seconds}s)")
    print(">>> Test 3 (Non-Health Topic) PASSED!\n")

    # -------------------------------------------------------------
    # TEST 4: Dynamic Duration-Aware Shot Planning (8s, 15s, 30s, 45s)
    # -------------------------------------------------------------
    print("--- TEST 4: Dynamic Duration-Aware Shot Planning (8s, 15s, 30s, 45s) ---")
    
    # 1. 8s -> 1 shot (~8.0s)
    script_8s = writer.write_script(analysis_1, selection_1, target_duration_seconds=8.0, force_fallback=True)
    plan_8s = planner.plan_shots(script_8s, analysis_1, selection_1, target_duration_seconds=8.0, force_fallback=True)
    assert plan_8s.shot_count == 1, f"Expected 1 shot for 8s, got {plan_8s.shot_count}"
    assert len(plan_8s.shots) == 1
    assert 4.0 <= plan_8s.shots[0].duration_seconds <= 8.0
    assert abs(plan_8s.total_duration_seconds - 8.0) <= 0.5
    print(f"  [OK] 8s Target -> {plan_8s.shot_count} shot, {plan_8s.total_duration_seconds}s total planned")

    # 2. 15s -> 2 shots (~15.0s, each 4-8s)
    script_15s = writer.write_script(analysis_1, selection_1, target_duration_seconds=15.0, force_fallback=True)
    plan_15s = planner.plan_shots(script_15s, analysis_1, selection_1, target_duration_seconds=15.0, force_fallback=True)
    assert plan_15s.shot_count == 2, f"Expected 2 shots for 15s, got {plan_15s.shot_count}"
    assert len(plan_15s.shots) == 2
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in plan_15s.shots)
    assert abs(plan_15s.total_duration_seconds - 15.0) <= 0.5
    print(f"  [OK] 15s Target -> {plan_15s.shot_count} shots, {plan_15s.total_duration_seconds}s total planned")

    # 3. 30s -> 4 shots (~30.0s, each 4-8s)
    script_30s = writer.write_script(analysis_1, selection_1, target_duration_seconds=30.0, force_fallback=True)
    plan_30s = planner.plan_shots(script_30s, analysis_1, selection_1, target_duration_seconds=30.0, force_fallback=True)
    assert plan_30s.shot_count == 4, f"Expected 4 shots for 30s, got {plan_30s.shot_count}"
    assert len(plan_30s.shots) == 4
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in plan_30s.shots)
    assert abs(plan_30s.total_duration_seconds - 30.0) <= 0.5
    print(f"  [OK] 30s Target -> {plan_30s.shot_count} shots, {plan_30s.total_duration_seconds}s total planned")

    # 4. 45s -> 6 shots (~45.0s, each 4-8s)
    script_45s = writer.write_script(analysis_1, selection_1, target_duration_seconds=45.0, force_fallback=True)
    plan_45s = planner.plan_shots(script_45s, analysis_1, selection_1, target_duration_seconds=45.0, force_fallback=True)
    assert plan_45s.shot_count == 6, f"Expected 6 shots for 45s, got {plan_45s.shot_count}"
    assert len(plan_45s.shots) == 6
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in plan_45s.shots)
    assert abs(plan_45s.total_duration_seconds - 45.0) <= 1.0
    print(f"  [OK] 45s Target -> {plan_45s.shot_count} shots, {plan_45s.total_duration_seconds}s total planned")

    # 5. Invalid shot count rejected (< 1)
    try:
        planner.plan_shots(script_1, shot_count=0, force_fallback=True)
        assert False, "Should reject shot_count=0"
    except ValueError as e:
        print(f"  [OK] shot_count=0 rejected: {e}")

    try:
        planner.plan_shots(script_1, shot_count=-1, force_fallback=True)
        assert False, "Should reject shot_count=-1"
    except ValueError as e:
        print(f"  [OK] shot_count=-1 rejected: {e}")

    print(">>> Test 4 (Dynamic Duration-Aware Shot Planning) PASSED!\n")

    # -------------------------------------------------------------
    # TEST 5: Deterministic Fallback & Validation Status
    # -------------------------------------------------------------
    print("--- TEST 5: Deterministic Fallback & Validation Status ---")
    fallback_plan = planner.plan_shots(
        reel_script=script_1,
        topic_analysis=analysis_1,
        framework_metadata=selection_1,
        target_duration_seconds=45.0,
        shot_count=5,
        force_fallback=True
    )
    assert fallback_plan.generation_source == "deterministic_fallback"
    assert fallback_plan.shot_count == 5
    assert fallback_plan.total_duration_seconds <= 40.0
    assert fallback_plan.validation_status.startswith("VALID")
    assert " ".join(" ".join(s.script_text for s in fallback_plan.shots).split()) == norm_expected
    for s in fallback_plan.shots:
        assert len(s.video_generation_prompt) > 30
        assert "Vertical 9:16" in s.video_generation_prompt
        assert s.duration_seconds <= 8.0
        assert s.validation_status.startswith("VALID")
    print(">>> Test 5 (Deterministic Fallback) PASSED!\n")

    # -------------------------------------------------------------
    # TEST 6: Narration Timing Limit (> 38 words rejected) & Safety Claim Rejection
    # -------------------------------------------------------------
    print("--- TEST 6: Narration Timing Limit & Safety Claim Rejection ---")
    # 1. Non-ReelScript rejected
    try:
        planner.plan_shots("not a script")
        assert False, "Should reject non-ReelScript"
    except TypeError as e:
        print(f"  [OK] Non-ReelScript rejected: {e}")

    # 2. Oversized section (> 38 words) rejected rather than truncated
    oversized_sections = [
        ScriptSection(stage_order=1, stage_name="Hook", text="Hook sentence here."),
        ScriptSection(stage_order=2, stage_name="Oversized Explanation", text=(
            "This section contains way too many words to fit in an eight second clip. "
            "One two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one "
            "twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight "
            "twenty-nine thirty thirty-one thirty-two thirty-three thirty-four thirty-five "
            "thirty-six thirty-seven thirty-eight thirty-nine forty."
        )),
        ScriptSection(stage_order=3, stage_name="Prevention", text="Ending sentence here.")
    ]
    oversized_script = ReelScript(
        topic="Testing Word Limit",
        framework_id=6,
        framework_name="Hook, Explanation, CTA",
        target_duration_seconds=45.0,
        hook=oversized_sections[0].text,
        sections=oversized_sections,
        cta="Follow for more tips!",
        safety_notes=["Complies with safety rules."]
    )
    try:
        planner.plan_shots(oversized_script, shot_count=5, force_fallback=True)
        assert False, "Should reject narration that exceeds 8-second speaking capacity"
    except ValueError as e:
        print(f"  [OK] Oversized narration rejected with clear error: {e}")
        assert "Oversized Explanation" in str(e) or "Shot" in str(e)

    # 3. Disallowed medical phrase rejected by validator
    try:
        ShotPlan(
            topic="Test",
            framework_id=1,
            framework_name="AIDA",
            target_duration_seconds=45.0,
            shot_count=5,
            total_duration_seconds=38.0,
            cta="CTA",
            shots=[
                Shot(
                    shot_number=i, duration_seconds=7.5, script_stage_order=1, script_stage_name="A",
                    script_text="Text", visual_goal="Goal",
                    visual_description="Shows insulin overload happening" if i == 1 else "Normal visual",
                    subject="Subject", setting="Kitchen", camera_direction="Cam", motion="Motion",
                    on_screen_text="Insulin overload drops health" if i == 1 else "Text", audio_direction="Audio",
                    video_generation_prompt="Vertical 9:16 prompt"
                )
                for i in range(1, 6)
            ]
        )
        assert False, "Should reject disallowed phrase 'insulin overload'"
    except ValueError as e:
        print(f"  [OK] Disallowed medical claim rejected: {e}")

    # 4. Shot duration > 8.0s rejected by Pydantic
    try:
        Shot(
            shot_number=1, duration_seconds=8.5, script_stage_order=1, script_stage_name="A",
            script_text="Text", visual_goal="Goal", visual_description="Visual",
            subject="Subject", setting="Kitchen", camera_direction="Cam", motion="Motion",
            on_screen_text="Caption", audio_direction="Audio", video_generation_prompt="Vertical 9:16 prompt"
        )
        assert False, "Should reject duration_seconds > 8.0"
    except ValueError as e:
        print(f"  [OK] Shot duration > 8.0s rejected: {e}")

    print(">>> Test 6 (Narration Timing & Validation Rejections) PASSED!\n")

    print("==================================================")
    print("ALL SHOT PLANNER TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_shot_planner()
