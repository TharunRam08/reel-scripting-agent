import sys
import os

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


def test_script_writer():
    analyzer = TopicAnalyzer()
    selector = FrameworkSelector()
    writer = ScriptWriter()

    print("==================================================")
    print("TEST 1: REVISED JUNK FOOD DEMO SCRIPT")
    print("==================================================")
    topic_1 = "What happens if you eat junk food every day?"
    analysis_1 = analyzer.analyze(topic_1)
    selection_1 = selector.select(analysis_1)
    script_1 = writer.write_script(analysis_1, selection_1, target_duration_seconds=45.0)

    print(f"Topic: {script_1.topic}")
    print(f"Framework: #{script_1.framework_id} {script_1.framework_name}")
    print(f"Target Duration: {script_1.target_duration_seconds}s")
    print(f"Hook: {script_1.hook}")
    print("Sections:")
    for s in script_1.sections:
        print(f"  Stage {s.stage_order} [{s.stage_name}]: {s.text}")
    print(f"Full Script: {script_1.full_script}")
    word_count = len(script_1.full_script.split())
    print(f"Word Count: {word_count} words")
    print(f"CTA: {script_1.cta}")
    print(f"Generation Source: {script_1.generation_source}")
    print(f"Safety Notes Count: {len(script_1.safety_notes)}")

    # Assertions for Test 1
    assert script_1.framework_id == 15
    assert script_1.framework_name == "What Happens If"
    assert len(script_1.sections) == 3
    assert [s.stage_name for s in script_1.sections] == ["What happens if", "Explanation", "Prevention"]
    assert script_1.target_duration_seconds == 45.0
    
    # Assert absence of any sensational / exaggerated phrases
    script_lower = script_1.full_script.lower()
    for phrase in SENSATIONAL_PHRASES:
        assert phrase not in script_lower, f"Sensational phrase found in script: '{phrase}'"
    
    # Assert presence of careful, evidence-based language
    evidence_phrases = ["can contribute", "may increase", "can affect", "regularly relying", "can alter", "support", "may", "can"]
    assert any(w in script_lower for w in evidence_phrases)
    print(">>> Test 1 (Revised Junk Food Script) PASSED!\n")

    print("==================================================")
    print("TEST 2: MYTH/DEBUNKING TOPIC USING FRAMEWORK #8")
    print("==================================================")
    topic_2 = "Curd at night causes cold: Myth or Truth?"
    analysis_2 = analyzer.analyze(topic_2)
    selection_2 = selector.select(analysis_2)
    script_2 = writer.write_script(analysis_2, selection_2, target_duration_seconds=45.0)

    print(f"Topic: {script_2.topic}")
    print(f"Framework: #{script_2.framework_id} {script_2.framework_name}")
    print(f"Hook: {script_2.hook}")
    for s in script_2.sections:
        print(f"  Stage {s.stage_order} [{s.stage_name}]: {s.text[:80]}...")
    assert script_2.framework_id == 8
    assert [s.stage_name for s in script_2.sections] == ["Myth", "Truth", "Explanation"]
    print(">>> Test 2 PASSED!\n")

    print("==================================================")
    print("TEST 3: '3 SIGNS' TOPIC USING FRAMEWORK #16")
    print("==================================================")
    topic_3 = "3 signs your liver is under stress"
    analysis_3 = analyzer.analyze(topic_3)
    selection_3 = selector.select(analysis_3)
    script_3 = writer.write_script(analysis_3, selection_3, target_duration_seconds=45.0)

    print(f"Topic: {script_3.topic}")
    print(f"Framework: #{script_3.framework_id} {script_3.framework_name}")
    for s in script_3.sections:
        print(f"  Stage {s.stage_order} [{s.stage_name}]: {s.text[:80]}...")
    assert script_3.framework_id == 16
    assert [s.stage_name for s in script_3.sections] == ["Hook", "Three signs", "Meaning", "Next step"]
    print(">>> Test 3 PASSED!\n")

    print("==================================================")
    print("TEST 4: VERIFY EXACT STAGE ORDER MATCHING")
    print("==================================================")
    topic_4 = "Normal fever vs Dengue red flags"
    analysis_4 = analyzer.analyze(topic_4)
    selection_4 = selector.select(analysis_4)
    script_4 = writer.write_script(analysis_4, selection_4, target_duration_seconds=45.0)

    print(f"Selected framework stages: {selection_4.stages}")
    script_stages = [s.stage_name for s in script_4.sections]
    print(f"Script section stages:    {script_stages}")
    assert script_stages == selection_4.stages
    for idx, s in enumerate(script_4.sections, 1):
        assert s.stage_order == idx
    print(">>> Test 4 PASSED!\n")

    print("==================================================")
    print("TEST 5: PYDANTIC VALIDATION & CONTENT QUALITY REJECTION")
    print("==================================================")
    # Valid script passes
    assert isinstance(script_1, ReelScript)

    # Rejection of sensational phrase via validator
    try:
        ReelScript(
            topic="Test",
            framework_id=1,
            framework_name="AIDA",
            target_duration_seconds=45.0,
            hook="Hook",
            sections=[
                ScriptSection(stage_order=1, stage_name="Attention", text="This foods your bloodstream with toxins!"),
                ScriptSection(stage_order=2, stage_name="Interest", text="It overloads your liver completely!"),
            ],
            full_script="This foods your bloodstream with toxins! It overloads your liver completely!",
            cta="Follow for more health updates!",
            safety_notes=["Safe"]
        )
        assert False, "Should have failed due to sensational phrase 'overloads your liver'"
    except ValueError as e:
        print(f"  [OK] Sensational phrase rejected by validator: {e}")

    # CTA containing medical consultation term rejected
    try:
        ReelScript(
            topic="Test",
            framework_id=1,
            framework_name="AIDA",
            target_duration_seconds=45.0,
            hook="Hook",
            sections=[
                ScriptSection(stage_order=1, stage_name="Attention", text="Take care of your health daily."),
                ScriptSection(stage_order=2, stage_name="Action", text="Consult a doctor if pain persists."),
            ],
            cta="Schedule a visit with your physician and follow for more tips!",
            safety_notes=["Safe"]
        )
        assert False, "Should have failed on CTA containing consultation term 'physician'"
    except ValueError as e:
        print(f"  [OK] CTA containing consultation term rejected: {e}")

    # CTA duplicating a sentence from script rejected
    try:
        ReelScript(
            topic="Test",
            framework_id=1,
            framework_name="AIDA",
            target_duration_seconds=45.0,
            hook="Hook",
            sections=[
                ScriptSection(stage_order=1, stage_name="Attention", text="Take care of your health daily."),
                ScriptSection(stage_order=2, stage_name="Action", text="If symptoms continue, see a clinic."),
            ],
            cta="If symptoms continue, see a clinic. Follow for more tips!",
            safety_notes=["Safe"]
        )
        assert False, "Should have failed on CTA duplicating a script sentence"
    except ValueError as e:
        print(f"  [OK] CTA duplicating script sentence rejected: {e}")

    # CTA missing action verb rejected
    try:
        ReelScript(
            topic="Test",
            framework_id=1,
            framework_name="AIDA",
            target_duration_seconds=45.0,
            hook="Hook",
            sections=[
                ScriptSection(stage_order=1, stage_name="Attention", text="Take care of your health daily."),
            ],
            cta="A very informative reel about wellness.",
            safety_notes=["Safe"]
        )
        assert False, "Should have failed on CTA without action verb"
    except ValueError as e:
        print(f"  [OK] CTA missing action verb rejected: {e}")

    # Negative target duration rejected
    try:
        writer.write_script(analysis_1, selection_1, target_duration_seconds=-5.0)
        assert False, "Should have failed on negative target duration"
    except ValueError as e:
        print(f"  [OK] Negative target duration rejected: {e}")

    # Disordered stages rejected
    try:
        ReelScript(
            topic="Test",
            framework_id=1,
            framework_name="AIDA",
            target_duration_seconds=45.0,
            hook="Hook",
            sections=[
                ScriptSection(stage_order=2, stage_name="Order2", text="Text"),
                ScriptSection(stage_order=1, stage_name="Order1", text="Text")
            ],
            full_script="Test full",
            cta="Follow for more health updates!",
            safety_notes=["Safe"]
        )
        assert False, "Should have failed on disordered stage orders"
    except Exception as e:
        print(f"  [OK] Disordered sections rejected: {e}")
    print(">>> Test 5 PASSED!\n")

    print("==================================================")
    print("TEST 6: DETERMINISTIC FALLBACK VERIFICATION")
    print("==================================================")
    fallback_script = writer.write_script(
        analysis_1, selection_1, target_duration_seconds=45.0, force_fallback=True
    )
    assert fallback_script.generation_source == "deterministic_fallback"
    assert len(fallback_script.full_script) > 50
    assert len(fallback_script.sections) == 3
    print(f"Fallback generation verified: {fallback_script.generation_source}")
    print(">>> Test 6 PASSED!\n")

    print("==================================================")
    print("TEST 7: UNIVERSAL SAFETY RULES COMPLIANCE")
    print("==================================================")
    full_spoken_text = (script_1.full_script + " " + script_1.cta).lower()
    assert not any(x in full_spoken_text for x in ["500mg", "100mg", "take two tablets", "prescribe"])
    assert any(x in full_spoken_text for x in ["doctor", "consult", "physician", "clinic", "healthcare"])
    assert script_1.cta != ""
    assert len(script_1.safety_notes) >= 6
    print("Safety rules compliance confirmed:")
    for note in script_1.safety_notes:
        print(f"  - {note}")
    print(">>> Test 7 PASSED!\n")

    print("==================================================")
    print("ALL SCRIPT WRITER TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_script_writer()
