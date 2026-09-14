import sys
import os
import json
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env file
load_dotenv()

from app.agents.topic_analyzer import TopicAnalyzer
from app.agents.framework_selector import FrameworkSelector
from app.agents.script_writer import ScriptWriter, ReelScript, SENSATIONAL_PHRASES


def test_groq_integration():
    print("==================================================")
    print("GROQ LLM CONFIGURATION & VERIFICATION TEST")
    print("==================================================")

    raw_key = os.environ.get("GROQ_API_KEY", "").strip()
    key_detected = bool(raw_key)

    # STRICT SAFETY: Never print the key value!
    print(f"GROQ_API_KEY detected: {'yes' if key_detected else 'no'}")

    analyzer = TopicAnalyzer()
    selector = FrameworkSelector()
    writer = ScriptWriter()

    demo_topic = "What happens if you eat junk food every day?"
    print(f"\nAnalyzing topic: '{demo_topic}'...")
    analysis = analyzer.analyze(demo_topic)
    print(f"  - Domain: {analysis.domain}")
    print(f"  - Topic Type: {analysis.topic_type}")
    print(f"  - Analysis Source: {analysis.analysis_source}")

    print("\nSelecting optimal framework...")
    selection = selector.select(analysis)
    print(f"  - Selected Framework: #{selection.selected_framework_id} {selection.selected_framework_name}")
    print(f"  - Compatibility Score: {selection.compatibility_score}/100")
    print(f"  - Mandatory Stages: {selection.stages}")

    if not key_detected:
        print("\n[SCENARIO A] GROQ_API_KEY is empty/missing:")
        print("Testing deterministic fallback behavior...")
        script = writer.write_script(analysis, selection, target_duration_seconds=45.0)

        assert isinstance(script, ReelScript)
        assert script.generation_source == "deterministic_fallback"
        assert script.framework_id == selection.selected_framework_id
        assert len(script.sections) == len(selection.stages)
        assert [s.stage_name for s in script.sections] == selection.stages
        assert len(script.full_script.split()) > 50

        print(f"  - Generation source: {script.generation_source}")
        print(f"  - Framework: #{script.framework_id} {script.framework_name}")
        print(f"  - Word count: {len(script.full_script.split())} words")
        print(f"  - Pydantic validation: PASSED")
        print("\n>>> Missing-key fallback test PASSED!")
        print(">>> NEXT STEP: Paste your Groq API key into .env (GROQ_API_KEY=gsk_...) and re-run:")
        print("    python tests/test_groq_integration.py")
    else:
        print("\n[SCENARIO B] GROQ_API_KEY detected:")
        print("Calling Groq API (model: llama-3.3-70b-versatile)...")
        try:
            script = writer.write_script(
                analysis,
                selection,
                target_duration_seconds=45.0,
                force_fallback=False,
                raise_on_llm_error=True
            )
        except Exception as e:
            # Sanitize error to prevent key exposure
            err_str = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", str(e))
            err_type = type(e).__name__
            print(f"\n[GROQ API FAILURE]")
            print(f"  - Error category: {err_type}")
            print(f"  - Error details: {err_str}")
            if "auth" in err_str.lower() or "401" in err_str or "unauthorized" in err_str.lower():
                print("  - Diagnosis: Authentication error. Please verify the Groq API key in .env.")
            elif "rate" in err_str.lower() or "429" in err_str:
                print("  - Diagnosis: Rate limit reached on Groq API.")
            elif "model" in err_str.lower() or "404" in err_str:
                print("  - Diagnosis: Model availability issue.")
            elif "connect" in err_str.lower():
                print("  - Diagnosis: Network connection failure.")
            else:
                print(f"  - Diagnosis: API error ({err_type}).")
            raise AssertionError(f"Groq API call failed: {err_str}") from e

        # 8. Verify generation_source == "groq"
        assert script.generation_source == "groq", (
            f"Expected generation_source == 'groq', but got '{script.generation_source}'"
        )

        # 9. Validate against all requirements
        assert isinstance(script, ReelScript), "Output must be a valid ReelScript instance"
        assert script.framework_id == selection.selected_framework_id
        assert script.framework_name == selection.selected_framework_name
        assert script.target_duration_seconds == 45.0
        assert len(script.sections) == len(selection.stages)
        assert [s.stage_name for s in script.sections] == selection.stages
        assert len(script.full_script.strip()) > 0
        assert script.cta.strip() != ""
        assert len(script.safety_notes) > 0

        # Check content quality
        script_lower = script.full_script.lower()
        for phrase in SENSATIONAL_PHRASES:
            assert phrase not in script_lower, f"Sensational phrase '{phrase}' found in script"

        model_used = getattr(writer, "groq_model_used", None) or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        print(f"  - Generation source: {script.generation_source}")
        print(f"  - Model used: {model_used}")
        print(f"  - Framework: #{script.framework_id} {script.framework_name}")
        print(f"  - Compatibility score: {selection.compatibility_score}/100")
        print(f"  - Word count: {len(script.full_script.split())} words")
        print(f"  - Hook: {script.hook}")
        print(f"  - CTA: {script.cta}")
        print("\n  --- STAGE-BY-STAGE OUTPUT ---")
        for s in script.sections:
            print(f"  Stage {s.stage_order} [{s.stage_name}]: {s.text}")
        print("\n  --- FULL SCRIPT ---")
        print(f"  {script.full_script}")
        print("\n  --- SAFETY NOTES ---")
        for sn in script.safety_notes:
            print(f"  - {sn}")
        print("\n  - Pydantic & content quality validation: PASSED")
        print("\n==================================================")
        print("COMPLETE GROQ-GENERATED REELSCRIPT JSON:")
        print("==================================================")
        print(script.model_dump_json(indent=2))
        print("\n>>> Live Groq generation test PASSED!")


if __name__ == "__main__":
    test_groq_integration()
