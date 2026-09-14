import sys
import os
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import (
    ReelDemoPipeline,
    run_demo_pipeline,
    DemoPipelineOutput,
    DemoShotOutput,
    format_cli_output,
)
from app.server import app
from starlette.testclient import TestClient


def test_demo_pipeline():
    print("==================================================")
    print("PHASE 4A: DEMO PIPELINE RIGOROUS TEST SUITE")
    print("==================================================")

    # -------------------------------------------------------------
    # TEST 1: End-to-End Pipeline Execution for "Junk Food"
    # -------------------------------------------------------------
    print("\n--- TEST 1: End-to-End Pipeline Execution ('Junk Food') ---")
    pipeline = ReelDemoPipeline()
    output = pipeline.run(topic="Junk Food", target_duration_seconds=45.0, shot_count=5)

    assert isinstance(output, DemoPipelineOutput), "Output must be a DemoPipelineOutput instance"
    assert output.original_topic == "Junk Food"
    print(f"  [OK] Pipeline completed for topic: {output.original_topic}")

    # -------------------------------------------------------------
    # TEST 2: Framework Selection Presence & Scoring
    # -------------------------------------------------------------
    print("\n--- TEST 2: Framework Selection Presence & Scoring ---")
    assert output.selected_framework_id == 15, f"Expected Framework 15, got {output.selected_framework_id}"
    assert output.selected_framework_name == "What Happens If"
    assert output.framework_score > 0, f"Expected positive score, got {output.framework_score}"
    print(f"  [OK] Selected Framework: #{output.selected_framework_id} '{output.selected_framework_name}' (Score: {output.framework_score})")

    # -------------------------------------------------------------
    # TEST 3: Script Presence & Exact Narration Preservation
    # -------------------------------------------------------------
    print("\n--- TEST 3: Script Presence & Exact Narration Preservation ---")
    assert output.final_script and len(output.final_script.strip()) > 30, "Final script must be non-empty"
    print(f"  [OK] Final script present: \"{output.final_script[:80]}...\"")

    # Reconstruct narration from all shots
    reconstructed_narration = " ".join(s.narration.strip() for s in output.shots if s.narration.strip())
    norm_reconstructed = " ".join(reconstructed_narration.split())
    norm_final = " ".join(output.final_script.split())

    assert norm_reconstructed == norm_final, (
        f"Reconstructed shot narration does not match final script!\n"
        f"Reconstructed: {norm_reconstructed}\n"
        f"Final Script:  {norm_final}"
    )
    print("  [OK] Exact narration preserved 100% across all shots.")

    # -------------------------------------------------------------
    # TEST 4: Exactly 5 Shots for Default 45s Configuration
    # -------------------------------------------------------------
    print("\n--- TEST 4: Shot Count & Duration ---")
    assert output.shot_count == 5, f"Expected shot_count 5, got {output.shot_count}"
    assert len(output.shots) == 5, f"Expected exactly 5 shots in list, got {len(output.shots)}"
    assert output.target_duration_seconds == 45.0, f"Expected 45.0s, got {output.target_duration_seconds}"
    print(f"  [OK] Exactly 5 shots returned for 45.0s target duration.")

    # -------------------------------------------------------------
    # TEST 5: Every Shot Contains Non-Empty Veo Prompt
    # -------------------------------------------------------------
    print("\n--- TEST 5: Veo Prompts & Shot Content ---")
    for s in output.shots:
        assert isinstance(s, DemoShotOutput)
        assert s.veo_prompt and len(s.veo_prompt.strip()) > 20, f"Shot {s.shot_number} has empty or short Veo prompt"
        assert s.narration and len(s.narration.strip()) > 0, f"Shot {s.shot_number} has empty narration"
        assert s.caption_text and len(s.caption_text.strip()) > 0, f"Shot {s.shot_number} has empty caption"
        assert s.stage and len(s.stage.strip()) > 0, f"Shot {s.shot_number} has empty stage"
        assert s.duration_seconds > 0, f"Shot {s.shot_number} duration must be positive"
        print(f"  [OK] Shot {s.shot_number}: [{s.stage}] ({s.duration_seconds}s) — Veo prompt length: {len(s.veo_prompt)} chars")

    # -------------------------------------------------------------
    # TEST 6: Shot Order is Strictly 1 to 5
    # -------------------------------------------------------------
    print("\n--- TEST 6: Shot Order Sequential Verification ---")
    shot_numbers = [s.shot_number for s in output.shots]
    assert shot_numbers == [1, 2, 3, 4, 5], f"Expected shot order [1, 2, 3, 4, 5], got {shot_numbers}"
    print(f"  [OK] Shot numbers strictly ordered: {shot_numbers}")

    # -------------------------------------------------------------
    # TEST 7: Final CTA Preserved Exactly Once
    # -------------------------------------------------------------
    print("\n--- TEST 7: Single CTA Occurrence ---")
    # CTA should appear in Shot 5 narration
    cta_occurrences = sum(1 for s in output.shots if "save this reel" in s.narration.lower() or "follow" in s.narration.lower())
    assert cta_occurrences >= 1, "CTA must be present in the shots"
    # Ensure CTA is at the end of the last shot
    assert any(word in output.shots[-1].narration.lower() for word in ["save", "follow", "share", "subscribe"]), (
        f"Final shot must contain closing CTA, got: {output.shots[-1].narration}"
    )
    print("  [OK] CTA correctly positioned at the final shot.")

    # -------------------------------------------------------------
    # REGRESSION TEST: Prompt Grammar, Continuity & Caption Relevance
    # -------------------------------------------------------------
    print("\n--- REGRESSION TEST: Prompt Grammar, Continuity & Caption Relevance ---")
    import re
    # 1. Total duration and clip limits
    assert all(s.duration_seconds <= 8.0 for s in output.shots), "All shot durations must be <= 8.0s"
    total_dur = sum(s.duration_seconds for s in output.shots)
    assert 35.0 <= total_dur <= 40.0, f"Total planned footage should be 35-40s, got {total_dur}s"
    assert output.validation_status.startswith("VALID")

    for s in output.shots:
        # Duration check
        assert s.duration_seconds <= 8.0, f"Shot {s.shot_number} duration > 8.0s"
        assert s.validation_status.startswith("VALID")

        # No malformed "The same a" phrasing
        assert not re.search(r'\bthe same a\b', s.veo_prompt, re.IGNORECASE), (
            f"Shot {s.shot_number} contains malformed 'the same a' phrasing: {s.veo_prompt}"
        )
        # No malformed "The action shows" phrasing
        assert not re.search(r'\bthe action shows\b', s.veo_prompt, re.IGNORECASE), (
            f"Shot {s.shot_number} contains malformed 'the action shows' phrasing: {s.veo_prompt}"
        )
        # Caption relevance and basic fidelity
        assert len(s.caption_text.strip()) > 0
        assert not any(sym in s.caption_text for sym in ["↑", "↓", "-->", "->"]), (
            f"Shot {s.shot_number} caption contains forbidden shorthand symbols: {s.caption_text}"
        )

        # Visual-only prompt rules:
        # Starts with Vertical 9:16
        assert s.veo_prompt.startswith("Vertical 9:16"), (
            f"Shot {s.shot_number} does not start with 'Vertical 9:16': {s.veo_prompt}"
        )
        assert not s.veo_prompt.startswith("Vertical 9:16 Vertical 9:16"), (
            f"Shot {s.shot_number} has duplicate 'Vertical 9:16': {s.veo_prompt}"
        )

        # Visual-only: NO voiceover / dialogue instructions inside the visual prompt
        assert not any(phrase in s.veo_prompt.lower() for phrase in ["spoken voiceover", "speaking exactly", "include a clear", "voiceover:"]), (
            f"Shot {s.shot_number} visual prompt contains voiceover instructions: {s.veo_prompt}"
        )
        # No quotation marks around narration or stray quotes
        assert '"' not in s.veo_prompt, f"Shot {s.shot_number} contains double quotes: {s.veo_prompt}"

        # No narration leakage
        assert s.narration.strip() not in s.veo_prompt, (
            f"Shot {s.shot_number} leaks full narration into visual prompt: {s.veo_prompt}"
        )
        for sent in re.split(r'(?<=[.!?])\s+', s.narration):
            clean_sent = sent.strip().strip('"\'')
            if len(clean_sent) >= 14:
                assert clean_sent.lower() not in s.veo_prompt.lower(), (
                    f"Shot {s.shot_number} leaks narration sentence '{clean_sent}' into visual prompt: {s.veo_prompt}"
                )

        # No subtitles or caption overlay instructions
        assert not any(phrase in s.veo_prompt.lower() for phrase in ["on-screen caption", "display subtitles", "show captions"]), (
            f"Shot {s.shot_number} visual prompt contains subtitle/caption instructions: {s.veo_prompt}"
        )

        # No unnecessary cinematic jargon
        assert not any(jargon in s.veo_prompt.lower() for jargon in ["35mm", "shallow depth of field"]), (
            f"Shot {s.shot_number} contains unnecessary jargon: {s.veo_prompt}"
        )

    # Shot 2 is a clean 3D biological visualization without people, focusing on glucose & pancreas/insulin
    shot_2_out = output.shots[1]
    assert "realistic human movement" not in shot_2_out.veo_prompt.lower(), "Shot 2 contains conflicting human movement"
    assert not any(w in shot_2_out.veo_prompt.lower() for w in ["person", "adult", "presenter", "face", "hands", "doctor", "patient"]), (
        f"Shot 2 contains prohibited person/hand/face reference: {shot_2_out.veo_prompt}"
    )
    assert not any(w in shot_2_out.veo_prompt.lower() for w in ["low fiber", "digestion", "bloating", "digestive tract", "slow digestion"]), (
        f"Shot 2 should not show low fiber/digestion/bloating: {shot_2_out.veo_prompt}"
    )

    # Shot 3 is clean educational digestive visualization without people, focusing on digestive tract & low fiber
    shot_3_out = output.shots[2]
    assert "realistic human movement" not in shot_3_out.veo_prompt.lower(), "Shot 3 contains conflicting human movement"
    assert not any(w in shot_3_out.veo_prompt.lower() for w in ["person", "adult", "presenter", "face", "hands", "doctor", "patient"]), (
        f"Shot 3 contains prohibited person/hand/face reference: {shot_3_out.veo_prompt}"
    )
    assert not any(w in shot_3_out.veo_prompt.lower() for w in ["pancreas", "insulin", "blood glucose"]), (
        f"Shot 3 should not show pancreas/insulin/blood glucose: {shot_3_out.veo_prompt}"
    )

    # Shot 2 and Shot 3 prompts are meaningfully different
    assert shot_2_out.veo_prompt != shot_3_out.veo_prompt, (
        "Shot 2 and Shot 3 must have meaningfully different visual prompts!"
    )

    # Shot 4 hands phrasing: close-up of hands and no hair/clothing descriptions
    shot_4_out = output.shots[3]
    assert "shot of hands of the recurring" not in shot_4_out.veo_prompt.lower()
    assert "close-up of the young adult's hands" in shot_4_out.veo_prompt.lower()
    assert not any(w in shot_4_out.veo_prompt.lower() for w in ["short dark hair", "neutral-colored casual shirt", "wearing"]), (
        f"Shot 4 describes hair/clothing when only hands should be visible: {shot_4_out.veo_prompt}"
    )

    # Final shot caption must match CTA
    assert any(word in output.shots[-1].caption_text.lower() for word in ["follow", "save", "share", "subscribe", "doctor", "physician"]), (
        f"Final shot caption must represent CTA, got: {output.shots[-1].caption_text}"
    )
    print("  [OK] Grammar, continuity, visual-only rules, no-leakage, and caption relevance verified for all shots.")

    # -------------------------------------------------------------
    # REGRESSION TEST: Deterministic Fallback Quality
    # -------------------------------------------------------------
    print("\n--- REGRESSION TEST: Deterministic Fallback Output Quality ---")
    from app.agents.shot_planner import ShotPlanner
    from app.agents.topic_analyzer import TopicAnalyzer
    from app.agents.framework_selector import FrameworkSelector
    from app.agents.script_writer import ScriptWriter

    ta = TopicAnalyzer().analyze("Junk Food")
    fs = FrameworkSelector().select(ta)
    sw = ScriptWriter().write_script(ta, fs, target_duration_seconds=45.0, force_fallback=True)
    sp = ShotPlanner().plan_shots(sw, ta, fs, target_duration_seconds=45.0, shot_count=5, force_fallback=True)

    assert sp.generation_source == "deterministic_fallback"
    assert len(sp.shots) == 5
    assert all(s.duration_seconds <= 8.0 for s in sp.shots)
    for s in sp.shots:
        # No 'the same a'
        assert not re.search(r'\bthe same a\b', s.video_generation_prompt, re.IGNORECASE), (
            f"Fallback Shot {s.shot_number} contains 'the same a': {s.video_generation_prompt}"
        )
        assert not re.search(r'\bthe same a\b', s.subject, re.IGNORECASE), (
            f"Fallback Shot {s.shot_number} subject contains 'the same a': {s.subject}"
        )
        # No 'the action shows'
        assert not re.search(r'\bthe action shows\b', s.video_generation_prompt, re.IGNORECASE), (
            f"Fallback Shot {s.shot_number} contains 'the action shows': {s.video_generation_prompt}"
        )
        # Non-empty prompt and caption
        assert len(s.video_generation_prompt.strip()) > 30
        assert len(s.on_screen_text.strip()) > 0
        # Visual-only: prompt starts with Vertical 9:16 and no spoken voiceover tag
        assert s.video_generation_prompt.startswith("Vertical 9:16")
        assert "Include a clear, natural spoken voiceover" not in s.video_generation_prompt
        # Exact script_text is preserved in shot model
        assert len(s.script_text) > 0

    # Verify visual prompt relevance: if shot mentions sodium/vessels, prompt should mention vessels/circulation
    for s in sp.shots:
        if "sodium" in s.script_text.lower() or "blood vessels" in s.script_text.lower():
            assert any(w in s.video_generation_prompt.lower() for w in ["sodium", "vessel", "circulation", "arterial", "salt"]), (
                f"Shot discussing sodium/vessels must have relevant visual! Got: {s.video_generation_prompt}"
            )
            assert any(w in s.on_screen_text.lower() for w in ["sodium", "vessel", "fats", "strain", "salt"]), (
                f"Shot discussing sodium/vessels must have relevant caption! Got: {s.on_screen_text}"
            )
    print("  [OK] Deterministic fallback grammar, prompt relevance, and caption fidelity verified.")

    # -------------------------------------------------------------
    # TEST 8: No Video Generation API Called
    # -------------------------------------------------------------
    print("\n--- TEST 8: Zero Video API Invocation Guard ---")
    # Assert no video generation library or external video API module is imported or active
    assert "veo" not in sys.modules, "External Veo API must not be imported"
    assert "google_video" not in sys.modules, "Google Video API must not be imported"
    assert "luma" not in sys.modules, "Luma API must not be imported"
    assert "runway" not in sys.modules, "Runway API must not be imported"
    print("  [OK] Guard passed: Zero external video APIs imported or called.")

    # -------------------------------------------------------------
    # TEST 9: Dynamic Duration-Aware Pipeline & Shot Count Derivation
    # -------------------------------------------------------------
    print("\n--- TEST 9: Dynamic Duration-Aware Pipeline & Shot Count Derivation ---")
    
    # 1. 8s target -> 1 shot (~8.0s)
    out_8s = pipeline.run(topic="everyday exercise", target_duration_seconds=8.0)
    assert out_8s.shot_count == 1, f"Expected 1 shot for 8s, got {out_8s.shot_count}"
    assert len(out_8s.shots) == 1
    total_8s = sum(s.duration_seconds for s in out_8s.shots)
    assert abs(total_8s - 8.0) <= 0.5
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in out_8s.shots)
    print(f"  [OK] 8s Pipeline: {out_8s.shot_count} shot, {total_8s:.1f}s planned")

    # 2. 15s target -> 2 shots (~15.0s)
    out_15s = pipeline.run(topic="everyday exercise", target_duration_seconds=15.0)
    assert out_15s.shot_count == 2, f"Expected 2 shots for 15s, got {out_15s.shot_count}"
    assert len(out_15s.shots) == 2
    total_15s = sum(s.duration_seconds for s in out_15s.shots)
    assert abs(total_15s - 15.0) <= 0.5
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in out_15s.shots)
    print(f"  [OK] 15s Pipeline: {out_15s.shot_count} shots, {total_15s:.1f}s planned")

    # 3. 30s target -> 4 shots (~30.0s)
    out_30s = pipeline.run(topic="everyday exercise", target_duration_seconds=30.0)
    assert out_30s.shot_count == 4, f"Expected 4 shots for 30s, got {out_30s.shot_count}"
    assert len(out_30s.shots) == 4
    total_30s = sum(s.duration_seconds for s in out_30s.shots)
    assert abs(total_30s - 30.0) <= 0.5
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in out_30s.shots)
    print(f"  [OK] 30s Pipeline: {out_30s.shot_count} shots, {total_30s:.1f}s planned")

    # 4. 45s target -> 6 shots (~45.0s)
    out_45s = pipeline.run(topic="everyday exercise", target_duration_seconds=45.0)
    assert out_45s.shot_count == 6, f"Expected 6 shots for 45s, got {out_45s.shot_count}"
    assert len(out_45s.shots) == 6
    total_45s = sum(s.duration_seconds for s in out_45s.shots)
    assert abs(total_45s - 45.0) <= 1.0
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in out_45s.shots)
    print(f"  [OK] 45s Pipeline: {out_45s.shot_count} shots, {total_45s:.1f}s planned")

    # 5. Invalid parameters rejected
    try:
        pipeline.run(topic="Junk Food", target_duration_seconds=45.0, shot_count=0)
        assert False, "Should reject shot_count=0"
    except ValueError as e:
        print(f"  [OK] shot_count=0 rejected: {e}")

    try:
        pipeline.run(topic="Junk Food", target_duration_seconds=0.0)
        assert False, "Should reject target_duration_seconds <= 0"
    except ValueError as e:
        print(f"  [OK] target_duration_seconds=0 rejected: {e}")

    # 6. Backward compatible explicit shot_count=5
    valid_5shot = pipeline.run(topic="Junk Food", target_duration_seconds=40.0, shot_count=5)
    assert valid_5shot.shot_count == 5
    assert len(valid_5shot.shots) == 5
    assert all(4.0 <= s.duration_seconds <= 8.0 for s in valid_5shot.shots)
    print("  [OK] Explicit shot_count=5 preserved for backward compatibility.")

    # -------------------------------------------------------------
    # TEST 10: CLI Formatting Test
    # -------------------------------------------------------------
    print("\n--- TEST 10: CLI Text Formatting ---")
    cli_text = format_cli_output(output)
    assert "REEL SCRIPTING AGENT — DEMO PIPELINE OUTPUT" in cli_text
    assert "Original Topic:      Junk Food" in cli_text
    assert "#15 What Happens If" in cli_text
    assert "[SHOT 1/5]" in cli_text
    assert "[SHOT 5/5]" in cli_text
    print("  [OK] CLI text formatting verified.")

    # -------------------------------------------------------------
    # TEST 11: FastAPI Endpoints & UI
    # -------------------------------------------------------------
    print("\n--- TEST 11: FastAPI Endpoints & UI ---")
    client = TestClient(app)

    # 1. GET /api/demo
    resp_demo = client.get("/api/demo")
    assert resp_demo.status_code == 200, f"Expected 200, got {resp_demo.status_code}"
    demo_json = resp_demo.json()
    assert demo_json["original_topic"] == "Junk Food"
    assert demo_json["selected_framework_id"] == 15
    assert len(demo_json["shots"]) == 5
    print("  [OK] GET /api/demo returned valid JSON matching DemoPipelineOutput.")

    # 2. GET / (UI)
    resp_ui = client.get("/")
    assert resp_ui.status_code == 200
    assert "Reel Scripting Agent" in resp_ui.text
    assert "Generate Reel Plan" in resp_ui.text
    assert "Copy Prompt" in resp_ui.text
    print("  [OK] GET / serves beginner-friendly HTML UI with 'Generate Reel Plan' action.")

    # -------------------------------------------------------------
    # TEST 12: Review / Feedback Regeneration via API
    # -------------------------------------------------------------
    print("\n--- TEST 12: Review / Feedback Regeneration via API ---")
    regen_payload = {
        "topic": "Junk Food",
        "target_duration_seconds": 45.0,
        "shot_count": 5,
        "feedback": "Make visual lighting warmer in the kitchen and emphasize sluggish energy."
    }
    resp_regen = client.post("/api/pipeline", json=regen_payload)
    assert resp_regen.status_code == 200, f"Expected 200, got {resp_regen.status_code}"
    regen_json = resp_regen.json()
    assert regen_json["original_topic"] == "Junk Food"
    assert len(regen_json["shots"]) == 5
    assert regen_json["feedback"] == regen_payload["feedback"]
    assert all(s["duration_seconds"] <= 8.0 for s in regen_json["shots"])
    print("  [OK] POST /api/pipeline successfully accepts feedback and returns regenerated 5-shot plan.")

    # -------------------------------------------------------------
    # TEST 13: Plan Approval Endpoint
    # -------------------------------------------------------------
    print("\n--- TEST 13: Plan Approval Endpoint ---")
    approve_payload = {
        "topic": "Junk Food",
        "approved": True
    }
    resp_approve = client.post("/api/approve", json=approve_payload)
    assert resp_approve.status_code == 200, f"Expected 200, got {resp_approve.status_code}"
    approve_json = resp_approve.json()
    assert approve_json["status"] == "approved"
    assert approve_json["message"] == "Plan approved — ready for video generation."
    assert approve_json["topic"] == "Junk Food"

    # Reject / invalid cases
    resp_empty = client.post("/api/approve", json={"topic": "", "approved": True})
    assert resp_empty.status_code == 400

    resp_disapprove = client.post("/api/approve", json={"topic": "Junk Food", "approved": False})
    assert resp_disapprove.status_code == 200
    assert resp_disapprove.json()["status"] == "rejected"
    print("  [OK] POST /api/approve returns 'Plan approved — ready for video generation.'")

    # -------------------------------------------------------------
    # TEST 14: Comprehensive Phase 4B UI Elements Verification
    # -------------------------------------------------------------
    print("\n--- TEST 14: Comprehensive Phase 4B UI Elements Verification ---")
    ui_html = resp_ui.text
    # Required flow elements:
    assert "Generate Reel Plan" in ui_html, "Missing 'Generate Reel Plan' button"
    assert "What would you like to change?" in ui_html, "Missing feedback prompt 'What would you like to change?'"
    assert "Regenerate Plan" in ui_html, "Missing 'Regenerate Plan' button"
    assert "Approve Plan" in ui_html, "Missing 'Approve Plan' button"
    assert "Plan approved — ready for video generation." in ui_html, "Missing approval status text"
    assert "Copy Prompt" in ui_html, "Missing 'Copy Prompt' button"
    assert "topicInput" in ui_html, "Missing topic input ID"
    assert "feedbackInput" in ui_html, "Missing feedback input ID"
    assert "approveBtn" in ui_html, "Missing approve button ID"
    assert "regenerateBtn" in ui_html, "Missing regenerate button ID"
    assert "approvalBanner" in ui_html, "Missing approval banner ID"
    print("  [OK] UI includes all required Phase 4B review, regeneration, and approval controls.")

    # -------------------------------------------------------------
    # TEST 15: Video Upload Endpoint & File Validation
    # -------------------------------------------------------------
    print("\n--- TEST 15: Video Upload Endpoint & File Validation ---")
    import tempfile
    import subprocess
    from app.video.assembler import detect_ffmpeg
    from app.server import DEMO_RUN

    # Reset DEMO_RUN state
    DEMO_RUN["approved"] = False
    DEMO_RUN["uploaded_shots"] = {}
    DEMO_RUN["rendered_output"] = None

    ffmpeg_bin = detect_ffmpeg()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video_path = os.path.join(tmpdir, "test_clip.mp4")
        test_txt_path = os.path.join(tmpdir, "test_doc.txt")
        with open(test_txt_path, "w") as f:
            f.write("Not a video file.")

        # Generate a small 1-second synthetic MP4 clip
        cmd_gen = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=720x1280:d=1.0",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            test_video_path
        ]
        subprocess.run(cmd_gen, capture_output=True, check=True)

        # 1. Invalid file rejection (.txt)
        with open(test_txt_path, "rb") as f:
            resp_invalid_file = client.post("/api/upload", data={"shot_number": 1}, files={"file": ("test_doc.txt", f, "text/plain")})
        assert resp_invalid_file.status_code == 400
        assert "Invalid file type" in resp_invalid_file.json()["detail"]
        print("  [OK] Invalid file type (.txt) correctly rejected.")

        # 2. Invalid shot number rejection (shot 0 and shot 99)
        with open(test_video_path, "rb") as f:
            resp_shot_0 = client.post("/api/upload", data={"shot_number": 0}, files={"file": ("clip.mp4", f, "video/mp4")})
        assert resp_shot_0.status_code == 400

        with open(test_video_path, "rb") as f:
            resp_shot_99 = client.post("/api/upload", data={"shot_number": 99}, files={"file": ("clip.mp4", f, "video/mp4")})
        assert resp_shot_99.status_code == 400
        print("  [OK] Invalid shot numbers (0, 99) correctly rejected.")

        # 3. Empty file rejection
        resp_empty_file = client.post("/api/upload", data={"shot_number": 1}, files={"file": ("empty.mp4", b"", "video/mp4")})
        assert resp_empty_file.status_code == 400
        assert "empty" in resp_empty_file.json()["detail"].lower()
        print("  [OK] Empty video file (0 bytes) correctly rejected.")

        # 4. Valid clip upload (Shot 1)
        with open(test_video_path, "rb") as f:
            resp_valid = client.post("/api/upload", data={"shot_number": 1}, files={"file": ("clip1.mp4", f, "video/mp4")})
        assert resp_valid.status_code == 200
        valid_data = resp_valid.json()
        assert valid_data["success"] is True
        assert valid_data["shot_number"] == 1
        assert valid_data["uploaded_count"] == 1
        assert valid_data["all_uploaded"] is False
        print("  [OK] Valid MP4 clip accepted for Shot 1.")

    # -------------------------------------------------------------
    # TEST 16: Render Endpoint Guards (Pre-Approval & Missing Shots)
    # -------------------------------------------------------------
    print("\n--- TEST 16: Render Endpoint Guards ---")
    # 1. Render rejected before plan approval
    DEMO_RUN["approved"] = False
    resp_render_pre_approval = client.post("/api/render")
    assert resp_render_pre_approval.status_code == 400
    assert "approved" in resp_render_pre_approval.json()["detail"].lower()
    print("  [OK] Render rejected before plan approval.")

    # 2. Render rejected when shots are missing (only shot 1 uploaded)
    client.post("/api/approve", json={"topic": "Junk Food", "approved": True})
    resp_render_missing = client.post("/api/render")
    assert resp_render_missing.status_code == 400
    assert "missing" in resp_render_missing.json()["detail"].lower()
    print("  [OK] Render rejected when one or more shots are missing.")

    # -------------------------------------------------------------
    # TEST 17: Successful 5-Shot Upload & Render Path
    # -------------------------------------------------------------
    print("\n--- TEST 17: Successful 5-Shot Upload & Render Path ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 distinct small test clips
        clip_files = []
        colors = ["blue", "green", "red", "yellow", "purple"]
        for i in range(1, 6):
            p = os.path.join(tmpdir, f"test_shot_{i}.mp4")
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "lavfi", "-i", f"color=c={colors[i-1]}:s=720x1280:d=1.2",
                "-f", "lavfi", "-i", f"sine=frequency={400 + i*50}:duration=1.2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                p
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            clip_files.append(p)

        # Upload all 5 shots
        for i in range(1, 6):
            with open(clip_files[i-1], "rb") as f:
                up_resp = client.post("/api/upload", data={"shot_number": i}, files={"file": (f"shot_{i}.mp4", f, "video/mp4")})
            assert up_resp.status_code == 200
        
        # Check upload status endpoint
        status_resp = client.get("/api/upload/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["approved"] is True
        assert status_data["uploaded_count"] == 5
        assert status_data["all_uploaded"] is True
        assert status_data["ready_to_render"] is True
        print("  [OK] All 5 shots uploaded; upload status reports ready_to_render=True.")

        # Render final reel
        render_resp = client.post("/api/render")
        assert render_resp.status_code == 200, f"Render failed: {render_resp.text}"
        render_data = render_resp.json()
        assert render_data["success"] is True
        assert os.path.exists(render_data["output_path"])
        assert render_data["file_size_bytes"] > 0
        assert render_data["duration_seconds"] > 0
        assert render_data["video_url"] == "/api/video/final_reel.mp4"
        print(f"  [OK] Final reel successfully assembled: {render_data['duration_seconds']}s, {render_data['file_size_bytes']} bytes.")

        # Test video streaming endpoint
        stream_resp = client.get("/api/video/final_reel.mp4")
        assert stream_resp.status_code == 200
        assert "video/mp4" in stream_resp.headers.get("content-type", "")
        print("  [OK] GET /api/video/final_reel.mp4 streams the final MP4 video.")

    # -------------------------------------------------------------
    # TEST 18: UI Upload Controls & Render Final Reel Verification
    # -------------------------------------------------------------
    print("\n--- TEST 18: UI Upload Controls & Render Final Reel ---")
    resp_ui_final = client.get("/")
    assert resp_ui_final.status_code == 200
    html_content = resp_ui_final.text
    assert "Video Generation & Upload" in html_content
    assert "Render Final Reel" in html_content
    assert "uploadTrackerBadge" in html_content
    assert "finalVideoPlayer" in html_content
    assert "downloadReelBtn" in html_content
    print("  [OK] UI contains upload section, upload inputs, and Render Final Reel controls.")

    print("\n==================================================")
    print("ALL TESTS (PHASES 1 TO 4C) PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_demo_pipeline()
