import sys
import os
import tempfile
import subprocess
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.shot_planner import ShotPlan, Shot
from app.agents.script_writer import ScriptSection, ReelScript
from app.video.assembler import (
    VideoAssembler,
    ClipMetadata,
    AssemblyPlan,
    AssemblyResult,
    detect_ffmpeg,
    detect_ffprobe,
)
from app.video.captions import (
    CaptionGenerator,
    CaptionEntry,
    BurnedCaptionConfig,
)


def create_sample_shot_plan() -> ShotPlan:
    """Creates an approved 5-shot ShotPlan for testing."""
    sections = [
        ScriptSection(stage_order=1, stage_name="What happens if", text="What happens if you eat junk food every day?"),
        ScriptSection(stage_order=2, stage_name="Explanation", text="Your body releases insulin to manage glucose surges. Unhealthy fats can raise blood pressure over time."),
        ScriptSection(stage_order=3, stage_name="Prevention", text="Swap chips for fresh fruit and nuts. If fatigue persists, consult a physician.")
    ]
    cta = "Follow for more science-backed nutrition tips!"
    script = ReelScript(
        topic="Junk Food",
        framework_id=15,
        framework_name="What Happens If",
        target_duration_seconds=45.0,
        hook=sections[0].text,
        sections=sections,
        cta=cta,
        safety_notes=["Universal rules compliant."]
    )

    shots = [
        Shot(
            shot_number=1, duration_seconds=6.8, script_stage_order=1, script_stage_name="What happens if",
            script_text="What happens if you eat junk food every day?", visual_goal="Hook", visual_description="Person holding chips",
            subject="Person", setting="Kitchen", camera_direction="Vertical 9:16 medium", motion="Holding chips",
            on_screen_text="What if you eat this daily?", audio_direction="Voiceover", video_generation_prompt="Vertical 9:16 prompt 1"
        ),
        Shot(
            shot_number=2, duration_seconds=7.8, script_stage_order=2, script_stage_name="Explanation",
            script_text="Your body releases insulin to manage glucose surges.", visual_goal="Metabolism", visual_description="Glucose animation",
            subject="Person", setting="Kitchen", camera_direction="Vertical 9:16 close-up", motion="Reaching for soda",
            on_screen_text="Insulin spike", audio_direction="Voiceover", video_generation_prompt="Vertical 9:16 prompt 2"
        ),
        Shot(
            shot_number=3, duration_seconds=7.5, script_stage_order=2, script_stage_name="Explanation",
            script_text="Unhealthy fats can raise blood pressure over time.", visual_goal="Long term", visual_description="Person resting",
            subject="Person", setting="Kitchen", camera_direction="Vertical 9:16 profile", motion="Breathing deeply",
            on_screen_text="Blood pressure", audio_direction="Voiceover", video_generation_prompt="Vertical 9:16 prompt 3"
        ),
        Shot(
            shot_number=4, duration_seconds=7.5, script_stage_order=3, script_stage_name="Prevention",
            script_text="Swap chips for fresh fruit and nuts.", visual_goal="Solutions", visual_description="Placing fruits",
            subject="Hands", setting="Counter", camera_direction="Vertical 9:16 top-down", motion="Placing berries",
            on_screen_text="Fresh fruit and nuts", audio_direction="Voiceover", video_generation_prompt="Vertical 9:16 prompt 4"
        ),
        Shot(
            shot_number=5, duration_seconds=7.8, script_stage_order=3, script_stage_name="Prevention",
            script_text="If fatigue persists, consult a physician. Follow for more science-backed nutrition tips!", visual_goal="Closing", visual_description="Smiling at camera",
            subject="Person", setting="Kitchen", camera_direction="Vertical 9:16 medium pull-back", motion="Affirmative nod",
            on_screen_text="Follow for more tips!", audio_direction="Voiceover", video_generation_prompt="Vertical 9:16 prompt 5"
        )
    ]

    return ShotPlan(
        topic="What happens if you eat junk food every day?",
        framework_id=15,
        framework_name="What Happens If",
        target_duration_seconds=45.0,
        shot_count=5,
        total_duration_seconds=37.4,
        cta=cta,
        continuity_notes="Consistent character in kitchen",
        shots=shots,
        generation_source="deterministic_fallback"
    )


def test_video_infrastructure():
    print("==================================================")
    print("PHASE 3A: VIDEO FINALIZATION TEST SUITE")
    print("==================================================")

    # -------------------------------------------------------------
    # TEST 1: FFmpeg and FFprobe Availability Detection
    # -------------------------------------------------------------
    print("\n--- TEST 1: FFmpeg & FFprobe Availability Detection ---")
    ffmpeg_path = detect_ffmpeg()
    ffprobe_path = detect_ffprobe()

    print(f"FFmpeg detected at:  {ffmpeg_path}")
    print(f"FFprobe detected at: {ffprobe_path}")

    assert os.path.exists(ffmpeg_path), f"FFmpeg path does not exist: {ffmpeg_path}"
    assert os.path.exists(ffprobe_path), f"FFprobe path does not exist: {ffprobe_path}"

    # Verify executable by running -version
    r_ffmpeg = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
    assert r_ffmpeg.returncode == 0, "ffmpeg -version failed"
    assert "ffmpeg version" in r_ffmpeg.stdout

    r_ffprobe = subprocess.run([ffprobe_path, "-version"], capture_output=True, text=True)
    assert r_ffprobe.returncode == 0, "ffprobe -version failed"
    assert "ffprobe version" in r_ffprobe.stdout

    print(">>> Test 1 (FFmpeg & FFprobe Detection) PASSED!")

    # -------------------------------------------------------------
    # TEST 2: Caption Generation, Timing, & Exact Fidelity
    # -------------------------------------------------------------
    print("\n--- TEST 2: Caption Generation, Timing, & Exact Narration Fidelity ---")
    shot_plan = create_sample_shot_plan()
    caption_gen = CaptionGenerator()

    entries = caption_gen.generate_from_shot_plan(shot_plan)
    assert len(entries) > 0

    # 1. Verify exact narration text preservation (zero words added, altered, or omitted)
    reconstructed_caption_text = " ".join(e.text for e in entries)
    # Expected narration from all shots
    expected_full_narration = " ".join(s.script_text for s in shot_plan.shots)

    norm_reconstructed = " ".join(reconstructed_caption_text.split())
    norm_expected = " ".join(expected_full_narration.split())

    print(f"Expected Narration:\n  \"{norm_expected}\"")
    print(f"Reconstructed Captions:\n  \"{norm_reconstructed}\"")

    assert norm_reconstructed == norm_expected, (
        f"Caption text was altered from approved narration!\nExpected:\n{norm_expected}\nGot:\n{norm_reconstructed}"
    )

    # 2. Verify caption timing aligns with shot durations
    current_time = 0.0
    for shot in shot_plan.shots:
        shot_entries = [e for e in entries if e.shot_number == shot.shot_number]
        assert len(shot_entries) > 0, f"No captions generated for shot {shot.shot_number}"
        # Start time of first entry in shot
        assert round(shot_entries[0].start_seconds, 2) == round(current_time, 2)
        # End time of last entry in shot
        assert round(shot_entries[-1].end_seconds, 2) == round(current_time + shot.duration_seconds, 2)
        current_time += shot.duration_seconds

    # 3. Verify SRT serialization
    srt_content = caption_gen.generate_srt(entries)
    assert "-->" in srt_content
    assert "What happens if you eat junk food every day?" in srt_content
    assert "Follow for more science-backed nutrition tips!" in srt_content

    # 4. Verify ASS serialization with mobile 9:16 vertical styling
    ass_content = caption_gen.generate_ass(entries, video_width=720, video_height=1280)
    assert "[Script Info]" in ass_content
    assert "PlayResX: 720" in ass_content
    assert "PlayResY: 1280" in ass_content
    assert "Style: ReelCaptions" in ass_content
    assert "MarginV" in ass_content
    assert "Dialogue: 0," in ass_content

    # 5. Verify Windows-safe subtitle filter construction
    filter_str = caption_gen.build_subtitles_filter("C:\\path\\to\\captions.ass")
    assert "ass=" in filter_str
    assert "C\\:/path/to/captions.ass" in filter_str or "C:/path/to/captions.ass" in filter_str

    print(">>> Test 2 (Captions & Exact Fidelity) PASSED!")

    # -------------------------------------------------------------
    # TEST 3: Assembler Command Construction & 9:16 Normalization
    # -------------------------------------------------------------
    print("\n--- TEST 3: Assembler Command Construction & 9:16 Normalization ---")
    assembler = VideoAssembler(target_width=720, target_height=1280, target_duration_seconds=45.0)

    fake_clips = [
        "c:\\videos\\clip1.mp4",
        "c:\\videos\\clip2.mp4",
        "c:\\videos\\clip3.mp4",
        "c:\\videos\\clip4.mp4",
        "c:\\videos\\clip5.mp4"
    ]

    cmd = assembler.build_assembly_command(
        clip_paths=fake_clips,
        output_path="c:\\output\\final_reel.mp4",
        burned_subtitles_path="c:\\output\\captions.ass"
    )

    cmd_str = " ".join(cmd)
    print(f"Generated FFmpeg Command:\n  {cmd_str[:120]}... [length: {len(cmd_str)} chars]")

    # Check command integrity
    assert assembler.ffmpeg_bin in cmd[0]
    assert "-y" in cmd
    # Check all 5 input clips are present in exact order
    for idx, clip in enumerate(fake_clips):
        assert clip in cmd_str
        # Verify clip appears before subsequent clip
        if idx > 0:
            assert cmd_str.find(clip) > cmd_str.find(fake_clips[idx - 1])

    # Check 9:16 scale and crop filters (avoids stretching)
    assert "scale=720:1280:force_original_aspect_ratio=increase" in cmd_str
    assert "crop=720:1280:" in cmd_str
    assert "concat=n=5:v=1:a=1" in cmd_str

    # Check audio normalization
    assert "aresample=48000" in cmd_str
    assert "aformat=channel_layouts=stereo" in cmd_str

    # Check web-compatible encoding parameters
    assert "-c:v libx264" in cmd_str
    assert "-pix_fmt yuv420p" in cmd_str
    assert "-c:a aac" in cmd_str
    assert "-b:a 192k" in cmd_str
    assert "-movflags +faststart" in cmd_str

    # Check burned subtitle filter inclusion
    assert "ass=" in cmd_str

    print(">>> Test 3 (Command Construction & 9:16 Filtering) PASSED!")

    # -------------------------------------------------------------
    # TEST 4: Configurable Resolution & Target Duration Handling
    # -------------------------------------------------------------
    print("\n--- TEST 4: Configurable Resolution & Target Duration ---")
    custom_assembler = VideoAssembler(target_width=1080, target_height=1920, target_duration_seconds=30.0)
    assert custom_assembler.target_width == 1080
    assert custom_assembler.target_height == 1920
    assert custom_assembler.target_duration_seconds == 30.0

    cmd_1080p = custom_assembler.build_assembly_command(
        clip_paths=["c:\\v1.mp4", "c:\\v2.mp4"],
        output_path="c:\\out.mp4"
    )
    cmd_1080p_str = " ".join(cmd_1080p)
    assert "scale=1080:1920" in cmd_1080p_str
    assert "crop=1080:1920" in cmd_1080p_str

    # Invalid dimension validation
    try:
        VideoAssembler(target_width=0, target_height=1280)
        assert False, "Should reject non-positive width"
    except ValueError:
        print("  [OK] Non-positive width rejected")

    try:
        VideoAssembler(target_width=720, target_height=-1280)
        assert False, "Should reject non-positive height"
    except ValueError:
        print("  [OK] Non-positive height rejected")

    try:
        VideoAssembler(target_duration_seconds=0)
        assert False, "Should reject zero duration"
    except ValueError:
        print("  [OK] Zero target duration rejected")

    print(">>> Test 4 (Configurable Parameters & Validation) PASSED!")

    # -------------------------------------------------------------
    # TEST 5: Missing Clip Handling & Error Raising
    # -------------------------------------------------------------
    print("\n--- TEST 5: Missing Clip Handling ---")
    try:
        assembler.inspect_clip("c:\\non_existent_clip_12345.mp4")
        assert False, "Should raise FileNotFoundError for missing clip"
    except FileNotFoundError as e:
        print(f"  [OK] Missing clip rejected: {e}")

    try:
        assembler.inspect_clips([])
        assert False, "Should reject empty clip list"
    except ValueError as e:
        print(f"  [OK] Empty clip list rejected: {e}")

    print(">>> Test 5 (Missing Clip Handling) PASSED!")

    # -------------------------------------------------------------
    # TEST 6: Real Synthetic Video Assembly End-to-End Test
    # -------------------------------------------------------------
    print("\n--- TEST 6: Real Synthetic Video Assembly (FFmpeg End-to-End) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        clip1_path = os.path.join(tmpdir, "shot1.mp4")
        clip2_path = os.path.join(tmpdir, "shot2.mp4")
        out_path = os.path.join(tmpdir, "assembled_reel.mp4")
        ass_path = os.path.join(tmpdir, "captions.ass")

        # Generate two small 1.5-second synthetic test clips with color + tone audio
        # Clip 1: 720x1280 vertical color test
        cmd_gen1 = [
            ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=720x1280:d=1.5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            clip1_path
        ]
        r1 = subprocess.run(cmd_gen1, capture_output=True, text=True)
        assert r1.returncode == 0, f"Failed to generate synthetic clip 1: {r1.stderr}"

        # Clip 2: 1280x720 horizontal color test (to verify scale/crop without stretching!)
        cmd_gen2 = [
            ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "color=c=green:s=1280x720:d=1.5",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=1.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            clip2_path
        ]
        r2 = subprocess.run(cmd_gen2, capture_output=True, text=True)
        assert r2.returncode == 0, f"Failed to generate synthetic clip 2: {r2.stderr}"

        # 1. Inspect clips via FFprobe
        meta1 = assembler.inspect_clip(clip1_path)
        meta2 = assembler.inspect_clip(clip2_path)

        assert meta1.width == 720 and meta1.height == 1280
        assert meta1.is_vertical_9_16 is True
        assert meta1.has_audio is True
        assert abs(meta1.duration_seconds - 1.5) < 0.2

        assert meta2.width == 1280 and meta2.height == 720
        assert meta2.is_vertical_9_16 is False
        assert meta2.has_audio is True

        print(f"  Clip 1 inspected: {meta1.width}x{meta1.height}, {meta1.duration_seconds}s, audio={meta1.has_audio}")
        print(f"  Clip 2 inspected: {meta2.width}x{meta2.height}, {meta2.duration_seconds}s, audio={meta2.has_audio}")

        # 2. Create test captions
        synthetic_entries = [
            CaptionEntry(index=1, shot_number=1, start_seconds=0.0, end_seconds=1.5, text="First shot narration."),
            CaptionEntry(index=2, shot_number=2, start_seconds=1.5, end_seconds=3.0, text="Second shot concluding narration.")
        ]
        caption_gen.write_ass_file(synthetic_entries, ass_path, video_width=720, video_height=1280)
        assert os.path.exists(ass_path)

        # 3. Plan assembly and verify duration delta reporting
        plan = assembler.plan_assembly([clip1_path, clip2_path], out_path, burned_subtitles_path=ass_path)
        print(f"  Total actual clip duration: {plan.total_clip_duration_seconds}s")
        print(f"  Target duration:           {plan.target_duration_seconds}s")
        print(f"  Duration delta:            {plan.duration_delta_seconds}s")
        assert plan.total_clip_duration_seconds > 2.5
        # Verify delta is exposed without blind padding
        assert plan.duration_delta_seconds == round(plan.total_clip_duration_seconds - plan.target_duration_seconds, 3)

        # 4. Execute actual assembly with FFmpeg
        result = assembler.assemble([clip1_path, clip2_path], out_path, burned_subtitles_path=ass_path)
        assert result.success is True
        assert os.path.exists(out_path)
        assert result.file_size_bytes > 0

        # 5. Inspect assembled output
        final_meta = assembler.inspect_clip(out_path)
        assert final_meta.width == 720
        assert final_meta.height == 1280
        assert final_meta.is_vertical_9_16 is True
        assert final_meta.has_audio is True
        assert final_meta.video_codec == "h264"
        assert final_meta.audio_codec == "aac"
        assert abs(final_meta.duration_seconds - 3.0) < 0.3

        print(f"  Output assembled: {final_meta.width}x{final_meta.height} (9:16), duration={final_meta.duration_seconds}s, size={result.file_size_bytes} bytes")
        print(">>> Test 6 (Real Synthetic Video Assembly) PASSED!")

    print("\n==================================================")
    print("ALL VIDEO FINALIZATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_video_infrastructure()
