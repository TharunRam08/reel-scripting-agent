import sys
import os
import hashlib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import run_demo_pipeline, ReelDemoPipeline
from app.agents.shot_planner import calculate_recommended_shot_count


def test_multi_duration_suite():
    print("==================================================")
    print("MULTI-DURATION PIPELINE TEST SUITE (8s, 15s, 30s, 45s)")
    print("==================================================")

    # -------------------------------------------------------------
    # STEP 0: Verify submission artifacts exist and record baseline
    # -------------------------------------------------------------
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    submission_dir = os.path.join(root_dir, "submission")
    zip_path = os.path.join(root_dir, "junk_food_reel_submission.zip")

    assert os.path.exists(submission_dir), "submission/ directory must exist"
    assert os.path.exists(zip_path), "junk_food_reel_submission.zip must exist"

    def get_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    initial_zip_hash = get_hash(zip_path)
    initial_sub_hashes = {
        fn: get_hash(os.path.join(submission_dir, fn))
        for fn in os.listdir(submission_dir)
        if os.path.isfile(os.path.join(submission_dir, fn))
    }
    print(f"  [Integrity Baseline] Protected submission artifacts: {len(initial_sub_hashes)} files + ZIP verified.")

    pipeline = ReelDemoPipeline()

    # -------------------------------------------------------------
    # TEST 1: 8 Seconds -> Exactly 1 Shot (~8.0s)
    # -------------------------------------------------------------
    print("\n--- TEST 1: 8 Seconds Target ---")
    assert calculate_recommended_shot_count(8.0) == 1
    
    topics_to_test = ["everyday exercise", "Junk Food", "stress management"]
    for topic in topics_to_test:
        out = pipeline.run(topic=topic, target_duration_seconds=8.0)
        assert out.shot_count == 1, f"Expected 1 shot for 8s, got {out.shot_count} ({topic})"
        assert len(out.shots) == 1
        shot = out.shots[0]
        assert 4.0 <= shot.duration_seconds <= 8.0, f"Shot duration {shot.duration_seconds}s not in [4.0, 8.0]"
        total_dur = sum(s.duration_seconds for s in out.shots)
        assert 7.5 <= total_dur <= 8.0, f"Total planned duration {total_dur}s not close to 8.0s"
        assert len(shot.narration.split()) <= 38, f"Word count {len(shot.narration.split())} > 38"
        print(f"  [OK] '{topic}' (8s): {out.shot_count} shot, {total_dur:.1f}s planned ({len(shot.narration.split())} words)")

    # -------------------------------------------------------------
    # TEST 2: 15 Seconds -> Exactly 2 Shots (~15.0s, each 4.0-8.0s)
    # -------------------------------------------------------------
    print("\n--- TEST 2: 15 Seconds Target ---")
    assert calculate_recommended_shot_count(15.0) == 2

    for topic in topics_to_test:
        out = pipeline.run(topic=topic, target_duration_seconds=15.0)
        assert out.shot_count == 2, f"Expected 2 shots for 15s, got {out.shot_count} ({topic})"
        assert len(out.shots) == 2
        total_dur = sum(s.duration_seconds for s in out.shots)
        assert 14.0 <= total_dur <= 15.5, f"Total planned duration {total_dur}s not close to 15.0s"
        for s in out.shots:
            assert 4.0 <= s.duration_seconds <= 8.0, f"Shot {s.shot_number} duration {s.duration_seconds}s not in [4.0, 8.0]"
            assert len(s.narration.split()) <= 38, f"Shot {s.shot_number} word count {len(s.narration.split())} > 38"
        print(f"  [OK] '{topic}' (15s): {out.shot_count} shots, {total_dur:.1f}s planned (durs: {[s.duration_seconds for s in out.shots]})")

    # -------------------------------------------------------------
    # TEST 3: 30 Seconds -> Exactly 4 Shots (~30.0s, each 4.0-8.0s)
    # -------------------------------------------------------------
    print("\n--- TEST 3: 30 Seconds Target ---")
    assert calculate_recommended_shot_count(30.0) == 4

    for topic in topics_to_test:
        out = pipeline.run(topic=topic, target_duration_seconds=30.0)
        assert out.shot_count == 4, f"Expected 4 shots for 30s, got {out.shot_count} ({topic})"
        assert len(out.shots) == 4
        total_dur = sum(s.duration_seconds for s in out.shots)
        assert 28.5 <= total_dur <= 31.0, f"Total planned duration {total_dur}s not close to 30.0s"
        for s in out.shots:
            assert 4.0 <= s.duration_seconds <= 8.0, f"Shot {s.shot_number} duration {s.duration_seconds}s not in [4.0, 8.0]"
            assert len(s.narration.split()) <= 38, f"Shot {s.shot_number} word count {len(s.narration.split())} > 38"
        print(f"  [OK] '{topic}' (30s): {out.shot_count} shots, {total_dur:.1f}s planned (durs: {[s.duration_seconds for s in out.shots]})")

    # -------------------------------------------------------------
    # TEST 4: 45 Seconds -> Exactly 6 Shots (~45.0s, each 4.0-8.0s)
    # -------------------------------------------------------------
    print("\n--- TEST 4: 45 Seconds Target ---")
    assert calculate_recommended_shot_count(45.0) == 6

    for topic in topics_to_test:
        out = pipeline.run(topic=topic, target_duration_seconds=45.0)
        assert out.shot_count == 6, f"Expected 6 shots for 45s, got {out.shot_count} ({topic})"
        assert len(out.shots) == 6
        total_dur = sum(s.duration_seconds for s in out.shots)
        assert 44.0 <= total_dur <= 46.0, f"Total planned duration {total_dur}s not close to 45.0s"
        for s in out.shots:
            assert 4.0 <= s.duration_seconds <= 8.0, f"Shot {s.shot_number} duration {s.duration_seconds}s not in [4.0, 8.0]"
            assert len(s.narration.split()) <= 38, f"Shot {s.shot_number} word count {len(s.narration.split())} > 38"
        print(f"  [OK] '{topic}' (45s): {out.shot_count} shots, {total_dur:.1f}s planned (durs: {[s.duration_seconds for s in out.shots]})")

    # -------------------------------------------------------------
    # TEST 5: Verify submission artifacts remain 100% untouched
    # -------------------------------------------------------------
    print("\n--- TEST 5: Submission Artifacts Integrity Guard ---")
    final_zip_hash = get_hash(zip_path)
    assert final_zip_hash == initial_zip_hash, "CRITICAL: junk_food_reel_submission.zip was modified!"

    for fn, orig_hash in initial_sub_hashes.items():
        sub_file = os.path.join(submission_dir, fn)
        assert os.path.exists(sub_file), f"CRITICAL: submission/{fn} is missing!"
        cur_hash = get_hash(sub_file)
        assert cur_hash == orig_hash, f"CRITICAL: submission/{fn} was modified!"
    print("  [OK] All submission artifacts verified 100% UNTOUCHED.")

    print("\n==================================================")
    print("ALL MULTI-DURATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_multi_duration_suite()
