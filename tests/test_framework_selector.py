import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.topic_analyzer import TopicAnalyzer
from app.agents.framework_selector import FrameworkSelector, FrameworkSelection


def test_framework_selector():
    analyzer = TopicAnalyzer()
    selector = FrameworkSelector()

    print("==================================================")
    print("TEST 1: 'Junk Food'")
    print("==================================================")
    topic_1 = "Junk Food"
    analysis_1 = analyzer.analyze(topic_1)
    selection_1 = selector.select(analysis_1)
    
    print(f"Topic: '{topic_1}'")
    print(f"Selected: #{selection_1.selected_framework_id} {selection_1.selected_framework_name}")
    print(f"Score: {selection_1.compatibility_score}/100")
    print(f"Stages: {selection_1.stages}")
    print("Top Candidates:")
    for c in selection_1.ranked_candidates[:3]:
        print(f"  #{c.framework_id:02d} {c.framework_name:<30} Score: {c.score:.1f} | {c.short_reason}")
    
    assert isinstance(selection_1, FrameworkSelection)
    assert selection_1.selected_framework_id in range(1, 31)
    assert selection_1.compatibility_score > 0
    assert len(selection_1.stages) >= 2
    assert selection_1.total_evaluated == 30
    print(">>> Test 1 PASSED!\n")

    print("==================================================")
    print("TEST 2: 'What happens if you eat junk food every day?'")
    print("==================================================")
    topic_2 = "What happens if you eat junk food every day?"
    analysis_2 = analyzer.analyze(topic_2)
    selection_2 = selector.select(analysis_2)

    print(f"Topic: '{topic_2}'")
    print(f"Selected: #{selection_2.selected_framework_id} {selection_2.selected_framework_name}")
    print(f"Score: {selection_2.compatibility_score}/100")
    print(f"Reason: {selection_2.reason}")
    print(f"Stages: {selection_2.stages}")
    print("Top Candidates:")
    for c in selection_2.ranked_candidates[:3]:
        print(f"  #{c.framework_id:02d} {c.framework_name:<30} Score: {c.score:.1f} | {c.short_reason}")

    # Naturally Framework 15 ("What Happens If") should be selected
    assert selection_2.selected_framework_id == 15
    assert selection_2.selected_framework_name == "What Happens If"
    assert selection_2.stages == ["What happens if", "Explanation", "Prevention"]
    assert selection_2.total_evaluated == 30
    print(">>> Test 2 PASSED!\n")

    print("==================================================")
    print("TEST 3: ARBITRARY 'Why...' TOPIC")
    print("==================================================")
    topic_3 = "Why do legs cramp at night?"
    analysis_3 = analyzer.analyze(topic_3)
    selection_3 = selector.select(analysis_3)

    print(f"Topic: '{topic_3}'")
    print(f"Selected: #{selection_3.selected_framework_id} {selection_3.selected_framework_name}")
    print(f"Score: {selection_3.compatibility_score}/100")
    print(f"Stages: {selection_3.stages}")
    print("Top Candidates:")
    for c in selection_3.ranked_candidates[:3]:
        print(f"  #{c.framework_id:02d} {c.framework_name:<30} Score: {c.score:.1f} | {c.short_reason}")

    # Natural fit should be an explainer or Why format (#6 Hook, Explanation, CTA or #28 Why Framework)
    assert selection_3.selected_framework_id in [6, 28, 7]
    assert selection_3.total_evaluated == 30
    print(">>> Test 3 PASSED!\n")

    print("==================================================")
    print("TEST 4: MYTH / DEBUNKING TOPIC")
    print("==================================================")
    topic_4 = "Curd at night causes cold: Myth or Truth?"
    analysis_4 = analyzer.analyze(topic_4)
    selection_4 = selector.select(analysis_4)

    print(f"Topic: '{topic_4}'")
    print(f"Selected: #{selection_4.selected_framework_id} {selection_4.selected_framework_name}")
    print(f"Score: {selection_4.compatibility_score}/100")
    print(f"Stages: {selection_4.stages}")
    print("Top Candidates:")
    for c in selection_4.ranked_candidates[:3]:
        print(f"  #{c.framework_id:02d} {c.framework_name:<30} Score: {c.score:.1f} | {c.short_reason}")

    # Myth framework (#8) should be top candidate
    assert selection_4.selected_framework_id in [8, 20, 24]
    assert selection_4.total_evaluated == 30
    print(">>> Test 4 PASSED!\n")

    print("==================================================")
    print("TEST 5: '3 SIGNS' OR SYMPTOM TOPIC")
    print("==================================================")
    topic_5 = "3 signs your liver is under stress"
    analysis_5 = analyzer.analyze(topic_5)
    selection_5 = selector.select(analysis_5)

    print(f"Topic: '{topic_5}'")
    print(f"Selected: #{selection_5.selected_framework_id} {selection_5.selected_framework_name}")
    print(f"Score: {selection_5.compatibility_score}/100")
    print(f"Stages: {selection_5.stages}")
    print("Top Candidates:")
    for c in selection_5.ranked_candidates[:3]:
        print(f"  #{c.framework_id:02d} {c.framework_name:<30} Score: {c.score:.1f} | {c.short_reason}")

    # 3 signs framework (#16) or symptom triage should be selected
    assert selection_5.selected_framework_id in [16, 10, 25]
    assert selection_5.total_evaluated == 30
    print(">>> Test 5 PASSED!\n")

    print("==================================================")
    print("TEST 6: VERIFY ALL 30 FRAMEWORKS ARE CONSIDERED")
    print("==================================================")
    # Check that selector internal evaluated pool is exactly 30 frameworks
    assert len(selector.frameworks) == 30
    framework_ids_seen = set(fw["framework_id"] for fw in selector.frameworks)
    assert framework_ids_seen == set(range(1, 31)), "Not all 30 framework IDs are present in selector pool"
    
    # Run an arbitrary general topic and check that all 30 are evaluated
    test_analysis = analyzer.analyze("General cardiovascular fitness and walking")
    res = selector.select(test_analysis, top_n=30)
    assert res.total_evaluated == 30
    assert len(res.ranked_candidates) == 30
    candidate_ids = set(c.framework_id for c in res.ranked_candidates)
    assert candidate_ids == set(range(1, 31)), "All 30 frameworks must be evaluated in ranked candidates"
    print(f"Verified: All {res.total_evaluated} frameworks (IDs 1 through 30) were evaluated and ranked.")
    print(">>> Test 6 PASSED!\n")

    print("==================================================")
    print("ALL FRAMEWORK SELECTOR TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_framework_selector()
