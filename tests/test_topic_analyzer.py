import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.topic_analyzer import TopicAnalyzer, TopicAnalysis


def test_topic_analyzer():
    analyzer = TopicAnalyzer()
    
    print("==================================================")
    print("TEST 1: DEMO TOPIC — 'Junk Food'")
    print("==================================================")
    result_junk = analyzer.analyze("Junk Food")
    print(result_junk.model_dump_json(indent=2))
    
    # Assertions for Junk Food
    assert result_junk.original_topic == "Junk Food"
    assert result_junk.domain == "Nutrition & Metabolism"
    assert "junk food" in [k.lower() for k in result_junk.keywords]
    assert len(result_junk.suggested_strategic_goals) > 0
    assert result_junk.audience != ""
    assert result_junk.intent != ""
    print(">>> Test 1 (Junk Food) PASSED!\n")

    print("==================================================")
    print("TEST 2: DETAILED TOPIC — 'What happens if you eat junk food every day?'")
    print("==================================================")
    result_what_happens = analyzer.analyze("What happens if you eat junk food every day?")
    print(result_what_happens.model_dump_json(indent=2))
    assert result_what_happens.topic_type == "habit_consequence"
    assert "Watch Time" in result_what_happens.suggested_strategic_goals
    print(">>> Test 2 (What happens if...) PASSED!\n")

    print("==================================================")
    print("TEST 3: ARBITRARY HEALTH TOPICS")
    print("==================================================")
    arbitrary_topics = [
        "Why do legs cramp at night?",
        "Curd at night causes cold: Myth or Truth?",
        "Normal fever vs Dengue red flags",
        "3 signs your liver is under stress"
    ]
    for topic in arbitrary_topics:
        res = analyzer.analyze(topic)
        print(f"Topic: '{topic}'")
        print(f"  Domain: {res.domain}")
        print(f"  Type:   {res.topic_type}")
        print(f"  Goals:  {res.suggested_strategic_goals}")
        print(f"  Intent: {res.intent[:60]}...")
        assert res.domain != ""
        assert res.topic_type != ""
        assert len(res.keywords) > 0
    print(">>> Test 3 (Arbitrary Topics) PASSED!\n")

    print("==================================================")
    print("TEST 4: INPUT VALIDATION & ERROR HANDLING")
    print("==================================================")
    # Empty string
    try:
        analyzer.analyze("")
        assert False, "Should have raised ValueError for empty string"
    except ValueError as e:
        print(f"  [OK] Empty string rejected: {e}")

    # Whitespace only
    try:
        analyzer.analyze("   ")
        assert False, "Should have raised ValueError for whitespace"
    except ValueError as e:
        print(f"  [OK] Whitespace rejected: {e}")

    # Non-string input
    try:
        analyzer.analyze(12345)  # type: ignore
        assert False, "Should have raised TypeError for non-string"
    except TypeError as e:
        print(f"  [OK] Non-string rejected: {e}")

    # Too short
    try:
        analyzer.analyze("x")
        assert False, "Should have raised ValueError for single character"
    except ValueError as e:
        print(f"  [OK] Single character rejected: {e}")

    print(">>> Test 4 (Input Validation) PASSED!\n")
    print("==================================================")
    print("ALL TOPIC ANALYZER TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_topic_analyzer()
