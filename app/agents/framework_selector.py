import os
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.agents.topic_analyzer import TopicAnalysis


class CandidateFramework(BaseModel):
    """Summarized framework candidate with score and rationale."""
    framework_id: int
    framework_name: str
    score: float
    short_reason: str
    stages: List[str]


class FrameworkSelection(BaseModel):
    """Final framework selection output ready for Script Writer ingestion."""
    selected_framework_id: int
    selected_framework_name: str
    compatibility_score: float
    reason: str
    stages: List[str] = Field(..., description="Ordered list of stage names to enforce in the script.")
    stage_details: List[Dict[str, Any]] = Field(..., description="Detailed stages with order, name, and descriptions.")
    framework_details: Dict[str, Any] = Field(..., description="Complete framework metadata from knowledge base.")
    ranked_candidates: List[CandidateFramework] = Field(..., description="Top ranked candidates evaluated.")
    total_evaluated: int = Field(30, description="Total frameworks evaluated from knowledge base.")
    selection_method: str = Field("deterministic_scoring", description="Mechanism used (deterministic_scoring or llm_tie_breaker).")


class FrameworkSelector:
    """Selects the optimal reel-writing framework from the 30-framework knowledge base
    using a principled, multi-factor deterministic scoring system."""

    DEFAULT_KB_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "frameworks.json")
    )

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path or self.DEFAULT_KB_PATH
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Framework knowledge base not found at: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.frameworks: List[Dict[str, Any]] = json.load(f)

        if len(self.frameworks) != 30:
            raise ValueError(f"Expected exactly 30 frameworks in knowledge base, found {len(self.frameworks)}")

        # Free LLM key check (for optional tie-breaking / explanation polish)
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

    def select(self, topic_analysis: TopicAnalysis, top_n: int = 5) -> FrameworkSelection:
        """Evaluates all 30 frameworks against the topic analysis and selects the best fit."""
        if not isinstance(topic_analysis, TopicAnalysis):
            raise TypeError(f"Expected TopicAnalysis instance, got {type(topic_analysis).__name__}")

        scored_candidates = []

        # Iterate through all 30 frameworks deterministically
        for fw in self.frameworks:
            score, reasons = self._score_framework(topic_analysis, fw)
            stage_names = [s["name"] for s in fw.get("stages", [])]
            candidate = CandidateFramework(
                framework_id=fw["framework_id"],
                framework_name=fw["name"],
                score=score,
                short_reason="; ".join(reasons[:2]) if reasons else "Compatible general health structure",
                stages=stage_names
            )
            scored_candidates.append((score, fw, candidate, reasons))

        # Sort descending by score; break equal ties by lower framework_id for determinism
        scored_candidates.sort(key=lambda x: (x[0], -x[1]["framework_id"]), reverse=True)

        top_candidates = [item[2] for item in scored_candidates[:top_n]]
        winner_tuple = scored_candidates[0]
        winner_score, winner_fw, winner_cand, winner_reasons = winner_tuple

        # Format human-readable reason
        detailed_reason = (
            f"Selected #{winner_fw['framework_id']} '{winner_fw['name']}' with a compatibility score of {winner_score:.1f}/100. "
            f"Key factors: {'; '.join(winner_reasons)}."
        )

        stage_names = [s["name"] for s in winner_fw.get("stages", [])]
        stage_details = winner_fw.get("stages", [])

        return FrameworkSelection(
            selected_framework_id=winner_fw["framework_id"],
            selected_framework_name=winner_fw["name"],
            compatibility_score=winner_score,
            reason=detailed_reason,
            stages=stage_names,
            stage_details=stage_details,
            framework_details=winner_fw,
            ranked_candidates=top_candidates,
            total_evaluated=len(self.frameworks),
            selection_method="deterministic_scoring"
        )

    def _score_framework(self, analysis: TopicAnalysis, fw: Dict[str, Any]) -> tuple[float, List[str]]:
        """Calculates multi-dimensional compatibility score (0 to 100 points):
        1. Topic Type & Intent Affinity (up to 35 pts)
        2. Keyword & Phrase Overlap (up to 25 pts)
        3. Strategic Goals Alignment (up to 20 pts)
        4. Use Case & Description Relevance (up to 15 pts)
        5. Structural Cues & Direct Triggers (up to 5 pts)
        """
        score = 0.0
        reasons = []

        fid = fw["framework_id"]
        tt = analysis.topic_type
        topic_lower = analysis.original_topic.lower()
        topic_keywords = [k.lower() for k in analysis.keywords]

        # -------------------------------------------------------------
        # 1. Structural & Intent Affinity (Max 35 points)
        # -------------------------------------------------------------
        intent_pts = 0.0
        if tt == "habit_consequence":
            if fid == 15:  # What Happens If
                intent_pts = 35.0
                reasons.append("Primary framework for habit consequence & internal bodily effects")
            elif fid in [27, 9]:  # Cause, Effect, Prevention / Mistake, Consequence, Fix
                intent_pts = 24.0
                reasons.append("High relevance to lifestyle habits and long-term health consequences")
            elif fid in [28, 18, 30]:
                intent_pts = 16.0
                reasons.append("Secondary relevance to habit change & physiological explanation")

        elif tt == "mechanism_inquiry":
            if fid in [28, 6]:  # Why Framework / Hook, Explanation, CTA
                intent_pts = 35.0
                reasons.append("Direct format for 'Why' and physiological mechanism explanation")
            elif fid == 7:  # Hook, Medical Explanation, Solution
                intent_pts = 28.0
                reasons.append("Clinical authority mechanism explanation")
            elif fid == 15:
                intent_pts = 18.0
                reasons.append("Explains internal body consequences")

        elif tt == "myth_debunk":
            if fid == 8:  # Myth, Truth, Explanation
                intent_pts = 35.0
                reasons.append("Dedicated framework for debunking health myths and home remedies")
            elif fid in [20, 24]:  # Doctor Reacts / Pattern Interrupt
                intent_pts = 28.0
                reasons.append("High-performing reaction or contrarian debunking format")

        elif tt == "symptom_triage":
            if fid == 16:  # 3 Signs
                intent_pts = 35.0
                reasons.append("High-save 3 Signs format for early detection")
            elif fid in [10, 11]:  # Symptom, Possible Reason, What To Do / Normal vs Red Flag
                intent_pts = 28.0
                reasons.append("Safe symptom differential or red flag triage")

        elif tt == "behavior_correction":
            if fid in [9, 17]:  # Mistake, Consequence, Fix / 3 Mistakes
                intent_pts = 35.0
                reasons.append("Direct format for identifying and correcting behavioral mistakes")
            elif fid == 18:
                intent_pts = 25.0
                reasons.append("Practical swap & habit substitution format")

        elif tt == "habit_substitution":
            if fid == 18:  # Do This, Not That
                intent_pts = 35.0
                reasons.append("Direct 'Do This, Not That' habit swap format")
            elif fid == 30:
                intent_pts = 25.0
                reasons.append("Before, After, Bridge transformation format")

        elif tt == "emergency_triage":
            if fid == 11:  # Normal vs Red Flag
                intent_pts = 35.0
                reasons.append("Urgent Normal vs Red Flag emergency triage")
            elif fid == 12:
                intent_pts = 25.0
                reasons.append("Fear to reassurance format for serious conditions")

        elif tt == "comparison":
            if fid == 26:  # Comparison
                intent_pts = 35.0
                reasons.append("Direct X vs Y comparative differentiation")

        elif tt == "practical_checklist":
            if fid == 29:  # Checklist
                intent_pts = 35.0
                reasons.append("Dedicated preparation checklist format")
            elif fid == 25:
                intent_pts = 25.0
                reasons.append("Numbered listicle format")

        elif tt == "lifestyle_health":
            # Broad lifestyle health topic
            if fid in [15, 5, 18, 27, 4]:
                intent_pts = 25.0
                reasons.append("Proven high-performing format for everyday lifestyle & dietary habits")
            elif fid in [1, 2, 6, 9]:
                intent_pts = 18.0
                reasons.append("General health awareness & behavioral correction structure")
        else:
            intent_pts = 10.0

        score += intent_pts

        # -------------------------------------------------------------
        # 2. Keyword & Phrase Overlap (Max 25 points)
        # -------------------------------------------------------------
        kw_pts = 0.0
        fw_kws = [k.lower() for k in fw.get("keywords", [])]

        for k in topic_keywords:
            if k in fw_kws:
                kw_pts += 12.0
                reasons.append(f"Direct keyword match: '{k}'")
            elif any(k in fk for fk in fw_kws) and len(k) > 3:
                kw_pts += 6.0
                reasons.append(f"Related keyword overlap: '{k}'")

        kw_pts = min(25.0, kw_pts)
        score += kw_pts

        # -------------------------------------------------------------
        # 3. Strategic Goals Alignment (Max 20 points)
        # -------------------------------------------------------------
        suggested_goals = set(analysis.suggested_strategic_goals)
        fw_goals = set(fw.get("strategic_goals", []))
        overlap = suggested_goals.intersection(fw_goals)
        if suggested_goals:
            sg_pts = (len(overlap) / len(suggested_goals)) * 20.0
            if overlap:
                reasons.append(f"Aligned strategic goals: {sorted(list(overlap))}")
            score += sg_pts

        # -------------------------------------------------------------
        # 4. Use Case & Description Relevance (Max 15 points)
        # -------------------------------------------------------------
        uc_pts = 0.0
        best_uses_text = " ".join(fw.get("best_use_cases", [])).lower()
        desc_text = fw.get("description", "").lower()

        for k in topic_keywords:
            if len(k) > 3 and (k in best_uses_text or k in desc_text):
                uc_pts += 6.0

        uc_pts = min(15.0, uc_pts)
        if uc_pts > 0:
            reasons.append("Documented use cases match topic domain")
        score += uc_pts

        # -------------------------------------------------------------
        # 5. Structural Cues & Trigger Phrases (Max 5 points)
        # -------------------------------------------------------------
        cue_pts = 0.0
        if "what happens" in topic_lower and fid == 15:
            cue_pts = 5.0
            reasons.append("Direct match for 'What happens if' phrasing")
        elif "why" in topic_lower and fid in [28, 6]:
            cue_pts = 5.0
            reasons.append("Direct match for 'Why' phrasing")
        elif ("myth" in topic_lower or "truth" in topic_lower) and fid in [8, 20]:
            cue_pts = 5.0
            reasons.append("Direct match for myth/truth phrasing")
        elif "3 signs" in topic_lower and fid == 16:
            cue_pts = 5.0
            reasons.append("Direct match for '3 signs' phrasing")
        elif ("red flag" in topic_lower or "normal vs" in topic_lower) and fid == 11:
            cue_pts = 5.0
            reasons.append("Direct match for 'normal vs red flag' phrasing")
        elif ("vs" in topic_lower or "versus" in topic_lower) and fid == 26:
            cue_pts = 5.0
            reasons.append("Direct match for comparison 'vs' phrasing")

        score += cue_pts

        # Final score bounded between 0.0 and 100.0
        final_score = round(min(100.0, score), 1)
        return final_score, reasons


if __name__ == "__main__":
    from app.agents.topic_analyzer import TopicAnalyzer
    analyzer = TopicAnalyzer()
    selector = FrameworkSelector()

    demo_topic = "What happens if you eat junk food every day?"
    print(f"Analyzing and Selecting framework for: '{demo_topic}'")
    analysis = analyzer.analyze(demo_topic)
    selection = selector.select(analysis)
    print(selection.model_dump_json(indent=2))
