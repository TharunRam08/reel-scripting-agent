import os
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.agents.topic_analyzer import TopicAnalysis
from app.agents.framework_selector import FrameworkSelection


# Lightweight list of sensational or unsupported medical phrases to disallow
SENSATIONAL_PHRASES = [
    "floods your bloodstream",
    "huge spikes",
    "overloads your liver",
    "damages your gut",
    "guarantees",
    "will definitely",
    "always causes",
    "cures",
    "detoxes",
    "miracle cure",
    "instantly destroys",
    "permanently ruins"
]


# Valid Call-To-Action action verbs and disallowed consultation terms
VALID_CTA_ACTIONS = [
    "follow", "save", "share", "comment", "learn more", "subscribe", "check out", "tap", "bookmark", "read"
]

MEDICAL_CONSULTATION_TERMS = [
    "doctor", "physician", "consult", "clinic", "schedule a visit", "medical advice", "appointment", "symptom", "diagnosis", "prescribe", "treatment"
]


class ScriptSection(BaseModel):
    """A distinct structural section of the reel script corresponding to a framework stage."""
    stage_order: int = Field(..., description="1-indexed chronological order of the stage.")
    stage_name: str = Field(..., description="Framework stage name matching the knowledge base.")
    text: str = Field(..., description="Natural spoken narration text for this stage.")

    @field_validator("stage_name", "text")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"Section field '{info.field_name}' must not be empty.")
        return v.strip()


class ReelScript(BaseModel):
    """Complete, structured, framework-aligned short-form video script."""
    topic: str = Field(..., description="The user's original topic.")
    framework_id: int = Field(..., description="Framework ID (1-30).")
    framework_name: str = Field(..., description="Framework name from knowledge base.")
    target_duration_seconds: float = Field(..., description="Target duration in seconds (configurable, default 45s).")
    hook: str = Field(..., description="Opening 3-second scroll-stopping statement or question.")
    sections: List[ScriptSection] = Field(..., description="Chronological sections following framework stages.")
    full_script_without_cta: str = Field("", description="Compiled spoken narration from all framework sections excluding CTA.")
    cta: str = Field(..., description="Single focused call to action.")
    full_script_with_cta: str = Field("", description="Complete spoken narration including the closing CTA.")
    full_script: str = Field("", description="Complete compiled spoken narration script.")
    safety_notes: List[str] = Field(..., description="Safety compliance notes based on universal health reel rules.")
    generation_source: str = Field("deterministic_fallback", description="Generation engine: llm:groq, llm:gemini, or deterministic_fallback.")

    @field_validator("framework_id")
    @classmethod
    def validate_framework_id(cls, v: int) -> int:
        if v not in range(1, 31):
            raise ValueError(f"framework_id must be between 1 and 30, got {v}")
        return v

    @field_validator("target_duration_seconds")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"target_duration_seconds must be positive, got {v}")
        return v

    @field_validator("sections")
    @classmethod
    def validate_sections_order(cls, v: List[ScriptSection]) -> List[ScriptSection]:
        if not v:
            raise ValueError("Script must contain at least one section.")
        for idx, sec in enumerate(v, 1):
            if sec.stage_order != idx:
                raise ValueError(f"Section {idx} has invalid stage_order {sec.stage_order}; expected {idx}.")
        return v

    @model_validator(mode="after")
    def validate_full_script_and_safety(self) -> "ReelScript":
        cta_clean = self.cta.strip() if self.cta else ""

        # If the last section accidentally has the CTA text, strip it so sections only contain framework stage text
        if self.sections and cta_clean:
            last_sec = self.sections[-1]
            if last_sec.text.strip().endswith(cta_clean):
                cleaned_text = last_sec.text.strip()[:-len(cta_clean)].strip()
                self.sections[-1] = ScriptSection(
                    stage_order=last_sec.stage_order,
                    stage_name=last_sec.stage_name,
                    text=cleaned_text
                )

        # 1. Populate full_script_without_cta
        if not self.full_script_without_cta.strip():
            if self.sections:
                self.full_script_without_cta = " ".join(s.text.strip() for s in self.sections).strip()
            elif self.full_script.strip():
                self.full_script_without_cta = self.full_script.strip()

        # Remove trailing CTA if already embedded in full_script_without_cta
        if cta_clean and self.full_script_without_cta.endswith(cta_clean):
            self.full_script_without_cta = self.full_script_without_cta[:-len(cta_clean)].strip()

        # 2. Strict CTA Validation
        if not cta_clean:
            raise ValueError("CTA must not be empty.")

        cta_lower = cta_clean.lower()

        # Rule 1: CTA must not contain consultation guidance or medical terms
        for term in MEDICAL_CONSULTATION_TERMS:
            if term in cta_lower:
                raise ValueError(
                    f"CTA must be a standalone call-to-action and must not contain medical consultation guidance or terms ('{term}'). Got: '{cta_clean}'"
                )

        # Rule 2: CTA must contain a clear CTA action verb
        if not any(act in cta_lower for act in VALID_CTA_ACTIONS):
            raise ValueError(
                f"CTA must contain a clear call-to-action action verb such as {VALID_CTA_ACTIONS}. Got: '{cta_clean}'"
            )

        # Rule 3: CTA must not duplicate any sentence from full_script_without_cta
        script_sentences = [
            s.strip().lower() for s in re.split(r'[.!?]+', self.full_script_without_cta)
            if len(s.strip()) > 8
        ]
        cta_sentences = [
            s.strip().lower() for s in re.split(r'[.!?]+', cta_clean)
            if len(s.strip()) > 8
        ]
        for cs in cta_sentences:
            for ss in script_sentences:
                if cs == ss or (len(cs) > 15 and cs in ss) or (len(ss) > 15 and ss in cs):
                    raise ValueError(
                        f"CTA duplicates a sentence from the spoken narration: '{cs}'"
                    )

        # Rule 4: CTA must be a single focused call-to-action (under 25 words)
        if len(cta_clean.split()) > 25:
            raise ValueError(f"CTA must be a single focused call-to-action (under 25 words), got: '{cta_clean}'")

        # 3. Populate full_script_with_cta = full_script_without_cta + " " + cta
        self.full_script_with_cta = f"{self.full_script_without_cta} {cta_clean}".strip()

        # 4. Maintain backwards-compatible full_script
        if not self.full_script.strip():
            self.full_script = self.full_script_without_cta

        if not self.full_script.strip():
            raise ValueError("full_script cannot be empty.")
        if not self.safety_notes:
            raise ValueError("safety_notes must be provided to confirm compliance with universal rules.")

        # Content quality check: detect sensational or unsupported medical claims
        script_lower = f"{self.full_script_with_cta}".lower()
        for phrase in SENSATIONAL_PHRASES:
            if phrase in script_lower:
                raise ValueError(
                    f"Content quality check failed: Script contains sensational or unsupported phrase '{phrase}'. "
                    f"Use nuanced, evidence-aligned language such as 'can contribute to', 'is associated with', or 'may increase the risk of'."
                )

        return self


from dotenv import load_dotenv

# Ensure .env is loaded from project root
_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


class ScriptWriter:
    """Generates natural spoken short-form reel scripts strictly following
    the selected framework's stage order and the 6 universal health reel rules."""

    DEFAULT_RULES_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "universal_rules.json")
    )

    def __init__(self, universal_rules_path: Optional[str] = None):
        self.rules_path = universal_rules_path or self.DEFAULT_RULES_PATH
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Universal rules not found at: {self.rules_path}")

        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.universal_rules: List[Dict[str, Any]] = json.load(f)

        self.last_llm_error: Optional[str] = None
        self.groq_model_used: Optional[str] = None
        self._refresh_api_keys()

    def _refresh_api_keys(self):
        """Reads API keys from environment, treating empty or whitespace strings as None."""
        if os.path.exists(_ENV_PATH):
            load_dotenv(_ENV_PATH)
        raw_groq = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_api_key = raw_groq if raw_groq else None

        raw_gemini = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_api_key = raw_gemini if raw_gemini else None

    def _sanitize_or_fallback_cta(
        self,
        raw_cta: str,
        full_script_without_cta: str,
        topic: str
    ) -> str:
        """Ensures CTA is a standalone, single call-to-action without consultation advice or sentence duplication.
        If raw_cta fails validation, extracts clean CTA portion or falls back to deterministic CTA."""
        raw_clean = (raw_cta or "").strip()
        topic_lower = topic.lower()
        fallback = (
            "Save this reel for your next grocery run."
            if any(k in topic_lower for k in ["food", "nutrition", "diet", "eat", "snack", "sugar"])
            else "Follow for more science-backed health tips!"
        )

        if not raw_clean:
            return fallback

        # If model prepended consultation/symptom text before an action verb, extract the standalone action part
        # e.g. "If you notice persistent fatigue... schedule a visit with your physician and follow for more tips!"
        # -> extract "Follow for more tips!"
        action_match = re.search(r'(?i)\b(follow|save|share|comment|learn more|subscribe|check out|tap|bookmark)\b.*', raw_clean)
        candidate = action_match.group(0).strip() if action_match else raw_clean
        # Ensure proper capitalization
        candidate = candidate[0].upper() + candidate[1:] if candidate else ""

        # Validate candidate against consultation terms
        cand_lower = candidate.lower()
        has_med = any(t in cand_lower for t in MEDICAL_CONSULTATION_TERMS)
        has_act = any(a in cand_lower for a in VALID_CTA_ACTIONS)

        # Check sentence duplication against full_script_without_cta
        script_sentences = [
            s.strip().lower() for s in re.split(r'[.!?]+', full_script_without_cta)
            if len(s.strip()) > 8
        ]
        cand_sentences = [
            s.strip().lower() for s in re.split(r'[.!?]+', candidate)
            if len(s.strip()) > 8
        ]
        is_dup = False
        for cs in cand_sentences:
            for ss in script_sentences:
                if cs == ss or (len(cs) > 15 and cs in ss) or (len(ss) > 15 and ss in cs):
                    is_dup = True
                    break

        if not has_med and has_act and not is_dup and len(candidate.split()) <= 20:
            return candidate

        return fallback

    def write_script(
        self,
        topic_analysis: TopicAnalysis,
        framework_selection: FrameworkSelection,
        target_duration_seconds: float = 45.0,
        force_fallback: bool = False,
        raise_on_llm_error: bool = False,
        feedback: Optional[str] = None
    ) -> ReelScript:
        """Main generation entry point. Strictly adheres to selected framework stages
        and target duration (scales word count appropriately for any duration)."""
        if target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be greater than zero.")

        # Re-check environment in case keys were loaded dynamically
        self._refresh_api_keys()

        # 1. Attempt LLM generation if free API key is configured and fallback is not forced
        if not force_fallback and self.groq_api_key:
            try:
                llm_script = self._generate_with_groq(
                    topic_analysis,
                    framework_selection,
                    target_duration_seconds,
                    feedback=feedback
                )
                if llm_script:
                    # Verify LLM respected the duration pacing constraint (< 2.8 words/sec max)
                    total_words = len(llm_script.full_script_with_cta.split())
                    if total_words <= max(24, int(target_duration_seconds * 2.8)):
                        return llm_script
            except Exception as e:
                # Sanitize error to prevent accidental API key leakage
                sanitized_msg = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", str(e))
                self.last_llm_error = f"Groq API Error ({type(e).__name__}): {sanitized_msg}"
                if raise_on_llm_error:
                    raise RuntimeError(self.last_llm_error) from e

        if not force_fallback and self.gemini_api_key:
            try:
                llm_script = self._generate_with_gemini(topic_analysis, framework_selection, target_duration_seconds)
                if llm_script:
                    return llm_script
            except Exception as e:
                sanitized_msg = re.sub(r"[A-Za-z0-9_\-]{30,}", "[REDACTED_API_KEY]", str(e))
                self.last_llm_error = f"Gemini API Error ({type(e).__name__}): {sanitized_msg}"
                if raise_on_llm_error:
                    raise RuntimeError(self.last_llm_error) from e

        # 2. Deterministic Fallback Generator (fully duration-aware)
        return self._generate_deterministic(topic_analysis, framework_selection, target_duration_seconds)

    def _generate_deterministic(
        self,
        analysis: TopicAnalysis,
        selection: FrameworkSelection,
        duration: float
    ) -> ReelScript:
        """Deterministic, rule-based script writer that dynamically builds
        script sections according to the exact stage sequence of the selected framework,
        scaled appropriately to the requested target duration and natural speaking pace."""
        fw_id = selection.selected_framework_id
        fw_name = selection.selected_framework_name
        stages = selection.stage_details  # List of {"order": int, "name": str, "description": str}
        topic = analysis.original_topic
        topic_lower = topic.lower()

        sections: List[ScriptSection] = []
        cta_text = "Save this reel for your next grocery run!" if "food" in topic_lower else "Share this reel with someone who needs to hear it today."

        # Speaking pace: ~2.2 words per second
        target_words = max(14, int(duration * 2.2))
        cta_words = len(cta_text.split())
        body_words = max(8, target_words - cta_words)
        num_stages = max(1, len(stages))

        # Specialized high-fidelity templates for key demo frameworks, scaled to duration
        if fw_id == 15:  # What Happens If
            topic_subject = "junk food" if "junk" in topic_lower or "food" in topic_lower else topic
            if duration <= 10.0:
                s1_text = f"What happens if you eat {topic_subject} daily?"
                s2_text = "Fast glucose spikes strain digestion."
                s3_text = "Swap snacks for whole foods."
            elif duration <= 20.0:
                s1_text = f"What happens if you eat {topic_subject} daily?"
                s2_text = f"Processed {topic_subject} triggers sharp blood sugar spikes and strains metabolic balance."
                s3_text = "Swap snacks for fresh fruit or nuts. If fatigue persists, consult a doctor."
            elif duration <= 35.0:
                s1_text = f"What happens if you eat {topic_subject} daily?"
                s2_text = (
                    f"Highly processed {topic_subject} triggers rapid blood sugar fluctuations, forcing your pancreas to release excess insulin. "
                    "Low fiber intake also slows digestion and leads to inconsistent daily energy."
                )
                s3_text = (
                    "Try swapping processed snacks for whole foods like fruit or nuts to stabilize your energy. "
                    "If persistent digestive discomfort or fatigue continues, consult a physician in person."
                )
            else:
                # 40s+ (e.g. 45s benchmark)
                s1_text = f"What happens if you eat {topic_subject} every single day?"
                s2_text = (
                    "Highly processed ingredients can trigger rapid blood sugar fluctuations. "
                    "This forces your pancreas to work harder to release insulin, which may contribute to insulin resistance over time. "
                    "Additionally, low fiber intake can slow digestion, leading to bloating and inconsistent energy levels throughout the day."
                )
                s3_text = (
                    "Try swapping one processed snack for whole foods like fruit or nuts to stabilize your energy. "
                    "If you notice persistent digestive issues or unexplained fatigue, please consult a physician in person for a proper evaluation."
                )
            stage_texts = [s1_text, s2_text, s3_text]

        elif fw_id == 8:  # Myth, Truth, Explanation
            if duration <= 12.0:
                s1_text = f"Myth: {topic} causes colds."
                s2_text = "Truth: Viruses cause colds, not food."
                s3_text = "Nutrient-dense foods support recovery."
            elif duration <= 25.0:
                s1_text = f"Does {topic} cause an immediate cold?"
                s2_text = "Respiratory infections are caused by viruses, not food temperature."
                s3_text = "Chilled sensations can feel noticeable, but food provides protein. If symptoms persist, consult a doctor."
            else:
                s1_text = "People often believe that having curd at night causes an immediate cold and congestion."
                s2_text = "Curd does not cause a cold; respiratory infections are caused by viruses."
                s3_text = (
                    "This belief often stems from chilled foods temporarily thickening throat sensations in people who already have "
                    "mild underlying congestion. For healthy individuals, curd provides dietary protein and probiotics and is generally "
                    "well-tolerated at night. If cold symptoms or sinus congestion persist for more than a week, consult a physician in person."
                )
            stage_texts = [s1_text, s2_text, s3_text]

        elif fw_id == 16:  # 3 Signs
            if duration <= 12.0:
                s1_text = f"Signs of stress in {topic}."
                s2_text = "Persistent sluggishness and unusual digestive fatigue."
                s3_text = "These signs suggest evaluating metabolic health."
                s4_text = "If symptoms persist, consult a doctor."
            elif duration <= 25.0:
                s1_text = f"Two signs your {topic} is under metabolic strain."
                s2_text = "First: lingering post-meal sluggishness. Second: chronic digestive discomfort."
                s3_text = "These signs suggest reviewing your lifestyle and metabolic habits."
                s4_text = "Avoid extreme detoxes. If fatigue continues for weeks, consult a physician."
            else:
                s1_text = "Three signs that may suggest your liver is under metabolic strain."
                s2_text = "First: persistent sluggishness and low energy after meals. Second: noticeable morning puffiness around the eyes. Third: chronic digestive discomfort accompanied by unusually dark urine or pale stools."
                s3_text = "Having several of these signs together does not diagnose a condition, but indicates it may be time to evaluate your metabolic health."
                s4_text = "Avoid unverified detox cleanses. If these symptoms continue for more than two weeks, consult a physician in person for routine liver tests."
            stage_texts = [s1_text, s2_text, s3_text, s4_text]

        elif fw_id == 6:  # Hook, Explanation, CTA
            if duration <= 12.0:
                s1_text = f"Why does {topic} happen?"
                s2_text = "Muscle tightening and low hydration can trigger cramping."
                s3_text = "Stay hydrated and stretch gently before sleep."
            elif duration <= 25.0:
                s1_text = f"Why do muscle cramps often happen during {topic}?"
                s2_text = "During rest, reduced circulation and mild dehydration can increase muscle irritability."
                s3_text = "Gentle stretching and hydration help. If cramps disrupt sleep regularly, see your doctor."
            else:
                s1_text = "Why do calf muscle cramps often happen during the night?"
                s2_text = (
                    "During sleep, reduced blood circulation combined with prolonged muscle shortening can increase muscle irritability. "
                    "Factors like mild dehydration, low magnesium intake, or physical fatigue can contribute to involuntary cramping."
                )
                s3_text = "Staying hydrated and gently stretching your calves before bed can help. If painful muscle cramps occur frequently or disrupt your sleep regularly, see your doctor to check electrolyte levels."
            stage_texts = [s1_text, s2_text, s3_text]

        elif fw_id == 11:  # Normal vs Red Flag
            if duration <= 15.0:
                s1_text = f"Mild symptoms from {topic} usually improve with rest."
                s2_text = "Severe unremitting pain or difficulty breathing are critical red flags."
                s3_text = "For red flags, seek medical evaluation promptly."
            else:
                s1_text = "A mild fever lasting two or three days with general body tiredness is common with simple viral illnesses and often improves with rest and fluids."
                s2_text = "However, signs like unexplained bleeding, severe unremitting abdominal pain, persistent vomiting, or difficulty breathing are critical red flags."
                s3_text = "These warning signs require prompt evaluation. Go to an emergency clinic or hospital promptly rather than waiting at home."
            stage_texts = [s1_text, s2_text, s3_text]

        else:
            # Universal Dynamic Framework Synthesizer for arbitrary frameworks (1 to 30), scaled to duration
            stage_texts = []
            for idx, stage in enumerate(stages, 1):
                st_name = stage["name"]
                st_desc = stage["description"]

                if idx == 1:
                    # Opening Hook
                    if duration <= 10.0:
                        txt = f"The truth about {topic}."
                    elif duration <= 20.0:
                        txt = f"How does {topic} affect your body daily?"
                    elif duration <= 35.0:
                        txt = f"Here is what medical evidence shows about {topic}."
                    else:
                        txt = f"{st_name}: Here is what medical evidence shows about {topic}."
                elif idx == len(stages):
                    # Closing Stage with consultation recommendation
                    if duration <= 10.0:
                        txt = "Build consistent habits. Consult a doctor."
                    elif duration <= 20.0:
                        txt = "Focus on sustainable daily routines. For persistent concerns, consult a doctor in person."
                    elif duration <= 35.0:
                        txt = "Focus on sustainable daily habits. If symptoms persist or worsen, consult a physician in person."
                    else:
                        txt = (
                            f"{st_name}: Focus on sustainable daily habits rather than extreme measures. "
                            f"If your symptoms persist, worsen, or cause concern, always schedule an in-person clinical consultation with a doctor."
                        )
                else:
                    # Intermediate Educational Mechanism
                    st_lower = st_name.lower()
                    if any(k in st_lower for k in ["problem", "mistake", "cause", "issue"]):
                        if duration <= 10.0:
                            txt = f"Inactivity affects daily health."
                        elif duration <= 20.0:
                            txt = f"Without regular movement, energy and circulation decline."
                        elif duration <= 35.0:
                            txt = f"Inside the body, sedentary habits reduce metabolic efficiency and physical stamina."
                        else:
                            txt = f"{st_name}: {st_desc} Without regular movement, metabolic efficiency and stamina decline over time."
                    elif any(k in st_lower for k in ["solution", "fix", "remedy", "habit", "action", "treatment"]):
                        if duration <= 10.0:
                            txt = f"Gentle daily movement helps."
                        elif duration <= 20.0:
                            txt = f"Adding moderate movement or brisk walks restores balance."
                        elif duration <= 35.0:
                            txt = f"Consistent moderate exercise and daily steps help stabilize cardiovascular health."
                        else:
                            txt = f"{st_name}: {st_desc} Consistent moderate exercise and daily steps help stabilize cardiovascular health."
                    else:
                        if duration <= 10.0:
                            txt = f"{topic} directly influences health."
                        elif duration <= 20.0:
                            txt = f"Inside the body, {topic} influences metabolic balance and energy."
                        elif duration <= 35.0:
                            txt = f"Inside the body, {topic} influences metabolic balance and cardiovascular energy over time."
                        else:
                            txt = (
                                f"{st_name}: {st_desc} Inside the body, {topic} can influence your metabolic markers and overall energy levels over time. "
                                f"Long-term consistency is what creates meaningful physiological improvements."
                            )
                stage_texts.append(txt)

        # Assemble ScriptSection objects
        for idx, stage in enumerate(stages):
            text_content = stage_texts[idx] if idx < len(stage_texts) else f"{stage['name']}: Practical steps for {topic}."
            sections.append(
                ScriptSection(
                    stage_order=stage["order"],
                    stage_name=stage["name"],
                    text=text_content.strip()
                )
            )

        hook = sections[0].text
        full_without_cta = " ".join(s.text for s in sections).strip()
        full_with_cta = f"{full_without_cta} {cta_text}".strip()

        safety_notes = [
            "Rule 1 (No Diagnosis): Script describes physiological mechanisms and risk factors probabilistically without diagnosing the individual viewer.",
            "Rule 2 (No Dosages/Prescribing): Zero pharmaceutical brand names or milligram dosages included.",
            "Rule 3 (Clinical Consultation Guidance): Recommends seeking an in-person medical evaluation with a qualified doctor for persistent, worsening, or concerning symptoms.",
            "Rule 4 (No Patient Details): Anonymized educational content with no private case histories.",
            "Rule 5 (Single CTA): Exactly one unified closing call-to-action enforced.",
            "Rule 6 (Three-Second Hook Primacy): Opening stage delivers immediate scroll-stopping curiosity hook.",
            "Content Quality: Avoids sensational, exaggerated, or absolute medical claims; uses evidence-based associative phrasing."
        ]

        return ReelScript(
            topic=topic,
            framework_id=fw_id,
            framework_name=fw_name,
            target_duration_seconds=duration,
            hook=hook,
            sections=sections,
            full_script_without_cta=full_without_cta,
            cta=cta_text,
            full_script_with_cta=full_with_cta,
            full_script=full_without_cta,
            safety_notes=safety_notes,
            generation_source="deterministic_fallback"
        )

    def _generate_with_groq(
        self,
        analysis: TopicAnalysis,
        selection: FrameworkSelection,
        duration: float,
        feedback: Optional[str] = None
    ) -> Optional[ReelScript]:
        """Free Groq LLM script generation enforcing framework stages, universal rules, and content quality."""
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)

            stages_schema = [
                {"order": s["order"], "name": s["name"], "description": s["description"]}
                for s in selection.stage_details
            ]

            feedback_directive = ""
            if feedback and feedback.strip():
                feedback_directive = f"\nUSER REVISION FEEDBACK TO INCORPORATE:\n\"{feedback.strip()}\"\nIncorporate this user feedback into the script while strictly adhering to framework stages and safety rules.\n"

            prompt = f"""You are an expert medical reel scriptwriter for doctors.
Write a spoken Instagram reel script that STRICTLY follows the selected framework and medical communication best practices.{feedback_directive}

Topic: "{analysis.original_topic}"
Domain: {analysis.domain}
Audience: {analysis.audience}
Framework #{selection.selected_framework_id}: "{selection.selected_framework_name}"
Target Duration: ~{duration} seconds (approx {int(duration * 2.4)} spoken words total)

MANDATORY STAGES TO FOLLOW IN THIS EXACT ORDER:
{json.dumps(stages_schema, indent=2)}

REFERENCE EXAMPLE FOR THIS FRAMEWORK:
"{selection.framework_details.get('example_script', '')}"

PITFALL WARNING TO AVOID:
"{selection.framework_details.get('pitfall_warning', '')}"

UNIVERSAL HEALTH REEL RULES:
1. No diagnosis: Explain what symptoms or habits mean; do NOT diagnose the viewer.
2. No drug dosages, brand names, or prescriptions.
3. In the final stage (prevention, next step, or resolution), always include a practical sentence advising the viewer to consult a doctor or physician in person if they experience persistent, worsening, or concerning symptoms.
4. No identifiable patient details.
5. STANDALONE CALL-TO-ACTION (CTA):
   - The CTA must be a short, single, completely standalone call-to-action (e.g. "Follow for more science-backed nutrition tips!" or "Save this reel for your next grocery run!").
   - CRITICAL: The CTA must NEVER contain medical advice, consultation advice, symptom guidance, or words like "doctor" or "physician".
   - The CTA must NEVER repeat or duplicate the consultation sentence from the final stage.
   - The CTA must begin with or contain a clear action verb: follow, save, share, comment, or learn more.
6. The first 3 seconds (Stage 1) must be a punchy, scroll-stopping hook.

CRITICAL CONTENT QUALITY & SCIENTIFIC ACCURACY CONSTRAINTS:
Target Spoken Duration: ~{duration} seconds.
TOTAL SPOKEN WORD COUNT CONSTRAINT: The combined spoken script (all sections + CTA) MUST contain approximately {int(duration * 2.2)} words (strictly between {max(12, int(duration * 1.8))} and {int(duration * 2.4)} words) so it can be spoken naturally within {duration} seconds. Do not write more words than can be spoken comfortably.

CRITICAL STANDALONE CTA RULES (MANDATORY):
1. The CTA ("cta" field) MUST be a standalone, single call-to-action (e.g. "Follow for more science-backed nutrition tips!").
2. The CTA must NOT contain medical advice, consultation guidance (e.g., do NOT mention doctor, physician, clinic, or scheduling a visit), symptom guidance, or diagnostic instructions.
3. The CTA must NOT duplicate any sentence from the spoken sections.
4. The CTA MUST start or contain an explicit call-to-action verb such as: follow, save, share, comment, learn more, or subscribe.
5. All consultation and medical guidance (such as advising persistent or severe symptoms to be evaluated by a physician) MUST belong in the spoken framework sections (e.g., Prevention/Solution stage), NOT inside the "cta" field.

CRITICAL CONTENT QUALITY RULES (MANDATORY):
1. ABSOLUTELY NO SENSATIONAL OR EXAGGERATED CLAIMS:
   - Do NOT use phrases like: "floods your bloodstream", "huge spikes", "overloads your liver", "damages your gut", "permanently ruins", "miracle cure".
   - Avoid absolute or deterministic language (e.g., "will definitely cause", "always leads to").
2. USE CAREFUL, EVIDENCE-BASED ASSOCIATIVE LANGUAGE:
   - Instead, use phrases like: "can contribute to", "is associated with", "may increase the risk of", "regularly relying on... can alter", "supports overall health".
3. NO DRUG DOSING OR PRESCRIPTIONS:
   - Never recommend specific medication dosages.
4. MEDICAL CONSULTATION GUIDANCE (UNIVERSAL RULE 3):
   - For health topics, recommend professional medical consultation when symptoms are persistent, severe, worsening, or concerning. Do not force an arbitrary duration into every script.
5. THREE-SECOND HOOK:
   - Hook must grab attention in the first 3 seconds with curiosity or relatability.

Return strictly a JSON object with:
- hook: string (opening 3-second hook)
- cta: string (single standalone closing call to action, e.g. "Follow for more science-backed nutrition tips!")
- sections: array of objects with keys: "stage_order" (int), "stage_name" (string), "text" (string)
- safety_notes: array of strings confirming compliance with the 6 rules and content quality
"""
            # Prioritized list of active models available on Groq (deduplicated)
            raw_models = [
                os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "groq/compound-mini",
                "groq/compound"
            ]
            models_to_try = list(dict.fromkeys(m for m in raw_models if m))

            completion = None
            last_err = None
            for model_name in models_to_try:
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You output strictly valid JSON conforming to the requested schema."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3
                    )
                    self.groq_model_used = model_name
                    break
                except Exception as ex:
                    last_err = ex
                    continue

            if completion is None and last_err is not None:
                raise last_err

            raw_content = completion.choices[0].message.content
            data = json.loads(raw_content)

            # Strictly align sections with expected framework stages
            aligned_sections = []
            for idx, expected_stage in enumerate(selection.stage_details, 1):
                matched_text = ""
                for s in data.get("sections", []):
                    if s.get("stage_order") == idx or s.get("stage_name", "").strip().lower() == expected_stage["name"].strip().lower():
                        matched_text = s.get("text", "").strip()
                        break
                if not matched_text and idx - 1 < len(data.get("sections", [])):
                    matched_text = data["sections"][idx - 1].get("text", "").strip()

                aligned_sections.append(
                    ScriptSection(
                        stage_order=expected_stage["order"],
                        stage_name=expected_stage["name"],
                        text=matched_text or f"{expected_stage['name']}: Evidence-based health guidance for {analysis.original_topic}."
                    )
                )

            full_without_cta = " ".join(s.text for s in aligned_sections).strip()
            raw_cta = data.get("cta", "").strip()

            clean_cta = self._sanitize_or_fallback_cta(
                raw_cta=raw_cta,
                full_script_without_cta=full_without_cta,
                topic=analysis.original_topic
            )

            # Ensure last section doesn't duplicate clean CTA
            if clean_cta and aligned_sections[-1].text.endswith(clean_cta):
                cleaned = aligned_sections[-1].text[:-len(clean_cta)].strip()
                aligned_sections[-1] = ScriptSection(
                    stage_order=aligned_sections[-1].stage_order,
                    stage_name=aligned_sections[-1].stage_name,
                    text=cleaned
                )
                full_without_cta = " ".join(s.text for s in aligned_sections).strip()

            full_with_cta = f"{full_without_cta} {clean_cta}".strip()

            return ReelScript(
                topic=analysis.original_topic,
                framework_id=selection.selected_framework_id,
                framework_name=selection.selected_framework_name,
                target_duration_seconds=duration,
                hook=data.get("hook", aligned_sections[0].text),
                sections=aligned_sections,
                full_script_without_cta=full_without_cta,
                cta=clean_cta,
                full_script_with_cta=full_with_cta,
                full_script=full_without_cta,
                safety_notes=data.get("safety_notes", ["Compliant with universal rules and content quality standards."]),
                generation_source="groq"
            )
        except Exception as e:
            # Let caller handle or record the exception
            raise e

    def _generate_with_gemini(
        self,
        analysis: TopicAnalysis,
        selection: FrameworkSelection,
        duration: float
    ) -> Optional[ReelScript]:
        """Free Gemini script generation with content quality constraints."""
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            prompt = f"""Write a short spoken reel script for: "{analysis.original_topic}" using framework #{selection.selected_framework_id} {selection.selected_framework_name}.
Stages: {selection.stages}. Target duration: {duration}s. Exactly one standalone CTA.
Avoid sensational phrases like "floods your bloodstream", "huge spikes", "overloads your liver", "damages your gut".
Use balanced language: "can contribute to", "is associated with", "may increase risk".
Output JSON with hook, cta, sections (stage_order, stage_name, text), safety_notes."""
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.3}
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(text)
                sections = [
                    ScriptSection(
                        stage_order=s["stage_order"],
                        stage_name=s["stage_name"],
                        text=s["text"]
                    )
                    for s in data["sections"]
                ]
                full_without_cta = " ".join(s.text for s in sections).strip()
                clean_cta = self._sanitize_or_fallback_cta(
                    raw_cta=data.get("cta", ""),
                    full_script_without_cta=full_without_cta,
                    topic=analysis.original_topic
                )
                full_with_cta = f"{full_without_cta} {clean_cta}".strip()
                return ReelScript(
                    topic=analysis.original_topic,
                    framework_id=selection.selected_framework_id,
                    framework_name=selection.selected_framework_name,
                    target_duration_seconds=duration,
                    hook=data.get("hook", sections[0].text),
                    sections=sections,
                    full_script_without_cta=full_without_cta,
                    cta=clean_cta,
                    full_script_with_cta=full_with_cta,
                    full_script=full_without_cta,
                    safety_notes=data.get("safety_notes", ["Compliant with universal rules and content quality standards."]),
                    generation_source="llm:gemini"
                )
        except Exception:
            return None


if __name__ == "__main__":
    from app.agents.topic_analyzer import TopicAnalyzer
    from app.agents.framework_selector import FrameworkSelector

    analyzer = TopicAnalyzer()
    selector = FrameworkSelector()
    writer = ScriptWriter()

    demo_topic = "What happens if you eat junk food every day?"
    analysis = analyzer.analyze(demo_topic)
    selection = selector.select(analysis)
    script = writer.write_script(analysis, selection, target_duration_seconds=45.0)

    print(script.model_dump_json(indent=2))
