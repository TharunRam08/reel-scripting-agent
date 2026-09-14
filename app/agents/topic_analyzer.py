import os
import re
import json
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TopicAnalysis(BaseModel):
    """Structured analytical representation of a user topic for reel scripting."""
    original_topic: str = Field(..., description="The exact raw input topic provided by the user.")
    domain: str = Field(..., description="Medical or health specialty area (e.g. Nutrition & Metabolism, Cardiovascular Health).")
    audience: str = Field(..., description="Primary intended viewer demographic or cohort.")
    intent: str = Field(..., description="The informational or behavioral objective of the reel.")
    keywords: List[str] = Field(..., description="Cleaned, relevant keyword tags for matching.")
    emotional_angle: str = Field(..., description="Dominant emotional tone (e.g. Curiosity, Urgency, Empowerment).")
    topic_type: str = Field(..., description="Format classification (e.g. habit_consequence, symptom_triage, myth_debunk).")
    suggested_strategic_goals: List[str] = Field(..., description="Target reel outcomes (e.g. Watch Time, Reach, Saves, Shares, Doctor Authority).")
    analysis_source: str = Field("deterministic_fallback", description="Engine used for analysis: llm:groq, llm:gemini, or deterministic_fallback.")

    @field_validator("original_topic", "domain", "audience", "intent", "emotional_angle", "topic_type")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"Field '{info.field_name}' must not be empty.")
        return v.strip()

    @field_validator("keywords", "suggested_strategic_goals")
    @classmethod
    def validate_non_empty_lists(cls, v: List[str], info) -> List[str]:
        if not v:
            raise ValueError(f"Field '{info.field_name}' must contain at least one item.")
        return v


from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()


class TopicAnalyzer:
    """Analyzes arbitrary user topics into structured health content metadata."""

    # Stopwords to filter out from general keyword extraction
    STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
        "are", "was", "were", "do", "does", "did", "can", "could", "should", "would",
        "what", "why", "how", "when", "where", "which", "who", "whom", "this", "that",
        "these", "those", "you", "your", "we", "our", "it", "its", "if", "be", "been",
        "every", "day", "days", "about", "with", "from", "by", "as"
    }

    # Domain keyword heuristics for deterministic analysis
    DOMAIN_PATTERNS = [
        (
            "Nutrition & Metabolism",
            [
                "junk food", "fast food", "processed food", "sugar", "diet", "food",
                "eating", "snack", "fat", "calories", "nutrition", "meal", "breakfast",
                "dinner", "lunch", "rice", "curd", "salt", "beverage", "soda", "drink",
                "protein", "carb", "carbohydrates", "weight", "obesity", "metabolism",
                "insulin", "glucose", "diabetes"
            ]
        ),
        (
            "Cardiovascular Health",
            [
                "heart", "cardiac", "blood pressure", "bp", "hypertension", "cholesterol",
                "artery", "stroke", "chest pain", "angina", "triglycerides", "lipid",
                "lipoprotein"
            ]
        ),
        (
            "Gastroenterology & Digestion",
            [
                "acid reflux", "gerd", "stomach", "gut", "indigestion", "constipation",
                "diarrhea", "liver", "bloating", "endoscopy", "heartburn", "gastritis",
                "antacid"
            ]
        ),
        (
            "Sleep & Circadian Health",
            [
                "sleep", "insomnia", "apnea", "snoring", "circadian", "body clock",
                "tired", "waking up", "fatigue", "nap", "melatonin", "screen time",
                "bedtime"
            ]
        ),
        (
            "Musculoskeletal & Physical Health",
            [
                "knee", "joint", "back pain", "spine", "muscle", "cramp", "sitting",
                "walking", "steps", "posture", "exercise", "leg raises", "ergonomics"
            ]
        ),
        (
            "Neurology & Mental Wellness",
            [
                "headache", "migraine", "brain", "stress", "anxiety", "cortisol",
                "memory", "concentration", "dizziness"
            ]
        ),
        (
            "Infectious Disease & Emergency Triage",
            [
                "fever", "dengue", "viral", "infection", "emergency", "red flag",
                "hospital", "cough", "cold", "flu", "bleeding"
            ]
        ),
        (
            "Endocrinology & Women's Health",
            [
                "thyroid", "tsh", "hormone", "pcos", "period", "iron", "calcium",
                "vitamin d", "anaemia"
            ]
        )
    ]

    def __init__(self):
        self._refresh_api_keys()

    def _refresh_api_keys(self):
        """Reads API keys from environment, treating empty or whitespace strings as None."""
        raw_groq = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_api_key = raw_groq if raw_groq else None

        raw_gemini = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_api_key = raw_gemini if raw_gemini else None

    def analyze(self, topic: str) -> TopicAnalysis:
        """Main entry point: validates topic, attempts LLM analysis if configured,
        or performs robust deterministic fallback analysis."""
        # 1. Input Validation
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be a string, got {type(topic).__name__}")

        topic_clean = topic.strip()
        if not topic_clean:
            raise ValueError("Topic cannot be empty or whitespace only.")

        if len(topic_clean) < 2:
            raise ValueError("Topic is too short to analyze meaningfully.")

        # Re-check environment in case keys were updated
        self._refresh_api_keys()

        # 2. Try LLM if API key exists
        if self.groq_api_key:
            try:
                llm_result = self._analyze_with_groq(topic_clean)
                if llm_result:
                    return llm_result
            except Exception:
                # Silently fall back to deterministic analyzer on any API failure
                pass

        if self.gemini_api_key:
            try:
                llm_result = self._analyze_with_gemini(topic_clean)
                if llm_result:
                    return llm_result
            except Exception:
                pass

        # 3. Deterministic Fallback Analysis
        return self._analyze_deterministic(topic_clean)

    def _analyze_deterministic(self, topic: str) -> TopicAnalysis:
        """Deterministic topic classification using domain keyword scoring,
        linguistic pattern recognition, and strategic objective mapping."""
        topic_lower = topic.lower()

        # 1. Keywords extraction
        raw_words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", topic_lower)
        keywords = [w for w in raw_words if w not in self.STOPWORDS and len(w) > 2]
        
        # Check for compound key phrases
        for phrase in ["junk food", "fast food", "blood pressure", "acid reflux", "sleep apnea",
                       "screen time", "heart attack", "red flag", "morning headache"]:
            if phrase in topic_lower and phrase not in keywords:
                keywords.insert(0, phrase)

        if not keywords:
            keywords = [topic_lower]

        # 2. Domain detection by keyword score
        domain = "General Preventive Health"
        max_matches = 0
        for dom_name, patterns in self.DOMAIN_PATTERNS:
            matches = sum(1 for p in patterns if p in topic_lower)
            if matches > max_matches:
                max_matches = matches
                domain = dom_name

        # 3. Intent & Topic Type & Emotional Angle detection
        intent = "Educate viewers on health impacts, biological mechanisms, and actionable habits."
        topic_type = "lifestyle_health"
        emotional_angle = "Curiosity and practical awareness"
        suggested_strategic_goals = ["Watch Time", "Reach"]

        if re.search(r"\b(what happens if|consequences?|effects? of|impacts? of)\b", topic_lower):
            intent = "Explain internal biological consequences and systemic effects of an everyday habit."
            topic_type = "habit_consequence"
            emotional_angle = "Curiosity and eye-opening self-reflection"
            suggested_strategic_goals = ["Watch Time", "Reach", "Saves"]

        elif re.search(r"\b(why do|why does|why is|why are|how does)\b", topic_lower):
            intent = "Explain the bodily scientific mechanism behind a common physical experience."
            topic_type = "mechanism_inquiry"
            emotional_angle = "Scientific curiosity and clinical clarity"
            suggested_strategic_goals = ["Doctor Authority", "Watch Time"]

        elif re.search(r"\b(normal\s+vs|red\s*flags?|emergenc(?:y|ies)|hospitals?|danger(?:ous)?)\b", topic_lower):
            intent = "Draw an urgent, unambiguous line between harmless symptoms and emergency red flags."
            topic_type = "emergency_triage"
            emotional_angle = "Urgency and decisive action"
            suggested_strategic_goals = ["Shares", "Urgency"]

        elif re.search(r"\b(myths?|truths?|fact or fiction|is it true|does .+ cause)\b", topic_lower):
            intent = "Debunk a widely circulated health misconception or remedy with clinical evidence."
            topic_type = "myth_debunk"
            emotional_angle = "Surprise and intrigue"
            suggested_strategic_goals = ["Reach", "Shares"]

        elif re.search(r"\b(signs?|symptoms?|how to know|tests?)\b", topic_lower):
            intent = "Highlight observable warning indicators and provide clear clinical thresholds."
            topic_type = "symptom_triage"
            emotional_angle = "Alertness and concern for loved ones"
            suggested_strategic_goals = ["Saves", "Shares", "Reach"]

        elif re.search(r"\b(mistakes?|wrong|avoid|stop doing)\b", topic_lower):
            intent = "Identify widespread behavioral blunders and deliver single-step corrections."
            topic_type = "behavior_correction"
            emotional_angle = "Realization and practical empowerment"
            suggested_strategic_goals = ["Reach", "Saves"]

        elif re.search(r"\b(swap|do this not that|instead of|replace)\b", topic_lower):
            intent = "Provide realistic, achievable habit substitutions rather than restrictive elimination."
            topic_type = "habit_substitution"
            emotional_angle = "Empowerment and achievable change"
            suggested_strategic_goals = ["Saves", "Reach"]

        elif re.search(r"\b(\bvs\b|versus|difference between|compare)\b", topic_lower):
            intent = "Differentiate two lookalike conditions or treatment choices using observable signs."
            topic_type = "comparison"
            emotional_angle = "Clarity and discernment"
            suggested_strategic_goals = ["Saves", "Doctor Authority"]

        elif re.search(r"\b(checklists?|what to carry|prepar(?:e|ation))\b", topic_lower):
            intent = "Provide a comprehensive, physically actionable preparation list for future reference."
            topic_type = "practical_checklist"
            emotional_angle = "Readiness and reassurance"
            suggested_strategic_goals = ["Saves"]

        # 4. Audience detection
        audience = "General public and social media viewers interested in everyday wellness."
        if any(w in topic_lower for w in ["women", "woman", "female"]):
            audience = "Adult women concerned with health, hormones, and prevention."
        elif any(w in topic_lower for w in ["men", "man", "male"]):
            audience = "Adult men seeking direct, no-nonsense health guidance."
        elif any(w in topic_lower for w in ["elderly", "parents", "father", "mother", "senior"]):
            audience = "Adult children managing elderly parents' health and seniors."
        elif any(w in topic_lower for w in ["office", "desk", "sitting", "screen"]):
            audience = "Office workers and desk employees with sedentary daily routines."
        elif any(w in topic_lower for w in ["junk food", "fast food", "eating", "snack", "sugar", "diet"]):
            audience = "Everyday consumers and young adults frequently eating processed foods and snacks."

        # Add domain-specific keywords if minimal
        if "junk food" in topic_lower or "fast food" in topic_lower:
            for extra in ["ultra-processed food", "metabolism", "diet habits"]:
                if extra not in keywords:
                    keywords.append(extra)

        return TopicAnalysis(
            original_topic=topic,
            domain=domain,
            audience=audience,
            intent=intent,
            keywords=keywords,
            emotional_angle=emotional_angle,
            topic_type=topic_type,
            suggested_strategic_goals=suggested_strategic_goals,
            analysis_source="deterministic_fallback"
        )

    def _analyze_with_groq(self, topic: str) -> Optional[TopicAnalysis]:
        """Free Groq LLM topic analysis if API key is present."""
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            prompt = f"""You are an expert health content strategist. Analyze this topic for a medical reel:
Topic: "{topic}"

Return a JSON object with EXACTLY these fields:
- domain: string (medical or health specialty, e.g. Nutrition & Metabolism)
- audience: string (target demographic)
- intent: string (core goal of reel)
- keywords: array of strings (5-8 relevant tags)
- emotional_angle: string (e.g. Curiosity, Urgency)
- topic_type: string (e.g. habit_consequence, symptom_triage, myth_debunk, etc.)
- suggested_strategic_goals: array of strings from: ["Reach", "Watch Time", "Saves", "Shares", "Doctor Authority"]
"""
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You output strictly valid JSON matching the requested schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            data = json.loads(completion.choices[0].message.content)
            return TopicAnalysis(
                original_topic=topic,
                domain=data.get("domain", "General Preventive Health"),
                audience=data.get("audience", "General public"),
                intent=data.get("intent", "Educate viewers on health mechanisms."),
                keywords=data.get("keywords", [topic]),
                emotional_angle=data.get("emotional_angle", "Curiosity"),
                topic_type=data.get("topic_type", "lifestyle_health"),
                suggested_strategic_goals=data.get("suggested_strategic_goals", ["Watch Time", "Reach"]),
                analysis_source="llm:groq"
            )
        except Exception:
            return None

    def _analyze_with_gemini(self, topic: str) -> Optional[TopicAnalysis]:
        """Free Gemini REST topic analysis if API key is present."""
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            prompt = f"""Analyze this medical/health topic for a short reel: "{topic}".
Output strictly valid JSON with keys: domain, audience, intent, keywords (list), emotional_angle, topic_type, suggested_strategic_goals (list from ["Reach", "Watch Time", "Saves", "Shares", "Doctor Authority"])."""
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
            }
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                result = resp.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(text)
                return TopicAnalysis(
                    original_topic=topic,
                    domain=data.get("domain", "General Preventive Health"),
                    audience=data.get("audience", "General public"),
                    intent=data.get("intent", "Educate viewers on health mechanisms."),
                    keywords=data.get("keywords", [topic]),
                    emotional_angle=data.get("emotional_angle", "Curiosity"),
                    topic_type=data.get("topic_type", "lifestyle_health"),
                    suggested_strategic_goals=data.get("suggested_strategic_goals", ["Watch Time", "Reach"]),
                    analysis_source="llm:gemini"
                )
        except Exception:
            return None


if __name__ == "__main__":
    analyzer = TopicAnalyzer()
    print("Running Topic Analyzer on demo topic 'Junk Food':")
    res = analyzer.analyze("Junk Food")
    print(res.model_dump_json(indent=2))
