import os
import json
import re
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

from app.agents.topic_analyzer import TopicAnalysis
from app.agents.framework_selector import FrameworkSelection
from app.agents.script_writer import ReelScript, SENSATIONAL_PHRASES

# Ensure .env is loaded from project root
_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()

# Extended list of medical and physiological claims that must never be fabricated in visual plans or captions
DISALLOWED_VISUAL_CLAIMS = SENSATIONAL_PHRASES + [
    "insulin overload",
    "gut microbiome diversity drops",
    "encourages fat storage",
    "guaranteed harm",
    "diagnose",
    "diagnosis"
]


class Shot(BaseModel):
    """A single structured visual shot designed for short-form reel video generation (Google Flow / Veo 3.1 Lite).
    
    The spoken narration (script_text), stage, and duration are strictly controlled by the application.
    The visual director (LLM or deterministic engine) determines what is visually shown."""
    shot_number: int = Field(..., description="1-indexed sequential shot number (1 to shot_count).")
    duration_seconds: float = Field(..., description="Application-calculated duration in seconds.")
    script_stage_order: int = Field(..., description="Framework stage order number corresponding to this shot.")
    script_stage_name: str = Field(..., description="Framework stage name matching the ReelScript section.")
    script_text: str = Field(..., description="Exact spoken narration text delivered during this shot.")
    visual_goal: str = Field(..., description="Strategic visual objective of this shot.")
    visual_description: str = Field(..., description="Concrete description of what appears visually on screen.")
    subject: str = Field(..., description="Primary visual subject or focal point.")
    setting: str = Field(..., description="Physical environment, background, and location.")
    camera_direction: str = Field(..., description="Framing, camera angle, and camera motion.")
    motion: str = Field(..., description="Action, physical movements, or dynamic elements.")
    on_screen_text: str = Field(..., description="Concise on-screen caption overlay faithful to spoken text.")
    audio_direction: str = Field(..., description="Audio tone, voiceover pacing, and subtle ambient sound.")
    video_generation_prompt: str = Field(..., description="Self-contained, production-ready video prompt for Google Flow / Veo 3.1 Lite.")

    validation_status: str = Field("VALID (duration <= 8.0s)", description="Validation status of this shot.")

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v < 4.0:
            raise ValueError(f"duration_seconds must be >= 4.0s for Google Flow / Veo 3.1 Lite minimum, got {v}")
        if v > 8.0:
            raise ValueError(f"duration_seconds must be <= 8.0s for Google Flow / Veo 3.1 Lite clip limit, got {v}")
        return round(v, 1)

    @field_validator(
        "script_stage_name",
        "script_text",
        "visual_goal",
        "visual_description",
        "subject",
        "setting",
        "camera_direction",
        "motion",
        "on_screen_text",
        "audio_direction",
        "video_generation_prompt"
    )
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"Field '{info.field_name}' must not be empty.")
        return v.strip()


class ShotPlan(BaseModel):
    """Complete, structured visual plan for a short-form video reel."""
    topic: str = Field(..., description="The reel topic.")
    framework_id: int = Field(..., description="Framework ID (1-30).")
    framework_name: str = Field(..., description="Framework name from knowledge base.")
    target_duration_seconds: float = Field(..., description="Target video duration in seconds.")
    shot_count: int = Field(..., description="Total planned shots (dynamically derived from target duration).")
    total_duration_seconds: float = Field(..., description="Sum of planned shot durations.")
    cta: str = Field(..., description="Approved single CTA from ReelScript.")
    continuity_notes: Optional[str] = Field(None, description="Visual and character continuity guidelines across shots.")
    shots: List[Shot] = Field(..., description="Sequential list of planned shots.")
    generation_source: str = Field("deterministic_fallback", description="Engine used: groq or deterministic_fallback.")
    validation_status: str = Field("VALID (all shots in [4.0s, 8.0s], narration timing verified)", description="Validation status of the shot plan.")
    feedback: Optional[str] = Field(None, description="Applied user revision feedback, if any.")

    @field_validator("shot_count")
    @classmethod
    def validate_shot_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"shot_count must be at least 1, got {v}")
        return v

    @field_validator("target_duration_seconds")
    @classmethod
    def validate_target_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"target_duration_seconds must be positive, got {v}")
        return v

    @field_validator("shots")
    @classmethod
    def validate_shots_sequence(cls, v: List[Shot]) -> List[Shot]:
        if not v:
            raise ValueError("ShotPlan must contain at least one shot.")
        for idx, shot in enumerate(v, 1):
            if shot.shot_number != idx:
                raise ValueError(f"Shot {idx} has invalid shot_number {shot.shot_number}; expected {idx}.")
        return v

    @model_validator(mode="after")
    def validate_consistency_and_safety(self) -> "ShotPlan":
        if len(self.shots) != self.shot_count:
            raise ValueError(f"shots length ({len(self.shots)}) must match shot_count ({self.shot_count}).")

        for s in self.shots:
            if s.duration_seconds < 4.0:
                raise ValueError(f"Shot {s.shot_number} duration ({s.duration_seconds}s) must be >= 4.0s.")
            if s.duration_seconds > 8.0:
                raise ValueError(f"Shot {s.shot_number} duration ({s.duration_seconds}s) exceeds the maximum 8.0s clip limit.")

        calc_total = sum(s.duration_seconds for s in self.shots)
        self.total_duration_seconds = round(calc_total, 1)

        # Medical safety check: ensure visuals and captions do not introduce unauthorized claims
        for s in self.shots:
            combined = f"{s.visual_description} {s.video_generation_prompt} {s.on_screen_text}".lower()
            for phrase in DISALLOWED_VISUAL_CLAIMS:
                if phrase in combined:
                    raise ValueError(f"Disallowed claim phrase '{phrase}' is not permitted in visual plan.")

        return self


def _sanitize_prompt_text(text: str) -> str:
    """Sanitizes prompt text and descriptions to ensure natural, grammatically correct phrasing."""
    if not text:
        return text
    # 1. Fix malformed hand continuity phrasing: never write 'shot of hands of the recurring...'
    text = re.sub(
        r'(?i)\b(shot of hands of the recurring|hands of the recurring|close-up of hands of the recurring)\s+young adult[^.]*',
        "close-up of the young adult's hands",
        text
    )
    text = re.sub(r'(?i)\bshot of hands of\b', 'close-up of', text)
    text = re.sub(r'(?i)\bhands of the recurring\b', "the young adult's hands", text)
    text = re.sub(r'(?i)\byoung adults hands\b', "young adult's hands", text)
    text = re.sub(r'(?i)\bhands gently moves\b', "hands gently moving", text)
    # 2. Fix duplicated articles or awkward recurring prefixes
    text = re.sub(r'\bThe same a\b', 'The same', text)
    text = re.sub(r'\bthe same a\b', 'the same', text)
    text = re.sub(r'\b[Aa]\s+[Aa]\b', 'a', text)
    text = re.sub(r'\b(the\s+)?recurring\s+a\b', 'the recurring', text, flags=re.IGNORECASE)
    # 3. Fix malformed action grammar like "The action shows holds" -> "The person holds"
    text = re.sub(
        r'\b[Tt]he action shows\s+(holds?|takes?|reaches?|drinks?|pours?|examines?|looks?|rubbing|rubs?|arranges?|organizes?|nods?|places?|sprinkles?|pairs?|swaps?)\b',
        r'The person \1',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\b[Tt]he action shows\b', 'The person', text)
    # 4. Clean any double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _assemble_clean_veo_prompt(
    visual_portion: str,
    narration: str,
    forbidden_leakage_strings: Optional[List[str]] = None,
    is_no_person: bool = False
) -> str:
    """Assembles a clean, production-ready visual prompt for Google Flow / Veo 3.1 Lite.

    Strict Visual-Only Rules:
    - Never include narration text inside the visual prompt.
    - Never tell the video model to display subtitles or captions.
    - Never put quotation marks around spoken narration.
    - Never use generic visual instructions that conflict with the shot (e.g. realistic human movement in no-person shots).
    - Do not use unnecessary cinematic jargon such as shallow depth of field or 35mm lens.
    - Keep prompts simple and generation-friendly for Veo/Google Flow.
    - Every prompt describes ONLY what visually happens on screen.
    """
    if not visual_portion:
        visual_portion = "Vertical 9:16 photorealistic cinematic shot."

    text = visual_portion.strip()

    # 1. Truncate at any voiceover/audio/dialogue instructions if present
    vo_match = re.search(
        r'(?i)\b(include a clear|speaking exactly|voiceover should be|voiceover:|spoken voiceover|voiceover\b|clean composition with no overlaid)\b',
        text
    )
    if vo_match:
        text = text[:vo_match.start()].strip()

    # 2. Strip any audio/sound directions accidentally placed into visuals
    text = re.sub(r'(?i)\b(soft ambient|ambient room tone|ambient sounds?|background room tone|kitchen ambient)[^.]*\.?', '', text)

    # 3. Strip any overlay / caption / subtitle instructions
    text = re.sub(r'(?i)(the\s+)?on-screen\s+caption\s+[^.,;]*[.,;]?', '', text)
    text = re.sub(r'(?i)(the\s+)?caption\s+(includes|reads|shows|displays)\s+[^.,;]*[.,;]?', '', text)
    text = re.sub(r'(?i)(overlay\s+text|text\s+overlay|subtitles?)[^.,;]*[.,;]?', '', text)
    text = re.sub(r'(?i)\bwith no overlaid text or logos\.?\b', '', text)

    # 4. Remove any narration leakage from forbidden strings and current narration
    if forbidden_leakage_strings:
        for leak_str in forbidden_leakage_strings:
            leak_clean = leak_str.strip().strip('"\'')
            if len(leak_clean) >= 8:
                pattern = re.escape(leak_clean)
                text = re.sub(rf'(?i)[\s"\'*]*' + pattern + rf'[\s"\'*.]*', ' ', text)

    curr_narration_clean = narration.strip().strip('"\'')
    if len(curr_narration_clean) >= 8:
        pattern = re.escape(curr_narration_clean)
        text = re.sub(rf'(?i)[\s"\'*]*' + pattern + rf'[\s"\'*.]*', ' ', text)
        for sentence in re.split(r'(?<=[.!?])\s+', curr_narration_clean):
            s_clean = sentence.strip().strip('"\'')
            if len(s_clean) >= 8:
                text = re.sub(rf'(?i)[\s"\'*]*' + re.escape(s_clean) + rf'[\s"\'*.]*', ' ', text)

    # 5. Remove unnecessary cinematic jargon (35mm lens, shallow depth of field)
    text = re.sub(r'(?i)\b(35mm\s+(lens\s+)?aesthetic|35mm\s+lens|shallow\s+depth\s+of\s+field)\b[,.]?', '', text)

    # 6. Remove conflicting "realistic human movement" in no-person shots
    if is_no_person or any(w in text.lower() for w in ["without people", "without human", "no people", "no human", "3d medical", "3d biological"]):
        text = re.sub(r'(?i)\b(with\s+)?realistic\s+human\s+movement\b[,.]?', '', text)
        text = re.sub(r'(?i)\b(with\s+)?human\s+movement\b[,.]?', '', text)

    # 7. Clean up quotes, backslashes, and broken punctuation while preserving internal apostrophes
    text = text.replace('"', '').replace('\\', '')
    text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '', text)
    text = _sanitize_prompt_text(text)
    text = re.sub(r'9\s*:\s*16', '9:16', text)
    text = re.sub(r'\s*([.,;!?])\s*', r'\1 ', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure exactly one clean 'Vertical 9:16' at the start
    text = re.sub(r'^(Vertical\s+9:16\s*)+', '', text, flags=re.IGNORECASE).strip()
    text = f"Vertical 9:16 {text}"

    # Ensure visual portion ends with a clean period
    if not text.endswith((".", "!", "?")):
        text = f"{text}."

    return text


def calculate_recommended_shot_count(target_duration_seconds: float) -> int:
    """Derives a recommended shot count from the requested reel target duration.
    
    Guarantees:
    - 8s -> 1 shot (~8.0s)
    - 15s -> 2 shots (~15.0s, 2 x 7.5s)
    - 30s -> 4 shots (~30.0s, 4 x 7.5s)
    - 45s -> 6 shots (~45.0s, 6 x 7.5s)
    - Every shot fits within Google Flow / Veo 3.1 Lite duration limits: [4.0s, 8.0s].
    """
    t = float(target_duration_seconds)
    if t <= 10.0:
        return 1
    elif t <= 20.0:
        return 2
    elif t <= 26.0:
        return 3
    elif t <= 36.0:
        return 4
    elif t <= 42.0:
        return 5
    elif t <= 50.0:
        return 6
    else:
        return max(6, int(round(t / 7.5)))


class ShotPlanner:
    """Converts an approved ReelScript into a structured, production-ready multi-shot visual plan
    optimized for AI video generation tools like Google Flow / Veo 3.1 Lite.
    
    Architectural Guarantee:
    - Script Writer decides WHAT is said (100% preserved narration).
    - Shot Planner decides WHAT is shown (visuals, camera, audio, captions).
    Spoken narration, framework stage mapping, and shot durations are strictly controlled by the application."""

    def __init__(self):
        self.last_llm_error: Optional[str] = None
        self.groq_model_used: Optional[str] = None

    def _refresh_api_keys(self) -> None:
        raw_groq = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_api_key = raw_groq if raw_groq else None

    def plan_shots(
        self,
        reel_script: ReelScript,
        topic_analysis: Optional[TopicAnalysis] = None,
        framework_metadata: Optional[Union[FrameworkSelection, Dict[str, Any]]] = None,
        target_duration_seconds: Optional[float] = None,
        shot_count: Optional[int] = None,
        force_fallback: bool = False,
        raise_on_llm_error: bool = False,
        feedback: Optional[str] = None
    ) -> ShotPlan:
        """Generates a structured ShotPlan for the given ReelScript.
        
        Guarantees that joining shot.script_text across all shots strictly reconstructs
        the original reel_script.full_script with 100% fidelity.
        """
        if not isinstance(reel_script, ReelScript):
            raise TypeError(f"Expected ReelScript instance, got {type(reel_script).__name__}")

        duration = target_duration_seconds or reel_script.target_duration_seconds
        if duration <= 0:
            raise ValueError(f"target_duration_seconds must be positive, got {duration}")

        if shot_count is None:
            shot_count = calculate_recommended_shot_count(duration)
        elif shot_count < 1:
            raise ValueError(f"shot_count must be at least 1, got {shot_count}")

        self._refresh_api_keys()
        self.last_llm_error = None

        # 1. Distribute exact script text and calculate durations deterministically in application
        distributed_script = self._distribute_script_exact(reel_script, shot_count)
        calculated_durations = self._calculate_durations(distributed_script, duration)

        # 2. Attempt Groq-powered visual generation if key is available and fallback not forced
        if not force_fallback and self.groq_api_key:
            try:
                llm_plan = self._generate_with_groq(
                    reel_script=reel_script,
                    topic_analysis=topic_analysis,
                    framework_metadata=framework_metadata,
                    target_duration=duration,
                    shot_count=shot_count,
                    distributed_script=distributed_script,
                    calculated_durations=calculated_durations,
                    feedback=feedback
                )
                if llm_plan:
                    return llm_plan
            except Exception as e:
                sanitized_msg = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", str(e))
                self.last_llm_error = f"Groq ShotPlanner Error ({type(e).__name__}): {sanitized_msg}"
                if raise_on_llm_error:
                    raise RuntimeError(self.last_llm_error) from e

        # 3. Deterministic Fallback Generator
        return self._generate_deterministic(
            reel_script=reel_script,
            topic_analysis=topic_analysis,
            framework_metadata=framework_metadata,
            target_duration=duration,
            shot_count=shot_count,
            distributed_script=distributed_script,
            calculated_durations=calculated_durations,
            feedback=feedback
        )

    def _distribute_script_exact(self, reel_script: ReelScript, shot_count: int) -> List[Dict[str, Any]]:
        """Extracts every sentence from the approved ReelScript sections and partitions them
        into shot_count contiguous buckets.
        
        Guarantee: ' '.join(item['text'] for item in result) == reel_script.full_script exactly."""
        # 1. Extract sentence-level items tagged with stage metadata
        sentence_items = []
        for sec in reel_script.sections:
            raw_sents = re.split(r'(?<=[.!?])\s+', sec.text.strip())
            for s in raw_sents:
                s_clean = s.strip()
                if s_clean:
                    sentence_items.append({
                        "stage_order": sec.stage_order,
                        "stage_name": sec.stage_name,
                        "text": s_clean
                    })

        total_sentences = len(sentence_items)

        # Fast-path for single-shot reels (e.g. 8s duration)
        if shot_count == 1:
            all_text = " ".join(item["text"] for item in sentence_items)
            first_sec = reel_script.sections[0] if reel_script.sections else None
            primary_order = first_sec.stage_order if first_sec else 1
            primary_name = first_sec.stage_name if first_sec else "Overview"
            distributed = [{
                "shot_number": 1,
                "stage_order": primary_order,
                "stage_name": primary_name,
                "text": all_text
            }]
            if reel_script.cta and reel_script.cta.strip():
                cta_clean = reel_script.cta.strip()
                if distributed[0]["text"]:
                    distributed[0]["text"] = f"{distributed[0]['text']} {cta_clean}".strip()
                else:
                    distributed[0]["text"] = cta_clean
            return distributed

        # 2. If fewer sentences than requested shots, subdivide longer sentences on clause boundaries
        if total_sentences < shot_count:
            expanded_items = []
            needed = shot_count - total_sentences
            for item in sentence_items:
                text = item["text"]
                # Try to split on clause boundaries like ', while ', ', and ', '; '
                clause_match = re.search(r'([,;])\s+(while|and|so|but|leading to|prompting|which)\b', text, re.IGNORECASE)
                if clause_match and needed > 0:
                    split_idx = clause_match.start(1) + 1
                    part1 = text[:split_idx].strip()
                    part2 = text[split_idx:].strip()
                    expanded_items.append({"stage_order": item["stage_order"], "stage_name": item["stage_name"], "text": part1})
                    expanded_items.append({"stage_order": item["stage_order"], "stage_name": item["stage_name"], "text": part2})
                    needed -= 1
                else:
                    expanded_items.append(item)
            sentence_items = expanded_items
            total_sentences = len(sentence_items)

        # 2b. Secondary split if still fewer sentences than shots
        if total_sentences < shot_count:
            expanded_items = []
            needed = shot_count - total_sentences
            for item in sentence_items:
                words = item["text"].split()
                if needed > 0 and len(words) >= 4:
                    mid = len(words) // 2
                    part1 = " ".join(words[:mid]).strip()
                    part2 = " ".join(words[mid:]).strip()
                    expanded_items.append({"stage_order": item["stage_order"], "stage_name": item["stage_name"], "text": part1})
                    expanded_items.append({"stage_order": item["stage_order"], "stage_name": item["stage_name"], "text": part2})
                    needed -= 1
                else:
                    expanded_items.append(item)
            sentence_items = expanded_items
            total_sentences = len(sentence_items)

        # 3. Partition sentence items into shot_count contiguous buckets
        distributed = []
        if total_sentences <= shot_count:
            for i in range(shot_count):
                if i < total_sentences:
                    it = sentence_items[i]
                    distributed.append({
                        "shot_number": i + 1,
                        "stage_order": it["stage_order"],
                        "stage_name": it["stage_name"],
                        "text": it["text"]
                    })
                else:
                    # Final shot holds remaining visual stage
                    last_it = sentence_items[-1]
                    distributed.append({
                        "shot_number": i + 1,
                        "stage_order": last_it["stage_order"],
                        "stage_name": last_it["stage_name"],
                        "text": ""
                    })
        else:
            # More sentences than shots (e.g. 6 to 15 sentences for 5 shots).
            # Partition sentence_items into exactly shot_count contiguous buckets.
            # Constraints & Optimization:
            # 1. Spoken narration for every shot MUST be <= 38 words (<= 8.0s natural speaking rate).
            # 2. Account for CTA added to the final shot (shot 5).
            # 3. Minimize maximum words and balance words evenly across shots.
            # 4. Respect framework stage continuity where possible.
            import itertools

            cta_words = len(reel_script.cta.strip().split()) if reel_script.cta and reel_script.cta.strip() else 0
            sent_word_counts = [len(item["text"].split()) for item in sentence_items]
            total_words = sum(sent_word_counts) + cta_words
            ideal_per_shot = total_words / float(shot_count)

            best_cuts = None
            best_score = float('inf')

            # Evaluate all combinations of cut points
            for cuts_tuple in itertools.combinations(range(1, total_sentences), shot_count - 1):
                full_cuts = (0,) + cuts_tuple + (total_sentences,)
                bucket_word_counts = []
                for b_idx in range(shot_count):
                    s_idx = full_cuts[b_idx]
                    e_idx = full_cuts[b_idx + 1]
                    w_count = sum(sent_word_counts[j] for j in range(s_idx, e_idx))
                    if b_idx == shot_count - 1:
                        w_count += cta_words
                    bucket_word_counts.append(w_count)

                # Primary penalty: severe penalty for any bucket > 38 words
                overflow = sum(max(0, w - 38) for w in bucket_word_counts)
                max_w = max(bucket_word_counts)
                variance = sum((w - ideal_per_shot) ** 2 for w in bucket_word_counts)

                # Stage cross penalty: prefer buckets that don't mix disparate stages
                stage_cross = 0
                for b_idx in range(shot_count):
                    s_idx = full_cuts[b_idx]
                    e_idx = full_cuts[b_idx + 1]
                    stages = set(sentence_items[j]["stage_order"] for j in range(s_idx, e_idx))
                    if len(stages) > 1:
                        stage_cross += (len(stages) - 1)

                score = (overflow * 100000.0) + (stage_cross * 2000.0) + (max_w * 10.0) + variance
                if score < best_score:
                    best_score = score
                    best_cuts = full_cuts

            if best_cuts is None:
                # Fallback to linear cuts if no combination found
                cuts = [int(round(i * total_sentences / shot_count)) for i in range(shot_count + 1)]
                best_cuts = tuple(cuts)

            for i in range(shot_count):
                start_i = best_cuts[i]
                end_i = best_cuts[i + 1]
                if end_i <= start_i:
                    end_i = start_i + 1
                end_i = min(end_i, total_sentences)

                bucket = sentence_items[start_i:end_i]
                bucket_text = " ".join(b["text"] for b in bucket)
                primary_order = bucket[0]["stage_order"]
                primary_name = bucket[0]["stage_name"]

                distributed.append({
                    "shot_number": i + 1,
                    "stage_order": primary_order,
                    "stage_name": primary_name,
                    "text": bucket_text
                })

        # Append exact ReelScript.cta to final shot's spoken script_text:
        # The final shot should contain: [approved final narration] + [exact ReelScript.cta]
        if reel_script.cta and reel_script.cta.strip():
            cta_clean = reel_script.cta.strip()
            if distributed[-1]["text"]:
                distributed[-1]["text"] = f"{distributed[-1]['text']} {cta_clean}".strip()
            else:
                distributed[-1]["text"] = cta_clean

        return distributed

    def _calculate_durations(self, distributed_shots: List[Dict[str, Any]], target_duration: float = 45.0) -> List[float]:
        """Calculates deterministic shot durations based on narration length and target duration.

        Rules:
        - Every shot duration is strictly in [4.0s, 8.0s] (Google Flow / Veo 3.1 Lite limit).
        - Total duration closely tracks target_duration.
        - If narration cannot fit naturally in 8.0s (> 38 words), raises ValueError identifying the affected shot.
        """
        N = len(distributed_shots)
        if N == 0:
            return []

        if N == 1:
            dur = min(8.0, max(4.0, round(float(target_duration), 1)))
            text = distributed_shots[0].get("text", "").strip()
            wc = len(text.split())
            if wc > 38:
                stage_name = distributed_shots[0].get("stage_name", "Unknown")
                raise ValueError(
                    f"Narration for Shot 1 ('{stage_name}') contains {wc} words, which cannot "
                    f"realistically fit within the 8.0-second clip duration limit (maximum 38 words). "
                    f"Section narration: \"{text}\""
                )
            return [dur]

        # Check word counts and speaking rate limit
        word_counts = []
        min_speech_durs = []
        for idx, item in enumerate(distributed_shots):
            text = item.get("text", "").strip()
            wc = len(text.split())
            word_counts.append(wc)
            if wc > 38:
                stage_name = item.get("stage_name", "Unknown")
                raise ValueError(
                    f"Narration for Shot {idx + 1} ('{stage_name}') contains {wc} words, which cannot "
                    f"realistically fit within the 8.0-second clip duration limit (maximum 38 words). "
                    f"Section narration: \"{text}\""
                )
            min_speech_durs.append(max(4.0, round(wc / 3.8, 1)))

        # Target total clamped to physical limits for N clips [N * 4.0, N * 8.0]
        clamped_target = min(N * 8.0, max(N * 4.0, float(target_duration)))
        even_dur = round(clamped_target / N, 1)

        # Initial allocation
        durations = []
        for idx in range(N):
            base = max(even_dur, min_speech_durs[idx])
            durations.append(min(8.0, max(4.0, round(base, 1))))

        # Fine-tune to match clamped_target exactly (in 0.1s increments)
        current_sum = round(sum(durations), 1)
        iterations = 0
        while round(current_sum, 1) != round(clamped_target, 1) and iterations < 200:
            iterations += 1
            diff = round(clamped_target - current_sum, 1)
            step = 0.1 if diff > 0 else -0.1

            best_idx = None
            if diff > 0:
                candidates = [i for i in range(N) if durations[i] + 0.05 < 8.0]
                if not candidates:
                    break
                candidates.sort(key=lambda i: (durations[i], -word_counts[i]))
                best_idx = candidates[0]
            else:
                candidates = [i for i in range(N) if durations[i] - 0.05 > max(4.0, min_speech_durs[i])]
                if not candidates:
                    candidates = [i for i in range(N) if durations[i] - 0.05 > 4.0]
                if not candidates:
                    break
                candidates.sort(key=lambda i: (-durations[i], word_counts[i]))
                best_idx = candidates[0]

            durations[best_idx] = round(durations[best_idx] + step, 1)
            current_sum = round(sum(durations), 1)

        # Final safety clamp
        return [round(min(8.0, max(4.0, d)), 1) for d in durations]

    def _generate_deterministic(
        self,
        reel_script: ReelScript,
        topic_analysis: Optional[TopicAnalysis],
        framework_metadata: Optional[Union[FrameworkSelection, Dict[str, Any]]],
        target_duration: float,
        shot_count: int,
        distributed_script: List[Dict[str, Any]],
        calculated_durations: List[float],
        feedback: Optional[str] = None
    ) -> ShotPlan:
        """Deterministic visual planner generating concrete, topic-aware shots,
        full narrative Veo prompts, concise captions, and audio directions."""
        topic = reel_script.topic
        topic_lower = topic.lower()

        is_nutrition = any(k in topic_lower for k in ["food", "eat", "diet", "nutrition", "sugar", "curd", "snack", "chip", "drink", "meal"])
        is_medical_symptom = any(k in topic_lower for k in ["cramp", "pain", "fever", "liver", "stress", "dengue", "cough", "muscle", "leg"])

        if is_nutrition:
            char_desc = "young adult with short dark hair, wearing a simple neutral-colored casual shirt, in a bright modern kitchen"
            base_setting = "a bright modern kitchen"
            theme = "nutrition"
        elif is_medical_symptom:
            char_desc = "relatable adult in everyday comfortable clothing"
            base_setting = "a cozy, softly lit home interior"
            theme = "symptom"
        else:
            char_desc = "articulate presenter in smart casual attire"
            base_setting = "a minimalist contemporary studio with warm ambient lighting"
            theme = "general"

        continuity_notes = (
            f"Consistent visual style: A {char_desc} featured across recurring character shots. "
            "Vertical 9:16 aspect ratio, warm cinematic color grading, natural diffused lighting, "
            "steady camera movement, and coherent interior atmosphere."
        )

        shots: List[Shot] = []

        all_forbidden_strings = [
            reel_script.cta or "",
            reel_script.full_script_with_cta,
            reel_script.full_script_without_cta
        ] + [s_item["text"] for s_item in distributed_script]

        for idx, item in enumerate(distributed_script):
            shot_num = idx + 1
            dur = calculated_durations[idx]
            st_order = item["stage_order"]
            st_name = item["stage_name"]
            st_text = item["text"]
            st_lower = st_text.lower()
            is_first_shot = (shot_num == 1)
            is_final_shot = (shot_num == shot_count)

            # Character reference: 'A ...' on first shot, 'The recurring ...' on subsequent shots
            if is_first_shot:
                char_subject = f"A {char_desc}"
            else:
                char_subject = f"The recurring {char_desc}"

            # Determine visual goal, subject, setting, camera_dir, motion, and caption based on narration content
            if is_first_shot:
                visual_goal = f"Introduce the core question and hook viewer attention regarding {topic}."
                if theme == "nutrition":
                    subject = char_subject
                    setting = "In a bright modern kitchen with morning sunlight streaming through windows"
                    camera_dir = "eye-level medium shot, subtle camera push-in"
                    motion = "The person holds an obviously processed snack package such as chips thoughtfully, looking toward the camera with a curious, questioning expression"
                    caption = "What happens if you eat this daily?"
                elif theme == "symptom":
                    subject = char_subject
                    setting = "Resting comfortably on a sofa in a warm, softly lit living room"
                    camera_dir = "medium shot with gentle camera drift"
                    motion = "The person pauses to notice a physical sensation in the lower leg, looking toward the camera with curious attention"
                    caption = "Why does this happen?"
                else:
                    subject = char_subject
                    setting = "In a minimalist studio with soft ambient background lighting"
                    camera_dir = "eye-level medium shot, steady framing"
                    motion = "The presenter engages directly with the camera, gesturing calmly with open hands"
                    caption = "Here is what happens:"

            elif is_final_shot:
                visual_goal = "Visually reinforce sustainable habits and doctor consultation guidance."
                if theme == "nutrition":
                    subject = char_subject
                    setting = "In the bright modern kitchen by the counter with natural window lighting"
                    camera_dir = "eye-level medium shot, steady composition"
                    motion = "The person faces the camera naturally with a calm, caring, and trustworthy expression, maintaining subtle, relaxed body language"
                    if any(w in st_lower for w in ["physician", "doctor", "consult", "symptom", "healthcare", "clinic"]):
                        caption = "Persistent symptoms? Consult a physician."
                    else:
                        caption = reel_script.cta or "Follow for more science-backed nutrition tips!"
                elif theme == "symptom":
                    subject = char_subject
                    setting = "In the peaceful living room by an open window with natural greenery"
                    camera_dir = "medium shot, steady composition"
                    motion = "The person drinks from a clear water glass and gestures affirmatively toward the camera with reassurance"
                    if any(w in st_lower for w in ["physician", "doctor", "consult", "symptom", "healthcare", "clinic"]):
                        caption = "Persistent symptoms? Consult a physician."
                    else:
                        caption = reel_script.cta or "Consult a doctor if symptoms persist."
                else:
                    subject = char_subject
                    setting = "In the contemporary studio space with warm ambient lighting"
                    camera_dir = "medium close-up, steady composition"
                    motion = "The presenter nods affirmatively and delivers closing guidance with confidence"
                    if any(w in st_lower for w in ["physician", "doctor", "consult", "symptom", "healthcare", "clinic"]):
                        caption = "Persistent symptoms? Consult a physician."
                    else:
                        caption = reel_script.cta or "Follow for more evidence-based insights."

            else:
                # Intermediate shots: dynamically tailored to the specific narration of this shot

                # 1. Healthy food swaps & alternatives (Prevention / Solution)
                if any(w in st_lower for w in ["swap", "popcorn", "nut", "nuts", "fruit", "veggie", "vegetable", "curd", "whole food", "apple", "berry", "substitute"]):
                    visual_goal = "Demonstrate practical, delicious whole-food swaps for processed snacks."
                    subject = "Close-up of the young adult's hands"
                    setting = "At a clean wooden kitchen counter bathed in warm natural daylight"
                    camera_dir = "medium close-up, downward 45-degree angle"
                    motion = "The person gently moves a bag of processed snacks aside and replaces it with a wholesome bowl of fresh blueberries, apple slices, and raw walnuts"
                    caption = "Swap processed snacks for whole foods"

                # 2. Sodium, salt, blood vessels, arterial workload, heart
                elif any(w in st_lower for w in ["sodium", "salt", "blood vessel", "vessel", "blood pressure", "heart", "arterial", "hypertension"]):
                    visual_goal = "Visually demonstrate the physiological impact of sodium and fats on circulation and vessel workload."
                    subject = "A close-up of a salty packaged snack on a counter alongside an artistic clinical visual of blood vessels"
                    setting = "On the kitchen island in clean, focused natural lighting"
                    camera_dir = "close-up tracking shot"
                    motion = "A hand sprinkles a pinch of salt beside the snack, while a clean artistic visual subtly illustrates blood vessel circulation and heart strain"
                    caption = "High sodium & fats increase vessel strain"

                # 3. Low fiber intake, slow digestion, bloating, and fluctuating energy levels (Explanation)
                elif any(w in st_lower for w in ["fiber", "bloating", "slow digestion", "slows digestion", "digestive tract"]) or ("digestion" in st_lower and not any(w in st_lower for w in ["insulin", "pancreas"])):
                    visual_goal = "Clean educational 3D digestive visualization showing food moving slowly through the digestive tract."
                    subject = "A clean educational 3D digestive visualization without people"
                    setting = "In a clean educational anatomical visualization space with focused soft studio lighting"
                    camera_dir = "smooth anatomical camera pan"
                    motion = "Low-fiber food matter moves with visually slower movement through the digestive tract, smoothly transitioning to a subtle visual metaphor of fluctuating energy levels"
                    caption = "Low fiber intake slows digestive transit"

                # 4. Pancreatic insulin release, glucose surge, and insulin resistance (Explanation)
                elif any(w in st_lower for w in ["pancreas", "insulin"]) or ("blood sugar" in st_lower and any(w in st_lower for w in ["insulin", "pancreas", "resistance", "spike", "fluctuation"])):
                    visual_goal = "Clean educational 3D biological visualization showing food-derived glucose entering bloodstream and pancreas releasing insulin."
                    subject = "A clean educational 3D biological visualization without people"
                    setting = "In a clean clinical visualization environment with neutral soft studio lighting"
                    camera_dir = "macro tracking shot with smooth anatomical camera pan"
                    motion = "Food-derived glucose molecules enter the bloodstream causing a noticeable rise in blood glucose levels, prompting the pancreas to release insulin in a clear visual representation of an active insulin response"
                    caption = "Blood sugar rises; pancreas releases insulin"

                # 5. Rapid blood sugar spikes / fluctuations WITHOUT insulin / pancreas (Explanation)
                elif any(w in st_lower for w in ["blood sugar", "fluctuation", "fluctuations", "surge"]):
                    visual_goal = "Clean educational 3D biological visualization showing glucose entering bloodstream and rapid blood sugar fluctuations."
                    subject = "A clean educational 3D biological visualization without people"
                    setting = "In a clean clinical visualization environment with neutral soft studio lighting"
                    camera_dir = "macro tracking shot"
                    motion = "Artistic biological visualization shows processed food breaking down, glucose molecules entering the bloodstream, causing blood sugar to rise rapidly before sharply fluctuating"
                    caption = "Processed ingredients trigger rapid blood sugar shifts"

                # 6. Immediate insulin response / sugar absorption
                elif any(w in st_lower for w in ["insulin", "sugar", "glucose", "carb", "candy", "sweet", "soda"]):
                    visual_goal = "Illustrate rapid glucose absorption and the body's natural insulin response."
                    subject = "A close-up of sweet packaged snacks on the kitchen counter alongside an artistic biological visual"
                    setting = "On the kitchen counter with natural soft daylight"
                    camera_dir = "close-up lateral slide"
                    motion = "Subtle lighting highlights sweet food textures while a clean artistic visual illustrates glucose absorption and insulin release"
                    caption = "Sugar surge triggers insulin response"

                # 7. Balanced meals / steady blood sugar solutions (when NOT matching low-fiber problems)
                elif any(w in st_lower for w in ["protein", "steady", "satiety", "stabilize", "balance", "balanced"]):
                    visual_goal = "Showcase balanced snack pairings that promote steady blood sugar and sustained satiety."
                    subject = "Close-up of the young adult's hands"
                    setting = "On the sunlit kitchen island"
                    camera_dir = "medium close-up, downward angle"
                    motion = "The hands pair crisp apple slices with natural curd and whole almonds, demonstrating a balanced plate"
                    caption = "Balanced meals keep blood sugar steady"

                # 8. Fats, triglycerides, cholesterol, liver
                elif any(w in st_lower for w in ["fat", "triglyceride", "cholesterol", "liver", "lipid"]):
                    visual_goal = "Illustrate how dietary fats and excess calories circulate and store in the body."
                    subject = "A close-up of rich, fried snacks on a plate with clear focus on food textures"
                    setting = "On the kitchen counter in natural daylight"
                    camera_dir = "macro close-up with gentle pan"
                    motion = "The camera slowly glides past rich, greasy snacks, highlighting texture before pulling back to a clean background"
                    caption = "Excess fats convert to triglycerides"

                # 9. Inflammation, fatigue, sluggishness, sleep
                elif any(w in st_lower for w in ["inflammation", "gut", "microbiome", "fatigue", "sluggish", "tired", "sleep", "craving", "headache"]):
                    visual_goal = "Depict the feeling of afternoon sluggishness, fatigue, or physical energy drop."
                    subject = char_subject
                    setting = "At a tidy home workspace in soft afternoon ambient lighting"
                    camera_dir = "medium profile shot, gentle push-in"
                    motion = "The person pauses at their desk, takes a tired breath, gently rubs their temples indicating an energy slump, and resets"
                    caption = "Can trigger inflammation & fatigue"

                # 10. Water & hydration
                elif any(w in st_lower for w in ["water", "hydrate", "hydration", "dehydration", "bottle", "thirst"]):
                    visual_goal = "Highlight hydration as an essential tool for managing daily appetite."
                    subject = char_subject
                    setting = "In the sunlit kitchen by the counter"
                    camera_dir = "medium shot with gentle camera motion"
                    motion = "The person pours fresh water from a glass carafe into a clear glass and takes a refreshing sip"
                    caption = "Stay hydrated to manage cravings"

                elif theme == "symptom":
                    visual_goal = "Demonstrate physical recovery and gentle relief."
                    subject = char_subject
                    setting = "In the comfortable living room in warm afternoon lighting"
                    camera_dir = "medium shot, steady composition"
                    motion = "The person stands up comfortably, moves with ease, and smiles toward the window"
                    caption = "Gentle movement aids recovery"

                else:
                    visual_goal = f"Present essential actionable context regarding {topic}."
                    subject = char_subject
                    setting = base_setting
                    camera_dir = "medium shot, steady"
                    motion = "The presenter delivers clear guidance with calm, natural hand gestures"
                    caption = "Actionable steps for better health"

            # Clean and sanitize attributes
            subject_clean = _sanitize_prompt_text(subject)
            setting_clean = _sanitize_prompt_text(setting)
            motion_clean = _sanitize_prompt_text(motion)
            caption_clean = _sanitize_prompt_text(caption)

            # Ensure subject starts lowercase when following 'shot of ' unless proper noun
            subject_for_prompt = subject_clean
            for pfx in ["A ", "The ", "Hands of "]:
                if subject_clean.startswith(pfx):
                    subject_for_prompt = pfx.lower() + subject_clean[len(pfx):]
                    break

            # Character continuity should ONLY be requested when the shot actually contains the recurring character
            is_no_person_shot = any(w in subject.lower() for w in ["without people", "without human", "no people", "no human", "3d medical", "3d biological", "3d digestive", "close-up of sweet", "close-up of rich"])
            has_character = (
                not is_no_person_shot
                and not any(w in subject.lower() for w in ["hands", "hand"])
                and (
                    char_desc in subject
                    or "recurring" in subject.lower()
                    or any(w in subject.lower() for w in ["person", "adult", "presenter"])
                )
            )
            continuity_clause = f"Continuity: The recurring character is a {char_desc}. " if has_character else ""

            # Build visual portion describing ONLY what visually happens
            if "close-up of the young adult's hands" in subject_clean.lower() or "hands of" in subject_clean.lower():
                clean_motion = motion_clean
                if clean_motion.lower().startswith("the person "):
                    clean_motion = clean_motion[11:].strip()
                elif clean_motion.lower().startswith("the hands "):
                    clean_motion = clean_motion[10:].strip()
                clean_motion = re.sub(r'^(gently\s+)?moves\b', r'\1moving', clean_motion, flags=re.IGNORECASE)
                clean_motion = re.sub(r'\breplaces\b', 'replacing', clean_motion, flags=re.IGNORECASE)
                visual_parts = [
                    f"Vertical 9:16 photorealistic cinematic close-up of the young adult's hands {clean_motion} "
                    f"on a clean wooden counter in natural daylight. Downward angle with warm natural lighting."
                ]
            elif is_no_person_shot:
                visual_parts = [
                    f"Vertical 9:16 {subject_clean}. {motion_clean}. {camera_dir} with neutral soft studio lighting and smooth biological motion."
                ]
            else:
                visual_parts = [f"Vertical 9:16 photorealistic cinematic shot of {subject_for_prompt}."]
                if setting_clean:
                    visual_parts.append(f"{setting_clean}.")
                if motion_clean:
                    visual_parts.append(f"{motion_clean}.")
                visual_parts.append(f"Framing: {camera_dir} with natural soft lighting.")
                if continuity_clause:
                    visual_parts.append(continuity_clause.strip())

            visual_portion = " ".join(visual_parts)

            # Assemble complete prompt with zero narration leakage and no quotation marks
            video_prompt = _assemble_clean_veo_prompt(
                visual_portion=visual_portion,
                narration=st_text,
                forbidden_leakage_strings=all_forbidden_strings,
                is_no_person=is_no_person_shot
            )

            audio_dir = (
                "Spoken narration delivered in a clear, warm, conversational doctor voiceover tone. "
                "Subtle, soothing ambient room acoustic in background; zero distracting sound effects or background dialogue."
            )

            shots.append(
                Shot(
                    shot_number=shot_num,
                    duration_seconds=dur,
                    script_stage_order=st_order,
                    script_stage_name=st_name,
                    script_text=st_text,
                    visual_goal=visual_goal,
                    visual_description=_sanitize_prompt_text(f"{subject_clean} in {setting_clean}. {motion_clean}."),
                    subject=subject_clean,
                    setting=setting_clean,
                    camera_direction=f"Vertical 9:16, {camera_dir}",
                    motion=motion_clean,
                    on_screen_text=caption_clean,
                    audio_direction=audio_dir,
                    video_generation_prompt=video_prompt,
                    validation_status=f"VALID ({dur}s <= 8.0s)"
                )
            )

        return ShotPlan(
            topic=topic,
            framework_id=reel_script.framework_id,
            framework_name=reel_script.framework_name,
            target_duration_seconds=target_duration,
            shot_count=shot_count,
            total_duration_seconds=round(sum(s.duration_seconds for s in shots), 1),
            cta=reel_script.cta,
            continuity_notes=continuity_notes,
            shots=shots,
            generation_source="deterministic_fallback",
            validation_status=f"VALID (all {shot_count} shots in [4.0s, 8.0s], narration timing verified)",
            feedback=feedback
        )

    def _generate_with_groq(
        self,
        reel_script: ReelScript,
        topic_analysis: Optional[TopicAnalysis],
        framework_metadata: Optional[Union[FrameworkSelection, Dict[str, Any]]],
        target_duration: float,
        shot_count: int,
        distributed_script: List[Dict[str, Any]],
        calculated_durations: List[float],
        feedback: Optional[str] = None
    ) -> Optional[ShotPlan]:
        """Groq LLM visual shot planner generating cinematic Veo prompts, visual descriptions,
        and faithful captions. Spoken script_text and durations are strictly application-controlled."""
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key, max_retries=1)

            shots_context = []
            for idx, item in enumerate(distributed_script):
                shots_context.append({
                    "shot_number": idx + 1,
                    "duration_seconds": calculated_durations[idx],
                    "framework_stage_order": item["stage_order"],
                    "framework_stage_name": item["stage_name"],
                    "script_text": item["text"]
                })

            feedback_directive = ""
            if feedback and feedback.strip():
                feedback_directive = f"\nUSER REVISION FEEDBACK TO INCORPORATE:\n\"{feedback.strip()}\"\nIncorporate this user feedback into visual goals, camera directions, and setting nuances while strictly maintaining the {shot_count}-shot structure, <=8.0s duration limit, and exact narration fidelity.\n"

            prompt = f"""You are an award-winning Creative Director specializing in short-form medical and educational reels.
Your job is to design the visual plan (what is SHOWN on screen) for an approved reel script suitable for Google Flow / Veo 3.1 Lite.{feedback_directive}

CRITICAL ARCHITECTURAL BOUNDARY:
- The Script Writer has ALREADY decided WHAT IS SAID.
- The application has ALREADY fixed the exact spoken script_text, stage, and duration for every shot.
- You must NOT rewrite, summarize, or modify the spoken narration.
- Your job is strictly to decide WHAT IS SHOWN (visuals, camera, motion, concise captions, and Veo prompts).

Topic: "{reel_script.topic}"
Framework #{reel_script.framework_id}: "{reel_script.framework_name}"
Total Video Target Duration: {target_duration}s
Approved Call-To-Action (CTA): "{reel_script.cta}"

PRE-ALLOCATED SHOT SCRIPT ASSIGNMENTS (DO NOT CHANGE):
{json.dumps(shots_context, indent=2)}

CRITICAL SHOT-BY-SHOT VISUAL DIRECTIVES:
- SHOT 1 (Hook): Strong human hook. A young adult with short dark hair, wearing a simple neutral-colored casual shirt, in a bright modern kitchen, holding an obviously processed snack package thoughtfully, looking toward the camera with a curious, questioning expression. Subtle camera push-in. Simple attention-grabbing opening. DO NOT put narration text inside prompt.
- SHOT 2 (Blood Sugar / Insulin): Clean educational 3D biological visualization without people. Food-derived glucose molecules entering bloodstream, noticeable rise in blood glucose levels, and pancreas releasing insulin in a clear visual representation of an active insulin response. DO NOT show digestive tract, slow digestion, low fiber, bloating, people, horror, organ damage, or disease imagery. Single achievable 7–8 second Veo clip.
- SHOT 3 (Low Fiber / Digestion / Energy): Clean educational 3D digestive visualization without people. Low-fiber food moving through the digestive tract with visually slower movement, with an optional simple visual transition representing inconsistent energy levels. DO NOT show pancreas, insulin release, blood glucose, people, horror, or organ damage. Single achievable 7–8 second Veo clip.
- SHOT 4 (Practical Swap): Close-up of the young adult's hands moving a processed snack aside and replacing it with fruit and/or nuts on a clean wooden kitchen counter bathed in warm natural daylight. Downward angle. Use EXACTLY: 'close-up of the young adult's hands...'. NEVER describe hair, shirt, or clothing when only hands are visible. Simple, realistic action within 7–8 seconds.
- SHOT 5 (Consultation + CTA): The recurring young adult with short dark hair, wearing a simple neutral-colored casual shirt, in a bright modern kitchen, facing the camera naturally with a calm, caring, and trustworthy expression and relaxed body language. Simple, reassuring framing. No doctor shown, no medical diagnosis, no extra claims. DO NOT put the CTA into the visual description.
- CHARACTER CONSISTENCY: For shots with the character (Shots 1, 4, 5), use the EXACT character descriptor: 'a young adult with short dark hair, wearing a simple neutral-colored casual shirt, in a bright modern kitchen'. Shots 2 and 3 must have NO character and NO person.
- ABSOLUTE SEPARATION: In 'video_generation_prompt', describe ONLY visual scene, subject, setting, motion, camera direction, lighting, and style. DO NOT write narration sentences, dialogue, voiceover instructions, or text captions into 'video_generation_prompt'. Spoken voiceover will be attached automatically by the application.

REQUIREMENTS FOR EACH OF THE {shot_count} SHOTS:
1. visual_goal: Specific communication purpose of this shot's visuals.
2. visual_description: Concrete description of what appears on screen. Avoid vague generalizations.
3. subject: Main person, object, or action in focus (maintain consistent recurring character across shots where appropriate).
4. setting: Realistic, authentic environment (e.g. bright modern kitchen, clean visualization space).
5. camera_direction: Camera framing and movement (e.g., 'Vertical 9:16, eye-level medium shot, subtle push-in').
6. motion: Concrete physical action occurring during the shot.
7. on_screen_text: Concise, natural on-screen caption (under 8 words) faithfully reflecting that shot's spoken narration.
   - Prefer natural, clear phrasing.
   - AVOID overly compressed telegram-style shorthand.
   - NEVER use arrows like "↑" or "↓".
   - In the final shot, if spoken narration contains consultation guidance, reflect it (e.g. "Persistent symptoms? Consult a physician."); otherwise feature the CTA.
8. audio_direction: Voiceover delivery tone and subtle ambient room acoustics (no distracting sound effects).
9. video_generation_prompt: Complete visual prompt for Google Flow / Veo 3.1 Lite describing the visual scene (subject, setting, motion, camera, lighting, realism). DO NOT include voiceover instructions or dialogue here.

CRITICAL MEDICAL SAFETY & CAPTION RULES:
- NEVER introduce medical claims that are absent from the ReelScript.
- Specifically NEVER use phrases like 'insulin overload', 'gut microbiome diversity drops', 'encourages fat storage', 'guaranteed harm', or any diagnostic claims.
- on_screen_text must represent ONLY the exact concepts in that shot's script_text. Do not introduce concepts from later or earlier shots. Do not turn explanatory narration into a solution caption.

Return valid JSON with:
- continuity_notes: string describing recurring character/setting consistency across the reel
- shots: array of {shot_count} objects with keys:
  "shot_number", "visual_goal", "visual_description", "subject", "setting",
  "camera_direction", "motion", "on_screen_text", "audio_direction", "video_generation_prompt"
"""

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
                            {"role": "system", "content": "You are an expert visual director producing structured JSON for AI video generation."},
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

            shots_data = data.get("shots", [])
            if len(shots_data) != shot_count:
                return None

            topic_lower = reel_script.topic.lower()
            is_nutrition = any(k in topic_lower for k in ["food", "eat", "diet", "nutrition", "sugar", "curd", "snack", "chip", "drink", "meal"])
            char_desc_stable = "young adult with short dark hair, wearing a simple neutral-colored casual shirt, in a bright modern kitchen"

            all_forbidden_strings = [
                reel_script.cta or "",
                reel_script.full_script_with_cta,
                reel_script.full_script_without_cta
            ] + [item["text"] for item in distributed_script]

            built_shots = []
            for idx, raw_shot in enumerate(shots_data):
                dist_item = distributed_script[idx]
                dur = calculated_durations[idx]
                st_lower = dist_item["text"].lower()
                is_final_shot = (idx == shot_count - 1 and idx > 0)

                # User Requirement Enforcement for Specific Shots:
                # Shot 1 (Hook): Strong human hook
                if idx == 0 and is_nutrition:
                    raw_shot["subject"] = f"A {char_desc_stable}"
                    raw_shot["setting"] = "In a bright modern kitchen with morning sunlight streaming through windows"
                    raw_shot["motion"] = "The person holds an obviously processed snack package such as chips thoughtfully, looking toward the camera with a curious, questioning expression"
                    raw_shot["camera_direction"] = "Vertical 9:16, eye-level medium shot, subtle camera push-in"
                    raw_shot["visual_description"] = f"A {char_desc_stable} in a bright modern kitchen with morning sunlight streaming through windows. The person holds an obviously processed snack package such as chips thoughtfully, looking toward the camera with a curious, questioning expression."
                    raw_shot["video_generation_prompt"] = (
                        f"Vertical 9:16 photorealistic cinematic shot of a {char_desc_stable}. "
                        "In a bright modern kitchen with morning sunlight streaming through windows, the person holds an obviously processed "
                        "snack package thoughtfully, looking toward the camera with a curious, questioning expression. "
                        "Eye-level medium shot with a subtle camera push-in and natural soft lighting."
                    )

                # Shot 2 (Blood Sugar / Insulin): Clean biological visualization without people
                is_blood_sugar_shot = (
                    (shot_count == 5 and idx == 1)
                    or (("pancreas" in st_lower or "insulin" in st_lower or "blood sugar" in st_lower) and "fiber" not in st_lower and "digestion" not in st_lower and "bloat" not in st_lower)
                ) and not is_final_shot
                if is_blood_sugar_shot and is_nutrition:
                    raw_shot["subject"] = "A clean educational 3D biological visualization without people"
                    raw_shot["setting"] = "In a clean clinical visualization environment with neutral soft studio lighting"
                    raw_shot["motion"] = "Food-derived glucose molecules enter the bloodstream causing a noticeable rise in blood glucose levels, prompting the pancreas to release insulin in a clear visual representation of an active insulin response"
                    raw_shot["camera_direction"] = "Vertical 9:16, macro tracking shot with smooth anatomical camera pan"
                    raw_shot["visual_description"] = "A clean educational 3D biological visualization without people. Food-derived glucose molecules enter the bloodstream causing a noticeable rise in blood glucose levels, prompting the pancreas to release insulin in a clear visual representation of an active insulin response."
                    raw_shot["video_generation_prompt"] = (
                        "Vertical 9:16 clean educational 3D biological visualization without people. "
                        "Food-derived glucose molecules enter the bloodstream causing a noticeable rise in blood glucose levels, "
                        "prompting the pancreas to release insulin in a clear visual representation of an active insulin response. "
                        "Macro tracking shot with smooth anatomical camera pan, neutral soft studio lighting, and smooth biological motion."
                    )
                    raw_shot["on_screen_text"] = "Blood sugar rises; pancreas releases insulin"

                # Shot 3 (Low Fiber / Digestion / Energy): Clean digestive visualization without people
                is_digestion_shot = (
                    (shot_count == 5 and idx == 2)
                    or ("fiber" in st_lower or "digestion" in st_lower or "bloat" in st_lower or "slows digestion" in st_lower)
                ) and not is_final_shot
                if is_digestion_shot and is_nutrition:
                    raw_shot["subject"] = "A clean educational 3D digestive visualization without people"
                    raw_shot["setting"] = "In a clean educational anatomical visualization space with focused soft studio lighting"
                    raw_shot["motion"] = "Low-fiber food matter moves with visually slower movement through the digestive tract, smoothly transitioning to a subtle visual metaphor of fluctuating energy levels"
                    raw_shot["camera_direction"] = "Vertical 9:16, smooth anatomical camera pan"
                    raw_shot["visual_description"] = "A clean educational 3D digestive visualization without people. Low-fiber food matter moves with visually slower movement through the digestive tract, smoothly transitioning to a subtle visual metaphor of fluctuating energy levels."
                    raw_shot["video_generation_prompt"] = (
                        "Vertical 9:16 clean educational 3D digestive visualization without people. "
                        "Low-fiber food matter moves with visually slower movement through the digestive tract, "
                        "smoothly transitioning to a subtle visual metaphor of fluctuating energy levels. "
                        "Smooth anatomical camera pan with soft clinical lighting and gentle biological motion."
                    )
                    raw_shot["on_screen_text"] = "Low fiber intake slows digestive transit"

                # Shot 4 (Practical Swap): Close-up of the young adult's hands
                is_swap_shot = (
                    (shot_count == 5 and idx == 3)
                    or any(w in st_lower for w in ["swap", "fruit", "nuts", "whole food", "replace"])
                ) and not is_final_shot
                if is_swap_shot and is_nutrition:
                    raw_shot["subject"] = "Close-up of the young adult's hands"
                    raw_shot["setting"] = "At a clean wooden kitchen counter bathed in warm natural daylight"
                    raw_shot["motion"] = "moving a bag of processed snacks aside and replacing it with a wholesome bowl of fresh blueberries, apple slices, and raw walnuts"
                    raw_shot["camera_direction"] = "Vertical 9:16, medium close-up, downward 45-degree angle"
                    raw_shot["visual_description"] = "Close-up of the young adult's hands moving a bag of processed snacks aside and replacing it with a wholesome bowl of fresh blueberries, apple slices, and raw walnuts on a clean wooden kitchen counter bathed in warm natural daylight."
                    raw_shot["video_generation_prompt"] = (
                        "Vertical 9:16 photorealistic cinematic close-up of the young adult's hands "
                        "moving a bag of processed snacks aside and replacing it with a wholesome bowl "
                        "of fresh blueberries, apple slices, and raw walnuts on a clean wooden kitchen counter in natural daylight. "
                        "Downward angle with warm natural lighting."
                    )
                    raw_shot["on_screen_text"] = "Swap processed snacks for whole foods"

                # Final Shot (Consultation + CTA): Recurring character facing camera naturally
                if is_final_shot and is_nutrition:
                    raw_shot["subject"] = f"The recurring {char_desc_stable}"
                    raw_shot["setting"] = "In the bright modern kitchen by the counter with natural window lighting"
                    raw_shot["motion"] = "The person faces the camera naturally with a calm, caring, and trustworthy expression, maintaining relaxed body language"
                    raw_shot["camera_direction"] = "Vertical 9:16, eye-level medium shot, steady composition"
                    raw_shot["visual_description"] = f"The recurring {char_desc_stable}. In the bright modern kitchen by the counter with natural window lighting, the person faces the camera naturally with a calm, caring, and trustworthy expression, maintaining relaxed body language."
                    raw_shot["video_generation_prompt"] = (
                        f"Vertical 9:16 photorealistic cinematic shot of the recurring {char_desc_stable}. "
                        "In the bright modern kitchen by the counter with natural window lighting, the person faces the camera naturally "
                        "with a calm, caring, and trustworthy expression, maintaining relaxed body language. "
                        "Eye-level medium shot with warm natural window lighting."
                    )

                # Format production-ready visual prompt with zero narration leakage
                raw_vgp = raw_shot.get("video_generation_prompt", "").strip()
                if not raw_vgp or not raw_vgp.startswith("Vertical 9:16"):
                    raw_vgp = (
                        f"Vertical 9:16 photorealistic cinematic shot of {raw_shot.get('subject', 'a relatable person')}. "
                        f"In {raw_shot.get('setting', 'a contemporary space')}, {raw_shot.get('motion', 'natural movement')}. "
                        f"Camera: {raw_shot.get('camera_direction', 'medium shot')} with natural lighting."
                    )

                is_no_person_shot = is_blood_sugar_shot or is_digestion_shot or any(w in str(raw_shot.get("subject", "")).lower() for w in ["without people", "without human", "no people", "no human", "3d medical", "3d biological", "3d digestive"])

                clean_vgp = _assemble_clean_veo_prompt(
                    visual_portion=raw_vgp,
                    narration=dist_item["text"],
                    forbidden_leakage_strings=all_forbidden_strings,
                    is_no_person=is_no_person_shot
                )

                # Clean captions: ensure faithful, natural wording without shorthand symbols or disallowed claims
                raw_caption = raw_shot.get("on_screen_text", dist_item["text"][:45]).strip()
                raw_caption = raw_caption.replace("↑", " rises ").replace("↓", " drops ")
                raw_caption = re.sub(r'\s+', ' ', raw_caption).strip()
                for disallowed in DISALLOWED_VISUAL_CLAIMS:
                    if disallowed in raw_caption.lower():
                        raw_caption = dist_item["text"][:40]

                # In the final shot, ensure caption cleanly reflects consultation guidance or approved CTA
                if idx == shot_count - 1:
                    if any(w in dist_item["text"].lower() for w in ["physician", "doctor", "consult", "healthcare", "clinic", "symptom"]):
                        raw_caption = "Persistent symptoms? Consult a physician."
                    elif reel_script.cta:
                        raw_caption = reel_script.cta

                raw_caption = _sanitize_prompt_text(raw_caption)

                mot = _sanitize_prompt_text(raw_shot.get("motion", "Delivers presentation calmly"))
                v_desc = _sanitize_prompt_text(raw_shot.get("visual_description", f"{raw_shot.get('subject', 'Presenter')} in {raw_shot.get('setting', 'Studio')}. {mot}."))

                built_shots.append(
                    Shot(
                        shot_number=idx + 1,
                        duration_seconds=dur,  # Strictly application-controlled
                        script_stage_order=dist_item["stage_order"],  # Strictly application-controlled
                        script_stage_name=dist_item["stage_name"],  # Strictly application-controlled
                        script_text=dist_item["text"],  # Strictly application-controlled (100% exact copy)
                        visual_goal=raw_shot.get("visual_goal", "Visually support the spoken message.").strip(),
                        visual_description=v_desc,
                        subject=_sanitize_prompt_text(raw_shot.get("subject", "Relatable presenter")),
                        setting=_sanitize_prompt_text(raw_shot.get("setting", "Studio setting")),
                        camera_direction=raw_shot.get("camera_direction", "Vertical 9:16, medium shot, slow push-in").strip(),
                        motion=mot,
                        on_screen_text=raw_caption,
                        audio_direction=raw_shot.get("audio_direction", "Clear conversational voiceover with subtle ambient acoustic").strip(),
                        video_generation_prompt=clean_vgp,
                        validation_status=f"VALID ({dur}s <= 8.0s)"
                    )
                )

            return ShotPlan(
                topic=reel_script.topic,
                framework_id=reel_script.framework_id,
                framework_name=reel_script.framework_name,
                target_duration_seconds=target_duration,
                shot_count=shot_count,
                total_duration_seconds=round(sum(s.duration_seconds for s in built_shots), 1),
                cta=reel_script.cta,
                continuity_notes=data.get("continuity_notes", "Consistent vertical 9:16 aesthetic, warm realistic lighting, and natural character progression."),
                shots=built_shots,
                generation_source="groq",
                validation_status=f"VALID (all {shot_count} shots in [4.0s, 8.0s], narration timing verified)",
                feedback=feedback
            )

        except Exception as e:
            raise e
