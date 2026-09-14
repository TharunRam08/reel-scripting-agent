"""Caption generation and styling infrastructure for vertical 9:16 short-form reels.

Derives captions STRICTLY from approved ShotPlan/ReelScript narration.
Guarantees 100% text fidelity: zero invented words, summaries, or paraphrasing.
"""

import os
import re
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from app.agents.shot_planner import ShotPlan, Shot
from app.agents.script_writer import ReelScript


class CaptionEntry(BaseModel):
    """A timed subtitle line locked to a specific shot's duration."""
    index: int = Field(..., description="1-indexed caption counter.")
    shot_number: int = Field(..., description="The corresponding shot number.")
    start_seconds: float = Field(..., description="Start timestamp in seconds.")
    end_seconds: float = Field(..., description="End timestamp in seconds.")
    text: str = Field(..., description="Exact narration text excerpt.")

    @field_validator("text")
    @classmethod
    def validate_text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Caption text cannot be empty.")
        return v.strip()

    @field_validator("end_seconds")
    @classmethod
    def validate_timestamps(cls, v: float, info) -> float:
        start = info.data.get("start_seconds")
        if start is not None and v <= start:
            raise ValueError(f"end_seconds ({v}) must be greater than start_seconds ({start})")
        return round(v, 3)

    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 3)


class BurnedCaptionConfig(BaseModel):
    """Styling configuration for burning readable captions onto 9:16 vertical reels."""
    font_name: str = Field("Arial", description="Font family for burned subtitles.")
    font_size: int = Field(32, description="Font size scaled for 720x1280 resolution.")
    primary_color: str = Field("&H00FFFFFF", description="ASS hex color: White (&H00FFFFFF).")
    outline_color: str = Field("&H00000000", description="ASS hex color: Black (&H00000000).")
    outline_width: int = Field(3, description="Outline border width for crisp legibility.")
    shadow_depth: int = Field(1, description="Soft drop shadow depth.")
    alignment: int = Field(2, description="ASS alignment: 2 = bottom center.")
    margin_vertical: int = Field(180, description="Bottom vertical margin in pixels (above mobile UI safe zone).")
    words_per_caption_target: int = Field(10, description="Target words per subtitle line.")
    split_sentences_only: bool = Field(True, description="When True, partitions shot narration by sentence boundaries.")


class CaptionGenerator:
    """Generates timed subtitles exclusively from approved narration."""

    def __init__(self, config: Optional[BurnedCaptionConfig] = None):
        self.config = config or BurnedCaptionConfig()

    def generate_from_shot_plan(
        self,
        shot_plan: Union[ShotPlan, Any],
        actual_clip_durations: Optional[List[float]] = None
    ) -> List[CaptionEntry]:
        """Derives sequential timed captions from each shot's approved script_text or narration.
        
        If actual_clip_durations are provided (e.g. from ffprobe inspection),
        captions use the actual clip timing; otherwise, shot.duration_seconds is used.
        """
        entries: List[CaptionEntry] = []
        current_time = 0.0
        caption_idx = 1

        for i, shot in enumerate(shot_plan.shots):
            # Determine shot duration: actual inspected clip duration takes priority if provided
            if actual_clip_durations and i < len(actual_clip_durations):
                shot_duration = actual_clip_durations[i]
            else:
                shot_duration = shot.duration_seconds

            shot_text = getattr(shot, "script_text", getattr(shot, "narration", "")).strip()
            if not shot_text:
                current_time += shot_duration
                continue

            # Split narration into natural, readable subtitle phrases without modifying a single word
            phrases = self._split_into_phrases(
                shot_text,
                target_words=self.config.words_per_caption_target,
                by_sentence=self.config.split_sentences_only
            )
            total_words = sum(len(p.split()) for p in phrases)

            shot_start = current_time
            elapsed_in_shot = 0.0

            for p_idx, phrase in enumerate(phrases):
                phrase_word_count = len(phrase.split())
                # Allocate duration proportionally by word count
                if total_words > 0:
                    weight = phrase_word_count / total_words
                    phrase_dur = shot_duration * weight
                else:
                    phrase_dur = shot_duration / len(phrases)

                p_start = shot_start + elapsed_in_shot
                p_end = p_start + phrase_dur

                # Last phrase in shot should precisely hit the end of the shot
                if p_idx == len(phrases) - 1:
                    p_end = shot_start + shot_duration

                entries.append(
                    CaptionEntry(
                        index=caption_idx,
                        shot_number=shot.shot_number,
                        start_seconds=round(p_start, 3),
                        end_seconds=round(p_end, 3),
                        text=phrase
                    )
                )
                caption_idx += 1
                elapsed_in_shot += phrase_dur

            current_time += shot_duration

        return entries

    def _split_into_phrases(self, text: str, target_words: int = 10, by_sentence: bool = True) -> List[str]:
        """Splits text into readable chunks while guaranteeing:
        ' '.join(result) == text exactly.
        """
        clean = text.strip()
        if not clean:
            return []

        if by_sentence:
            # Split by sentence boundaries while preserving exact characters
            raw_sents = re.split(r'(?<=[.!?])\s+', clean)
            sents = [s.strip() for s in raw_sents if s.strip()]
            if sents:
                return sents

        words = clean.split()
        if len(words) <= target_words:
            return [clean]

        phrases: List[str] = []
        current_chunk: List[str] = []

        for word in words:
            current_chunk.append(word)
            has_punct = bool(re.search(r'[,.!?;:]$', word))
            if (len(current_chunk) >= target_words) or (len(current_chunk) >= 5 and has_punct):
                phrases.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            if phrases and len(current_chunk) <= 2:
                phrases[-1] = f"{phrases[-1]} {' '.join(current_chunk)}"
            else:
                phrases.append(" ".join(current_chunk))

        return phrases

    @staticmethod
    def format_timestamp_srt(seconds: float) -> str:
        """Converts float seconds to SRT timestamp format: HH:MM:SS,mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            secs += 1
            millis = 0
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def format_timestamp_ass(seconds: float) -> str:
        """Converts float seconds to ASS timestamp format: H:MM:SS.cc"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int(round((seconds - int(seconds)) * 100))
        if centis >= 100:
            secs += 1
            centis = 0
        return f"{hrs:01d}:{mins:02d}:{secs:02d}.{centis:02d}"

    def generate_srt(self, entries: List[CaptionEntry]) -> str:
        """Serializes caption entries into standard SubRip (.srt) format."""
        blocks = []
        for e in entries:
            start_fmt = self.format_timestamp_srt(e.start_seconds)
            end_fmt = self.format_timestamp_srt(e.end_seconds)
            blocks.append(f"{e.index}\n{start_fmt} --> {end_fmt}\n{e.text}\n")
        return "\n".join(blocks).strip() + "\n"

    def generate_ass(
        self,
        entries: List[CaptionEntry],
        video_width: int = 720,
        video_height: int = 1280
    ) -> str:
        """Serializes caption entries into Advanced SubStation Alpha (.ass) format
        with customized mobile-friendly styling for 9:16 vertical reels."""
        cfg = self.config

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelCaptions,{cfg.font_name},{cfg.font_size},{cfg.primary_color},&H000000FF,{cfg.outline_color},{cfg.outline_color},-1,0,0,0,100,100,0,0,1,{cfg.outline_width},{cfg.shadow_depth},{cfg.alignment},40,40,{cfg.margin_vertical},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogues = []
        for e in entries:
            s_fmt = self.format_timestamp_ass(e.start_seconds)
            e_fmt = self.format_timestamp_ass(e.end_seconds)
            # Escape curly braces in text if any
            clean_text = e.text.replace("{", "\\{").replace("}", "\\}")
            dialogues.append(f"Dialogue: 0,{s_fmt},{e_fmt},ReelCaptions,,0,0,0,,{clean_text}")

        return header + "\n".join(dialogues) + "\n"

    def write_srt_file(self, entries: List[CaptionEntry], output_path: str) -> str:
        """Writes SRT subtitle file to disk and returns the absolute path."""
        content = self.generate_srt(entries)
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return abs_path

    def write_ass_file(
        self,
        entries: List[CaptionEntry],
        output_path: str,
        video_width: int = 720,
        video_height: int = 1280
    ) -> str:
        """Writes ASS subtitle file to disk and returns the absolute path."""
        content = self.generate_ass(entries, video_width=video_width, video_height=video_height)
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return abs_path

    @staticmethod
    def build_subtitles_filter(subtitles_path: str) -> str:
        """Builds a robust, Windows-safe FFmpeg subtitles filter string.
        Handles backslashes and drive letter colons correctly for FFmpeg.
        """
        abs_path = os.path.abspath(subtitles_path).replace("\\", "/")
        # On Windows, escape colon after drive letter: C:/path -> C\:/path
        if len(abs_path) > 1 and abs_path[1] == ":":
            escaped = abs_path[0] + "\\:" + abs_path[2:]
        else:
            escaped = abs_path
        
        # If it's an .ass file, use ass= filter, otherwise use subtitles= filter
        if abs_path.lower().endswith(".ass"):
            return f"ass='{escaped}'"
        return f"subtitles='{escaped}'"
