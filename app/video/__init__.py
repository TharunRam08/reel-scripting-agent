"""Local video finalization infrastructure for short-form video reels."""

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

__all__ = [
    "VideoAssembler",
    "ClipMetadata",
    "AssemblyPlan",
    "AssemblyResult",
    "detect_ffmpeg",
    "detect_ffprobe",
    "CaptionGenerator",
    "CaptionEntry",
    "BurnedCaptionConfig",
]
