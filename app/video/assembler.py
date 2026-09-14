"""Local video assembly infrastructure for short-form video reels.

Concatenates sequential video clips, normalizes resolution into 9:16 (default 720x1280),
handles scale/crop without stretching, normalizes audio streams (H.264 + AAC),
burns approved narration captions, and exposes duration metrics.
"""

import os
import sys
import json
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


def detect_ffmpeg() -> str:
    """Detects the path to the ffmpeg executable on the system.
    Searches standard PATH and common installation locations (e.g. WinGet Gyan.FFmpeg).
    Raises FileNotFoundError if ffmpeg is unavailable.
    """
    cmd = shutil.which("ffmpeg")
    if cmd:
        return os.path.abspath(cmd)

    # Check Windows user winget packages location
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_dir = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
            if os.path.exists(winget_dir):
                for root, dirs, files in os.walk(winget_dir):
                    if "ffmpeg.exe" in files:
                        candidate = os.path.join(root, "ffmpeg.exe")
                        return os.path.abspath(candidate)

    raise FileNotFoundError(
        "FFmpeg executable not found on system PATH or standard locations. "
        "Please ensure FFmpeg is installed and accessible."
    )


def detect_ffprobe() -> str:
    """Detects the path to the ffprobe executable on the system.
    Searches standard PATH and common installation locations (e.g. WinGet Gyan.FFmpeg).
    Raises FileNotFoundError if ffprobe is unavailable.
    """
    cmd = shutil.which("ffprobe")
    if cmd:
        return os.path.abspath(cmd)

    # Check Windows user winget packages location
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_dir = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
            if os.path.exists(winget_dir):
                for root, dirs, files in os.walk(winget_dir):
                    if "ffprobe.exe" in files:
                        candidate = os.path.join(root, "ffprobe.exe")
                        return os.path.abspath(candidate)

    raise FileNotFoundError(
        "FFprobe executable not found on system PATH or standard locations. "
        "Please ensure FFprobe is installed and accessible."
    )


class ClipMetadata(BaseModel):
    """Inspected technical metadata for an individual video clip."""
    file_path: str = Field(..., description="Absolute path to the video clip.")
    duration_seconds: float = Field(..., description="Exact duration in seconds.")
    width: int = Field(..., description="Video stream pixel width.")
    height: int = Field(..., description="Video stream pixel height.")
    aspect_ratio: str = Field(..., description="Aspect ratio description (e.g., 9:16, 16:9).")
    has_audio: bool = Field(False, description="Whether an audio stream is present.")
    audio_codec: Optional[str] = Field(None, description="Audio codec if present.")
    video_codec: str = Field(..., description="Video codec (e.g., h264, hevc).")
    frame_rate: float = Field(24.0, description="Calculated frame rate in fps.")

    @property
    def is_vertical_9_16(self) -> bool:
        """Determines if the clip matches 9:16 vertical within 1% tolerance."""
        if self.height <= 0:
            return False
        ratio = self.width / self.height
        target = 9.0 / 16.0
        return abs(ratio - target) < 0.02


class AssemblyPlan(BaseModel):
    """Execution plan for concatenating and finalizing clips into a reel."""
    clips: List[ClipMetadata] = Field(..., description="Sequential clip metadata in shot order.")
    target_width: int = Field(720, description="Target pixel width.")
    target_height: int = Field(1280, description="Target pixel height.")
    target_duration_seconds: float = Field(45.0, description="Configurable target reel duration in seconds.")
    total_clip_duration_seconds: float = Field(..., description="Sum of actual clip durations.")
    duration_delta_seconds: float = Field(..., description="total_clip_duration_seconds - target_duration_seconds.")
    output_path: str = Field(..., description="Target output MP4 file path.")
    captions_file: Optional[str] = Field(None, description="Path to burned subtitle file if configured.")
    ffmpeg_command: List[str] = Field(..., description="The complete self-contained FFmpeg command.")


class AssemblyResult(BaseModel):
    """Result of video assembly execution."""
    success: bool = Field(..., description="Whether FFmpeg completed successfully.")
    output_path: str = Field(..., description="Path to generated output file.")
    duration_seconds: float = Field(..., description="Final reel duration in seconds.")
    duration_delta_seconds: float = Field(..., description="Difference from target duration.")
    file_size_bytes: int = Field(0, description="File size of output video.")
    ffmpeg_command: List[str] = Field(..., description="Command that was executed.")
    stderr: str = Field("", description="Captured FFmpeg stderr output.")


class VideoAssembler:
    """Assembles sequential video clips into a normalized vertical 9:16 reel."""

    DEFAULT_TARGET_WIDTH = 720
    DEFAULT_TARGET_HEIGHT = 1280
    DEFAULT_TARGET_DURATION = 45.0

    def __init__(
        self,
        target_width: int = DEFAULT_TARGET_WIDTH,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        target_duration_seconds: float = DEFAULT_TARGET_DURATION,
        ffmpeg_bin: Optional[str] = None,
        ffprobe_bin: Optional[str] = None
    ):
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"Target dimensions must be positive, got {target_width}x{target_height}")
        if target_duration_seconds <= 0:
            raise ValueError(f"Target duration must be positive, got {target_duration_seconds}")

        self.target_width = target_width
        self.target_height = target_height
        self.target_duration_seconds = target_duration_seconds
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin

    @property
    def ffmpeg_bin(self) -> str:
        if not self._ffmpeg_bin:
            self._ffmpeg_bin = detect_ffmpeg()
        return self._ffmpeg_bin

    @property
    def ffprobe_bin(self) -> str:
        if not self._ffprobe_bin:
            self._ffprobe_bin = detect_ffprobe()
        return self._ffprobe_bin

    def inspect_clip(self, clip_path: str) -> ClipMetadata:
        """Uses ffprobe to inspect actual duration, dimensions, and audio presence of a clip."""
        abs_path = os.path.abspath(clip_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Video clip not found at: {abs_path}")

        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-show_entries", "stream=index,codec_name,codec_type,width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            abs_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed to inspect clip '{abs_path}': {result.stderr.strip()}")

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        format_info = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not video_stream:
            raise ValueError(f"No video stream found in '{abs_path}'")

        # Determine duration: prefer format duration, fallback to stream duration
        duration_str = format_info.get("duration") or video_stream.get("duration") or "0"
        duration = float(duration_str)

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        video_codec = video_stream.get("codec_name", "unknown")

        # Calculate frame rate
        fps_str = video_stream.get("r_frame_rate", "24/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 24.0
        except Exception:
            fps = 24.0

        has_audio = audio_stream is not None
        audio_codec = audio_stream.get("codec_name") if audio_stream else None

        # Describe aspect ratio
        if height > 0:
            ratio = width / height
            if abs(ratio - (9.0 / 16.0)) < 0.02:
                aspect_ratio = "9:16"
            elif abs(ratio - (16.0 / 9.0)) < 0.02:
                aspect_ratio = "16:9"
            elif abs(ratio - 1.0) < 0.02:
                aspect_ratio = "1:1"
            else:
                aspect_ratio = f"{width}:{height}"
        else:
            aspect_ratio = "unknown"

        return ClipMetadata(
            file_path=abs_path,
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            has_audio=has_audio,
            audio_codec=audio_codec,
            video_codec=video_codec,
            frame_rate=round(fps, 2)
        )

    def inspect_clips(self, clip_paths: List[str]) -> List[ClipMetadata]:
        """Validates that all required clips exist in order and inspects their metadata."""
        if not clip_paths:
            raise ValueError("clip_paths list cannot be empty.")
        
        inspected = []
        for p in clip_paths:
            inspected.append(self.inspect_clip(p))
        return inspected

    def build_assembly_command(
        self,
        clip_paths: List[str],
        output_path: str,
        burned_subtitles_path: Optional[str] = None,
        inspected_clips: Optional[List[ClipMetadata]] = None
    ) -> List[str]:
        """Constructs the deterministic FFmpeg assembly command without executing.
        
        Enforces:
        - Exact sequential shot ordering
        - Scale/crop to target 9:16 resolution without stretching
        - Resampling audio to 48kHz stereo AAC (synthesizes silence if audio track is missing)
        - Web-compatible H.264 video with yuv420p pixel format
        - Optional burned captions using Windows-safe subtitle filter
        """
        if not clip_paths:
            raise ValueError("clip_paths cannot be empty.")

        abs_output = os.path.abspath(output_path)
        num_clips = len(clip_paths)
        w = self.target_width
        h = self.target_height

        cmd = [self.ffmpeg_bin, "-y"]

        # Add all input clips
        for p in clip_paths:
            cmd.extend(["-i", os.path.abspath(p)])

        # Construct filter_complex
        filter_parts = []

        # For each clip, scale and crop to 9:16 without stretching:
        # scale=w:h:force_original_aspect_ratio=increase,crop=w:h:(in_w-w)/2:(in_h-h)/2,setsar=1
        for i in range(num_clips):
            v_filter = (
                f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}:(in_w-{w})/2:(in_h-{h})/2,setsar=1,fps=24[v{i}]"
            )
            filter_parts.append(v_filter)

            # Audio stream handling: normalize to 48kHz stereo
            # If inspected metadata is available and clip lacks audio, synthesize silence
            clip_has_audio = True
            if inspected_clips and i < len(inspected_clips):
                clip_has_audio = inspected_clips[i].has_audio

            if clip_has_audio:
                a_filter = f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]"
            else:
                a_filter = f"anullsrc=r=48000:cl=stereo,atrim=duration={inspected_clips[i].duration_seconds if inspected_clips else 5.0}[a{i}]"
            filter_parts.append(a_filter)

        # Concatenate all normalized video and audio streams
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(num_clips))
        filter_parts.append(f"{concat_inputs}concat=n={num_clips}:v=1:a=1[vconcat][aconcat]")

        # Apply burned captions if provided
        final_v_label = "[vconcat]"
        if burned_subtitles_path:
            from app.video.captions import CaptionGenerator
            sub_filter = CaptionGenerator.build_subtitles_filter(burned_subtitles_path)
            filter_parts.append(f"[vconcat]{sub_filter}[vfinal]")
            final_v_label = "[vfinal]"

        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(["-map", final_v_label, "-map", "[aconcat]"])

        # Encoding settings for pristine web compatibility (H.264 + AAC)
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            abs_output
        ])

        return cmd

    def plan_assembly(
        self,
        clip_paths: List[str],
        output_path: str,
        burned_subtitles_path: Optional[str] = None
    ) -> AssemblyPlan:
        """Inspects all clips and constructs an AssemblyPlan detailing actual duration vs target."""
        inspected = self.inspect_clips(clip_paths)
        total_duration = round(sum(c.duration_seconds for c in inspected), 3)
        delta = round(total_duration - self.target_duration_seconds, 3)

        cmd = self.build_assembly_command(
            clip_paths=clip_paths,
            output_path=output_path,
            burned_subtitles_path=burned_subtitles_path,
            inspected_clips=inspected
        )

        return AssemblyPlan(
            clips=inspected,
            target_width=self.target_width,
            target_height=self.target_height,
            target_duration_seconds=self.target_duration_seconds,
            total_clip_duration_seconds=total_duration,
            duration_delta_seconds=delta,
            output_path=os.path.abspath(output_path),
            captions_file=os.path.abspath(burned_subtitles_path) if burned_subtitles_path else None,
            ffmpeg_command=cmd
        )

    def assemble(
        self,
        clip_paths: List[str],
        output_path: str,
        burned_subtitles_path: Optional[str] = None
    ) -> AssemblyResult:
        """Executes the full video assembly pipeline using FFmpeg and returns AssemblyResult."""
        plan = self.plan_assembly(clip_paths, output_path, burned_subtitles_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        process = subprocess.run(
            plan.ffmpeg_command,
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg assembly failed (exit code {process.returncode}):\n{process.stderr}"
            )

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Expected output file not found after assembly: {output_path}")

        file_size = os.path.getsize(output_path)
        # Inspect final assembled output duration
        final_meta = self.inspect_clip(output_path)

        return AssemblyResult(
            success=True,
            output_path=os.path.abspath(output_path),
            duration_seconds=final_meta.duration_seconds,
            duration_delta_seconds=round(final_meta.duration_seconds - self.target_duration_seconds, 3),
            file_size_bytes=file_size,
            ffmpeg_command=plan.ffmpeg_command,
            stderr=process.stderr
        )
