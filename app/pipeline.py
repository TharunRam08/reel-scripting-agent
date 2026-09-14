import os
import sys
import json
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.topic_analyzer import TopicAnalyzer, TopicAnalysis
from app.agents.framework_selector import FrameworkSelector, FrameworkSelection
from app.agents.script_writer import ScriptWriter, ReelScript
from app.agents.shot_planner import ShotPlanner, ShotPlan, Shot


class DemoShotOutput(BaseModel):
    """Structured representation of a single shot's output for Google Flow / Veo 3.1 Lite."""
    shot_number: int = Field(..., description="Sequential shot number (1 to shot_count).")
    stage: str = Field(..., description="Framework stage name.")
    duration_seconds: float = Field(..., description="Duration of this shot in seconds.")
    narration: str = Field(..., description="Exact spoken narration text delivered during this shot.")
    caption_text: str = Field(..., description="Exact caption text overlay faithful to the narration.")
    veo_prompt: str = Field(..., description="Full self-contained visual generation prompt for Google Flow / Veo 3.1 Lite.")
    validation_status: str = Field("VALID (duration <= 8.0s)", description="Validation status of this shot.")


class DemoPipelineOutput(BaseModel):
    """Complete structured output of the Reel Scripting Agent pipeline ready for demo inspection."""
    original_topic: str = Field(..., description="Original user topic input.")
    selected_framework_id: int = Field(..., description="Selected framework ID (1 to 30).")
    selected_framework_name: str = Field(..., description="Selected framework name.")
    framework_score: float = Field(..., description="Framework selection compatibility score.")
    final_script: str = Field(..., description="Final approved spoken script including closing CTA.")
    target_duration_seconds: float = Field(..., description="Target duration in seconds.")
    shot_count: int = Field(..., description="Number of shots planned.")
    shots: List[DemoShotOutput] = Field(..., description="List of structured shots in chronological sequence.")
    generation_source: str = Field(..., description="Source engine for generation (e.g., groq, deterministic_fallback).")
    validation_status: str = Field("VALID (5 shots, all <= 8.0s, narration timing verified)", description="Overall validation status.")
    feedback: Optional[str] = Field(None, description="Applied revision feedback, if any.")
    approval_status: str = Field("pending_review", description="Plan review and approval status.")


class ReelDemoPipeline:
    """Orchestrates the complete 4-stage Reel Scripting Agent pipeline:
    Topic -> Topic Analyzer -> Framework Selector -> Script Writer -> Shot Planner -> Demo Output.
    """

    def __init__(
        self,
        topic_analyzer: Optional[TopicAnalyzer] = None,
        framework_selector: Optional[FrameworkSelector] = None,
        script_writer: Optional[ScriptWriter] = None,
        shot_planner: Optional[ShotPlanner] = None,
    ):
        self.topic_analyzer = topic_analyzer or TopicAnalyzer()
        self.framework_selector = framework_selector or FrameworkSelector()
        self.script_writer = script_writer or ScriptWriter()
        self.shot_planner = shot_planner or ShotPlanner()

    def run(
        self,
        topic: str = "Junk Food",
        target_duration_seconds: float = 45.0,
        shot_count: int = 5,
        feedback: Optional[str] = None,
    ) -> DemoPipelineOutput:
        """Runs the complete agent pipeline end-to-end and returns a structured DemoPipelineOutput."""
        if not topic or not topic.strip():
            raise ValueError("Topic must not be empty.")
        if target_duration_seconds <= 0:
            raise ValueError(f"target_duration_seconds must be positive, got {target_duration_seconds}")
        if shot_count != 5:
            raise ValueError(f"Locked reel architecture requires exactly 5 shots, got {shot_count}")

        # Stage 1: Topic Analyzer
        topic_analysis = self.topic_analyzer.analyze(topic.strip())

        # Stage 2: Framework Selector
        framework_selection = self.framework_selector.select(topic_analysis)

        # Stage 3: Script Writer
        reel_script = self.script_writer.write_script(
            topic_analysis=topic_analysis,
            framework_selection=framework_selection,
            target_duration_seconds=target_duration_seconds,
            feedback=feedback,
        )

        # Stage 4: Shot Planner
        shot_plan = self.shot_planner.plan_shots(
            reel_script=reel_script,
            topic_analysis=topic_analysis,
            framework_metadata=framework_selection,
            target_duration_seconds=target_duration_seconds,
            shot_count=shot_count,
            feedback=feedback,
        )

        # Map to user-specified DemoPipelineOutput format
        shot_outputs = [
            DemoShotOutput(
                shot_number=shot.shot_number,
                stage=shot.script_stage_name,
                duration_seconds=shot.duration_seconds,
                narration=shot.script_text,
                caption_text=shot.on_screen_text,
                veo_prompt=shot.video_generation_prompt,
                validation_status=shot.validation_status,
            )
            for shot in shot_plan.shots
        ]

        return DemoPipelineOutput(
            original_topic=topic.strip(),
            selected_framework_id=framework_selection.selected_framework_id,
            selected_framework_name=framework_selection.selected_framework_name,
            framework_score=framework_selection.compatibility_score,
            final_script=reel_script.full_script_with_cta,
            target_duration_seconds=target_duration_seconds,
            shot_count=shot_count,
            shots=shot_outputs,
            generation_source=shot_plan.generation_source,
            validation_status=shot_plan.validation_status,
            feedback=feedback,
            approval_status="pending_review",
        )


def run_demo_pipeline(
    topic: str = "Junk Food",
    target_duration_seconds: float = 45.0,
    shot_count: int = 5,
    feedback: Optional[str] = None,
) -> DemoPipelineOutput:
    """Convenience helper to instantiate and run the demo pipeline."""
    pipeline = ReelDemoPipeline()
    return pipeline.run(
        topic=topic,
        target_duration_seconds=target_duration_seconds,
        shot_count=shot_count,
        feedback=feedback,
    )


def format_cli_output(output: DemoPipelineOutput) -> str:
    """Formats the DemoPipelineOutput into a clean, human-readable terminal report."""
    lines = []
    lines.append("=" * 70)
    lines.append("REEL SCRIPTING AGENT — DEMO PIPELINE OUTPUT (PHASE 4A)")
    lines.append("=" * 70)
    lines.append(f"Original Topic:      {output.original_topic}")
    lines.append(f"Selected Framework:  #{output.selected_framework_id} {output.selected_framework_name}")
    lines.append(f"Framework Score:     {output.framework_score:.1f}")
    lines.append(f"Target Duration:     {output.target_duration_seconds:.1f}s")
    lines.append(f"Number of Shots:     {output.shot_count}")
    lines.append(f"Generation Engine:   {output.generation_source}")
    lines.append(f"Validation Status:   {output.validation_status}")
    lines.append("-" * 70)
    lines.append("FINAL APPROVED SCRIPT:")
    lines.append(f"  \"{output.final_script}\"")
    lines.append("=" * 70)
    lines.append("STRUCTURED SHOT PLAN FOR GOOGLE FLOW / VEO 3.1 LITE")
    lines.append("=" * 70)

    for s in output.shots:
        lines.append(f"\n[SHOT {s.shot_number}/{output.shot_count}] — Stage: {s.stage} ({s.duration_seconds}s) [{s.validation_status}]")
        lines.append(f"  Spoken Narration: \"{s.narration}\"")
        lines.append(f"  Caption Text:     \"{s.caption_text}\"")
        lines.append("  Google Flow / Veo Prompt:")
        lines.append("  " + "-" * 66)
        lines.append(f"  {s.veo_prompt}")
        lines.append("  " + "-" * 66)

    lines.append("\n" + "=" * 70)
    lines.append("END OF DEMO PLAN")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the Reel Scripting Agent Demo Pipeline")
    parser.add_argument("--topic", type=str, default="Junk Food", help="Topic to generate reel for (default: 'Junk Food')")
    parser.add_argument("--duration", type=float, default=45.0, help="Target duration in seconds (default: 45.0)")
    parser.add_argument("--shots", type=int, default=5, help="Number of shots (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable text")
    parser.add_argument("--out", type=str, default=None, help="Optional file path to save output")

    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    output = run_demo_pipeline(
        topic=args.topic,
        target_duration_seconds=args.duration,
        shot_count=args.shots,
    )

    if args.json:
        result_text = json.dumps(output.model_dump(), indent=2, ensure_ascii=False)
    else:
        result_text = format_cli_output(output)

    print(result_text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            if args.json:
                f.write(result_text)
            else:
                json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"\n[Saved to {args.out}]")


if __name__ == "__main__":
    main()
