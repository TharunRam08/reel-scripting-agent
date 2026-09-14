# AI Reel Scripting Agent Application Package
import os
from dotenv import load_dotenv

# Automatically load environment variables from .env file at project root
load_dotenv()


def __getattr__(name: str):
    if name in ("DemoShotOutput", "DemoPipelineOutput", "ReelDemoPipeline", "run_demo_pipeline"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DemoShotOutput",
    "DemoPipelineOutput",
    "ReelDemoPipeline",
    "run_demo_pipeline",
]
