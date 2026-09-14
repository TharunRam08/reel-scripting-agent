import os
import sys
import shutil
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import run_demo_pipeline, DemoPipelineOutput
from app.video.assembler import VideoAssembler
from app.video.captions import CaptionGenerator

app = FastAPI(
    title="Reel Scripting Agent — Review, Upload & Finalization",
    description="End-to-end web UI for 5-shot reel planning, evaluator video upload, and FFmpeg finalization.",
    version="1.2.0",
)

# Output directory configuration
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
UPLOADS_DIR = os.path.join(OUTPUTS_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# In-memory run state tracking for demo evaluation runs
DEMO_RUN: Dict[str, Any] = {
    "topic": "Junk Food",
    "plan": None,
    "approved": False,
    "uploaded_shots": {},  # {1: {"path": str, "filename": str, "size": int}, ...}
    "rendered_output": None,
}


class PipelineRequest(BaseModel):
    topic: str = Field("Junk Food", description="Topic for the reel.")
    target_duration_seconds: float = Field(45.0, description="Target duration in seconds.")
    shot_count: int = Field(5, description="Number of shots to plan (strictly 5).")
    feedback: Optional[str] = Field(None, description="Optional revision feedback for plan regeneration.")


class ApprovalRequest(BaseModel):
    topic: str = Field(..., description="Topic of the approved reel plan.")
    approved: bool = Field(True, description="Whether the user approves the reel plan.")
    feedback: Optional[str] = Field(None, description="Optional reviewer notes.")


class ApprovalResponse(BaseModel):
    status: str = Field("approved", description="Approval state: approved or rejected.")
    message: str = Field("Plan approved — ready for video generation.", description="Approval status message.")
    topic: str = Field(..., description="Topic of the approved plan.")


class RenderResponse(BaseModel):
    success: bool
    output_path: str
    video_url: str
    download_url: str
    duration_seconds: float
    duration_delta_seconds: float
    file_size_bytes: int
    message: str


@app.get("/api/demo", response_model=DemoPipelineOutput)
def get_junk_food_demo():
    """Runs the full pipeline for the 'Junk Food' demo and returns the complete structured plan."""
    output = run_demo_pipeline(
        topic="Junk Food",
        target_duration_seconds=45.0,
        shot_count=5,
    )
    DEMO_RUN["plan"] = output
    DEMO_RUN["topic"] = "Junk Food"
    return output


@app.post("/api/pipeline", response_model=DemoPipelineOutput)
def run_custom_pipeline(req: PipelineRequest):
    """Runs the pipeline for a specified topic, duration, shot count, and optional feedback."""
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must not be empty.")
    if req.shot_count != 5:
        raise HTTPException(status_code=400, detail="Locked reel architecture requires exactly 5 shots.")
    
    try:
        output = run_demo_pipeline(
            topic=req.topic.strip(),
            target_duration_seconds=req.target_duration_seconds,
            shot_count=req.shot_count,
            feedback=req.feedback.strip() if req.feedback and req.feedback.strip() else None,
        )
        DEMO_RUN["plan"] = output
        DEMO_RUN["topic"] = output.original_topic
        DEMO_RUN["approved"] = False
        DEMO_RUN["rendered_output"] = None
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/approve", response_model=ApprovalResponse)
def approve_plan(req: ApprovalRequest):
    """Marks the reel plan as approved and unlocks video upload and finalization."""
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must not be empty.")
    
    if req.approved:
        DEMO_RUN["approved"] = True
        DEMO_RUN["topic"] = req.topic.strip()
        # If plan not yet cached, generate default demo plan
        if DEMO_RUN["plan"] is None:
            DEMO_RUN["plan"] = run_demo_pipeline(topic=req.topic.strip(), target_duration_seconds=45.0, shot_count=5)
        return ApprovalResponse(
            status="approved",
            message="Plan approved — ready for video generation.",
            topic=req.topic.strip(),
        )
    DEMO_RUN["approved"] = False
    return ApprovalResponse(
        status="rejected",
        message="Plan review rejected.",
        topic=req.topic.strip(),
    )


@app.post("/api/upload")
async def upload_shot_clip(
    shot_number: int = Form(..., description="Shot number from 1 to 5."),
    file: UploadFile = File(..., description="Generated video file for this shot."),
):
    """Uploads an externally generated video clip for a specific shot (1 to 5)."""
    if shot_number not in range(1, 6):
        raise HTTPException(status_code=400, detail=f"shot_number must be between 1 and 5, got {shot_number}.")
    
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    valid_exts = [".mp4", ".mov", ".webm", ".m4v"]
    if ext not in valid_exts and not (file.content_type and "video" in file.content_type.lower()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only MP4 video files are accepted."
        )
    
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")
    
    # Save video clip to safe working directory
    save_filename = f"shot_{shot_number}.mp4"
    save_path = os.path.join(UPLOADS_DIR, save_filename)
    with open(save_path, "wb") as f:
        f.write(content)
    
    DEMO_RUN["uploaded_shots"][shot_number] = {
        "path": save_path,
        "filename": filename,
        "size": len(content),
    }
    
    uploaded_count = len(DEMO_RUN["uploaded_shots"])
    all_uploaded = uploaded_count == 5
    
    return {
        "success": True,
        "shot_number": shot_number,
        "filename": filename,
        "saved_as": save_filename,
        "file_size": len(content),
        "uploaded_count": uploaded_count,
        "all_uploaded": all_uploaded,
        "ready_to_render": DEMO_RUN.get("approved", False) and all_uploaded,
        "message": f"Shot {shot_number} video uploaded successfully."
    }


@app.get("/api/upload/status")
def get_upload_status():
    """Returns the current upload status of all 5 shots and rendering readiness."""
    uploaded_shots = DEMO_RUN.get("uploaded_shots", {})
    uploaded_dict = {}
    for i in range(1, 6):
        if i in uploaded_shots:
            item = uploaded_shots[i]
            uploaded_dict[str(i)] = {
                "uploaded": True,
                "filename": item["filename"],
                "size": item["size"],
            }
        else:
            uploaded_dict[str(i)] = {
                "uploaded": False,
                "filename": None,
                "size": 0,
            }
    
    count = sum(1 for v in uploaded_dict.values() if v["uploaded"])
    all_up = count == 5
    is_appr = DEMO_RUN.get("approved", False)
    
    return {
        "approved": is_appr,
        "uploaded_shots": uploaded_dict,
        "uploaded_count": count,
        "all_uploaded": all_up,
        "ready_to_render": is_appr and all_up,
        "rendered_output": DEMO_RUN.get("rendered_output") is not None,
    }


@app.post("/api/render", response_model=RenderResponse)
def render_final_reel():
    """Assembles the 5 uploaded clips in shot order (1 -> 5) with burned captions into the final MP4."""
    if not DEMO_RUN.get("approved"):
        raise HTTPException(status_code=400, detail="Plan must be approved before rendering.")
    
    uploaded = DEMO_RUN.get("uploaded_shots", {})
    missing_shots = [i for i in range(1, 6) if i not in uploaded or not os.path.exists(uploaded[i]["path"])]
    if missing_shots:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot render reel: missing video clips for shots: {missing_shots}. All 5 shots must be uploaded."
        )
    
    # Ensure plan is loaded
    plan = DEMO_RUN.get("plan")
    if plan is None:
        plan = run_demo_pipeline(topic=DEMO_RUN.get("topic", "Junk Food"), target_duration_seconds=45.0, shot_count=5)
        DEMO_RUN["plan"] = plan
    
    # Sequential clip paths preserving strict order 1 -> 5
    clip_paths = [uploaded[i]["path"] for i in range(1, 6)]
    
    try:
        assembler = VideoAssembler(target_width=720, target_height=1280, target_duration_seconds=45.0)
        inspected = assembler.inspect_clips(clip_paths)
        actual_durations = [c.duration_seconds for c in inspected]
        
        # Burn captions using existing CaptionGenerator derived from approved narration
        caption_gen = CaptionGenerator()
        entries = caption_gen.generate_from_shot_plan(plan, actual_clip_durations=actual_durations)
        ass_path = os.path.join(OUTPUTS_DIR, "reel_captions.ass")
        caption_gen.write_ass_file(entries, ass_path, video_width=720, video_height=1280)
        
        # Execute assembly
        final_video_path = os.path.join(OUTPUTS_DIR, "final_reel.mp4")
        result = assembler.assemble(clip_paths, final_video_path, burned_subtitles_path=ass_path)
        DEMO_RUN["rendered_output"] = final_video_path
        
        return RenderResponse(
            success=True,
            output_path=final_video_path,
            video_url="/api/video/final_reel.mp4",
            download_url="/api/video/final_reel.mp4",
            duration_seconds=result.duration_seconds,
            duration_delta_seconds=result.duration_delta_seconds,
            file_size_bytes=result.file_size_bytes,
            message="Final reel assembled successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg assembly error: {str(e)}")


@app.get("/api/video/{filename}")
def get_video_file(filename: str):
    """Streams or downloads an assembled or uploaded video file."""
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUTS_DIR, safe_name)
    if not os.path.exists(path):
        # Check in uploads
        up_path = os.path.join(UPLOADS_DIR, safe_name)
        if os.path.exists(up_path):
            path = up_path
        else:
            raise HTTPException(status_code=404, detail=f"Video file '{safe_name}' not found.")
    
    return FileResponse(path, media_type="video/mp4", filename=safe_name)


@app.get("/", response_class=HTMLResponse)
def get_ui():
    """Serves a beginner-friendly, clean, readable Vanilla HTML/CSS/JS UI to review, regenerate, approve, upload clips, and finalize the reel."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reel Scripting Agent — Review, Upload & Finalization</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #131c2e;
      --card-alt: #18243b;
      --border: #23324d;
      --border-focus: #38bdf8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #22c55e;
      --accent-hover: #16a34a;
      --accent-dim: rgba(34, 197, 94, 0.15);
      --text: #f8fafc;
      --muted: #94a3b8;
      --danger: #ef4444;
      --danger-dim: rgba(239, 68, 68, 0.15);
      --code-bg: #0b1120;
    }
    * {
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 32px 16px;
      display: flex;
      justify-content: center;
      line-height: 1.5;
    }
    .container {
      max-width: 960px;
      width: 100%;
    }
    header {
      margin-bottom: 28px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
    }
    .header-badge {
      display: inline-block;
      background: rgba(56, 189, 248, 0.12);
      color: var(--primary);
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      margin-bottom: 8px;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 1.8rem;
      color: #ffffff;
      letter-spacing: -0.5px;
    }
    p.subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }
    .flow-steps {
      display: flex;
      gap: 8px;
      margin-top: 14px;
      font-size: 0.82rem;
      color: var(--muted);
      align-items: center;
      flex-wrap: wrap;
    }
    .flow-step {
      padding: 3px 10px;
      border-radius: 4px;
      background: var(--code-bg);
      border: 1px solid var(--border);
    }
    .flow-step.active {
      background: rgba(56, 189, 248, 0.18);
      color: var(--primary);
      border-color: var(--primary);
      font-weight: 600;
    }
    .flow-step.completed {
      background: var(--accent-dim);
      color: var(--accent);
      border-color: var(--accent);
      font-weight: 600;
    }
    .flow-arrow {
      color: var(--border);
    }
    .controls-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .controls-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: flex-end;
    }
    .field {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-width: 240px;
    }
    label {
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 6px;
      color: var(--muted);
    }
    input, textarea {
      background: var(--code-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.95rem;
      transition: border-color 0.2s;
    }
    input:focus, textarea:focus {
      outline: none;
      border-color: var(--border-focus);
    }
    textarea {
      width: 100%;
      min-height: 80px;
      resize: vertical;
      font-family: inherit;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: none;
      border-radius: 8px;
      padding: 10px 20px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      height: 42px;
      gap: 8px;
      text-decoration: none;
    }
    .btn-primary {
      background: var(--primary);
      color: #090d16;
    }
    .btn-primary:hover {
      background: var(--primary-hover);
      color: #ffffff;
    }
    .btn-secondary {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover {
      background: #334155;
      border-color: var(--primary);
    }
    .btn-accent {
      background: var(--accent);
      color: #090d16;
    }
    .btn-accent:hover {
      background: var(--accent-hover);
      color: #ffffff;
    }
    .btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      pointer-events: none;
    }
    .error-banner {
      background: var(--danger-dim);
      border: 1px solid var(--danger);
      color: #fca5a5;
      padding: 14px 18px;
      border-radius: 8px;
      margin-bottom: 24px;
      font-size: 0.95rem;
      display: none;
    }
    .status-msg {
      color: var(--muted);
      font-style: italic;
      padding: 24px;
      text-align: center;
      background: var(--card-bg);
      border: 1px dashed var(--border);
      border-radius: 12px;
    }
    .summary-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .meta-item {
      background: var(--code-bg);
      padding: 12px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .meta-label {
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }
    .meta-value {
      font-size: 1.05rem;
      font-weight: 700;
      color: #ffffff;
      margin-top: 4px;
    }
    .script-box-container {
      margin-top: 14px;
    }
    .script-box {
      background: var(--code-bg);
      border-left: 4px solid var(--accent);
      padding: 14px 18px;
      border-radius: 6px;
      font-size: 0.98rem;
      line-height: 1.6;
      color: #e2e8f0;
      border-top: 1px solid var(--border);
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
    }
    .shots-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .shots-title {
      font-size: 1.3rem;
      font-weight: 700;
      color: #ffffff;
      margin: 0;
    }
    .shots-count-badge {
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 700;
    }
    .shot-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      transition: border-color 0.2s;
    }
    .shot-card:hover {
      border-color: rgba(56, 189, 248, 0.4);
    }
    .shot-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }
    .shot-title-group {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .shot-number-badge {
      background: var(--primary);
      color: #090d16;
      font-weight: 800;
      font-size: 0.85rem;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .shot-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: #ffffff;
    }
    .shot-duration-badge {
      background: #1e293b;
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--primary);
    }
    .field-row {
      margin-bottom: 14px;
    }
    .field-label {
      font-weight: 600;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }
    .field-text {
      color: var(--text);
      background: var(--code-bg);
      padding: 10px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.95rem;
      line-height: 1.5;
    }
    .field-text.spoken {
      font-style: italic;
      color: #f1f5f9;
      border-left: 3px solid var(--primary);
    }
    .prompt-box-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      margin-bottom: 8px;
    }
    .copy-btn {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .copy-btn:hover {
      background: var(--primary);
      color: #090d16;
      border-color: var(--primary);
    }
    .copy-btn.copied {
      background: var(--accent);
      color: #090d16;
      border-color: var(--accent);
    }
    .prompt-content {
      background: var(--code-bg);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.88rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      color: #cbd5e1;
    }
    .review-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-top: 32px;
      margin-bottom: 24px;
    }
    .review-title {
      font-size: 1.2rem;
      font-weight: 700;
      color: #ffffff;
      margin: 0 0 8px 0;
    }
    .review-desc {
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 16px;
    }
    .action-row {
      display: flex;
      gap: 16px;
      margin-top: 18px;
      flex-wrap: wrap;
      align-items: center;
    }
    .approval-banner {
      background: var(--accent-dim);
      border: 2px solid var(--accent);
      border-radius: 12px;
      padding: 20px;
      margin-top: 24px;
      display: none;
      align-items: center;
      gap: 16px;
    }
    .approval-banner-icon {
      background: var(--accent);
      color: #090d16;
      width: 44px;
      height: 44px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      font-weight: 800;
      flex-shrink: 0;
    }
    .approval-banner-text {
      flex: 1;
    }
    .approval-banner-title {
      font-size: 1.15rem;
      font-weight: 800;
      color: #ffffff;
      margin: 0 0 4px 0;
    }
    .approval-banner-sub {
      margin: 0;
      font-size: 0.9rem;
      color: #cbd5e1;
    }
    .feedback-applied-tag {
      display: inline-block;
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      margin-top: 8px;
    }

    /* Video Upload & Finalization Section */
    .upload-section {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-top: 32px;
      display: none;
    }
    .upload-section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 14px;
    }
    .upload-section-title {
      font-size: 1.3rem;
      font-weight: 800;
      color: #ffffff;
      margin: 0;
    }
    .upload-tracker-badge {
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 700;
    }
    .upload-instructions {
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 24px;
      line-height: 1.6;
    }
    .upload-shot-card {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .upload-shot-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .upload-control-row {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .upload-file-btn {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: all 0.2s;
    }
    .upload-file-btn:hover {
      background: #334155;
      border-color: var(--primary);
    }
    .upload-badge {
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
    }
    .upload-badge.not-uploaded {
      background: rgba(148, 163, 184, 0.15);
      color: var(--muted);
      border: 1px solid var(--border);
    }
    .upload-badge.uploaded {
      background: var(--accent-dim);
      color: var(--accent);
      border: 1px solid var(--accent);
    }
    .render-area {
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      margin-top: 24px;
      text-align: center;
    }
    .render-note {
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 10px;
    }
    .final-video-container {
      margin-top: 28px;
      text-align: center;
      display: none;
      padding: 20px;
      background: var(--code-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    video {
      max-width: 360px;
      width: 100%;
      height: auto;
      aspect-ratio: 9 / 16;
      border-radius: 12px;
      background: #000;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      border: 2px solid var(--accent);
      margin-bottom: 16px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-badge">Reel Scripting Agent</div>
      <h1>Human-in-the-Loop Reel Finalization</h1>
      <p class="subtitle">Plan 5 shots → Copy prompts to Google Flow → Upload generated clips → FFmpeg Finalization.</p>
      <div class="flow-steps">
        <div class="flow-step">1. Topic Analysis</div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">2. Framework Selection</div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">3. Script Writing</div>
        <div class="flow-arrow">→</div>
        <div class="flow-step active" id="stepPlan">4. 5-Shot Planning & Review</div>
        <div class="flow-arrow">→</div>
        <div class="flow-step" id="stepFinalize">5. Clip Upload & Final Reel</div>
      </div>
    </header>

    <div id="errorBanner" class="error-banner"></div>

    <!-- Step 1: Planning Controls -->
    <div class="controls-card">
      <div class="controls-grid">
        <div class="field" style="flex: 3;">
          <label for="topicInput">Reel Topic</label>
          <input id="topicInput" type="text" value="Junk Food" placeholder="Enter topic (e.g. Junk Food, Why do legs cramp at night?)" />
        </div>
        <div class="field" style="max-width: 140px;">
          <label for="durationInput">Target Duration</label>
          <input id="durationInput" type="number" value="45" min="15" max="60" />
        </div>
        <div class="field" style="max-width: 120px;">
          <label for="shotsInput">Shots (Locked)</label>
          <input id="shotsInput" type="number" value="5" min="5" max="5" readonly disabled style="opacity: 0.7;" />
        </div>
        <button id="generateBtn" class="btn btn-primary" onclick="generatePlan()">
          <span>Generate Reel Plan</span>
        </button>
      </div>
    </div>

    <div id="statusDiv" class="status-msg">
      Click <strong>"Generate Reel Plan"</strong> to run the agent pipeline and inspect all 5 shots.
    </div>

    <!-- Step 2: 5-Shot Plan Results Display -->
    <div id="resultsDiv" style="display: none;"></div>

    <!-- Step 3: Review, Feedback & Approval Section -->
    <div id="reviewCard" class="review-card" style="display: none;">
      <h3 class="review-title">Review & Feedback</h3>
      <p class="review-desc">Review the spoken script and prompts above. You can provide feedback to regenerate the plan or approve it for video generation.</p>
      
      <div class="field" style="margin-bottom: 12px;">
        <label for="feedbackInput">What would you like to change?</label>
        <textarea id="feedbackInput" placeholder="What would you like to change? (e.g., make visual lighting warmer, adjust camera framing, emphasize fatigue)"></textarea>
      </div>

      <div class="action-row">
        <button id="regenerateBtn" class="btn btn-secondary" onclick="regeneratePlan()">
          <span>↻ Regenerate Plan</span>
        </button>
        <button id="approveBtn" class="btn btn-accent" onclick="approvePlan()">
          <span>✓ Approve Plan</span>
        </button>
      </div>

      <div id="approvalBanner" class="approval-banner">
        <div class="approval-banner-icon">✓</div>
        <div class="approval-banner-text">
          <div class="approval-banner-title">Plan approved — ready for video generation.</div>
          <p class="approval-banner-sub">The 5-shot reel plan and visual prompts are approved. Scroll down to the Video Generation & Upload section below.</p>
        </div>
      </div>
    </div>

    <!-- Step 4: Video Generation & Upload Section (Unlocked upon plan approval) -->
    <div id="uploadSection" class="upload-section">
      <div class="upload-section-header">
        <h3 class="upload-section-title">Video Generation & Upload</h3>
        <span class="upload-tracker-badge" id="uploadTrackerBadge">0 / 5 Uploaded</span>
      </div>
      <p class="upload-instructions">
        For each shot, copy the generated prompt into Google Flow / Veo 3.1 Lite. Once generated, upload each video clip (.mp4) below. When all 5 clips are uploaded, the <strong>"Render Final Reel"</strong> button will enable.
      </p>

      <div id="uploadShotsList"></div>

      <div class="render-area">
        <button id="renderReelBtn" class="btn btn-accent" disabled onclick="renderFinalReel()" style="padding: 12px 28px; font-size: 1.05rem;">
          <span>Render Final Reel</span>
        </button>
        <div id="renderHelperNote" class="render-note">Please upload all 5 video clips to enable rendering (0/5 uploaded).</div>
        <div id="renderingStatus" style="display:none;" class="status-msg">Assembling 5 clips with FFmpeg & burning approved captions...</div>
      </div>

      <!-- Step 5: Final Reel Output Player & Download Link -->
      <div id="finalVideoContainer" class="final-video-container">
        <h3 style="color: var(--accent); margin: 0 0 6px 0; font-size: 1.3rem;">✓ Final Reel Assembled Successfully</h3>
        <p style="color: var(--muted); font-size: 0.9rem; margin-bottom: 18px;" id="finalVideoMetrics"></p>
        
        <div>
          <video id="finalVideoPlayer" controls playsinline>
            <source id="finalVideoSource" src="" type="video/mp4">
            Your browser does not support HTML5 video.
          </video>
        </div>
        
        <div style="margin-top: 12px;">
          <a id="downloadReelBtn" href="" download="final_reel.mp4" class="btn btn-primary" style="padding: 12px 24px;">
            <span>⬇ Download Final Reel (.mp4)</span>
          </a>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentPlanData = null;
    let uploadedShots = { 1: false, 2: false, 3: false, 4: false, 5: false };

    async function generatePlan() {
      const topic = document.getElementById("topicInput").value.trim();
      const duration = parseFloat(document.getElementById("durationInput").value) || 45.0;
      
      const errorBanner = document.getElementById("errorBanner");
      errorBanner.style.display = "none";

      if (!topic) {
        showError("Topic cannot be empty. Please enter a topic for the reel.");
        return;
      }

      const btn = document.getElementById("generateBtn");
      const statusDiv = document.getElementById("statusDiv");
      const resultsDiv = document.getElementById("resultsDiv");
      const reviewCard = document.getElementById("reviewCard");
      const approvalBanner = document.getElementById("approvalBanner");
      const uploadSection = document.getElementById("uploadSection");
      const finalVideoContainer = document.getElementById("finalVideoContainer");

      btn.disabled = true;
      btn.innerText = "Running Pipeline...";
      statusDiv.style.display = "block";
      statusDiv.innerText = "Running Topic Analyzer → Framework Selector → Script Writer → Shot Planner (5 shots)...";
      resultsDiv.style.display = "none";
      reviewCard.style.display = "none";
      approvalBanner.style.display = "none";
      uploadSection.style.display = "none";
      finalVideoContainer.style.display = "none";

      try {
        const response = await fetch("/api/pipeline", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topic: topic,
            target_duration_seconds: duration,
            shot_count: 5,
            feedback: null
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned status ${response.status}`);
        }

        const data = await response.json();
        currentPlanData = data;
        renderResults(data);

        statusDiv.style.display = "none";
        resultsDiv.style.display = "block";
        reviewCard.style.display = "block";
      } catch (err) {
        showError("Error running pipeline: " + err.message);
        statusDiv.style.display = "block";
        statusDiv.innerText = "Pipeline execution failed. Please check the error above and try again.";
      } finally {
        btn.disabled = false;
        btn.innerText = "Generate Reel Plan";
      }
    }

    async function regeneratePlan() {
      if (!currentPlanData) return;

      const topic = document.getElementById("topicInput").value.trim();
      const duration = parseFloat(document.getElementById("durationInput").value) || 45.0;
      const feedback = document.getElementById("feedbackInput").value.trim();

      const btn = document.getElementById("regenerateBtn");
      const approvalBanner = document.getElementById("approvalBanner");
      const uploadSection = document.getElementById("uploadSection");
      approvalBanner.style.display = "none";
      uploadSection.style.display = "none";

      btn.disabled = true;
      btn.innerText = "Regenerating with Feedback...";

      try {
        const response = await fetch("/api/pipeline", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topic: topic,
            target_duration_seconds: duration,
            shot_count: 5,
            feedback: feedback || null
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned status ${response.status}`);
        }

        const data = await response.json();
        currentPlanData = data;
        renderResults(data);

        if (feedback) {
          document.getElementById("feedbackInput").value = "";
        }
      } catch (err) {
        showError("Regeneration failed: " + err.message);
      } finally {
        btn.disabled = false;
        btn.innerText = "↻ Regenerate Plan";
      }
    }

    async function approvePlan() {
      if (!currentPlanData) return;

      const topic = currentPlanData.original_topic;
      const approveBtn = document.getElementById("approveBtn");
      const approvalBanner = document.getElementById("approvalBanner");
      const uploadSection = document.getElementById("uploadSection");

      approveBtn.disabled = true;
      approveBtn.innerText = "Approving...";

      try {
        const response = await fetch("/api/approve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topic: topic,
            approved: true
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned status ${response.status}`);
        }

        const data = await response.json();
        approvalBanner.style.display = "flex";
        approveBtn.innerText = "✓ Plan Approved";

        // Reveal Step 4: Video Generation & Upload Section
        renderUploadCards(currentPlanData);
        uploadSection.style.display = "block";
        uploadSection.scrollIntoView({ behavior: "smooth" });

        // Update step flow indicator
        document.getElementById("stepPlan").className = "flow-step completed";
        document.getElementById("stepFinalize").className = "flow-step active";
      } catch (err) {
        showError("Approval failed: " + err.message);
        approveBtn.disabled = false;
        approveBtn.innerText = "✓ Approve Plan";
      }
    }

    function renderResults(data) {
      const div = document.getElementById("resultsDiv");
      let feedbackHtml = "";
      if (data.feedback) {
        feedbackHtml = `<div class="feedback-applied-tag">Active Feedback: "${escapeHtml(data.feedback)}"</div>`;
      }

      let html = `
        <div class="summary-card">
          <div class="meta-grid">
            <div class="meta-item">
              <div class="meta-label">Original Topic</div>
              <div class="meta-value">${escapeHtml(data.original_topic)}</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">Selected Framework</div>
              <div class="meta-value">#${data.selected_framework_id} ${escapeHtml(data.selected_framework_name)}</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">Framework Score</div>
              <div class="meta-value">${data.framework_score.toFixed(1)}</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">Target Duration</div>
              <div class="meta-value">${data.target_duration_seconds}s (5 shots)</div>
            </div>
          </div>
          ${feedbackHtml}
          <div class="script-box-container">
            <div class="meta-label" style="margin-bottom: 6px;">Full Approved Script (with CTA)</div>
            <div class="script-box">${escapeHtml(data.final_script)}</div>
          </div>
        </div>

        <div class="shots-header">
          <h2 class="shots-title">Structured 5-Shot Plan for Google Flow / Veo 3.1 Lite</h2>
          <span class="shots-count-badge">Exactly 5 Shots (Each &le; 8.0s)</span>
        </div>
      `;

      data.shots.forEach((shot) => {
        html += `
          <div class="shot-card" id="shot-${shot.shot_number}">
            <div class="shot-header">
              <div class="shot-title-group">
                <div class="shot-number-badge">${shot.shot_number}</div>
                <div class="shot-title">Shot ${shot.shot_number}: ${escapeHtml(shot.stage)}</div>
              </div>
              <div class="shot-duration-badge">${shot.duration_seconds}s &le; 8.0s</div>
            </div>
            
            <div class="field-row">
              <div class="field-label">Spoken Narration (Exact):</div>
              <div class="field-text spoken">"${escapeHtml(shot.narration)}"</div>
            </div>
            
            <div class="field-row">
              <div class="field-label">On-Screen Caption:</div>
              <div class="field-text">"${escapeHtml(shot.caption_text)}"</div>
            </div>
            
            <div class="prompt-box-header">
              <div class="field-label">Google Flow / Veo 3.1 Lite Generation Prompt:</div>
              <button class="copy-btn" onclick="copyPrompt(this, 'prompt-${shot.shot_number}')">
                <span>Copy Prompt</span>
              </button>
            </div>
            <div class="prompt-content" id="prompt-${shot.shot_number}">${escapeHtml(shot.veo_prompt)}</div>
          </div>
        `;
      });

      div.innerHTML = html;
    }

    function renderUploadCards(data) {
      const listDiv = document.getElementById("uploadShotsList");
      let html = "";

      data.shots.forEach((shot) => {
        const isUp = uploadedShots[shot.shot_number];
        const statusClass = isUp ? "upload-badge uploaded" : "upload-badge not-uploaded";
        const statusText = isUp ? "✓ Uploaded" : "Not uploaded";

        html += `
          <div class="upload-shot-card" id="upload-card-${shot.shot_number}">
            <div class="upload-shot-header">
              <div style="font-weight: 700; color: #fff;">Shot ${shot.shot_number}: ${escapeHtml(shot.stage)} (${shot.duration_seconds}s)</div>
              <span id="shotUploadStatus-${shot.shot_number}" class="${statusClass}">${statusText}</span>
            </div>

            <div class="prompt-box-header" style="margin-top: 6px;">
              <div class="field-label" style="font-size: 0.78rem;">Visual Prompt for Google Flow:</div>
              <button class="copy-btn" onclick="copyPrompt(this, 'upload-prompt-${shot.shot_number}')">
                <span>Copy Prompt</span>
              </button>
            </div>
            <div class="prompt-content" id="upload-prompt-${shot.shot_number}" style="font-size: 0.82rem; padding: 10px;">${escapeHtml(shot.veo_prompt)}</div>

            <div class="upload-control-row">
              <label for="shotFileInput-${shot.shot_number}" class="btn btn-secondary upload-file-btn">
                <span>📁 Choose Clip (${shot.shot_number}/5)</span>
              </label>
              <input type="file" id="shotFileInput-${shot.shot_number}" accept="video/mp4,video/*,.mp4,.mov,.webm" style="display:none;" onchange="handleFileSelect(${shot.shot_number}, this.files[0])">
              <span id="uploadFileName-${shot.shot_number}" style="font-size: 0.85rem; color: var(--muted);">No file chosen</span>
            </div>
          </div>
        `;
      });

      listDiv.innerHTML = html;
      updateUploadTracker();
    }

    async function handleFileSelect(shotNumber, file) {
      if (!file) return;

      const fileNameLabel = document.getElementById(`uploadFileName-${shotNumber}`);
      const statusBadge = document.getElementById(`shotUploadStatus-${shotNumber}`);
      
      fileNameLabel.innerText = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      statusBadge.className = "upload-badge not-uploaded";
      statusBadge.innerText = "Uploading...";

      const formData = new FormData();
      formData.append("shot_number", shotNumber);
      formData.append("file", file);

      try {
        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        uploadedShots[shotNumber] = true;
        statusBadge.className = "upload-badge uploaded";
        statusBadge.innerText = `✓ Uploaded (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        updateUploadTracker();
      } catch (err) {
        showError(`Upload failed for Shot ${shotNumber}: ` + err.message);
        statusBadge.className = "upload-badge not-uploaded";
        statusBadge.innerText = "Upload failed";
      }
    }

    function updateUploadTracker() {
      const count = Object.values(uploadedShots).filter(Boolean).length;
      const trackerBadge = document.getElementById("uploadTrackerBadge");
      const renderBtn = document.getElementById("renderReelBtn");
      const renderNote = document.getElementById("renderHelperNote");

      trackerBadge.innerText = `${count} / 5 Uploaded`;

      if (count === 5) {
        trackerBadge.style.background = "var(--accent-dim)";
        trackerBadge.style.color = "var(--accent)";
        trackerBadge.style.borderColor = "var(--accent)";
        renderBtn.disabled = false;
        renderNote.innerText = "✓ All 5 clips uploaded. Ready to assemble the final reel!";
        renderNote.style.color = "var(--accent)";
      } else {
        trackerBadge.style.background = "rgba(56, 189, 248, 0.15)";
        trackerBadge.style.color = "var(--primary)";
        trackerBadge.style.borderColor = "rgba(56, 189, 248, 0.3)";
        renderBtn.disabled = true;
        renderNote.innerText = `Please upload all 5 video clips to enable rendering (${count}/5 uploaded).`;
        renderNote.style.color = "var(--muted)";
      }
    }

    async function renderFinalReel() {
      const renderBtn = document.getElementById("renderReelBtn");
      const renderingStatus = document.getElementById("renderingStatus");
      const finalContainer = document.getElementById("finalVideoContainer");
      const videoPlayer = document.getElementById("finalVideoPlayer");
      const videoSource = document.getElementById("finalVideoSource");
      const downloadBtn = document.getElementById("downloadReelBtn");
      const metricsText = document.getElementById("finalVideoMetrics");

      renderBtn.disabled = true;
      renderBtn.innerText = "Assembling Reel...";
      renderingStatus.style.display = "block";
      finalContainer.style.display = "none";

      try {
        const response = await fetch("/api/render", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned status ${response.status}`);
        }

        const data = await response.json();
        renderingStatus.style.display = "none";

        // Setup player
        videoSource.src = data.video_url + "?t=" + new Date().getTime();
        videoPlayer.load();
        downloadBtn.href = data.download_url;

        metricsText.innerText = `Duration: ${data.duration_seconds}s | File Size: ${(data.file_size_bytes / (1024 * 1024)).toFixed(2)} MB | Normalized: 720x1280 (9:16) with burned captions`;

        finalContainer.style.display = "block";
        finalContainer.scrollIntoView({ behavior: "smooth" });
      } catch (err) {
        showError("Assembly failed: " + err.message);
        renderingStatus.style.display = "none";
      } finally {
        renderBtn.disabled = false;
        renderBtn.innerText = "Render Final Reel";
      }
    }

    function copyPrompt(buttonEl, elementId) {
      const text = document.getElementById(elementId).innerText;
      navigator.clipboard.writeText(text).then(() => {
        const originalText = buttonEl.innerHTML;
        buttonEl.innerHTML = "<span>✓ Copied!</span>";
        buttonEl.classList.add("copied");
        setTimeout(() => {
          buttonEl.innerHTML = originalText;
          buttonEl.classList.remove("copied");
        }, 2000);
      }).catch(err => {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
        buttonEl.innerHTML = "<span>✓ Copied!</span>";
        buttonEl.classList.add("copied");
        setTimeout(() => {
          buttonEl.innerHTML = "<span>Copy Prompt</span>";
          buttonEl.classList.remove("copied");
        }, 2000);
      });
    }

    function showError(msg) {
      const errBanner = document.getElementById("errorBanner");
      errBanner.innerText = msg;
      errBanner.style.display = "block";
      errBanner.scrollIntoView({ behavior: "smooth" });
    }

    function escapeHtml(str) {
      if (!str) return "";
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
