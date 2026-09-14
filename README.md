# Reel Scripting Agent

An autonomous, multi-agent AI system designed to research, structure, script, plan, and assemble high-retention 9:16 short-form video reels.

Built with a modular multi-agent pipeline, an authoritative 30-framework knowledge base, Groq LLM integration with deterministic safety fallbacks, a modern review/approval web UI, and an automated FFmpeg video assembly engine.

---

## Architecture & Workflow

```
User Input (Any Topic)
         │
         ▼
┌─────────────────────────┐
│     Topic Analyzer      │  Deconstructs domain, keywords, audience, intent & goals
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│   Framework Selector    │  Scores & ranks topic across 30 short-form frameworks
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│      Script Writer      │  Structures spoken script strictly to framework stages
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│      Shot Planner       │  Plans exactly 5 shots (<=8.0s each) with Veo/Flow prompts
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│   Review & Approval UI  │  Human-in-the-loop review, feedback revisions & plan approval
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│ Google Flow Generation  │  Video clips generated via Google Flow (Veo 3.1 Lite)
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│    UI Video Upload      │  Direct web UI upload for shots 1 through 5
└───────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│ FFmpeg Video Assembler  │  Clip validation, ASS caption synthesis, 9:16 video render
└─────────────────────────┘
```

---

## Arbitrary Topic Adaptability & Generalization

While demonstrated end-to-end on the **Junk Food** topic, the pipeline is fully generalized and natively handles **arbitrary topics** across diverse domains:

- **Habit Consequence Topics**: e.g., *"What happens if you eat junk food every day?"* -> Selects Framework #15 (*What Happens If*) with cause-and-effect progression.
- **Physiological Mechanism Topics**: e.g., *"Why do legs cramp at night?"* -> Selects Framework #6 (*Hook, Explanation, CTA*) or #28 (*Why Framework*).
- **Myth Debunking Topics**: e.g., *"Curd at night causes cold: Myth or Truth?"* -> Selects Framework #8 (*Myth, Truth, Explanation*) or #24 (*Pattern Interrupt*).
- **Symptom & Triage Topics**: e.g., *"3 signs your liver is under stress"* -> Selects Framework #16 (*3 Signs*) with clear clinical disclaimers.
- **Urgent Differential Triage**: e.g., *"Normal fever vs Dengue red flags"* -> Selects Framework #11 (*Normal vs Red Flag*).
- **Non-Health Domains**: Supports productivity, technology, lifestyle, and educational topics without modification.

### How Generalization Works:
1. **Dynamic Topic Analysis**: Analyzes linguistic intent, domain category, keywords, emotional angle, and suggested engagement metrics (Watch Time, Saves, Shares, Authority).
2. **Comprehensive Framework Evaluation**: Evaluates all 30 industry-tested short-form frameworks from `knowledge_base/frameworks_knowledge_base.json` using weighted semantic and structural scoring.
3. **Structured Scripting**: Adapts script sections to the exact stages of whatever framework is chosen.
4. **Locked 5-Shot Architecture**: Automatically breaks any approved script into 5 chronological shots, each strictly $\le 8.0$ seconds, matching the Google Flow / Veo 3.1 Lite duration constraints.

---

## Project Structure

```
reel-scripting-agent/
├── app/
│   ├── agents/
│   │   ├── topic_analyzer.py      # Stage 1: Topic analysis & entity extraction
│   │   ├── framework_selector.py  # Stage 2: 30-framework scoring & selection
│   │   ├── script_writer.py       # Stage 3: Stage-bound script generation
│   │   └── shot_planner.py        # Stage 4: 5-shot visual & prompt planning
│   ├── video/
│   │   └── assembler.py           # FFmpeg assembly, ASS subtitles, normalization
│   ├── pipeline.py                # End-to-end orchestrator & CLI
│   └── server.py                  # FastAPI server & endpoints
├── static/
│   ├── index.html                 # Responsive web UI for review & clip upload
│   ├── styles.css                 # Dark-mode styling and layout
│   └── app.js                     # Frontend state management & async API calls
├── knowledge_base/
│   ├── frameworks_knowledge_base.json  # 30-framework knowledge base
│   └── frameworks_summary.md           # Framework overview & use cases
├── reference/                     # Source documentation and guidelines
├── tests/                         # Complete automated test suite
├── submission/                    # Submission package deliverables
│   ├── reel.json                  # Structured metadata (topic, framework, script)
│   ├── framework_selected.txt     # Framework name, ID, and score
│   ├── script.txt                 # Exact approved script with CTA
│   ├── junk_food_reel.mp4         # Final rendered 9:16 reel (<=10 MB)
│   └── README_OUTPUT.txt          # Submission documentation
├── outputs/                       # Render outputs, captions, and uploads
├── junk_food_reel_submission.zip  # Packaged submission archive (<=10 MB)
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git exclusion rules (secrets, venv, cache)
└── README.md                      # Project documentation
```

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (tested with Python 3.10 through 3.14)
- **FFmpeg & FFprobe**: Installed and available in system `PATH` (or winget default location).

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory (optional for demo fallback, required for live Groq LLM generation):
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
> **Note**: If `GROQ_API_KEY` is not provided, the system automatically falls back to deterministic rule-based generation with zero external API dependencies.

---

## Running the Application

### 1. Web Application (Recommended)
Start the FastAPI server:
```bash
uvicorn app.server:app --reload --port 8000
```
Open your browser and navigate to:
```
http://localhost:8000
```

#### Web UI Features:
1. **Plan Generation**: Enter any topic (default: *"Junk Food"*) and click **"Generate Reel Plan"**.
2. **Interactive Review**: Inspect the selected framework, score, full script, and 5 structured shots.
3. **Copy Prompts**: One-click prompt copying for Google Flow / Veo 3.1 Lite.
4. **Feedback Revisions**: Submit revision requests to regenerate the plan dynamically.
5. **Approve Plan**: Lock the plan and transition to the upload workflow.
6. **Upload Video Clips**: Upload 5 generated clips (`.mp4`) directly through the browser.
7. **Render Final Reel**: Trigger automated FFmpeg assembly, subtitle burn-in, and video streaming.

### 2. Command Line Interface (CLI)
Run the pipeline directly from the terminal:
```bash
python -m app.pipeline --topic "Junk Food" --json
```
For arbitrary topics:
```bash
python -m app.pipeline --topic "Why do legs cramp at night?" --json
```

---

## Running Automated Tests

Execute the comprehensive test suite covering all modules:
```bash
python tests/test_topic_analyzer.py
python tests/test_framework_selector.py
python tests/test_script_writer.py
python tests/test_shot_planner.py
python tests/test_video_assembler.py
python tests/test_demo_pipeline.py
```

---

## Submission Deliverables

The submission package for the **Junk Food** demo is stored in `submission/` and `junk_food_reel_submission.zip`:
- `reel.json`: Exact JSON containing `"topic"`, `"framework"`, and `"script"`.
- `framework_selected.txt`: Framework #15 ("What Happens If"), Score: 75.0.
- `script.txt`: Complete approved script including CTA.
- `junk_food_reel.mp4`: Rendered 9:16 vertical reel ($720\times 1280$, 40.0s, AAC audio), compressed to 6.68 MB to satisfy the $\le 10$ MB upload limit.
- `README_OUTPUT.txt`: Submission summary.
- `junk_food_reel_submission.zip`: Complete archive (6.65 MB, $\le 10$ MB limit).
- `outputs/final_reel.mp4`: Original uncompressed master render (13.09 MB) preserved separately.
