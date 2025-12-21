# R&D Tax Credit — AI Agent (MVP)

AI-assisted tool to classify projects for IRS §41 R&D tax credit eligibility, capture audit-ready traces, and export Form 6765 / audit packages. Now includes Human-in-the-Loop (HITL) review and local-LLM support.

## Quick start (Windows / PowerShell)
1) Create venv & install deps
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2) Configure secrets locally (do NOT commit)
   # .env
   VALID_API_KEYS=demo-key
   USER_ROLES={"admin":"demo-key"}
   # For OpenAI cloud (optional)
   # OPENAI_API_KEY=sk-...
   # OPENAI_MODEL=gpt-4o-mini
   # For local LLM (e.g., LM Studio / Ollama)
   # LLM_BASE_URL=http://localhost:11434/v1

3) Start backend (FastAPI)
   uvicorn app.main:app --reload --port 8000

4) Start Streamlit UI
   streamlit run streamlit_app.py
   (defaults Backend URL to http://127.0.0.1:8000 and pre-fills API key from VALID_API_KEYS)

## Features
- Tiered classifier: rule-out → heuristic → LLM (with local-LLM support via `LLM_BASE_URL` and dummy key)
- HITL review: low-confidence items (confidence < 0.75) queue for expert review; human override sets confidence=1.0
- Exports: Form 6765 PDF and audit package ZIP use the final (human-overridden if present) decision
- Trace logging: immutable traces with pointers included in audit package

## Human-in-the-Loop workflow
- After classify, items below confidence threshold appear in **Expert Review Queue** in Streamlit.
- Reviewer selects a project, sets Final Decision (Eligible/Not Eligible), and adds rationale.
- On commit: status becomes Reviewed, confidence set to 1.0, and exports use the human decision.

## Local LLM usage
- Set `LLM_BASE_URL` (default `http://localhost:11434/v1`).
- If no `OPENAI_API_KEY` is present and base URL is local, a dummy key (`lm-studio`) is used to avoid 401s.
- If the local LLM is offline, the app falls back to heuristics without crashing.

## Notes
- Keep `.env` and API keys local; do not commit secrets.
- For CI/CD, store secrets in GitHub Secrets.
- See IMPLEMENTATION_SUMMARY.md and QUICK_START.md for deeper details.
