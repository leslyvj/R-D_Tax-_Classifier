import streamlit as st
import pandas as pd
import requests
import os
import json
import zipfile
from io import BytesIO

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI R&D Tax Credit Agent - MVP", layout="wide")

st.title("AI R&D Tax Credit Agent - MVP (Phase 1)")
st.write("Upload a CSV of project descriptions to classify R&D tax credit eligibility and export supporting documents.")

CONFIDENCE_THRESHOLD = 0.75

if "projects" not in st.session_state:
    st.session_state["projects"] = {}  # project_id -> project dict with ai_decision/human_decision
if "needs_review_ids" not in st.session_state:
    st.session_state["needs_review_ids"] = []


def get_final_decision(project: dict):
    """Return (label, confidence, source) preferring human override when present."""
    human = project.get("human_decision")
    if human:
        return human.get("final_label"), human.get("confidence", 1.0), "Human"
    ai = project.get("ai_decision", {})
    return ai.get("label"), ai.get("confidence", 0.0), "AI"


def build_display_df():
    rows = []
    for pid, project in st.session_state["projects"].items():
        label, conf, source = get_final_decision(project)
        rows.append(
            {
                "project_id": pid,
                "project_name": project.get("project_name", ""),
                "eligible": label == "Eligible",
                "confidence": conf,
                "decision_source": source,
                "status": project.get("status", "AI Classified"),
                "region": project.get("region"),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def ingest_results(results):
    """Sync backend results into session state and flag low-confidence items."""
    projects = st.session_state["projects"]
    st.session_state["needs_review_ids"] = []
    for row in results:
        pid = str(row.get("project_id"))
        ai_decision = {
            "label": "Eligible" if row.get("eligible") else "Not Eligible",
            "confidence": float(row.get("confidence", 0.0)),
            "rationale": row.get("rationale", ""),
        }
        project = projects.get(pid, {})
        project.update(
            {
                "project_id": pid,
                "project_name": row.get("project_name", ""),
                "region": row.get("region"),
                "trace_path": row.get("trace_path"),
                "ai_decision": ai_decision,
            }
        )

        if project.get("human_decision"):
            project["status"] = "Reviewed"
        else:
            if ai_decision["confidence"] < CONFIDENCE_THRESHOLD:
                project["status"] = "Needs Review"
                st.session_state["needs_review_ids"].append(pid)
            else:
                project["status"] = "AI Classified"

        projects[pid] = project

    st.session_state["results_df"] = build_display_df()


def make_export_payload(project_id: str) -> dict:
    project = st.session_state["projects"][project_id]
    label, conf, source = get_final_decision(project)
    rationale = project.get("human_decision", {}).get("rationale") or project.get("ai_decision", {}).get("rationale", "")
    return {
        "project_id": project_id,
        "project_name": project.get("project_name", ""),
        "region": project.get("region"),
        "eligible": label == "Eligible",
        "eligible_label": label,
        "confidence": conf,
        "decision_source": source,
        "status": project.get("status", ""),
        "rationale": rationale,
        "trace_path": project.get("trace_path", ""),
    }


def render_form_6765_pdf(payload: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "Form 6765 (MVP) - Credit for Increasing Research Activities")

    c.setFont("Helvetica", 11)
    c.drawString(72, 720, f"Project ID: {payload['project_id']}")
    c.drawString(72, 705, f"Project Name: {payload.get('project_name', '')}")
    c.drawString(72, 690, f"Region: {payload.get('region', '')}")
    c.drawString(72, 675, f"Eligible: {payload.get('eligible_label', '')}")
    c.drawString(72, 660, f"Confidence: {payload.get('confidence', 0.0):.2f} ({payload.get('decision_source', '')})")

    rationale = (payload.get("rationale") or "")[:1200]
    c.drawString(72, 640, "Rationale (truncated):")
    text_obj = c.beginText(72, 625)
    text_obj.setFont("Helvetica", 10)
    for line in rationale.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def render_audit_zip(payload: dict) -> bytes:
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classification_summary.json", json.dumps(payload, indent=2))
        zf.writestr("trace_pointer.txt", payload.get("trace_path", ""))
    zip_buf.seek(0)
    return zip_buf.getvalue()

backend_url = st.text_input(
    "Backend URL",
    value=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"),
)
user_id = st.text_input("User ID (for trace)", value="demo-user")

_default_api_key = os.environ.get("API_KEY_DEFAULT")
if not _default_api_key:
    _valid_keys_env = os.environ.get("VALID_API_KEYS", "")
    if _valid_keys_env:
        _default_api_key = _valid_keys_env.split(",")[0].strip()
api_key = st.text_input("API Key (X-API-Key)", value=_default_api_key or "", type="password")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

results_df = st.session_state.get("results_df", None)

if uploaded and st.button("Analyze"):
    if not api_key:
        st.error("API key is required.")
    else:
        with st.spinner("Classifying... (this may take a minute)"):
            try:
                resp = requests.post(
                    f"{backend_url}/classify_rnd",
                    files={"file": uploaded},
                    data={"user_id": user_id},
                    headers={"X-API-Key": api_key},
                    timeout=300,  # 5 minute timeout for classification
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    if "results" in payload:
                        ingest_results(payload["results"])
                        results_df = st.session_state.get("results_df")
                        st.success(f"Processed {payload.get('count', len(payload['results']))} rows.")
                        if results_df is not None and not results_df.empty:
                            st.dataframe(results_df, use_container_width=True)
                    else:
                        st.error(payload)
                else:
                    st.error(f"Error: {resp.status_code} -> {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# Reload results if they exist in session
if results_df is not None and not results_df.empty:
    st.markdown("### Classification Results (final)")
    st.dataframe(results_df, use_container_width=True)

    st.markdown("### Expert Review Queue")
    needs_review = [pid for pid in st.session_state.get("needs_review_ids", []) if pid in st.session_state["projects"]]
    if needs_review:
        selected_pid = st.selectbox(
            "Select a project requiring review",
            needs_review,
            format_func=lambda pid: f"{pid} - {st.session_state['projects'][pid].get('project_name', '')}",
            key="needs_review_select",
        )
        selected_proj = st.session_state["projects"][selected_pid]
        ai_decision = selected_proj.get("ai_decision", {})
        st.write("AI decision:", ai_decision.get("label"))
        st.write("AI confidence:", f"{ai_decision.get('confidence', 0.0):.2f}")
        st.write("AI rationale:")
        st.write(ai_decision.get("rationale", ""))

        with st.form(key=f"review_form_{selected_pid}"):
            default_idx = 0 if ai_decision.get("label") == "Eligible" else 1
            final_label = st.radio("Final Decision", ["Eligible", "Not Eligible"], index=default_idx)
            final_rationale = st.text_area("Expert Rationale", height=160)
            commit = st.form_submit_button("Commit Review")
            if commit:
                st.session_state["projects"][selected_pid]["human_decision"] = {
                    "final_label": final_label,
                    "rationale": final_rationale,
                    "confidence": 1.0,
                }
                st.session_state["projects"][selected_pid]["status"] = "Reviewed"
                if selected_pid in st.session_state["needs_review_ids"]:
                    st.session_state["needs_review_ids"].remove(selected_pid)
                st.session_state["results_df"] = build_display_df()
                results_df = st.session_state["results_df"]
                st.success(f"Review committed for project {selected_pid}")

    st.markdown("### Export Tools")
    project_ids = results_df["project_id"].astype(str).tolist()
    selected_project = st.selectbox("Select Project ID", project_ids)

    col1, col2 = st.columns(2)

    with col1:
        payload = make_export_payload(selected_project)
        pdf_bytes = render_form_6765_pdf(payload)
        st.download_button(
            label="Download Form 6765 PDF",
            data=pdf_bytes,
            file_name=f"form6765_{selected_project}.pdf",
            mime="application/pdf",
            key=f"dl_pdf_{selected_project}",
        )

    with col2:
        payload = make_export_payload(selected_project)
        zip_bytes = render_audit_zip(payload)
        st.download_button(
            label="Download Audit Package ZIP",
            data=zip_bytes,
            file_name=f"audit_package_{selected_project}.zip",
            mime="application/zip",
            key=f"dl_zip_{selected_project}",
        )

st.markdown("---")
st.caption("Tip: Start the backend with `uvicorn app.main:app --reload --port 8000` before running Streamlit.")