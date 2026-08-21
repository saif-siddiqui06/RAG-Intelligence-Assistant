"""Settings / diagnostics page.

Read-only: the backend loads configuration once at startup from
environment variables (see .env.example), so there's nothing here to
edit at runtime — this page just surfaces what the backend is actually
running with, for debugging a deployment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from api_client import API_BASE_URL, get_health, get_stats

st.set_page_config(page_title="Settings — Agentic RAG", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")

st.subheader("Backend connection")
st.code(f"API_BASE_URL = {API_BASE_URL}", language="text")

try:
    health = get_health()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", health["status"])
    col2.metric("Environment", health["environment"])
    col3.metric("Version", health["version"])
    col4.metric("Database", health["database"])
except Exception as exc:
    st.error(f"Could not reach backend: {exc}")

st.subheader("Storage")
try:
    stats = get_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", stats["total_documents"])
    col2.metric("Chunks", stats["total_chunks"])
    col3.metric("Vectors in index", stats["vector_count"])
except Exception as exc:
    st.caption(f"Stats unavailable: {exc}")

st.divider()
st.caption(
    "All other configuration (retrieval mode, chunk size, rate limits, model "
    "names, etc.) is environment-variable driven on the backend — see "
    "`.env.example` and the README's configuration reference."
)
