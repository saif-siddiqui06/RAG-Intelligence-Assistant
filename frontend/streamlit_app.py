"""Agentic RAG Research Assistant — Streamlit frontend (Chat page).

Milestone 1: document ingestion UI (now on the Documents page).
Milestone 2: conversational RAG chat, streamed, with metadata filters,
citations, confidence and a retrieval debug panel.
Milestone 6: multi-page app (Chat / Documents / Evaluation / Settings),
conversation history in the sidebar, and an agent mode with a
tool-usage indicator alongside the streaming RAG mode.

Run with: streamlit run frontend/streamlit_app.py
"""
import requests
import streamlit as st
from api_client import (
    API_BASE_URL,
    api_error_detail,
    ask_chat_stream,
    delete_conversation,
    get_conversation,
    get_health,
    get_stats,
    list_conversations,
    list_documents,
    run_agent,
)

st.set_page_config(page_title="Agentic RAG Research Assistant", page_icon="📚", layout="wide")

st.title("📚 Agentic RAG Research Assistant")
st.caption("Conversational RAG with citations, hybrid retrieval and an agent that can reach for tools.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Backend")
    st.code(API_BASE_URL, language="text")
    if st.button("Check connection", use_container_width=True):
        try:
            health = get_health()
            status_icon = "✅" if health.get("database") == "ok" else "⚠️"
            st.success(f"{status_icon} {health['app_name']} v{health['version']} — db: {health.get('database', '?')}")
        except Exception as exc:
            st.error(f"Could not reach backend: {exc}")

    st.divider()
    st.subheader("Storage stats")
    try:
        stats = get_stats()
        st.metric("Documents", stats["total_documents"])
        st.metric("Chunks", stats["total_chunks"])
        st.metric("Vectors in index", stats["vector_count"])
    except Exception as exc:
        st.caption(f"Stats unavailable: {exc}")

    st.divider()
    if st.button("➕ New conversation", use_container_width=True, type="primary"):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.rerun()

    st.subheader("Conversation history")
    try:
        conversations = list_conversations()
    except Exception as exc:
        conversations = []
        st.caption(f"History unavailable: {exc}")

    if not conversations:
        st.caption("No saved conversations yet.")
    for conv in conversations:
        label = conv.get("title") or "(untitled)"
        is_active = conv["conversation_id"] == st.session_state.session_id
        row = st.columns([4, 1])
        if row[0].button(
            f"{'🟢 ' if is_active else ''}{label}",
            key=f"load-{conv['conversation_id']}",
            use_container_width=True,
        ):
            try:
                detail = get_conversation(conv["conversation_id"])
                st.session_state.session_id = detail["conversation_id"]
                st.session_state.chat_history = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "meta": {"sources": m["sources"]} if m["role"] == "assistant" else None,
                    }
                    for m in detail["messages"]
                ]
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load conversation: {exc}")
        if row[1].button("🗑️", key=f"delete-{conv['conversation_id']}"):
            try:
                delete_conversation(conv["conversation_id"])
                if is_active:
                    st.session_state.session_id = None
                    st.session_state.chat_history = []
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete conversation: {exc}")

    st.divider()
    st.page_link("pages/1_Documents.py", label="📄 Document library")
    st.page_link("pages/2_Evaluation.py", label="📊 Evaluation")
    st.page_link("pages/3_Settings.py", label="⚙️ Settings")


def _render_meta(meta: dict) -> None:
    if meta.get("sources"):
        lines = [
            f"[{s['index']}] {s['document_name']} — Page {s['page_number']}" for s in meta["sources"]
        ]
        st.caption("**Sources:**  \n" + "  \n".join(lines))
    else:
        st.caption("**Sources:** none")
    if "confidence" in meta:
        st.caption(f"Confidence: **{meta.get('confidence', '?')}**")

    diagnostics = meta.get("retrieval_diagnostics")
    retrieved_chunks = meta.get("retrieved_chunks")
    if "rewritten_query" in meta or diagnostics or retrieved_chunks:
        with st.expander("Retrieval details"):
            if "rewritten_query" in meta:
                st.caption(f"Rewritten query: `{meta.get('rewritten_query', '')}`")

            if diagnostics:
                st.caption("**Hybrid retrieval pipeline** — vector + BM25 → fusion → rerank:")
                stage_tabs = st.tabs(["Vector", "Keyword (BM25)", "Fused", "Reranked (final)"])
                stage_keys = ["vector_results", "keyword_results", "fused_results", "reranked_results"]
                for tab, key in zip(stage_tabs, stage_keys):
                    with tab:
                        stage_chunks = diagnostics.get(key, [])
                        if not stage_chunks:
                            st.caption("(no candidates at this stage)")
                        for chunk in stage_chunks:
                            st.text(
                                f"score={chunk['score']:.3f}  ·  {chunk['filename']}  ·  page {chunk['page_number']}"
                            )
                            st.caption(chunk["content"][:300])
            else:
                for chunk in retrieved_chunks or []:
                    st.text(
                        f"score={chunk['score']:.3f}  ·  {chunk['filename']}  ·  page {chunk['page_number']}"
                    )
                    st.caption(chunk["content"][:300])


def _render_tool_usage(meta: dict) -> None:
    tools_used = meta.get("tools_used") or []
    st.caption("**Tools used:** " + (" ".join(f"`{t}`" for t in tools_used) if tools_used else "none"))
    if meta.get("sources"):
        lines = [
            f"[{s['tool']}] " + (s.get("title") or s.get("document_name") or s.get("url") or "")
            for s in meta["sources"]
        ]
        st.caption("**Sources:**  \n" + "  \n".join(lines))
    with st.expander("Reasoning summary"):
        st.caption(meta.get("reasoning_summary", ""))
    st.caption(f"Execution time: {meta.get('execution_time', 0):.2f}s")


try:
    completed_docs = [d for d in list_documents().get("documents", []) if d["status"] == "completed"]
except Exception:
    completed_docs = []

mode_col, scope_col, type_col = st.columns([1.2, 1, 1])
mode = mode_col.radio(
    "Mode", ["💬 Direct RAG (streaming)", "🤖 Agent (tools)"], horizontal=True, label_visibility="collapsed"
)

doc_options = {"All documents": None}
doc_options.update(
    {f"{d['filename']} ({d['document_id'][:8]})": d["document_id"] for d in completed_docs}
)
selected_doc_label = scope_col.selectbox("Search scope", list(doc_options.keys()))
selected_document_id = doc_options[selected_doc_label]

doc_type_options = {"Any type": None, "PDF": "pdf"}
selected_type_label = type_col.selectbox("Document type", list(doc_type_options.keys()))
selected_document_type = doc_type_options[selected_type_label]

if mode.startswith("🤖"):
    st.caption("Agent-mode turns are not saved to conversation history (only Direct RAG turns are).")

if not completed_docs:
    st.info("Upload and ingest at least one document on the Document library page before chatting.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("meta"):
            if "tools_used" in turn["meta"]:
                _render_tool_usage(turn["meta"])
            else:
                _render_meta(turn["meta"])

question = st.chat_input("Ask a question about your documents...", disabled=not completed_docs)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if mode.startswith("🤖"):
            with st.spinner("Agent is choosing tools and working..."):
                try:
                    result = run_agent(
                        question,
                        session_id=st.session_state.session_id,
                        document_id=selected_document_id,
                        document_type=selected_document_type,
                    )
                    answer = result["answer"]
                    st.markdown(answer)
                    _render_tool_usage(result)
                    meta = result
                except requests.exceptions.HTTPError as exc:
                    answer, meta = "", None
                    st.error(f"Agent request failed: {api_error_detail(exc)}")
                except Exception as exc:
                    answer, meta = "", None
                    st.error(f"Agent request failed: {exc}")
        else:
            stream, result = ask_chat_stream(
                question,
                session_id=st.session_state.session_id,
                document_id=selected_document_id,
                document_type=selected_document_type,
            )
            try:
                answer = st.write_stream(stream)
            except requests.exceptions.HTTPError as exc:
                answer = ""
                st.error(f"Chat request failed: {api_error_detail(exc)}")

            meta = result.get("meta")
            if result.get("error"):
                st.error(result["error"])
            elif meta:
                st.session_state.session_id = meta["session_id"]
                _render_meta(meta)

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "meta": meta})
