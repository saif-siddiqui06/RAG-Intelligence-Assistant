"""Agentic RAG Research Assistant — Streamlit frontend.

Milestone 1: document ingestion UI (upload, list, preview chunks,
delete, re-index).
Milestone 2: conversational RAG chat, streamed, with metadata filters,
citations, confidence and a retrieval debug panel.

Run with: streamlit run frontend/streamlit_app.py
"""
import requests
import streamlit as st
from api_client import (
    API_BASE_URL,
    api_error_detail,
    ask_chat_stream,
    delete_document,
    get_document_chunks,
    get_health,
    get_stats,
    list_documents,
    reindex_document,
    upload_documents,
)

st.set_page_config(page_title="Agentic RAG Research Assistant", page_icon="📚", layout="wide")

st.title("📚 Agentic RAG Research Assistant")
st.caption("Milestone 2 — conversational RAG (retrieval, query rewriting, citations).")

with st.sidebar:
    st.header("Backend")
    st.code(API_BASE_URL, language="text")
    if st.button("Check connection", use_container_width=True):
        try:
            health = get_health()
            st.success(f"Connected — {health['app_name']} v{health['version']}")
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

chat_tab, documents_tab = st.tabs(["💬 Chat", "📄 Documents"])


def _render_meta(meta: dict) -> None:
    if meta.get("sources"):
        lines = [
            f"[{s['index']}] {s['document_name']} — Page {s['page_number']}" for s in meta["sources"]
        ]
        st.caption("**Sources:**  \n" + "  \n".join(lines))
    else:
        st.caption("**Sources:** none")
    st.caption(f"Confidence: **{meta.get('confidence', '?')}**")

    with st.expander("Retrieval details"):
        st.caption(f"Rewritten query: `{meta.get('rewritten_query', '')}`")

        diagnostics = meta.get("retrieval_diagnostics")
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
            for chunk in meta.get("retrieved_chunks", []):
                st.text(
                    f"score={chunk['score']:.3f}  ·  {chunk['filename']}  ·  page {chunk['page_number']}"
                )
                st.caption(chunk["content"][:300])


with chat_tab:
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        completed_docs = [
            d for d in list_documents().get("documents", []) if d["status"] == "completed"
        ]
    except Exception:
        completed_docs = []

    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

    doc_options = {"All documents": None}
    doc_options.update(
        {f"{d['filename']} ({d['document_id'][:8]})": d["document_id"] for d in completed_docs}
    )
    selected_doc_label = filter_col1.selectbox("Search scope", list(doc_options.keys()))
    selected_document_id = doc_options[selected_doc_label]

    doc_type_options = {"Any type": None, "PDF": "pdf"}
    selected_type_label = filter_col2.selectbox("Document type", list(doc_type_options.keys()))
    selected_document_type = doc_type_options[selected_type_label]

    if filter_col3.button("New conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.rerun()

    if not completed_docs:
        st.info("Upload and ingest at least one document (Documents tab) before chatting.")

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                _render_meta(turn["meta"])

    question = st.chat_input(
        "Ask a question about your documents...", disabled=not completed_docs
    )
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
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

with documents_tab:
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns(2)
    chunk_size = col1.number_input(
        "Chunk size (chars)", min_value=100, max_value=8000, value=1000, step=100
    )
    chunk_overlap = col2.number_input(
        "Chunk overlap (chars)", min_value=0, max_value=2000, value=150, step=50
    )

    if st.button("Ingest", disabled=not uploaded_files, type="primary"):
        with st.spinner("Extracting, chunking and embedding..."):
            try:
                results = upload_documents(uploaded_files, int(chunk_size), int(chunk_overlap))
                st.success(f"Ingested {len(results)} document(s).")
                st.json(results)
            except requests.exceptions.HTTPError as exc:
                st.error(f"Ingestion failed: {api_error_detail(exc)}")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Documents")

    try:
        documents = list_documents().get("documents", [])
    except Exception as exc:
        documents = []
        st.error(f"Could not load documents: {exc}")

    if not documents:
        st.info("No documents ingested yet.")

    for doc in documents:
        with st.container(border=True):
            cols = st.columns([3, 1, 1, 1, 1, 1])
            cols[0].markdown(f"**{doc['filename']}**  \n`{doc['document_id']}`")
            cols[1].write(doc["status"])
            cols[2].write(f"{doc.get('num_pages') or '-'} pages")
            cols[3].write(f"{doc.get('num_chunks') or '-'} chunks")

            if cols[4].button("Reindex", key=f"reindex-{doc['document_id']}"):
                try:
                    reindex_document(doc["document_id"])
                    st.success("Reindexed.")
                    st.rerun()
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Reindex failed: {api_error_detail(exc)}")
                except Exception as exc:
                    st.error(f"Reindex failed: {exc}")

            if cols[5].button("Delete", key=f"delete-{doc['document_id']}"):
                try:
                    delete_document(doc["document_id"])
                    st.success("Deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")

            if doc["status"] == "failed" and doc.get("error_message"):
                st.error(doc["error_message"])

            with st.expander("Preview chunks"):
                try:
                    chunks = get_document_chunks(doc["document_id"])
                    for chunk in chunks[:20]:
                        st.caption(
                            f"Page {chunk['page_number']} · chunk {chunk['chunk_index']} · `{chunk['chunk_id']}`"
                        )
                        st.text(chunk["content"][:500])
                    if len(chunks) > 20:
                        st.caption(f"...and {len(chunks) - 20} more chunks.")
                except Exception as exc:
                    st.caption(f"Could not load chunks: {exc}")
