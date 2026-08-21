"""Document library page — upload, list, preview chunks, re-index, delete."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from api_client import (
    api_error_detail,
    delete_document,
    get_document_chunks,
    list_documents,
    reindex_document,
    upload_documents,
)
import requests

st.set_page_config(page_title="Documents — Agentic RAG", page_icon="📄", layout="wide")
st.title("📄 Document library")

st.subheader("Upload documents")
uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns(2)
chunk_size = col1.number_input("Chunk size (chars)", min_value=100, max_value=8000, value=1000, step=100)
chunk_overlap = col2.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=150, step=50)

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
