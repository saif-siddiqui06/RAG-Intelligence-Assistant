"""Evaluation page — retrieval quality benchmark (vector-only vs
BM25-only vs hybrid fusion vs hybrid+rerank) against the built-in
22-question dataset (see app.evaluation.dataset).

Each run makes ~20+ real Gemini embedding calls, so this is a manual,
button-triggered action, not something that runs on page load — the
backend also rate-limits it (RATE_LIMIT_EVALUATION, default 3/hour) as
a second line of defense against burning the free-tier daily quota.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
from api_client import api_error_detail, run_evaluation_benchmark

st.set_page_config(page_title="Evaluation — Agentic RAG", page_icon="📊", layout="wide")
st.title("📊 Retrieval evaluation")
st.caption(
    "Compares vector-only, BM25-only, hybrid fusion and hybrid+rerank retrieval "
    "against a hand-built 22-question dataset. Makes real embedding-API calls — "
    "run sparingly."
)

if st.button("▶️ Run benchmark", type="primary"):
    with st.spinner("Embedding corpus, running BM25, reranking... (can take ~30-60s)"):
        try:
            st.session_state.eval_result = run_evaluation_benchmark()
        except requests.exceptions.HTTPError as exc:
            st.error(f"Benchmark failed: {api_error_detail(exc)}")
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")

result = st.session_state.get("eval_result")
if not result:
    st.info("No results yet — click **Run benchmark** above.")
else:
    results = result["results"]
    per_question = result["per_question"]
    strategies = list(results.keys())
    k = len(per_question[0].get(strategies[0], [])) if per_question else 0

    st.subheader("Overall metrics")
    table = [
        {
            "Strategy": name,
            f"Recall@{k}": round(sum(results[name]["recall"]) / len(results[name]["recall"]), 3),
            "MRR": round(sum(results[name]["mrr"]) / len(results[name]["mrr"]), 3),
        }
        for name in strategies
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("By question category")
    categories = sorted({row["note"] for row in per_question})
    for category in categories:
        indices = [i for i, row in enumerate(per_question) if row["note"] == category]
        with st.expander(f"{category}  (n={len(indices)})"):
            cat_table = [
                {
                    "Strategy": name,
                    f"Recall@{k}": round(
                        sum(results[name]["recall"][i] for i in indices) / len(indices), 3
                    ),
                    "MRR": round(sum(results[name]["mrr"][i] for i in indices) / len(indices), 3),
                }
                for name in strategies
            ]
            st.dataframe(cat_table, use_container_width=True, hide_index=True)

    st.subheader("Failed queries (relevant passage missed in top-k)")
    for name in strategies:
        misses = [
            row["question"] for i, row in enumerate(per_question) if results[name]["recall"][i] == 0.0
        ]
        with st.expander(f"{name}: {len(misses)} miss(es)"):
            if not misses:
                st.caption("None — every relevant passage was retrieved in the top-k.")
            for question in misses:
                st.caption(f"- {question}")
