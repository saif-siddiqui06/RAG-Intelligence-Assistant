"""Retrieval benchmark: vector-only vs BM25-only vs hybrid (fusion) vs
hybrid+reranking, measured against app.evaluation.dataset's 22
hand-built question/passage pairs.

This is a real experiment, not a mocked unit test: it embeds the
14-passage corpus with the actual configured Gemini embedding model
and reranks with the actual cross-encoder. Requires a real
GEMINI_API_KEY in .env.

Run with:
    python -m app.evaluation.retrieval_benchmark
"""
import sys
import tempfile
import time
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.dataset import CORPUS, QUESTIONS
from app.evaluation.metrics import mean, reciprocal_rank, recall_at_k
from app.rag.embeddings.gemini_embedder import GeminiEmbedder
from app.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrieval.bm25_retriever import BM25Retriever
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.faiss_store import FaissVectorStore

STRATEGIES = ["vector-only", "bm25-only", "hybrid (fusion)", "hybrid + rerank"]


def _build_vector_index(embedder, index_dir: Path) -> tuple[VectorRetriever, dict[int, str]]:
    doc_ids = list(CORPUS.keys())
    vectors = embedder.embed_documents(list(CORPUS.values()))
    store = FaissVectorStore(index_path=index_dir / "bench_index.faiss", dimension=embedder.dimension)
    vector_ids = store.add(vectors)
    id_map = dict(zip(vector_ids, doc_ids))  # int vector_id -> corpus doc id
    return VectorRetriever(embedder, store), id_map


def _fuse(vector_ranked: list[str], bm25_ranked: list[str], rrf_k: int) -> list[str]:
    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=rrf_k)
    return [doc_id for doc_id, _ in fused]


def run_benchmark() -> dict:
    settings = get_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set in .env — this benchmark embeds a real")
        print("corpus and needs a real key. See README §5 for a free key.")
        sys.exit(1)

    print(f"Corpus: {len(CORPUS)} passages | Questions: {len(QUESTIONS)}")
    print("Embedding corpus, building BM25 index, loading cross-encoder...")

    embedder = GeminiEmbedder(settings)
    with tempfile.TemporaryDirectory(prefix="retrieval_benchmark_") as tmp:
        retriever, id_map = _build_vector_index(embedder, Path(tmp))
        bm25 = BM25Retriever(list(CORPUS.items()))
        reranker = CrossEncoderReranker(settings.reranker_model)

        k = settings.final_context_k
        vector_top_k = settings.vector_top_k
        bm25_top_k = settings.bm25_top_k
        rerank_top_k = settings.rerank_top_k

        results: dict[str, dict[str, list[float]]] = {
            name: {"recall": [], "mrr": []} for name in STRATEGIES
        }
        per_question: list[dict] = []
        elapsed: dict[str, float] = dict.fromkeys(STRATEGIES, 0.0)

        for q in QUESTIONS:
            t0 = time.perf_counter()
            vector_ranked_k = [id_map[h.vector_id] for h in retriever.search(q.question, top_k=k)]
            elapsed["vector-only"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            bm25_ranked_k = [h.chunk_id for h in bm25.search(q.question, top_k=k)]
            elapsed["bm25-only"] += time.perf_counter() - t0

            # Hybrid needs the wider over-fetched lists to fuse from.
            # Timed as one block, then reused for the rerank strategy
            # below — reranking's own reported time is *additional* to
            # this, not standalone, so each strategy's printed total is
            # its real, complete, end-to-end wall-clock cost.
            t0 = time.perf_counter()
            vector_wide = [id_map[h.vector_id] for h in retriever.search(q.question, top_k=vector_top_k)]
            bm25_wide = [h.chunk_id for h in bm25.search(q.question, top_k=bm25_top_k)]
            fused = _fuse(vector_wide, bm25_wide, settings.rrf_k)
            fused_k = fused[:k]
            fusion_time = time.perf_counter() - t0
            elapsed["hybrid (fusion)"] += fusion_time

            t0 = time.perf_counter()
            fused_for_rerank = fused[:rerank_top_k]
            if fused_for_rerank:
                scores = reranker.score(q.question, [CORPUS[doc_id] for doc_id in fused_for_rerank])
                reranked_k = [
                    doc_id
                    for doc_id, _ in sorted(
                        zip(fused_for_rerank, scores), key=lambda pair: pair[1], reverse=True
                    )
                ][:k]
            else:
                reranked_k = []
            rerank_time = time.perf_counter() - t0
            # Cumulative: fusion's cost is real work "hybrid + rerank"
            # also depends on, not something it gets for free.
            elapsed["hybrid + rerank"] += fusion_time + rerank_time

            run = {
                "vector-only": vector_ranked_k,
                "bm25-only": bm25_ranked_k,
                "hybrid (fusion)": fused_k,
                "hybrid + rerank": reranked_k,
            }
            for name, ranked in run.items():
                results[name]["recall"].append(recall_at_k(ranked, q.relevant_ids, k))
                results[name]["mrr"].append(reciprocal_rank(ranked, q.relevant_ids))
            per_question.append({"question": q.question, "note": q.note, "relevant": q.relevant_ids, **run})

    _print_report(results, per_question, k, elapsed)
    return {"results": results, "per_question": per_question}


def _print_report(results, per_question, k: int, elapsed: dict[str, float]) -> None:
    print()
    print(f"Overall (n={len(per_question)} questions, k={k}):")
    print(f"{'Strategy':<20} {'Recall@' + str(k):<12} {'MRR':<8} {'Total time (s)':<15}")
    print("-" * 58)
    for name in STRATEGIES:
        recall = mean(results[name]["recall"])
        mrr = mean(results[name]["mrr"])
        print(f"{name:<20} {recall:<12.3f} {mrr:<8.3f} {elapsed[name]:<15.2f}")

    print()
    print("By question category:")
    categories = sorted({row["note"] for row in per_question})
    for category in categories:
        indices = [i for i, row in enumerate(per_question) if row["note"] == category]
        print(f"\n  {category}  (n={len(indices)}):")
        for name in STRATEGIES:
            cat_recall = mean([results[name]["recall"][i] for i in indices])
            cat_mrr = mean([results[name]["mrr"][i] for i in indices])
            print(f"    {name:<20} Recall@{k} = {cat_recall:.3f}   MRR = {cat_mrr:.3f}")

    print()
    print("Questions each strategy got wrong (relevant id not in top-k):")
    for name in STRATEGIES:
        misses = [
            row["question"]
            for i, row in enumerate(per_question)
            if results[name]["recall"][i] == 0.0
        ]
        print(f"\n  {name}: {len(misses)} miss(es)")
        for question in misses:
            print(f"    - {question}")


if __name__ == "__main__":
    run_benchmark()
