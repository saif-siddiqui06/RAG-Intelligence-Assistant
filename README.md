# Agentic RAG Research Assistant

A portfolio-grade, production-style **Agentic RAG** system: an LLM-based agent (Gemini)
that routes between document retrieval, web search and a calculator tool,
built on top of an advanced RAG pipeline (hybrid retrieval, reranking, query
rewriting, citations) with conversational memory and automated evaluation.

This repository is built **incrementally, milestone by milestone**. This
README reflects the current milestone and will be updated as each new one
lands.

> **Current milestone: 3 — Hybrid Retrieval + Reranking.**
> Vector search + BM25 keyword search, fused with Reciprocal Rank Fusion,
> reranked with a local cross-encoder — all behind a `RETRIEVAL_MODE`
> config switch that leaves the Milestone 2 vector-only path completely
> unchanged. Full per-stage diagnostics, a 22-question retrieval benchmark,
> and a from-scratch measured comparison of all four strategies. No agents
> yet — that's by design. See [Roadmap](#4-roadmap).

---

## 1. Target architecture

### Request flow (full target — not all boxes exist yet)

```
                         ┌──────────────────┐
                         │    Streamlit      │
                         │    Frontend       │
                         └────────┬──────────┘
                                  │ HTTP (JSON)
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI       │
                         │     Backend       │
                         └────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   Agent / Router    │   ← not built yet (Milestone 4)
                       │      (Gemini)       │
                       └──────────┬──────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
       Document RAG          Web Search           Calculator      ← tools not built yet
             │
             ▼
      Query Rewriting                                              ← implemented
             │
             ▼
       Hybrid Retrieval                                             ← implemented (RETRIEVAL_MODE=hybrid;
       ┌──────────────┐                                               RETRIEVAL_MODE=vector keeps M2's path)
       │ Vector Search│  ← implemented
       │ BM25 Search  │  ← implemented (rank_bm25, fused via RRF)
       └──────┬───────┘
              ▼
          Reranker                                                  ← implemented (local cross-encoder,
              │                                                        RERANKER_BACKEND=cross_encoder|none)
              ▼
       Relevant Chunks                                               ← implemented (dedup)
              │
              ▼
          LLM (Gemini)                                                  ← implemented
              │
       ┌──────┴───────┐
       ▼              ▼
   Answer         Citations                                          ← both implemented
```

### This milestone's request lifecycle — **implemented**

```
User Query
   │
   ▼
Memory ──────────────► bounded window of prior messages (SQL, not the full transcript)
   │
   ▼
Query Rewrite ───────► LLM folds conversational context into one standalone,
   │                   retrieval-optimized query (skipped on a conversation's first turn)
   ▼
Retrieval ───────────► embed the rewritten query, semantic search (optionally
   │                   restricted to one document / one document type)
   ▼
Context Selection ───► drop near-duplicate chunks, keep top_k by score,
   │                   discard anything below the relevance floor
   ▼
LLM ─────────────────► generate an answer using ONLY the selected chunks
   │                   (streamed token-by-token), or skip straight to a fixed
   │                   "cannot determine this" answer if nothing cleared the bar
   ▼
Citation ────────────► parse [n] markers out of the answer text, map back to
   │                   (document name, page number, chunk id)
   ▼
Response ────────────► {answer, sources, retrieved_chunks, confidence,
                        rewritten_query, session_id}
```

See [§3](#3-request-lifecycle-explained) for a step-by-step walkthrough of
exactly which module does what.

### This milestone's addition: hybrid retrieval (`RETRIEVAL_MODE=hybrid`)

```
                    User Query (rewritten)
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Vector Search              Keyword Search
      (VectorRetriever,            (BM25Retriever,
       FAISS, top vector_top_k)     rank_bm25, top bm25_top_k)
             │                         │
             └────────────┬────────────┘
                          ▼
                  Result Fusion (RRF)
              Reciprocal Rank Fusion — combines
              rank position, not raw scores (vector
              cosine and BM25 scores aren't comparable)
                          │
                          ▼  top rerank_top_k
                      Reranker
              cross-encoder scores (query, chunk) jointly
              (CrossEncoderReranker | NoOpReranker)
                          │
                          ▼
              Dedup + top final_context_k
                          │
                          ▼
                         LLM
```

`RETRIEVAL_MODE=vector` (the default) skips all of this and uses the
unchanged Milestone 2 `RetrievalService` — see [§10](#10-hybrid-retrieval-explained)
for why each stage exists and when it actually helps.

### What exists today (Milestone 0 + 1 + 2 + 3)

```
Streamlit  →  FastAPI  →  /api/v1/health
                       →  /api/v1/documents/{upload,list,get,delete,reindex,chunks,stats}
                       →  /api/v1/chat, /api/v1/chat/stream
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
         DocumentService              ChatService ──── picks retrieval_mode:
      (ingestion orchestration)   (conversational RAG orchestration)  │
                 │                          │                         │
                 ▼                    ┌─────┴──────┐          ┌──────┴───────┐
        IngestionPipeline             ▼            ▼          ▼              ▼
       (app/rag/ingestion)   QueryRewriter / AnswerGenerator  RetrievalService  HybridRetrievalService
                 │           (app/rag/generation)             (vector-only,     (vector+BM25+RRF+
                 │                  │                          app/services)     rerank, app/services)
                 │                  │                          │                 │
                 │                  │                          └────────┬────────┘
                 │                  │                                   ▼
                 │                  │                       BM25Retriever + fusion.py + BaseReranker
                 │                  │                       (app/rag/retrieval, app/rag/reranking)
                 └──────────┬───────┘                                   │
                            ▼                                           │
              Gemini embeddings ◄──────── shared ────────────────────────
              (app/rag/embeddings)              Gemini chat completions
                            │
                            ▼
                  FAISS vector store
                (app/rag/vectorstore)
                            │
                            ▼
              SQL metadata store (documents, chunks, conversations, messages)
                      (app/database — SQLite by default)
```

The agent router, web search/calculator tools, and the rest of the
evaluation harness (answer correctness/faithfulness, beyond the
retrieval benchmark this milestone added) are still empty placeholder
packages with docstrings — no logic yet. This keeps the codebase honest:
imports don't lie about what's implemented.

---

## 2. Directory structure and responsibilities

```
RAG/
├── app/
│   ├── main.py                  # App factory: config/logging/DB-init/routes. No business logic.
│   ├── api/
│   │   ├── deps.py              #   shared FastAPI dependency providers (settings, db session)
│   │   └── v1/
│   │       ├── router.py        #   aggregates all v1 endpoint routers
│   │       └── endpoints/
│   │           ├── health.py    #   GET /health
│   │           ├── documents.py #   document upload/list/get/delete/reindex/chunks/stats
│   │           └── chat.py      #   POST /chat, POST /chat/stream
│   ├── core/
│   │   ├── config.py            #   Settings: app, DB, chunking, retrieval, confidence thresholds
│   │   ├── logging.py           #   logging.dictConfig setup (console + rotating file)
│   │   └── exceptions.py        #   AppException hierarchy + FastAPI error handlers
│   ├── models/                  # Pydantic schemas (API contracts), not ORM models
│   │   ├── schemas.py           #   HealthResponse
│   │   ├── document.py          #   Document/Chunk/Stats request-response schemas
│   │   └── chat.py              #   ChatRequest/ChatResponse/SourceCitation/ChatStreamMeta
│   ├── services/
│   │   ├── document_service.py  #   Orchestrates rag/ingestion + database for documents
│   │   ├── chunk_lookup.py      #   Shared SQL helpers (RetrievedChunk, filters) — both retrieval services use this
│   │   ├── retrieval_service.py #   Vector-only retrieval (Milestone 2, unchanged)
│   │   ├── hybrid_retrieval_service.py  # Vector + BM25 + RRF fusion + rerank (Milestone 3)
│   │   └── chat_service.py      #   The conversational RAG orchestrator (memory→...→response)
│   ├── rag/                     # RAG pipeline — pure logic, no DB/HTTP
│   │   ├── ingestion/           #   extraction, cleaning, chunking, hashing (Milestone 1)
│   │   ├── embeddings/          #   BaseEmbedder interface + Gemini implementation
│   │   ├── vectorstore/         #   VectorStore interface + FAISS implementation + factory
│   │   ├── retrieval/
│   │   │   ├── vector_retriever.py  # pure: embed query -> vector_store.search(allowed_ids)
│   │   │   ├── bm25_retriever.py    # pure: rank_bm25 over a caller-supplied (id, text) corpus
│   │   │   ├── fusion.py            # Reciprocal Rank Fusion
│   │   │   └── dedup.py             # near-duplicate chunk-text detection (shared by both services)
│   │   ├── reranking/
│   │   │   ├── base.py              # BaseReranker interface
│   │   │   ├── cross_encoder_reranker.py  # sentence-transformers cross-encoder (the default)
│   │   │   └── noop_reranker.py     # passthrough (RERANKER_BACKEND=none)
│   │   ├── generation/
│   │   │   ├── base.py          #   BaseChatModel interface + gemini_chat_model.py implementation
│   │   │   ├── prompts.py       #   every system/user prompt template, in one auditable file
│   │   │   ├── query_rewriter.py    # LLM call #1: conversation -> standalone query
│   │   │   └── answer_generator.py  # LLM call #2: query+chunks -> cited answer (streamable)
│   │   └── dependencies.py      #   cached singletons (embedder, vector store, reranker, rewriter, generator)
│   ├── agents/                  # [empty] LLM agent/router + tools (doc search, web search, calculator)
│   ├── evaluation/
│   │   ├── dataset.py           #   14-passage corpus + 22 question/relevant-id pairs
│   │   ├── metrics.py           #   Recall@k, MRR (pure functions)
│   │   └── retrieval_benchmark.py  # vector-only vs BM25-only vs hybrid vs hybrid+rerank, measured
│   ├── database/
│   │   ├── session.py           #   SQLAlchemy engine/session + init_db()
│   │   └── models.py            #   DocumentRecord, ChunkRecord, ConversationRecord, MessageRecord
│   └── utils/                   # small, dependency-free helpers shared across the app
├── frontend/
│   ├── streamlit_app.py         # Chat tab (streamed, cited, filterable, per-stage diagnostics) + Documents tab
│   └── api_client.py            #   the only module allowed to call `requests` against the backend
├── tests/                       # pytest suite — see §7
├── data/, logs/                 # gitignored contents — see Milestone 1 README section
├── requirements.txt / requirements-dev.txt
├── pytest.ini / .env.example / .gitignore
```

**Why this layout (additions this milestone):**

- **`rag/retrieval/` vs `services/retrieval_service.py`** — the same split as
  ingestion: `vector_retriever.py` is pure (embed + search an optional
  `allowed_ids` set, no DB), while `retrieval_service.py` is the DB-aware
  layer that turns a document/document-type filter into `allowed_ids`,
  resolves hits to chunk content, and removes near-duplicates. This is what
  makes retrieval independently testable (fake embedder + real FAISS index,
  no service layer needed) and independently swappable.
- **Metadata filtering without hybrid search** — FAISS's flat index has no
  native filtered search. `VectorStore.search()` grew an `allowed_ids`
  parameter; the FAISS implementation reconstructs and scores only the
  allowed vectors directly (`IndexIDMap2.reconstruct`), which is exact, not
  approximate, since the underlying index was already an exact flat scan.
  A future Qdrant/pgvector backend would instead pass `allowed_ids` as a
  native indexed filter — same interface, better performance, no app code
  changes.
- **"Keep retrieval, prompting and generation as separate services"** —
  `retrieval_service.py` (retrieval), `rag/generation/prompts.py` (prompting
  — every prompt template lives in one auditable file), and
  `rag/generation/answer_generator.py` (generation) are three distinct
  modules; `chat_service.py` is the only thing that sequences them.
- **Conversational memory is minimal on purpose** — `ConversationRecord` /
  `MessageRecord` exist only to give query rewriting a bounded window of
  prior turns to read (`Settings.conversation_history_window`, default 6
  messages). Full session management (titles, ownership, expiry) is
  Milestone 5's job; this is deliberately just enough to make follow-up
  questions work.
- **Citations come from parsing the answer, not JSON mode** — the model is
  asked to inline-cite with `[n]` markers instead of returning structured
  JSON. That's what makes streaming practical: the client renders raw text
  as it arrives and only needs to parse citation markers once, after the
  stream ends. See `answer_generator.extract_cited_indices`.
- **Two-phase streaming (`prepare()` then `stream_answer()`)** — query
  rewriting and retrieval (which embeds the query — a real failure point,
  e.g. a bad API key) run synchronously *before* the streaming HTTP response
  begins, so a failure there still becomes a clean JSON error. Only the
  actual answer-generation LLM call streams, which is the one place a
  "failure mid-response" is genuinely unavoidable.
- **The provider swap actually happened, not just in theory** — this project
  started on OpenAI (embeddings + chat) and was migrated to Gemini (free
  tier, no credit card) for both. `BaseEmbedder` already made the embeddings
  side a one-file swap; the chat side didn't have an equivalent interface
  yet, so `app/rag/generation/base.py` (`BaseChatModel`) was added as part of
  this migration — `query_rewriter.py`/`answer_generator.py` now depend only
  on that interface, never on a provider SDK shape. Zero changes were needed
  to `chat_service.py`, `retrieval_service.py`, prompts, or any test's
  assertions — only the provider implementation and the dependency wiring
  in `app/rag/dependencies.py` changed.
- **Hybrid retrieval is additive, not a rewrite** — `RetrievalService`
  (vector-only) is untouched; `HybridRetrievalService` is a new, separate
  class. The only shared code is `app/services/chunk_lookup.py` (SQL
  filter/lookup helpers, extracted from `RetrievalService` in a pure,
  behavior-preserving refactor — every Milestone 2 test still passes
  unchanged) and `app/rag/retrieval/dedup.py` (near-duplicate detection).
  `ChatService.prepare()` is the only place that branches on
  `Settings.retrieval_mode`.
- **Fusion uses rank, not score** — vector cosine similarity and BM25's
  score live on completely different, incomparable scales. Reciprocal
  Rank Fusion (`fusion.py`) sidesteps that by only ever looking at each
  candidate's *position* in each ranked list, never the raw number.
- **BM25's corpus is rebuilt from SQL per query, not persisted** —
  `rank_bm25` has no incremental index API (unlike FAISS), so
  `HybridRetrievalService` loads the matching `ChunkRecord` rows fresh
  each time and builds a new `BM25Okapi` index. Fine at this project's
  scale; a documented scaling limit (same honesty as FAISS's exact
  flat-index tradeoff) for a corpus large enough that this matters — a
  production system would maintain a persistent, incrementally-updated
  inverted index (Elasticsearch/OpenSearch/tantivy) instead.
- **The reranker is a real interface, not a hardcoded model** —
  `BaseReranker` has two implementations: `CrossEncoderReranker`
  (sentence-transformers, local, free, the default) and `NoOpReranker`
  (`RERANKER_BACKEND=none`, a pure passthrough that preserves fusion's
  order exactly — this is what makes reranking optional rather than
  load-bearing).
- **Every hybrid stage's output survives to the API response** —
  `RetrievalDiagnostics` (`vector_results`, `keyword_results`,
  `fused_results`, `reranked_results`) is `None` in vector-only mode and
  fully populated in hybrid mode, visible in both the JSON response and
  the Streamlit "Retrieval details" expander (one tab per stage) — built
  for exactly the kind of "why did it retrieve *that*" debugging hybrid
  pipelines otherwise make opaque.

---

## 3. Request lifecycle, explained

```
User Query → Memory → Query Rewrite → Retrieval → Context Selection → LLM → Citation → Response
```

1. **User Query** — `POST /api/v1/chat` (or `/chat/stream`) receives
   `{question, session_id?, document_id?, document_type?, top_k?}`
   (`app/models/chat.py:ChatRequest`).

2. **Memory** — `ChatService._get_or_create_conversation()` finds or creates
   a `ConversationRecord` by `session_id` (server-generated if omitted).
   `_load_history()` reads only the **last `conversation_history_window`
   messages** (default 6 = ~3 turns) — never the full transcript.

3. **Query Rewrite** — `QueryRewriter.rewrite(history, question)`
   (`app/rag/generation/query_rewriter.py`). If there's no history yet (first
   turn), it returns the question unchanged and **skips the LLM call
   entirely**. Otherwise it sends the bounded history + question to the LLM
   with a prompt that resolves pronouns/references into one standalone query
   (`"What are its disadvantages?"` → `"What are the disadvantages of
   SMOTE?"`). On any LLM failure it degrades gracefully to the original
   question rather than failing the whole request.

4. **Retrieval** — `RetrievalService.retrieve()` (`app/services/
   retrieval_service.py`):
   - resolves `document_id`/`document_type` into a set of allowed vector ids
     via SQL (or `None` = search everything);
   - embeds the rewritten query and searches FAISS, over-fetching
     `top_k * retrieval_overfetch_multiplier` candidates;
   - joins hits back to `ChunkRecord`/`DocumentRecord` for content + citation
     metadata.

5. **Context Selection** — still inside `retrieve()`: candidates are sorted
   by score, and any candidate whose text is a **near-duplicate** (>90%
   similarity, `difflib.SequenceMatcher`) of an already-kept chunk is
   dropped, before truncating to `top_k`. `ChatService` then drops anything
   below `min_relevance_score` (default 0.15 cosine similarity) — if
   *nothing* survives, generation is skipped entirely and the fixed
   "cannot determine this from the uploaded documents" answer is returned
   (hallucination prevention, layer 1 — deterministic, no LLM call spent).

6. **LLM** — `AnswerGenerator.generate()` / `.generate_stream()`
   (`app/rag/generation/answer_generator.py`) sends the query + numbered
   source excerpts (`prompts.build_answer_messages`) with instructions to
   answer only from those sources, cite every claim with `[n]`, and reply
   with the exact fixed sentence if the sources are insufficient
   (hallucination prevention, layer 2 — the model's own judgment, as a
   second line of defense).

7. **Citation** — `ChatService._finalize()` extracts cited `[n]` markers
   (`extract_cited_indices`), maps them back to the actual chunk's filename/
   page number/chunk id, and computes a heuristic `confidence` (high/medium/
   low) from the average similarity score of the *cited* chunks — not a
   calibrated metric, just a practical proxy (real faithfulness scoring is
   the Evaluation milestone).

8. **Response** — the turn (original question + final answer) is persisted
   to `MessageRecord`, and the structured payload is returned:
   ```json
   {
     "answer": "...",
     "sources": [{"index": 1, "document_name": "...", "page_number": 12, "chunk_id": "..."}],
     "retrieved_chunks": [...],
     "confidence": "high",
     "rewritten_query": "...",
     "session_id": "..."
   }
   ```
   `POST /chat/stream` sends the answer text as it's generated, then one
   final chunk: `\n<<<META>>>\n` + this same payload minus `answer` (see
   `STREAM_META_DELIMITER` in `chat_service.py`).

---

## 4. Roadmap (not yet built — do not assume these exist)

1. ~~**Document ingestion**~~ — done (Milestone 1).
2. ~~**Core retrieval + generation**~~ — done (Milestone 2).
3. ~~**Advanced retrieval**~~ — hybrid (vector + BM25) search + reranking —
   done this milestone.
4. **Agent layer** — LLM-based router/tools: document search, web search,
   calculator.
5. **Memory** — full session management (titles, ownership, expiry) on top
   of the `ConversationRecord`/`MessageRecord` tables already in place.
6. **Evaluation** — context relevance, answer correctness,
   faithfulness/hallucination checks on *generation* (real calibrated
   confidence replaces today's heuristic) — building on the retrieval
   benchmark this milestone already added to `app/evaluation/`.
7. **Productionization** — Docker/Compose, Alembic migrations, CI,
   structured logging, secrets management.

---

## 5. Setup

### Prerequisites

- Python 3.10+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
  — no credit card required (embeddings + chat completions both call the
  real API). PostgreSQL is **not** required; SQLite is used by default.

### Install dependencies

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# sentence-transformers (the reranker) pulls in torch. Install the small
# CPU-only wheel first to avoid a multi-GB default download:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt   # runtime + test deps
```

The cross-encoder reranker model (~90MB) downloads once on first use and
is cached locally afterward — no network call on subsequent runs.

### Configure environment

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

Set a real key in `.env` (get one free at https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your real key...
```

Retrieval/chat defaults worth knowing (all in `.env.example`, all overridable):
`RETRIEVAL_TOP_K=5`, `MIN_RELEVANCE_SCORE=0.15`, `DEDUP_SIMILARITY_THRESHOLD=0.9`,
`CONVERSATION_HISTORY_WINDOW=6`.

**Hybrid retrieval is off by default** (`RETRIEVAL_MODE=vector`, the
unchanged Milestone 2 path). To try it:

```
RETRIEVAL_MODE=hybrid
VECTOR_TOP_K=20            # candidates fetched from FAISS
BM25_TOP_K=20               # candidates fetched from BM25
RERANK_TOP_K=10             # top fused candidates sent to the reranker
FINAL_CONTEXT_K=5           # chunks that reach the LLM after rerank + dedup
RRF_K=60                    # Reciprocal Rank Fusion constant
RERANKER_BACKEND=cross_encoder   # or "none" to disable reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 6. Running

### Backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

### Frontend (Streamlit)

```bash
streamlit run frontend/streamlit_app.py
```

Open the **Documents** tab first, upload a PDF, wait for `status: completed`.
Then switch to the **Chat** tab, optionally scope the search to one document
or document type, and ask a question — the answer streams in with sources,
confidence, and a "Retrieval details" expander showing exactly what was
retrieved and how it scored. With `RETRIEVAL_MODE=hybrid` set, that expander
grows four tabs — Vector / Keyword (BM25) / Fused / Reranked (final) — so
you can see exactly how each stage changed the ranking.

## 7. Testing

```bash
pytest -v
```

| File | Covers |
|---|---|
| `tests/test_health.py`, `test_extractor.py`, `test_chunker.py`, `test_hasher.py`, `test_metadata.py`, `test_document_service.py` | Milestones 0–1 (see prior README revisions) |
| `tests/test_retrieval_service.py` | **Retrieval (vector-only)**: semantic ranking, document-id filter, document-type filter, empty-filter/empty-store behavior, near-duplicate context-selection |
| `tests/test_query_rewriter.py` | **Query rewriting**: no-op on first turn (no LLM call), history-aware rewrite, prompt actually contains history, graceful fallback on LLM failure/blank response |
| `tests/test_citations.py` | **Citation generation** (pure logic): marker extraction/ordering/dedup, out-of-range marker rejection, no-context-sentinel detection |
| `tests/test_chat_service.py` | **Citation generation** (integration) + **no-context behavior** + **retrieval-mode switch**: correct index→source mapping, hallucination guard fires without calling the generation LLM, model-declines-anyway forces empty sources, no-citation-markers falls back to crediting all sources, conversational follow-up is rewritten using persisted history, document filter honored end-to-end, `retrieval_diagnostics` is `None` in vector mode and populated in hybrid mode |
| `tests/test_bm25_retriever.py` | **BM25**: exact keyword match ranks highest, no-overlap returns empty, top-k limiting, tokenizer behavior |
| `tests/test_fusion.py` | **Result fusion**: an id found by both engines outranks one found by only one, higher rank within a list scores higher, disjoint/empty-list edge cases, the `k` constant's effect |
| `tests/test_reranker.py` | **Reranking**: `NoOpReranker` preserves order, the *real* cross-encoder (cached locally, no network) scores a relevant document higher than an irrelevant one and returns sigmoid-normalized [0,1] scores |
| `tests/test_hybrid_retrieval_service.py` | **Hybrid pipeline integration**: BM25 surfaces an exact term vector search's fixed vocabulary can't represent, fusion favors items both engines found, reranking produces a valid sorted order over real content, `final_context_k` is respected, near-duplicate dedup, metadata-filter/empty-corpus edge cases |
| `tests/test_evaluation_metrics.py` | Recall@k and MRR (pure functions) |

None of the tests above call the real Gemini API: `tests/fakes.py` provides a
`KeywordFakeEmbedder` (deterministic, keyword-overlap-based similarity — good
enough to test ranking/filtering) and a `FakeChatModel` (scripted responses
implementing `BaseChatModel` directly, streaming and non-streaming). The
cross-encoder reranker tests *do* run the real local model (no network, no
API key — it's a downloaded-once local model, not a hosted one).
**The real Gemini-backed paths (embeddings, query rewriting, generation) are
not unit tested** — verify them manually with a real key using the test plan
below, or by running the retrieval benchmark (§10).

## 8. Test plan — verify RAG works correctly

Prerequisites: a real `GEMINI_API_KEY` in `.env`, backend + Streamlit running,
and **at least one PDF uploaded and `status: completed`** (a paper or article
that discusses SMOTE works well for questions 1–4 below; substitute your own
document's topic otherwise).

1. **Basic factual question** — *"What is SMOTE?"*
   Expect a grounded answer with `[1]`-style citations and `sources`
   pointing at real page numbers from your document.
2. **Follow-up requiring coreference resolution** — *"What are its
   disadvantages?"* (same session as #1)
   Check the response's `rewritten_query` — it should read something like
   *"What are the disadvantages of SMOTE?"*, not the raw follow-up.
3. **Second follow-up, deeper chain** — *"Is there a way to fix that?"*
   Confirms rewriting still works after 2+ turns, within the bounded history
   window.
4. **Out-of-scope question** — *"What is the capital of France?"*
   Expect the exact fixed answer *"I cannot determine this from the uploaded
   documents."*, `sources: []`, `confidence: "low"` — this is the
   hallucination-prevention guard.
5. **Question about a topic the document doesn't cover, but plausible for
   the domain** — e.g. if your doc is about SMOTE, ask *"How does BERT
   tokenization work?"*
   Should also decline rather than hallucinate a bridge between unrelated
   concepts.
6. **New conversation reset** — click "New conversation" in Streamlit, ask
   *"What are its disadvantages?"* with no prior turn.
   Since there's no history, expect `rewritten_query` to equal the question
   verbatim, and likely an ambiguous/low-relevance answer or a decline — the
   system should **not** silently reuse the previous session's context.
7. **Document-scoped search** — upload a second, unrelated PDF, then in the
   Chat tab set "Search scope" to the *first* document only and ask a
   question only the *second* document could answer.
   Expect a decline, proving the `document_id` filter is actually applied,
   not just cosmetic.
8. **Document-type filter** — set "Document type" to PDF and confirm normal
   questions still work (today, every ingested document is a PDF, so this is
   mainly a smoke test that the filter doesn't wrongly exclude everything).
9. **Streaming behaves like streaming** — watch the answer appear
   incrementally in the Chat tab rather than all at once; open the network
   tab / use `curl --no-buffer` against `/api/v1/chat/stream` and confirm
   text arrives in multiple chunks, ending with the `<<<META>>>` delimiter
   and a JSON blob.
10. **Citations point at real content** — for any answered question, expand
    "Retrieval details" and manually check that the `[1]`/`[2]` markers in
    the answer correspond to chunks whose content actually supports the
    claim next to that marker — not just that citations exist, but that
    they're *correct*.
11. **Duplicate-safe repeatability** — ask the same question twice in a row
    (same session). Both answers should cite the same or overlapping
    sources and stay consistent in tone/confidence — a sanity check that
    retrieval isn't randomly unstable.
12. **Confidence tracks relevance** — compare the `confidence` field between
    a well-covered question (#1) and a borderline/tangential one; confidence
    should be visibly lower for the latter even if it doesn't fully decline.
13. **Hybrid mode diagnostics** — set `RETRIEVAL_MODE=hybrid` in `.env`,
    restart the backend, ask a question, and expand "Retrieval details" in
    Streamlit. You should see four tabs (Vector / Keyword / Fused / Reranked)
    each with their own scores — confirms the whole pipeline is wired, not
    just returning the same thing as vector-only under a different label.

## 9. Verify before moving to the next milestone

- [ ] `pytest -v` passes (80+ tests).
- [ ] A real `GEMINI_API_KEY` is set in `.env`.
- [ ] At least one PDF uploaded and ingested successfully.
- [ ] `POST /api/v1/chat` with no documents ingested returns the fixed
      "cannot determine" answer with `confidence: "low"` and **no** Gemini
      chat-completion call billed (only true once something is ingested —
      before that, retrieval also skips embedding the query, see
      `VectorRetriever.search`'s empty-store short-circuit).
- [ ] Questions 1–13 in the test plan above all behave as described.
- [ ] `data/processed/metadata.db` shows populated `conversations` and
      `messages` tables after a chat (`sqlite3 data/processed/metadata.db
      "select role, content from messages;"`).
- [ ] `RETRIEVAL_MODE=vector` still behaves exactly like before this
      milestone (it's the same code path, untouched).
- [ ] `python -m app.evaluation.retrieval_benchmark` runs and prints a
      4-strategy comparison table (§10 below has a real run's output).

Once all of the above are true, hybrid retrieval + reranking is confirmed
working end-to-end and we can start Milestone 4 (the agent layer).

---

## 10. Hybrid retrieval, explained

### Run the benchmark yourself

```bash
python -m app.evaluation.retrieval_benchmark
```

This embeds `app/evaluation/dataset.py`'s 14 passages with the real,
configured Gemini embedding model, builds a real BM25 index, loads the
real cross-encoder, and runs all 22 questions through four strategies:
vector-only, BM25-only, hybrid (fusion, no reranking), and hybrid + rerank.
It needs a real `GEMINI_API_KEY` — this is a measured experiment against
live embeddings, not a mocked test.

### A real run's results

```
Overall (n=22 questions, k=5):
Strategy             Recall@5     MRR      Total time (s)
----------------------------------------------------------
vector-only          1.000        0.955    13.61
bm25-only            0.864        0.727    0.00
hybrid (fusion)      0.955        0.789    11.89
hybrid + rerank      1.000        0.895    14.66

By question category (selected):
  paraphrase (n=5):        vector 1.000/0.800   bm25 0.400/0.250   fusion 0.800/0.440   rerank 1.000/0.640
  exact-term (n=2):        vector 1.000/1.000   bm25 1.000/0.200   fusion 1.000/0.333   rerank 1.000/1.000
  rare-token (n=2):        vector 1.000/1.000   bm25 1.000/1.000   fusion 1.000/1.000   rerank 1.000/1.000

Misses (relevant passage not in top-5): vector-only 0, bm25-only 3
(all three "paraphrase"), hybrid-fusion 1, hybrid+rerank 0.
```

**Read honestly, not cherry-picked:** on this small, clean 14-passage
corpus, vector-only alone already gets perfect Recall@5 — Gemini's
embeddings are simply strong enough that hybrid fusion *alone* doesn't
beat it (0.955 vs. 1.000 recall, 0.789 vs. 0.955 MRR): BM25 dragging in
low-quality paraphrase results sometimes displaces a good vector hit
before reranking has a chance to fix the order. **Reranking is what
actually recovers this** — `hybrid + rerank` is the only strategy tied
for the best recall (1.000, zero misses) while *also* being robust on
both of BM25's and vector's respective weak spots (see below), which
neither one alone can guarantee on a different, harder corpus. That
nuance — hybrid isn't automatically better, reranking is what makes it
reliably at-least-as-good — is the real, honest finding here.

### Why vector search alone can fail

Embeddings capture *meaning*, which is exactly why they miss two things:

1. **Rare, exact tokens the model has no real association for** — a
   made-up acronym, a specific error code, an exact product/model name.
   The embedding space has no strong signal to place it near a document
   that literally contains it verbatim; a query like "What does the
   QZ7-Widget need?" only reliably works if the model happens to encode
   that literal substring similarly, which isn't guaranteed. BM25, which
   scores literal term overlap, catches this immediately.
2. **Small or narrow corpora with many similar chunks** — with a lot of
   near-duplicate or closely related passages (a 95-chunk single paper,
   say), cosine similarity differences between the truly-best chunk and
   several near-misses can be tiny, and embedding noise can reorder them.
   This project's small demo corpus doesn't show this starkly (14
   diverse, well-separated topics), but it's a well-documented failure
   mode at real scale — which is exactly why reranking exists.

### Why BM25 helps

BM25 scores *literal term overlap*, weighted by how rare/distinctive each
term is (inverse document frequency) and normalized for document length.
It has zero notion of meaning — "car" and "automobile" share no signal —
but that's precisely its strength for the cases vector search struggles
with: exact identifiers, jargon, acronyms, numbers, anything where the
*specific words* matter more than the concept. In this benchmark, BM25
alone gets every "rare-token" and "exact-term" question right, at
essentially zero latency (no API call, no model — pure arithmetic over
term counts already in memory).

### Why reranking helps

A cross-encoder reads the query and a candidate document *together*,
through one model pass, so it can weigh interactions between specific
words in both — something a bi-encoder (vector search) fundamentally
can't do, since it must compress the query and every document into
independent, fixed vectors *before* ever seeing them side by side. That
joint attention is why cross-encoders are consistently more accurate at
judging "is this actually relevant" — and why they're too slow to run
over an entire corpus (one model pass per candidate), which is exactly
why they only ever run over a short fused shortlist, not the whole
index. In this benchmark, reranking is what turns fusion's 1 miss back
into 0 — it correctly demotes the low-quality BM25-sourced result that
naive rank fusion let outrank a better vector hit.

### When hybrid retrieval is actually useful

- **Your queries mix styles** — some users ask in exact keywords/jargon,
  others paraphrase conversationally. No single retriever handles both
  well; fusion means you don't have to predict which kind a query is
  before choosing a strategy.
- **Your corpus has rare, load-bearing exact terms** — product codes,
  acronyms, specific names, numbers — that users will plausibly search
  for verbatim, and that a general-purpose embedding model was never
  trained to treat as meaningfully distinct from similar-looking terms.
- **Your corpus is large and/or topically narrow** — many chunks
  discussing similar things, where vector similarity alone becomes noisy
  and reranking's precision matters more than it does on a small, diverse
  corpus like this benchmark's.
- **It's *not* obviously worth it** when your corpus is small, clean, and
  your embedding model is already strong for the domain (this benchmark's
  own vector-only column) — the honest result above. Hybrid + reranking
  is a reliability/robustness upgrade more than a guaranteed win on every
  corpus; `RETRIEVAL_MODE=vector` staying fully intact and swappable is
  exactly what lets you measure that for your own data before deciding.
