# Agentic RAG Research Assistant

A portfolio-grade, production-style **Agentic RAG** system: an LLM-based agent (Gemini)
that routes between document retrieval, web search and a calculator tool,
built on top of an advanced RAG pipeline (hybrid retrieval, reranking, query
rewriting, citations) with conversational memory and automated evaluation.

This repository is built **incrementally, milestone by milestone**. This
README reflects the current milestone and will be updated as each new one
lands.

> **Current milestone: 2 — Core Advanced RAG (retrieval + generation).**
> Semantic retrieval, metadata filtering, LLM query rewriting, bounded
> conversational memory, context selection/dedup, grounded + cited answer
> generation, hallucination prevention, and streaming. No hybrid search,
> reranking or agents yet — that's by design. See [Roadmap](#4-roadmap).

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
       Hybrid Retrieval                                             ← vector-only so far;
       ┌──────────────┐                                               BM25 + reranker not yet
       │ Vector Search│  ← implemented
       │ BM25 Search  │  ← not built yet
       └──────┬───────┘
              ▼
          Reranker                                                  ← not built yet
              │
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

### What exists today (Milestone 0 + 1 + 2)

```
Streamlit  →  FastAPI  →  /api/v1/health
                       →  /api/v1/documents/{upload,list,get,delete,reindex,chunks,stats}
                       →  /api/v1/chat, /api/v1/chat/stream
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
         DocumentService              ChatService
      (ingestion orchestration)   (conversational RAG orchestration)
                 │                          │
                 ▼                    ┌─────┴──────┐
        IngestionPipeline             ▼            ▼
       (app/rag/ingestion)   RetrievalService   QueryRewriter / AnswerGenerator
                 │           (app/services)     (app/rag/generation)
                 │                  │                    │
                 └──────────┬───────┘                    │
                            ▼                             │
              Gemini embeddings ◄──────── shared ─────────┘
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

The agent router, web search/calculator tools, hybrid (BM25) retrieval,
reranking, and the evaluation harness are all still empty placeholder
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
│   │   ├── retrieval_service.py #   DB-aware retrieval: filters, chunk resolution, dedup
│   │   └── chat_service.py      #   The conversational RAG orchestrator (memory→...→response)
│   ├── rag/                     # RAG pipeline — pure logic, no DB/HTTP
│   │   ├── ingestion/           #   extraction, cleaning, chunking, hashing (Milestone 1)
│   │   ├── embeddings/          #   BaseEmbedder interface + Gemini implementation
│   │   ├── vectorstore/         #   VectorStore interface + FAISS implementation + factory
│   │   ├── retrieval/
│   │   │   └── vector_retriever.py  # pure: embed query -> vector_store.search(allowed_ids)
│   │   ├── generation/
│   │   │   ├── base.py          #   BaseChatModel interface + gemini_chat_model.py implementation
│   │   │   ├── prompts.py       #   every system/user prompt template, in one auditable file
│   │   │   ├── query_rewriter.py    # LLM call #1: conversation -> standalone query
│   │   │   └── answer_generator.py  # LLM call #2: query+chunks -> cited answer (streamable)
│   │   └── dependencies.py      #   cached singletons (embedder, vector store, rewriter, generator)
│   ├── agents/                  # [empty] GPT agent/router + tools (doc search, web search, calculator)
│   ├── evaluation/               # [empty] retrieval/faithfulness/correctness evaluation harness
│   ├── database/
│   │   ├── session.py           #   SQLAlchemy engine/session + init_db()
│   │   └── models.py            #   DocumentRecord, ChunkRecord, ConversationRecord, MessageRecord
│   └── utils/                   # small, dependency-free helpers shared across the app
├── frontend/
│   ├── streamlit_app.py         # Chat tab (streamed, cited, filterable) + Documents tab
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
2. ~~**Core retrieval + generation**~~ — done this milestone.
3. **Advanced retrieval** — hybrid (vector + BM25) search, reranking.
4. **Agent layer** — LLM-based router/tools: document search, web search,
   calculator.
5. **Memory** — full session management (titles, ownership, expiry) on top
   of the `ConversationRecord`/`MessageRecord` tables already in place.
6. **Evaluation** — retrieval quality, context relevance, answer
   correctness, faithfulness/hallucination checks (real calibrated
   confidence replaces today's heuristic).
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

pip install -r requirements-dev.txt   # runtime + test deps
```

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
retrieved and how it scored.

## 7. Testing

```bash
pytest -v
```

| File | Covers |
|---|---|
| `tests/test_health.py`, `test_extractor.py`, `test_chunker.py`, `test_hasher.py`, `test_metadata.py`, `test_document_service.py` | Milestones 0–1 (see prior README revisions) |
| `tests/test_retrieval_service.py` | **Retrieval**: semantic ranking, document-id filter, document-type filter, empty-filter/empty-store behavior, near-duplicate context-selection |
| `tests/test_query_rewriter.py` | **Query rewriting**: no-op on first turn (no LLM call), history-aware rewrite, prompt actually contains history, graceful fallback on LLM failure/blank response |
| `tests/test_citations.py` | **Citation generation** (pure logic): marker extraction/ordering/dedup, out-of-range marker rejection, no-context-sentinel detection |
| `tests/test_chat_service.py` | **Citation generation** (integration) + **no-context behavior**: correct index→source mapping, hallucination guard fires without ever calling the generation LLM, model-declines-anyway case forces empty sources, no-citation-markers falls back to crediting all sources, conversational follow-up is actually rewritten using persisted history, document filter is honored end-to-end |

None of the new tests call the real LLM provider: `tests/fakes.py` provides a
`KeywordFakeEmbedder` (deterministic, keyword-overlap-based similarity — good
enough to test ranking/filtering) and a `FakeChatClient` (scripted responses
for the `.chat.completions.create(...)` surface, streaming and non-streaming).
**The real Gemini-backed paths are not unit tested** — verify them manually
with a real key using the test plan below.

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

## 9. Verify before moving to the next milestone

- [ ] `pytest -v` passes (48+ tests).
- [ ] A real `GEMINI_API_KEY` is set in `.env`.
- [ ] At least one PDF uploaded and ingested successfully.
- [ ] `POST /api/v1/chat` with no documents ingested returns the fixed
      "cannot determine" answer with `confidence: "low"` and **no** Gemini
      chat-completion call billed (only true once something is ingested —
      before that, retrieval also skips embedding the query, see
      `VectorRetriever.search`'s empty-store short-circuit).
- [ ] Questions 1–12 in the test plan above all behave as described.
- [ ] `data/processed/metadata.db` shows populated `conversations` and
      `messages` tables after a chat (`sqlite3 data/processed/metadata.db
      "select role, content from messages;"`).

Once all of the above are true, core retrieval + generation is confirmed
working end-to-end and we can start Milestone 3 (hybrid search + reranking).
