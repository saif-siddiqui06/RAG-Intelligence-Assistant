# Agentic RAG Research Assistant

A portfolio-grade, production-style **Agentic RAG** system: a GPT-based agent
that routes between document retrieval, web search and a calculator tool,
built on top of an advanced RAG pipeline (hybrid retrieval, reranking, query
rewriting, citations) with conversational memory and automated evaluation.

This repository is built **incrementally, milestone by milestone**. This
README reflects the current milestone and will be updated as each new one
lands.

> **Current milestone: 1 — Document Ingestion Pipeline.**
> PDF upload/list/delete/re-index, extraction, cleaning, configurable
> chunking, OpenAI embeddings, a persistent FAISS vector store and a SQL
> metadata store. No hybrid search, reranking, agents or evaluation yet —
> that's by design. See [Roadmap](#4-roadmap).

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
                       │   Agent / Router    │
                       │        (GPT)        │
                       └──────────┬──────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
       Document RAG          Web Search           Calculator
             │
             ▼
      Query Rewriting
             │
             ▼
       Hybrid Retrieval
       ┌──────────────┐
       │ Vector Search│
       │ BM25 Search  │
       └──────┬───────┘
              ▼
          Reranker
              │
              ▼
       Relevant Chunks
              │
              ▼
          GPT / LLM
              │
       ┌──────┴───────┐
       ▼              ▼
   Answer         Citations
```

### Ingestion flow — **implemented this milestone**

```
PDF → extraction → cleaning → chunking → metadata → embeddings → vector database
                                              │
                                              ▼
                                   SQL metadata store (SQLite/Postgres)
```

### What exists today (Milestone 0 + 1)

```
Streamlit  →  FastAPI  →  /api/v1/health
                       →  /api/v1/documents/{upload,list,get,delete,reindex,chunks,stats}
                              │
                              ▼
                     DocumentService (app/services)
                       │                    │
                       ▼                    ▼
              IngestionPipeline      SQL metadata store
              (app/rag/ingestion)    (app/database — SQLite by default)
                       │
                       ▼
          OpenAI embeddings → FAISS vector store
          (app/rag/embeddings)  (app/rag/vectorstore)
```

The agent router, web search/calculator tools, hybrid retrieval/reranking,
query rewriting, chat memory and evaluation harness are all still empty
placeholder packages with docstrings — no logic yet. This keeps the
codebase honest: imports don't lie about what's implemented.

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
│   │           └── documents.py #   document upload/list/get/delete/reindex/chunks/stats
│   ├── core/
│   │   ├── config.py            #   Settings (pydantic-settings): app, DB, RAG/chunking, OpenAI
│   │   ├── logging.py           #   logging.dictConfig setup (console + rotating file)
│   │   └── exceptions.py        #   AppException hierarchy + FastAPI error handlers
│   ├── models/                  # Pydantic schemas (API contracts), not ORM models
│   │   ├── schemas.py           #   HealthResponse
│   │   └── document.py          #   Document/Chunk/Stats request-response schemas
│   ├── services/
│   │   └── document_service.py  # Orchestrates rag/ + database/ for upload/list/delete/reindex
│   ├── rag/                     # RAG pipeline — ingestion implemented, retrieval/chat not yet
│   │   ├── ingestion/
│   │   │   ├── extractor.py     #   PDF → per-page text (pypdf)
│   │   │   ├── cleaner.py       #   whitespace/artifact normalization
│   │   │   ├── chunker.py       #   configurable recursive-separator chunking
│   │   │   ├── hasher.py        #   sha256 file hash (duplicate detection)
│   │   │   └── pipeline.py      #   orchestrates extract→clean→chunk→embed→store
│   │   ├── embeddings/
│   │   │   ├── base.py          #   BaseEmbedder interface
│   │   │   └── openai_embedder.py
│   │   ├── vectorstore/
│   │   │   ├── base.py          #   VectorStore interface (add/search/delete/count)
│   │   │   ├── faiss_store.py   #   FAISS implementation
│   │   │   └── factory.py       #   the one place a backend swap happens
│   │   └── dependencies.py      #   cached singletons (embedder, vector store, pipeline)
│   ├── agents/                  # [empty] GPT agent/router + tools (doc search, web search, calculator)
│   ├── evaluation/               # [empty] retrieval/faithfulness/correctness evaluation harness
│   ├── database/
│   │   ├── session.py           #   SQLAlchemy engine/session + init_db()
│   │   └── models.py            #   DocumentRecord, ChunkRecord ORM models
│   └── utils/                   # small, dependency-free helpers shared across the app
├── frontend/
│   ├── streamlit_app.py         # upload / list / preview chunks / delete / re-index UI
│   └── api_client.py            #   the only module allowed to call `requests` against the backend
├── tests/                       # pytest suite — see §7
├── data/
│   ├── uploads/                 # raw uploaded PDFs, named {document_id}.pdf (gitignored contents)
│   ├── processed/                # metadata.db (SQLite) lives here by default (gitignored contents)
│   └── vectorstore/              # index.faiss + index.meta.json (gitignored contents)
├── logs/                        # rotating app.log (gitignored contents)
├── requirements.txt              # runtime deps (+ openai, faiss-cpu, pypdf this milestone)
├── requirements-dev.txt          # + pytest, httpx, reportlab (test-only, generates PDF fixtures)
├── pytest.ini
├── .env.example
└── .gitignore
```

**Why this layout (additions this milestone):**

- **`rag/ingestion/` vs `rag/embeddings/` vs `rag/vectorstore/`** — three
  independent concerns. `ingestion/` never imports FAISS or OpenAI directly;
  it depends only on the `BaseEmbedder` and `VectorStore` interfaces. This is
  what makes "swap FAISS for Chroma/Qdrant/pgvector without rewriting the
  app" literally true — add a new class in `vectorstore/`, add one branch to
  `vectorstore/factory.py`, done.
- **Ingestion is independent from the future chat/RAG pipeline** — as
  instructed. `rag/ingestion/pipeline.py` only turns files into embedded,
  stored chunks; it has no notion of a query, an answer, or a conversation.
  The future retrieval/chat pipeline will reuse `embeddings/` and
  `vectorstore/` (to embed a query and search the same index) but will never
  import anything from `ingestion/`.
- **Vector store holds vectors + integer ids only; SQL holds everything
  else** — `ChunkRecord.vector_id` is the join key. This means the metadata
  (filename, document_id, page number, chunk_id, timestamps, chunk text
  itself) is 100% decoupled from which vector backend is active.
- **`app/database/` is now wired in** — `init_db()` runs at startup and
  creates tables via `Base.metadata.create_all()` (no Alembic yet; that's a
  deliberate later "productionization" concern). `DATABASE_URL` defaults to
  a local SQLite file so ingestion works with zero extra infra — switching
  to Postgres later is only an env var change, not a code change.
- **`document_service.py` is the only thing that talks to both `rag/` and
  `database/`** — endpoints in `api/v1/endpoints/documents.py` never touch
  either directly.

---

## 3. What's implemented in this milestone

- **Upload** (`POST /api/v1/documents/upload`, multiple files) — validates
  PDF type, computes a SHA-256 content hash, rejects exact duplicates
  (`409`), saves the raw file under `data/uploads/{document_id}.pdf`, then
  runs the full pipeline synchronously.
- **Extraction** (`app/rag/ingestion/extractor.py`) — per-page text via
  `pypdf`, with typed failures for corrupt/encrypted/unreadable files.
- **Cleaning** (`cleaner.py`) — strips null bytes, collapses whitespace,
  normalizes blank lines.
- **Chunking** (`chunker.py`) — a from-scratch recursive-separator splitter
  (paragraph → line → sentence → word → character fallback), fully
  configurable (`chunk_size`, `chunk_overlap`, `separators`), with overlap
  carried between consecutive chunks. Overridable per-request via
  `?chunk_size=&chunk_overlap=` query params, or globally via `.env`.
- **Metadata** — every chunk keeps `document_id`, `chunk_id`, `page_number`,
  `chunk_index`, plus document-level `filename`, `document_type`,
  `upload_timestamp`, `status`, all persisted in SQL (`app/database/models.py`).
- **Embeddings** (`openai_embedder.py`) — OpenAI `text-embedding-3-small` by
  default, batched requests, clean `502` on provider failure.
- **Vector store** (`faiss_store.py`) — persistent FAISS `IndexIDMap2` +
  `IndexFlatIP` (cosine via L2-normalized vectors), surviving restarts.
- **Duplicate prevention** — SHA-256 file hash is unique-indexed in SQL;
  re-uploading identical bytes is rejected without re-running the pipeline.
- **Document management** — list (`GET /documents`), get one (`GET
  /documents/{id}`), preview chunks (`GET /documents/{id}/chunks`), delete
  (`DELETE /documents/{id}`, removes DB rows + vectors + the stored file),
  re-index (`POST /documents/{id}/reindex`, re-runs the pipeline on the
  already-stored file with fresh/updated chunking config).
- **Stats** (`GET /documents/stats/summary`) — total documents, total
  chunks, total vectors in the index — the quickest way to verify storage.
- **Streamlit UI** — upload with configurable chunk size/overlap, live
  document list with status/page/chunk counts, per-document chunk preview,
  delete and re-index buttons, sidebar storage stats.
- **Error handling** — typed `AppException` subclasses for duplicate
  uploads (`409`), not-found (`404`), bad file type (`415`), empty file
  (`400`), corrupt/encrypted/text-less PDFs (`422`), missing API key or
  provider failure (`500`/`502`).
- **Logging** — every upload/delete/reindex/failure logged via the
  rotating `logs/app.log` handler from Milestone 0.
- **Tests** — see [§7 Testing](#7-testing).

## 4. Roadmap (not yet built — do not assume these exist)

1. ~~**Document ingestion**~~ — done this milestone.
2. **Core retrieval** — embed a query, similarity search, basic QA endpoint
   (the first thing to consume `rag/embeddings` + `rag/vectorstore` from the
   *query* side).
3. **Advanced retrieval** — hybrid (vector + BM25) search, reranking, query
   rewriting, citation-attributed answers.
4. **Agent layer** — GPT-based router/tools: document search, web search,
   calculator.
5. **Memory** — conversation/session management (Postgres in production,
   same SQLAlchemy models pattern as documents/chunks).
6. **Evaluation** — retrieval quality, context relevance, answer
   correctness, faithfulness/hallucination checks.
7. **Productionization** — Docker/Compose, Alembic migrations, CI,
   structured logging, secrets management.

---

## 5. Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key (**required this milestone** — embeddings call the real
  API). PostgreSQL is **not** required; SQLite is used by default.

### Install dependencies

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt   # runtime + test deps
# or, without test tooling:
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

Then set a real key in `.env`:

```
OPENAI_API_KEY=sk-...your real key...
```

Everything else has a working default (SQLite metadata store, FAISS vector
store, 1000/150 char chunk size/overlap).

## 6. Running

### Backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

### Frontend (Streamlit)

In a second terminal (with the venv activated):

```bash
streamlit run frontend/streamlit_app.py
```

Open the URL Streamlit prints (default http://localhost:8501). Upload one or
more PDFs, watch status/page/chunk counts populate, expand "Preview chunks"
to see exactly what got embedded, and try Delete/Reindex.

## 7. Testing

```bash
pytest -v
```

| File | Covers |
|---|---|
| `tests/test_health.py` | Milestone 0 smoke tests |
| `tests/test_extractor.py` | PDF text extraction (multi-page, corrupt file) — generates real PDFs with `reportlab` |
| `tests/test_chunker.py` | Chunking: size limits, overlap, custom separators, config validation, hard-slice fallback |
| `tests/test_hasher.py` | SHA-256 hashing determinism/uniqueness (duplicate-detection primitive) |
| `tests/test_metadata.py` | `DocumentRecord`/`ChunkRecord` defaults, API response mapping |
| `tests/test_document_service.py` | Full service integration — duplicate rejection, multi-document upload, delete removes vectors, non-PDF rejection — using a fake network-free embedder |

`test_document_service.py` never calls OpenAI: it monkeypatches the
pipeline factory with a deterministic `FakeEmbedder`, so the full
upload→chunk→embed→store→delete flow is tested without needing a real API
key or network access. **The real `OpenAIEmbedder` path is not unit tested**
— verify it manually (see below) with a real key.

## 8. Verify documents and embeddings were actually stored

1. **Via the API directly:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/documents/upload \
     -F "files=@/path/to/your/paper.pdf"
   ```
   A `201` with a JSON body containing `"status": "completed"` and a
   non-null `num_chunks` means extraction, chunking, embedding and storage
   all succeeded.

2. **Check the stats endpoint:**
   ```bash
   curl http://localhost:8000/api/v1/documents/stats/summary
   ```
   `vector_count` should equal the sum of `num_chunks` across all completed
   documents — this is the FAISS index's real vector count, not just a DB
   row count, so if these ever diverge something is inconsistent.

3. **Preview the actual stored chunks:**
   ```bash
   curl http://localhost:8000/api/v1/documents/{document_id}/chunks
   ```
   Confirms page numbers/chunk indices/content look right.

4. **Inspect the files on disk:**
   - `data/uploads/{document_id}.pdf` — the raw file you uploaded.
   - `data/processed/metadata.db` — SQLite file; open it with any SQLite
     browser (or `sqlite3 data/processed/metadata.db "select * from documents;"`)
     to see the `documents` and `chunks` tables directly.
   - `data/vectorstore/index.faiss` + `index.meta.json` — the persisted
     FAISS index and its id counter. Delete these two files (with the app
     stopped) to reset the vector store from scratch.

5. **Restart the server and re-check** `GET /api/v1/documents` — if your
   documents are still listed with the same chunk counts, persistence
   across restarts is confirmed (this is why SQLite + FAISS-on-disk were
   chosen over in-memory structures for this milestone).

## 9. Verify before moving to the next milestone

- [ ] `pip install -r requirements-dev.txt` completes with no errors.
- [ ] A real `OPENAI_API_KEY` is set in `.env`.
- [ ] `uvicorn app.main:app --reload` starts without exceptions.
- [ ] Uploading a real PDF via `/docs` or Streamlit returns `status:
      completed` with `num_pages`/`num_chunks` populated.
- [ ] Re-uploading the same file returns `409 Conflict`.
- [ ] `GET /api/v1/documents/stats/summary` shows `vector_count > 0`.
- [ ] `data/processed/metadata.db` and `data/vectorstore/index.faiss` exist
      on disk after an upload.
- [ ] Deleting a document drops its rows and its vectors (`vector_count`
      decreases accordingly).
- [ ] `pytest -v` passes.

Once all of the above are true, ingestion is confirmed working end-to-end
and we can start Milestone 2 (core retrieval: embedding a query and
searching this same vector store).
