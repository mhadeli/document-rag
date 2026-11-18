# Document Intelligence RAG

A production-minded Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **PostgreSQL (`pgvector`)**, **Google Gemini**, and **Streamlit**.

Upload PDF documents or submit website URLs, index their contents into vector embeddings, and ask conversational natural-language questions grounded strictly in your sources with page-level and URL citations.

---

## Core Features

- **PDF Ingestion (`PyMuPDF`):** Page-by-page extraction preserving 1-indexed page numbers, with `%PDF-` header validation, file size limits, and path traversal protection.
- **Website Ingestion (`Trafilatura` + `httpx`):** Extracts clean article text, page titles, and canonical URLs while blocking private/loopback IP ranges to prevent SSRF.
- **SHA-256 Deduplication:** Prevents re-embedding identical PDFs or webpages.
- **Semantic Vector Search (`pgvector`):** Stores 768-dimensional Google Gemini embeddings (`models/gemini-embedding-001`) in PostgreSQL with HNSW cosine indexing and `ON DELETE CASCADE`.
- **Grounded Answers & Structured Citations (`gemini-2.5-flash`):** Uses Gemini Structured Outputs to map `cited_chunk_ids` deterministically back to exact PDF pages and URLs, plus conversational query rewriting for multi-turn follow-up questions.
- **Interactive Streamlit UI:** Manage indexed documents, filter chat scope to specific sources, and inspect expandable source snippets.

---

## Architecture

```text
Streamlit UI (Port 8501)
       |
       v
FastAPI Backend (Port 8000)
  ├── Ingestion Pipeline: PDF/Web Loader -> Chunker -> Gemini Embeddings -> PostgreSQL/pgvector
  └── Query Pipeline:     Query Rewrite -> Gemini Query Embedding -> pgvector Search -> Gemini Structured Answer
```

---

## Quickstart (Option A — Docker Compose)

1. **Configure `.env`:**
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY in .env
   ```

2. **Start all services (PostgreSQL + `pgvector`, FastAPI backend, Streamlit frontend):**
   ```bash
   docker compose up --build
   ```

3. **Open in your browser:**
   - **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
   - **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## Quickstart (Option B — Local Development with `uv`)

1. **Install dependencies and configure `.env`:**
   ```bash
   uv sync
   cp .env.example .env
   # Add your GEMINI_API_KEY in .env
   ```

2. **Start PostgreSQL with `pgvector` and run migrations:**
   ```bash
   docker compose up -d postgres
   uv run alembic upgrade head
   ```

3. **Start the FastAPI backend:**
   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

4. **Start the Streamlit frontend (in a second terminal):**
   ```bash
   uv run streamlit run frontend/streamlit_app.py
   ```

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check and active Gemini model configuration |
| `POST` | `/documents/upload` | Upload and index a PDF file (`multipart/form-data`) |
| `POST` | `/documents/url` | Fetch and index a public webpage (`{"url": "https://..."}`) |
| `GET` | `/documents` | List all indexed documents with status and chunk counts |
| `GET` | `/documents/{id}` | Inspect a single document's metadata and status |
| `DELETE` | `/documents/{id}` | Delete a document, its stored PDF, and all vector chunks |
| `POST` | `/chat` | Ask a question with optional `document_ids` and conversation `history` |

---

## Testing, Linting & RAG Evaluation

```bash
# Run Ruff linter
uv run ruff check .

# Run automated unit & API integration tests (57 tests)
uv run pytest -v

# Run the offline RAG evaluation benchmark (Recall@K, MRR, Citation Accuracy)
uv run python -m scripts.evaluate_rag
```
