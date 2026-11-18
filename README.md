# Document Intelligence RAG

A simple yet fully-featured Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **PostgreSQL (`pgvector`)**, **Google Gemini**, and **Streamlit**.

## Features

- Upload and extract text from PDF documents (preserving page numbers).
- Extract readable content from website URLs.
- Chunk content and generate embeddings using Google Gemini Embeddings (`gemini-embedding-001`).
- Store documents, chunks, and vectors in PostgreSQL with `pgvector`.
- Ask natural-language questions (including conversational follow-ups) grounded in your documents with structured source citations.

## Quickstart

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run Tests & Linter

```bash
uv run ruff check .
uv run pytest
```

### 4. Run Backend Server

```bash
uv run uvicorn app.main:app --reload
```
