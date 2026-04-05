# RAG Document Intelligence Backend

A modular FastAPI backend with two REST APIs:
1. **Document Ingestion** — upload PDFs/TXT, chunk with two strategies, embed, store in Qdrant Cloud
2. **Conversational RAG** — multi-turn Q&A with Redis Cloud memory and LLM-driven interview booking

---

## Tech Stack

| Layer          | Choice                              |
|----------------|-------------------------------------|
| Framework      | FastAPI                             |
| Embeddings     | gemini `gemini-embedding`     |
| Vector DB      | Qdrant Cloud                        |
| LLM            | Groq                     |
| Chat Memory    | Redis Cloud                         |
| Metadata DB    | MySQL (via MySQL Workbench)         |
| PDF Extraction | `unstructured`                      |
| Config         | `pydantic-settings`                 |

---

## Project Structure

```
rag_backend/
├── app/
│   ├── main.py                  # App factory, lifespan, router registration
│   ├── config.py                # All settings loaded from .env
│   ├── models/
│   │   ├── sql.py               # ORM models: Document, InterviewBooking
│   │   └── database.py          # MySQL engine, SessionLocal, get_db dependency
│   ├── schemas/
│   │   ├── ingest.py            # Pydantic schemas for ingestion endpoints
│   │   └── chat.py              # Pydantic schemas for chat, booking, history
│   ├── services/
│   │   ├── extractor.py         # unstructured PDF/TXT text extraction
│   │   ├── chunker.py           # Semantic + context-header chunking
│   │   ├── embedder.py          # Batched gemini embedding
│   │   ├── vector_store.py      # Qdrant Cloud upsert + similarity search
│   │   ├── memory.py            # Redis Cloud rolling-window session store
│   │   ├── rag.py               # Custom RAG pipeline (no RetrievalQAChain)
│   │   └── booking.py           # LLM intent detection + booking persistence
│   └── routers/
│       ├── ingest.py            # POST /ingest, GET /ingest, GET /ingest/{doc_id}
│       └── chat.py              # POST /chat, POST /chat/book-interview,
│                                #   GET /chat/sessions/{id}/history, GET /chat/bookings
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

All external services are cloud-hosted — nothing to install locally except Python.

| Service      | Where to sign up                        | Free tier |
|--------------|-----------------------------------------|-----------|
| Qdrant Cloud | https://cloud.qdrant.io                 | Yes       |
| Redis Cloud  | https://redis.com/try-free              | Yes       |
| MySQL        | Already running via MySQL Workbench     | Local     |

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the MySQL database

Open MySQL Workbench and run:
```sql
CREATE DATABASE rag_db;
```
Tables (`documents`, `interview_bookings`) are created automatically when the app starts.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Fill in your `.env`:


### 5. Run the app

```bash
uvicorn app.main:app --reload
```


