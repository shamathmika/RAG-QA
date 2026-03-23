# RAG Q&A - Ask My Docs

A Retrieval-Augmented Generation (RAG) question-answering system that lets users upload PDF documents and ask natural language questions about their content. Built as a personal portfolio tool where recruiters can ask questions to learn more about you.

## Architecture

```
POST /api/upload/
+------------------------------------------------------------+
| PDF -> extract_text -> chunk_text -> embed_and_store -> END |
|         (PyMuPDF)      (splitter)    (OpenAI + MongoDB)     |
+------------------------------------------------------------+

POST /api/search/
+------------------------------------------------------------+
| Query -> embed_query -> check_cache --> HIT -> END          |
|                             |                               |
|                            MISS                             |
|                             v                               |
|          vector_search -> rerank -> call_llm -> update_cache|
|          (MongoDB Atlas)  (0.7v     (GPT-4o     (Redis)     |
|                           +0.3k)    -mini)                  |
+------------------------------------------------------------+

Tech: Django + LangGraph + OpenAI + MongoDB Atlas + Redis
```

## Tech Stack

| Technology | Role |
|---|---|
| **Python 3.9+** | Core language |
| **Django + DRF** | REST API framework (endpoints for search and upload) |
| **LangGraph** | Workflow orchestration (state graphs with nodes and edges) |
| **LangChain** | OpenAI embeddings wrapper, text splitting utilities |
| **OpenAI API** | Embeddings (`text-embedding-3-small`, 1536D) and chat (`gpt-4o-mini`) |
| **MongoDB Atlas** | Document storage + vector search (HNSW index with cosine similarity) |
| **Redis Stack** | Semantic cache (HNSW vector index via RediSearch module) |
| **PyMuPDF (fitz)** | PDF text extraction |

## How It Works

### Upload Pipeline (`api/graphs/upload_graph.py`)

A LangGraph state graph with 3 nodes:

1. **`extract_text`** - Opens the PDF using PyMuPDF and extracts text from each page. Produces a list of `{text, page_number}` dicts.

2. **`chunk_text`** - Splits each page's text into overlapping chunks (~1000 characters, 200 character overlap) using LangChain's `RecursiveCharacterTextSplitter`. Overlap ensures context isn't lost at chunk boundaries. Produces a flat list of `{text, page_number}` dicts.

3. **`embed_and_store`** - Generates a 1536-dimensional vector embedding for each chunk using OpenAI's `text-embedding-3-small` model, then stores each chunk in MongoDB Atlas as a document with fields: `text`, `embedding`, `file_name`, `page_number`.

### Search Pipeline (`api/graphs/search_graph.py`)

A LangGraph state graph with 6 nodes and 1 conditional edge:

1. **`embed_query`** - Converts the user's question into a 1536D vector embedding (same model as upload).

2. **`check_cache`** - Searches Redis for a previously cached question with similar meaning (cosine similarity >= 0.80). If found, returns the cached answer immediately and skips all remaining nodes.

3. **`vector_search`** - Uses MongoDB Atlas `$vectorSearch` aggregation to find the top 5 document chunks most similar to the query vector. Uses the HNSW index for approximate nearest neighbor search.

4. **`rerank`** - Re-scores the results using a weighted formula: `0.7 * vector_score + 0.3 * keyword_score`, where keyword_score measures word overlap between the query and chunk text. Returns the top 3 results.

5. **`call_llm`** - Assembles a RAG prompt with the top 3 chunks as context and sends it to GPT-4o-mini. The system prompt instructs the LLM to answer ONLY from the provided context, preventing hallucination.

6. **`update_cache`** - Stores the question, answer, and embedding in Redis with a 24-hour TTL for future cache hits.

### Semantic Cache (`api/cache.py`)

The `RedisSemanticCache` class provides meaning-based caching:

- **Not exact match** - "What are her skills?" and "What skills does she have?" can hit the same cache entry because their vector embeddings are similar.
- **HNSW vector index** - Uses RediSearch's HNSW index on the `embedding` field for fast KNN similarity search.
- **Cosine similarity threshold** - Only returns a cached answer if similarity >= 0.80 (configurable).
- **Auto-expiry** - Cached entries expire after 24 hours via Redis TTL.

## Setup

### Prerequisites

- Python 3.9+
- Docker (for Redis Stack)
- MongoDB Atlas account (free tier works)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG-QA

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
MONGODB_URI=your-mongodb-atlas-connection-string
REDIS_URL=redis://localhost:6379
```

### MongoDB Atlas Setup

1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a database called `rag_db` with a collection called `documents`
3. Go to **Search & Vector Search** -> **Create Search Index**
4. Select **Vector Search** and **JSON Editor**
5. Name the index `vector_index`, select `rag_db.documents`, and use this definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

6. Make sure to whitelist your IP in **Network Access**

### Run Redis

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### Run the App

```bash
source venv/bin/activate
python manage.py runserver
```

The API will be available at `http://localhost:8000`.

### Testing Without Django

You can test the LangGraph pipelines directly without running the Django server:

```bash
# Upload a PDF (place a test.pdf in the project root first)
python test_upload.py

# Ask a question about the uploaded PDF
python test_search.py
```

`test_search.py` will prompt you to type a question. It runs the full search pipeline and prints the answer, cache hit status, and similarity scores.

## API Endpoints

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/upload/ \
  -F "file=@your-document.pdf"
```

Response:
```json
{
  "message": "Successfully uploaded and processed 12 chunks"
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the full name?"}'
```

Response:
```json
{
  "answer": "The full name is FNU Shamathmika.",
  "cache_hit": false
}
```

Asking a similar question again will return `"cache_hit": true` with a faster response.

## Project Structure

```
RAG-QA/
├── rag_qa/                  # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── api/                     # Django app
│   ├── views.py             # API endpoints (SearchView, UploadView)
│   ├── urls.py              # URL routing
│   ├── cache.py             # RedisSemanticCache class
│   └── graphs/              # LangGraph workflows
│       ├── upload_graph.py  # PDF upload pipeline
│       └── search_graph.py  # RAG search pipeline
├── requirements.txt
├── .env                     # Environment variables (not committed)
├── .gitignore
└── manage.py
```
