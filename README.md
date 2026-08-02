# Nixi RAG — Clinical RAG on German gynecology guidelines (Option B)

Local retrieval + generation over AWMF/DGGG guideline PDFs. Answers in German with citations, and **refuses** when the retrieved context does not support an answer. Everything runs on-premise via [Ollama](https://ollama.com) (no hosted LLM/embedding APIs).

**Frozen eval numbers** are under [`reports/`](reports/).

---

## Quick start (examiner)

This repo includes a ready-made vector index (`data/chroma/`). Clone → install → pull Ollama models → start the UI. You do **not** need to run `build_index.py` for normal use.

### 1. Clone the repository

```bash
git clone https://github.com/PremSavariya/RAG-Clinical-RAG-on-German-gynecology-guidelines-Option-B-.git
cd RAG-Clinical-RAG-on-German-gynecology-guidelines-Option-B-
```

Install Python deps ([uv](https://docs.astral.sh/uv/), Python **3.12**):

```bash
uv sync
```

### 2. Check the environment and update `.env`

```bash
cp .env.example .env
```

Open `.env` and confirm (or change) these values so they match the models you pull in step 3:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible API |
| `OLLAMA_API_KEY` | `ollama` | Placeholder (Ollama ignores auth locally) |
| `LLM_MODEL_NAME` | `qwen2.5:3b` | Chat model used by the UI |
| `EMBEDDINGS_MODEL_NAME` | `bge-m3:latest` | Embedding model for retrieval |

If Ollama runs in Docker on another host/port, set `OLLAMA_BASE_URL` accordingly (still must end with `/v1`).

### 3. Download Ollama models, then run only the UI

Pick **one** of the Ollama setups below, pull the two models, then start the UI.

#### Option A — Ollama on the host

1. Install [Ollama](https://ollama.com) and ensure it is running (`http://localhost:11434`).
2. Pull models once:

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
```

#### Option B — Ollama via Docker

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec -it ollama ollama pull qwen2.5:3b
docker exec -it ollama ollama pull bge-m3
```

Keep `OLLAMA_BASE_URL=http://localhost:11434/v1` in `.env` (port published to the host).

#### Start the UI only

```bash
uv run uvicorn scripts.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

You can ask the brief questions, inspect retrieved chunks, and check refusals on trap questions (stroke / pneumonia / hypertension).

> **Note:** A **prebuilt** Chroma index is included under `data/chroma/`. You do **not** need to run `build_index.py` or provide PDFs to use the UI. Keep `EMBEDDINGS_MODEL_NAME=bge-m3:latest` so queries match the index. The first retrieval may download the local reranker `BAAI/bge-reranker-v2-m3` from Hugging Face if rerank is enabled in config.
