# Steam Game Recommender (RAG Pipeline)

A retrieval-augmented generation system that recommends Steam games based on natural language queries. Uses ChromaDB for vector search, sentence-transformers for embeddings, and LiteLLM for multi-model LLM integration.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Ollama](https://ollama.ai/)** with a pulled model (e.g. `ollama pull qwen3.5:9b`)
- Extract the data `steam_games_reviews_25.sqlite` and save it to the root
- **Apple Silicon (M-series)** recommended. Ollama uses Metal automatically; no extra setup needed.

## Quick Start

Run all commands from this directory (the project root).

```bash
# 1. Pull the LLM model
ollama pull qwen3.5:9b

# 2. Install dependencies
uv sync

# 3. Build the vector database (one-time, ~15 min)
uv run python build_vector_db.py

# 4. Start the web app
uv run flask --app app run --debug

# 5. Open http://127.0.0.1:5000
```

## Project Structure

```
app.py                  Flask web server (GET /, POST /api/search)
recommender.py          RAG pipeline (retrieve -> rank -> generate)
build_vector_db.py      Offline script: embed 39K games into ChromaDB
steam_sqlite.py         SQLite data loader
system_prompt.txt       LLM system prompt (editable without code changes)
.env.example            Environment variable reference
pyproject.toml          Project metadata and dependencies
notebook/
  qwen-thinking-benchmark.ipynb   Latency/output benchmark: think=False vs think=True
chroma_db/              Vector database (generated, gitignored)
static/                 Frontend CSS + JS
templates/              HTML template
steam_games_reviews_25.sqlite   Source database (39K games, 7.7M reviews)
```

## How It Works

1. **Build phase** (`build_vector_db.py`): Reads all games from SQLite, fetches top 5 positive English reviews per game, creates a rich text document per game (title, genres, tags, description, player feedback), embeds with `all-MiniLM-L6-v2` into ChromaDB.

2. **Query phase** (`recommender.py`):
   - `retrieve_candidates()` embeds the user query, retrieves 15 nearest neighbors from ChromaDB via cosine similarity.
   - `rank_candidates()` reranks with a blended score: 70% semantic similarity + 30% Steam review approval ratio. Returns top 5.
   - `generate_answer()` formats ranked candidates as context, sends to the LLM with a gaming-journalist system prompt, returns a natural-language recommendation.

3. **Frontend**: Single-page interface. Users type a game description and get the LLM recommendation plus matched game cards with images, genres, tags, and Steam store links.

## GPU Setup (Apple Silicon)

On M-series Macs, Ollama uses Metal for GPU acceleration automatically — no drivers or configuration needed. The model runs in unified memory shared between CPU and GPU.

Qwen3.5:9b defaults to a **262,144-token context window**. The KV cache for that is large enough to crowd out the model weights in unified memory on chips with 16 GB. `num_ctx=4096` is already set in `recommender.py` to keep the KV cache small and leave headroom for the model.

Verify the model is GPU-accelerated:

```bash
ollama ps
```

You want to see a non-zero value under `PROCESSOR` (e.g. `100% GPU`), not `100% CPU`.

## Benchmarks

`notebook/qwen-thinking-benchmark.ipynb` measures latency and output for `qwen3.5:9b` with reasoning off (`think=False`) vs on (`think=True`) on the same query, reusing the production retrieval pipeline so the `think` flag is the only variable. On an Apple M5 Pro (48 GB, 100% GPU, `num_ctx=8192`): `think=False` runs ~11s / 313 tokens, while `think=True` runs ~75s / 2499 tokens for a near-identical final answer. The notebook captures both generations verbatim, including the hidden reasoning trace.

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

### LLM Models

Set `LLM_MODEL` in `.env` to any LiteLLM-compatible model string:

| Provider | Model | Env Vars Needed |
|---|---|---|
| Ollama (local) | `ollama/qwen3.5:9b` | None |
| DeepSeek | `deepseek/deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| Google Gemini | `gemini/gemini-3.1-flash-lite` | `GEMINI_API_KEY` |
| Zhipu AI | `zhipu/glm-5.1` | `ZHIPUAI_API_KEY` |
| Minimax | `minimax/minimax-m2.7` | `MINIMAX_API_KEY` |
| Moonshot | `moonshot/kimi-k2.6` | `MOONSHOT_API_KEY` |

### System Prompt

Edit `system_prompt.txt` in any text editor. Changes take effect on the next query (no restart needed).
