"""
RAG-based Steam game recommender: ChromaDB retrieval + LiteLLM generation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

# Bypass any macOS/system proxy for local Ollama calls (else httpx routes localhost -> proxy -> 503).
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

# Modualize the paths and constants for easy configuration and maintenance
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"
DEFAULT_MODEL = "ollama/qwen3.5:9b"


def create_search_engine() -> GameSearchEngine:
    return GameSearchEngine()


class GameSearchEngine:
    def __init__(self) -> None:
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
        self.indexed_count = self.collection.count()

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked = self.rank_candidates(candidates)

        results = []
        for c in ranked:
            results.append({
                "app_id": c["app_id"],
                "name": c["name"],
                "score": c["score"],
                "short_description": c["short_description"],
                "genres": c["genres"],
                "tags": c["tags"],
                "price": c["price"],
                "release_date": c["release_date"],
                "header_image": c["header_image"],
                "store_page": f"https://store.steampowered.com/app/{c['app_id']}",
                "platforms": {
                    "windows": c["windows"],
                    "mac": c["mac"],
                    "linux": c["linux"],
                },
            })

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked),
            "meta": {
                "indexed_games": self.indexed_count,
                "retrieval_mode": "chromadb-rag",
            },
        }

    def retrieve_candidates(self, query: str, k: int = 15) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["metadatas", "distances", "documents"],
        )

        candidates = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = round(1.0 - distance / 2.0, 4)

            candidates.append({
                "app_id": meta["app_id"],
                "name": meta["name"],
                "short_description": meta["short_description"],
                "genres": json.loads(meta["genres"]),
                "tags": json.loads(meta["tags"]),
                "price": meta["price"],
                "release_date": meta["release_date"],
                "header_image": meta["header_image"],
                "windows": meta["windows"],
                "mac": meta["mac"],
                "linux": meta["linux"],
                "positive": meta["positive"],
                "negative": meta["negative"],
                "score": score,
                "document": results["documents"][0][i],
            })

        return candidates

    def rank_candidates(
        self, candidates: list[dict[str, Any]], top_n: int = 5
    ) -> list[dict[str, Any]]:
        for c in candidates:
            total = c["positive"] + c["negative"]
            ratio = c["positive"] / total if total > 0 else 0.5
            c["score"] = round(c["score"] * 0.7 + ratio * 0.3, 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    def generate_answer(self, query: str, ranked: list[dict[str, Any]]) -> str:
        if not ranked:
            return "No games matched your description."

        system_prompt = self._load_system_prompt()
        context = self._format_context(ranked)
        user_message = f"User query: {query}\n\nRetrieved games:\n{context}"

        try:
            import litellm

            model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=8192,
                temperature=0.7,
                # thinking is stripped anyway; leaving it on overflows num_ctx -> empty, slow answers
                think=False,
                num_ctx=4096,
            )
            answer = self._strip_thinking(response.choices[0].message.content)
            if answer:
                return answer
            # Model returned no usable answer (e.g. only reasoning); list the matches.
            names = ", ".join(c["name"] for c in ranked[:3])
            return f"Top matches: {names}."
        except Exception as exc:
            names = ", ".join(c["name"] for c in ranked[:3])
            return f"LLM unavailable ({type(exc).__name__}: {exc}). Top matches: {names}."

    def _load_system_prompt(self) -> str:
        try:
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "You are a knowledgeable gaming expert. Recommend games based on the retrieved context."

    @staticmethod
    def _format_context(candidates: list[dict[str, Any]]) -> str:
        parts = []
        for i, c in enumerate(candidates, 1):
            genres = ", ".join(c["genres"]) if c["genres"] else "N/A"
            tags = ", ".join(c["tags"][:5]) if c["tags"] else "N/A"
            total = c["positive"] + c["negative"]
            approval = f"{c['positive'] / total * 100:.0f}%" if total > 0 else "N/A"
            parts.append(
                f"{i}. {c['name']} (${c['price']:.2f})\n"
                f"   Genres: {genres} | Tags: {tags}\n"
                f"   Approval: {approval} ({total:,} reviews)\n"
                f"   {c['short_description'][:200]}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        if not text:
            return ""
        # Drop complete <think>...</think> blocks plus any unclosed trailing <think>
        # (truncated output). Whatever remains is the user-facing answer; if the model
        # emitted only reasoning this is empty -- never surface raw reasoning as the answer.
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*\Z", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()
