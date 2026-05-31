"""Build ChromaDB vector store from Steam games SQLite database."""

import json
import sqlite3
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVIEWS_PER_GAME = 5
BATCH_SIZE = 500


def load_games(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT
            appid, name, short_description, about_the_game,
            release_date, price, header_image,
            windows, mac, linux,
            genres_json, tags_json,
            positive, negative, recommendations,
            developers_json, categories_json
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != ''
    """)
    return [dict(row) for row in cursor.fetchall()]


def load_reviews(conn: sqlite3.Connection) -> dict[int, str]:
    print("Loading top reviews per game (this may take a few minutes)...")
    start = time.time()

    cursor = conn.execute("""
        WITH ranked AS (
            SELECT
                appid, review,
                ROW_NUMBER() OVER (
                    PARTITION BY appid
                    ORDER BY CAST(weighted_vote_score AS REAL) DESC, votes_up DESC
                ) AS rn
            FROM reviews
            WHERE language = 'english'
              AND LENGTH(TRIM(review)) > 30
              AND voted_up = 1
        )
        SELECT appid, GROUP_CONCAT(review, ' ||| ') AS feedback
        FROM ranked
        WHERE rn <= ?
        GROUP BY appid
    """, (REVIEWS_PER_GAME,))

    reviews = {row[0]: row[1] for row in cursor.fetchall()}
    elapsed = time.time() - start
    print(f"  Loaded reviews for {len(reviews)} games in {elapsed:.1f}s")
    return reviews


def build_document(game: dict, feedback: str | None) -> str:
    name = game["name"]

    genres = json.loads(game["genres_json"]) if game["genres_json"] else []
    tags_raw = json.loads(game["tags_json"]) if game["tags_json"] else {}
    tag_names = list(tags_raw.keys())[:10] if isinstance(tags_raw, dict) else tags_raw[:10]

    desc = game["short_description"] or game["about_the_game"] or ""
    if len(desc) > 500:
        desc = desc[:500] + "..."

    parts = [f"Title: {name}"]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if tag_names:
        parts.append(f"Tags: {', '.join(tag_names)}")
    if desc:
        parts.append(f"Description: {desc}")
    if feedback:
        parts.append(f"Player Feedback: {feedback[:800]}")

    return "\n".join(parts)


def build_metadata(game: dict) -> dict:
    genres = json.loads(game["genres_json"]) if game["genres_json"] else []
    tags_raw = json.loads(game["tags_json"]) if game["tags_json"] else {}
    tag_names = list(tags_raw.keys())[:8] if isinstance(tags_raw, dict) else tags_raw[:8]

    return {
        "app_id": str(game["appid"]),
        "name": game["name"] or "Unknown title",
        "short_description": (game["short_description"] or "")[:500],
        "genres": json.dumps(genres),
        "tags": json.dumps(tag_names),
        "price": float(game["price"]) if game["price"] is not None else 0.0,
        "release_date": game["release_date"] or "",
        "header_image": game["header_image"] or "",
        "windows": bool(game["windows"]),
        "mac": bool(game["mac"]),
        "linux": bool(game["linux"]),
        "positive": int(game["positive"] or 0),
        "negative": int(game["negative"] or 0),
    }


def main():
    print(f"Database: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))

    games = load_games(conn)
    print(f"Loaded {len(games)} games")

    reviews = load_reviews(conn)
    conn.close()

    print(f"Initializing ChromaDB at {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(games)
    print(f"Embedding and upserting {total} games in batches of {BATCH_SIZE}...")
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = games[i : i + BATCH_SIZE]
        ids, documents, metadatas = [], [], []

        for game in batch:
            appid = game["appid"]
            ids.append(str(appid))
            documents.append(build_document(game, reviews.get(appid)))
            metadatas.append(build_metadata(game))

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {done / total * 100:.0f}% | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")

    elapsed = time.time() - start
    print(f"\nDone. {collection.count()} games indexed in {elapsed:.1f}s")
    print(f"Vector DB stored at: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
