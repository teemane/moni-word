"""MoniWord — Semantic Search.

Search your vector database by meaning, not keywords.
"""

import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tools.config import EMBEDDING_MODEL, SEMANTIC_DB_PATH


def load_model() -> SentenceTransformer:
    print(f"  Loading model: {EMBEDDING_MODEL}...")
    return SentenceTransformer(EMBEDDING_MODEL)


def search(query: str, model: SentenceTransformer, db_path: Path, top_k: int = 5) -> list[dict]:
    """Search the semantic map for the most relevant documents."""
    # Encode the query into a vector
    q_vec = model.encode(query, convert_to_numpy=True).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)  # normalize for cosine similarity

    conn = sqlite3.connect(str(db_path))

    # Load all vectors and compute cosine similarity
    # For large databases, consider using FAISS instead
    rows = conn.execute("SELECT doc_id, text, vector, source FROM embeddings").fetchall()
    conn.close()

    if not rows:
        return []

    results = []
    for doc_id, text, vec_blob, source in rows:
        vec = np.frombuffer(vec_blob, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)  # normalize
        score = float(np.dot(q_vec, vec))
        results.append({
            "doc_id": doc_id,
            "score": score,
            "source": source,
            "text": text,
        })

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def print_results(query: str, results: list[dict]) -> None:
    print()
    print(f"  Query: \"{query}\"")
    print(f"  {'=' * 60}")

    if not results:
        print("  No results found. Run the vectorizer first (option [4] in app.py).")
        return

    for i, r in enumerate(results, 1):
        score_pct = r["score"] * 100
        preview = r["text"][:300].replace("\n", " ")
        title = ""

        # Try to extract title from source
        if "wikipedia" in r["source"].lower():
            title = f" [Wikipedia]"

        print(f"\n  #{i}  (similarity: {score_pct:.1f}%){title}")
        print(f"  ID: {r['doc_id']}  |  Source: {r['source']}")
        print(f"  {'-' * 60}")
        print(f"  {preview}...")
        print()


def interactive_mode(model: SentenceTransformer, db_path: Path) -> None:
    print()
    print("  MoniWord Semantic Search")
    print("  Type a question or topic. Type 'quit' to exit.")
    print()

    while True:
        try:
            query = input("  Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        results = search(query, model, db_path)
        print_results(query, results)
