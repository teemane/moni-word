"""MoniWord Pipeline — Embedding & Semantic Map Builder.

Reads deduped JSONL, computes embeddings in batches of 100 using
sentence-transformers, and stores vectors in SQLite. Optionally builds
a clustered semantic map via MiniBatchKMeans.
"""

import base64
import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from tools.config import (
    BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_TEXT_LENGTH,
    N_CLUSTERS,
)

log = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────

def vectorize(
    input_path: Path,
    output_path: Path,
    db_path: Path,
) -> int:
    """Vectorize *input_path* (JSONL), write results to *output_path* and SQLite.

    Returns number of documents vectorized.
    """
    input_path, output_path, db_path = Path(input_path), Path(output_path), Path(db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _init_db(db_path)
    model = _load_model()
    dim = model.get_sentence_embedding_dimension()
    total = 0

    log.info("Vectorizing with model=%s, batch_size=%d, dim=%d",
             EMBEDDING_MODEL, BATCH_SIZE, dim)

    with open(output_path, "w", encoding="utf-8") as writer:
        for batch in tqdm(_batch_reader(input_path, BATCH_SIZE),
                          desc="Vectorizing", unit="batch"):
            texts = [
                r["text"][:MAX_TEXT_LENGTH] for r in batch
            ]
            vectors = _encode_batch(model, texts)
            _store_vectors(conn, batch, vectors, dim)
            _write_output(writer, batch, vectors)
            total += len(batch)

    conn.close()
    log.info("Vectorization complete: %d documents", total)

    # Build semantic map (clustering)
    log.info("Building semantic map with %d clusters...", N_CLUSTERS)
    build_semantic_map(db_path, n_clusters=N_CLUSTERS)

    return total


def build_semantic_map(db_path: Path, n_clusters: int = N_CLUSTERS) -> None:
    """Cluster all stored vectors using MiniBatchKMeans and save results."""
    from sklearn.cluster import MiniBatchKMeans

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
    doc_count = cursor.fetchone()[0]

    if doc_count < n_clusters:
        n_clusters = max(2, doc_count // 2)
        log.info("Adjusted clusters to %d (fewer docs than requested clusters)", n_clusters)

    if doc_count == 0:
        log.warning("No vectors to cluster")
        conn.close()
        return

    # Get embedding dimension
    row = conn.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()
    dim = row[0]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, n_init=3)

    # Partial fit in chunks
    chunk_size = 10_000
    offset = 0
    while True:
        rows = conn.execute(
            "SELECT doc_id, vector FROM embeddings LIMIT ? OFFSET ?",
            (chunk_size, offset),
        ).fetchall()
        if not rows:
            break
        vectors = np.array(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )
        kmeans.partial_fit(vectors)
        offset += chunk_size

    # Store centroids
    conn.execute("DROP TABLE IF EXISTS clusters")
    conn.execute("""
        CREATE TABLE clusters (
            cluster_id  INTEGER PRIMARY KEY,
            centroid    BLOB,
            label       TEXT,
            doc_count   INTEGER DEFAULT 0
        )
    """)
    for i, centroid in enumerate(kmeans.cluster_centers_):
        conn.execute(
            "INSERT INTO clusters (cluster_id, centroid) VALUES (?, ?)",
            (i, centroid.astype(np.float32).tobytes()),
        )

    # Assign documents to clusters
    conn.execute("DROP TABLE IF EXISTS doc_clusters")
    conn.execute("""
        CREATE TABLE doc_clusters (
            doc_id      TEXT,
            cluster_id  INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_cluster ON doc_clusters(cluster_id)")

    offset = 0
    cluster_counts = [0] * n_clusters
    while True:
        rows = conn.execute(
            "SELECT doc_id, vector FROM embeddings LIMIT ? OFFSET ?",
            (chunk_size, offset),
        ).fetchall()
        if not rows:
            break
        doc_ids = [r[0] for r in rows]
        vectors = np.array(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )
        labels = kmeans.predict(vectors)
        conn.executemany(
            "INSERT INTO doc_clusters (doc_id, cluster_id) VALUES (?, ?)",
            list(zip(doc_ids, (int(l) for l in labels))),
        )
        for l in labels:
            cluster_counts[int(l)] += 1
        offset += chunk_size

    # Update cluster doc counts
    for i, cnt in enumerate(cluster_counts):
        conn.execute(
            "UPDATE clusters SET doc_count = ? WHERE cluster_id = ?", (cnt, i)
        )

    conn.commit()
    conn.close()
    log.info("Semantic map built: %d clusters from %d documents", n_clusters, doc_count)


# ── Model Loading ─────────────────────────────────────────────────────

_model_cache = None


def _load_model() -> SentenceTransformer:
    """Load the sentence-transformer model (cached)."""
    global _model_cache
    if _model_cache is None:
        log.info("Loading model: %s", EMBEDDING_MODEL)
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


# ── Batch Processing ──────────────────────────────────────────────────

def _batch_reader(input_path: Path, batch_size: int) -> Iterator[list[dict]]:
    """Yield batches of JSONL records."""
    batch: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                batch.append(record)
            except json.JSONDecodeError:
                continue
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _encode_batch(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode a batch of texts into embeddings."""
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# ── SQLite Persistence ────────────────────────────────────────────────

def _init_db(db_path: Path) -> sqlite3.Connection:
    """Create the embeddings table if needed, return connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id  TEXT PRIMARY KEY,
            text    TEXT,
            vector  BLOB,
            source  TEXT,
            dim     INTEGER
        )
    """)
    conn.commit()
    return conn


def _store_vectors(
    conn: sqlite3.Connection,
    records: list[dict],
    vectors: np.ndarray,
    dim: int,
) -> None:
    """Bulk insert vectors into SQLite."""
    rows = [
        (
            str(rec["id"]),
            rec.get("text", ""),
            vec.astype(np.float32).tobytes(),
            rec.get("source", ""),
            dim,
        )
        for rec, vec in zip(records, vectors)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (doc_id, text, vector, source, dim) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ── Output ────────────────────────────────────────────────────────────

def _write_output(writer, records: list[dict], vectors: np.ndarray) -> None:
    """Append vectorized records to JSONL with base64-encoded vectors."""
    for rec, vec in zip(records, vectors):
        out = {
            "id": rec["id"],
            "source": rec.get("source", ""),
            "text": rec.get("text", "")[:200],  # Truncated preview
            "vector": base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii"),
        }
        writer.write(json.dumps(out, ensure_ascii=False) + "\n")
