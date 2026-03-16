"""MoniWord Pipeline — MinHash LSH Fuzzy Deduplication.

Streams through unified JSONL, computes MinHash signatures via shingling,
and uses Locality Sensitive Hashing to discard near-duplicates (Jaccard > 0.85).

All signatures are persisted to SQLite for auditability. The LSH index only
holds unique documents to keep RAM under ~900 MB.

Also exposes ``DedupChecker`` for inline single-document dedup (used by
wiki_ingestor and any other streaming source).

Wikipedia optimization: articles are shingled from word 20 onward to skip
boilerplate intros like "X is a Y located in Z" that cause false collisions.
"""

import json
import logging
import os
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Iterator

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from tools.config import (
    CHUNK_LINES,
    HASH_DB_PATH,
    LSH_THRESHOLD,
    MEMORY_LIMIT_MB,
    NUM_PERM,
    SHINGLE_SIZE,
)

log = logging.getLogger(__name__)

_CLEAN_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Wikipedia articles often share boilerplate first sentences.
# Skip the first N words before shingling to reduce false collisions.
_WIKI_INTRO_SKIP = 20


# ── Reusable single-doc checker ──────────────────────────────────────

class DedupChecker:
    """Stateful dedup checker backed by MinHash LSH + SQLite.

    Usage::

        checker = DedupChecker(db_path)
        is_dup = checker.is_duplicate(doc_id, text, source)
        checker.flush()   # flush remaining SQLite buffer
        checker.close()

    For Wikipedia-scale ingestion, set ``flush_every`` to a small number
    (e.g. 25) so checkpoints are written after every batch.
    """

    def __init__(
        self,
        db_path: Path = HASH_DB_PATH,
        flush_every: int = CHUNK_LINES,
        skip_intro: bool = False,
    ) -> None:
        self._conn = _init_db(db_path)
        self._lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
        self._buffer: list[tuple] = []
        self._flush_every = flush_every
        self._skip_intro = skip_intro
        self.total = 0
        self.duplicates = 0

    # ── core check ────────────────────────────────────────────────────

    def is_duplicate(self, doc_id: str, text: str, source: str = "") -> bool:
        """Return *True* if *text* is a near-duplicate of something already seen."""
        self.total += 1
        doc_id = str(doc_id)

        shingles = shingle(text, skip_intro=self._skip_intro)
        if not shingles:
            return False  # too short to shingle -> treat as unique

        mh = compute_minhash(shingles)

        is_dup = False
        try:
            if self._lsh.query(mh):
                is_dup = True
        except ValueError:
            pass  # empty index on first query

        if not is_dup:
            try:
                self._lsh.insert(doc_id, mh)
            except ValueError:
                is_dup = True  # duplicate key

        if is_dup:
            self.duplicates += 1

        # Buffer for SQLite
        self._buffer.append((doc_id, pickle.dumps(mh), source, int(is_dup)))
        if len(self._buffer) >= self._flush_every:
            self.flush()

        return is_dup

    def flush(self) -> None:
        """Write buffered signatures to SQLite."""
        if self._buffer:
            _store_signatures(self._conn, self._buffer)
            self._buffer.clear()

    def close(self) -> None:
        """Flush and close the SQLite connection."""
        self.flush()
        self._conn.close()


# ── Public API (file-to-file) ─────────────────────────────────────────

def deduplicate(
    input_path: Path,
    output_path: Path,
    db_path: Path,
) -> tuple[int, int]:
    """Run fuzzy deduplication on *input_path*, write clean docs to *output_path*.

    Returns (total_documents, duplicates_found).
    """
    input_path, output_path, db_path = Path(input_path), Path(output_path), Path(db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checker = DedupChecker(db_path)

    log.info("Starting deduplication (threshold=%.2f, shingle=%d, perms=%d)",
             LSH_THRESHOLD, SHINGLE_SIZE, NUM_PERM)

    with (
        open(input_path, "r", encoding="utf-8") as reader,
        open(output_path, "w", encoding="utf-8") as writer,
    ):
        for record in tqdm(_jsonl_reader(reader), desc="Deduplicating", unit="doc"):
            doc_id = str(record["id"])
            text = record.get("text", "")
            source = record.get("source", "")

            if not checker.is_duplicate(doc_id, text, source):
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    checker.close()
    log.info("Deduplication complete: %d total, %d duplicates removed, %d unique",
             checker.total, checker.duplicates, checker.total - checker.duplicates)
    return checker.total, checker.duplicates


# ── Shingling & Hashing (public for reuse) ────────────────────────────


def shingle(text: str, k: int = SHINGLE_SIZE, skip_intro: bool = False) -> set[str]:
    """Split *text* into k-word shingles. Lowercased, punctuation stripped.

    When *skip_intro* is True, the first ~20 words are dropped before
    shingling to avoid false positives from Wikipedia boilerplate intros.
    """
    text = _CLEAN_RE.sub("", text.lower())
    words = text.split()
    if skip_intro and len(words) > _WIKI_INTRO_SKIP + k:
        words = words[_WIKI_INTRO_SKIP:]
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def compute_minhash(shingles: set[str], num_perm: int = NUM_PERM) -> MinHash:
    """Create a datasketch MinHash from a set of shingles."""
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


# ── SQLite Persistence ────────────────────────────────────────────────

def _init_db(db_path: Path) -> sqlite3.Connection:
    """Create the signatures table if needed, return connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signatures (
            doc_id       TEXT PRIMARY KEY,
            minhash      BLOB,
            source_file  TEXT,
            is_duplicate INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON signatures(source_file)")
    conn.commit()
    return conn


def _store_signatures(conn: sqlite3.Connection, buffer: list[tuple]) -> None:
    """Batch-insert signatures into SQLite."""
    conn.executemany(
        "INSERT OR REPLACE INTO signatures (doc_id, minhash, source_file, is_duplicate) VALUES (?, ?, ?, ?)",
        buffer,
    )
    conn.commit()


# ── Streaming Helpers ─────────────────────────────────────────────────

def _jsonl_reader(f) -> Iterator[dict]:
    """Yield parsed JSON objects from a file handle, skipping bad lines."""
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _check_memory() -> int:
    """Return current process RSS in MB (cross-platform)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        except ImportError:
            return 0
