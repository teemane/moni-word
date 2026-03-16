"""MoniWord Pipeline — Central Configuration."""

import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = BASE_DIR / "state"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

HASH_DB_PATH = STATE_DIR / "processed_hashes.db"
SEMANTIC_DB_PATH = STATE_DIR / "semantic_map.db"

UNIFIED_JSONL = OUTPUT_DIR / "unified.jsonl"
DEDUPED_JSONL = OUTPUT_DIR / "deduped.jsonl"
VECTORS_JSONL = OUTPUT_DIR / "vectors.jsonl"

# ── Deduplication ──────────────────────────────────────────────────────
SHINGLE_SIZE = 3          # 3-word shingles
NUM_PERM = 128            # MinHash permutations
LSH_THRESHOLD = 0.85      # Jaccard similarity cutoff

# ── Vectorizer ─────────────────────────────────────────────────────────
BATCH_SIZE = 100           # Embedding batch size (sweet spot for 8GB RAM)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_TEXT_LENGTH = 512      # Max tokens per document

# ── Streaming ──────────────────────────────────────────────────────────
CHUNK_LINES = 10_000       # Lines per streaming chunk
MEMORY_LIMIT_MB = 900      # Trigger fallback if RSS exceeds this
TXT_CHUNK_WORDS = 500      # When a TXT has no blank lines, chunk every N words

# ── Wikipedia ZIM (tuned for 115GB on 8GB RAM) ────────────────────────
WIKI_BATCH_SIZE = 200      # Articles per batch
RAM_MIN_AVAILABLE_MB = 1024  # Pause if available RAM drops below 1GB
RAM_COOLDOWN_SECS = 10     # Seconds to sleep when RAM threshold is hit
WIKI_VELOCITY_INTERVAL = 1000  # Print speed report every N articles
WIKI_SKIP_PREFIXES = ("Category:", "Portal:", "Template:", "Wikipedia:",
                       "Help:", "Draft:", "Module:", "MediaWiki:", "File:",
                       "Talk:", "User:", "User talk:", "Special:")
WIKI_MIN_WORDS = 10        # Skip articles shorter than this

# ── Clustering ─────────────────────────────────────────────────────────
N_CLUSTERS = 50            # MiniBatchKMeans clusters for semantic map


def setup_logging(verbose: bool = False) -> None:
    """Configure file + console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_directories() -> None:
    """Create runtime directories if they don't exist."""
    for d in (STATE_DIR, OUTPUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
