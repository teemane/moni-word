"""MoniWord Pipeline — Wikipedia ZIM Ingestion (Phase 4, 115GB-scale).

Streams articles from Wikipedia ZIM files with:
  - Checkpoint-based resume (survives crashes at any point)
  - Batch-of-25 processing with SQLite commit after each batch
  - Aggressive RAM guard (pause when available < 1GB)
  - gc.collect() + posix_fadvise page-cache hints on WSL
  - Velocity reports every 1000 articles
  - Namespace 'A' filtering, skip Category/Portal/Template/etc.
  - Wikipedia intro-skip dedup to handle similar opening sentences
"""

import gc
import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import psutil
from bs4 import BeautifulSoup, SoupStrainer
from libzim.reader import Archive

from tools.config import (
    DEDUPED_JSONL,
    HASH_DB_PATH,
    RAM_COOLDOWN_SECS,
    RAM_MIN_AVAILABLE_MB,
    STATE_DIR,
    WIKI_BATCH_SIZE,
    WIKI_MIN_WORDS,
    WIKI_SKIP_PREFIXES,
    WIKI_VELOCITY_INTERVAL,
    ensure_directories,
)
from tools.deduplicator import DedupChecker

log = logging.getLogger(__name__)

# Regex for collapsing whitespace after HTML stripping
_MULTI_WS = re.compile(r"\s+")
# Regex to detect redirect pages (mediawiki-style)
_REDIRECT_RE = re.compile(r"^\s*#REDIRECT", re.IGNORECASE)
# Only parse body content, skip <script>, <style> etc.
_BODY_STRAINER = SoupStrainer("body")

# Checkpoint DB path
_CHECKPOINT_DB = STATE_DIR / "wiki_checkpoint.db"


# ── Public API ─────────────────────────────────────────────────────────

def ingest_zim(
    zim_path: Path,
    output_path: Path = DEDUPED_JSONL,
    db_path: Path = HASH_DB_PATH,
) -> dict:
    """Ingest a Wikipedia ZIM file with inline dedup and checkpointing.

    Resumes from the last checkpoint automatically if interrupted.
    Returns a stats dict: {total_scanned, accepted, duplicates, skipped, resumed_from}.
    """
    zim_path = Path(zim_path)
    output_path = Path(output_path)
    ensure_directories()

    if not zim_path.exists():
        raise FileNotFoundError(f"ZIM file not found: {zim_path}")

    # Open ZIM (lazy — no data loaded into RAM)
    archive = Archive(str(zim_path))
    entry_count = archive.entry_count

    # Set up checkpoint DB
    ckpt_conn = _init_checkpoint_db()
    resume_idx, doc_id_counter = _get_checkpoint(ckpt_conn, str(zim_path))

    # Set up dedup checker — flush every batch (25) for checkpoint safety
    # skip_intro=True to handle Wikipedia's similar opening sentences
    checker = DedupChecker(db_path, flush_every=WIKI_BATCH_SIZE, skip_intro=True)

    stats = {
        "total_scanned": 0,
        "accepted": 0,
        "duplicates": 0,
        "skipped": 0,
        "resumed_from": resume_idx,
    }

    if resume_idx > 0:
        log.info("RESUMING from entry %d / %d (%.1f%% done previously)",
                 resume_idx, entry_count, resume_idx / entry_count * 100)
    else:
        log.info("Starting fresh ZIM ingestion: %s (%d entries)", zim_path.name, entry_count)

    log.info("Batch size: %d | RAM guard: available < %d MB | Velocity report every %d articles",
             WIKI_BATCH_SIZE, RAM_MIN_AVAILABLE_MB, WIKI_VELOCITY_INTERVAL)

    batch: list[dict] = []
    velocity_start = time.time()
    velocity_count = 0
    articles_processed = 0

    with open(output_path, "a", encoding="utf-8") as writer:
        for entry_idx in range(resume_idx, entry_count):
            # ── Extract article ───────────────────────────────────
            result = _extract_article(archive, entry_idx)
            if result is None:
                continue

            title, text = result
            stats["total_scanned"] += 1
            articles_processed += 1

            # Skip too-short articles
            if len(text.split()) < WIKI_MIN_WORDS:
                stats["skipped"] += 1
                continue

            doc_id = f"wiki_{doc_id_counter}"
            doc_id_counter += 1

            # Inline fuzzy dedup
            if checker.is_duplicate(doc_id, text, source=zim_path.name):
                stats["duplicates"] += 1
                continue

            record = {
                "id": doc_id,
                "source": f"wikipedia:{zim_path.name}",
                "title": title,
                "text": text,
            }
            batch.append(record)

            # ── Flush batch ───────────────────────────────────────
            if len(batch) >= WIKI_BATCH_SIZE:
                _write_batch(writer, batch)
                stats["accepted"] += len(batch)
                batch.clear()

                # Commit dedup signatures
                checker.flush()

                # Save checkpoint AFTER EVERY BATCH
                _save_checkpoint(ckpt_conn, str(zim_path), entry_idx + 1, doc_id_counter)

                # RAM guard
                _ram_guard()

                # Force garbage collection every batch
                gc.collect()

            # ── Velocity report ───────────────────────────────────
            velocity_count += 1
            if velocity_count >= WIKI_VELOCITY_INTERVAL:
                _velocity_report(
                    velocity_start, velocity_count, articles_processed,
                    entry_idx, entry_count, stats,
                )
                velocity_start = time.time()
                velocity_count = 0

        # ── Flush remaining ───────────────────────────────────────────
        if batch:
            _write_batch(writer, batch)
            stats["accepted"] += len(batch)
            checker.flush()
            _save_checkpoint(ckpt_conn, str(zim_path), entry_count, doc_id_counter)

    checker.close()
    ckpt_conn.close()

    log.info(
        "ZIM ingestion complete: %d scanned, %d accepted, %d duplicates, %d skipped",
        stats["total_scanned"], stats["accepted"],
        stats["duplicates"], stats["skipped"],
    )
    return stats


# ── Article Extraction ────────────────────────────────────────────────

def _extract_article(archive: Archive, entry_idx: int) -> tuple[str, str] | None:
    """Extract a single article by index. Returns (title, text) or None."""
    try:
        entry = archive._get_entry_by_id(entry_idx)
    except (KeyError, IndexError, RuntimeError):
        return None

    # Skip redirects
    if entry.is_redirect:
        return None

    title = entry.title or entry.path

    # Filter: skip non-article namespaces by title prefix
    for prefix in WIKI_SKIP_PREFIXES:
        if title.startswith(prefix):
            return None

    try:
        item = entry.get_item()
    except Exception:
        return None

    # Skip non-HTML content (images, CSS, JS, etc.)
    mimetype = item.mimetype
    if "html" not in mimetype:
        return None

    try:
        raw_content = bytes(item.content).decode("utf-8", errors="replace")
    except Exception:
        return None

    # Skip redirect pages embedded in HTML
    if _REDIRECT_RE.search(raw_content[:200]):
        return None

    # Clean HTML -> plain text
    text = _html_to_plain(raw_content)
    if not text:
        return None

    # Release raw_content reference immediately
    del raw_content

    return title, text


# ── HTML Cleaning ─────────────────────────────────────────────────────

def _html_to_plain(html: str) -> str:
    """Convert raw HTML to clean plain text.

    Uses SoupStrainer to only parse <body>, reducing memory usage.
    Strips all tags, scripts, styles, and collapses whitespace.
    """
    soup = BeautifulSoup(html, "html.parser", parse_only=_BODY_STRAINER)

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # Remove Wikipedia boilerplate (infoboxes, navboxes, sidebars, references)
    for el in soup.find_all(
        class_=re.compile(
            r"infobox|sidebar|navbox|metadata|reflist|references|mw-references|"
            r"mw-editsection|catlinks|printfooter|mw-jump-link|noprint"
        )
    ):
        el.decompose()

    text = soup.get_text(separator=" ")

    # Explicitly free the soup tree
    soup.decompose()

    text = _MULTI_WS.sub(" ", text).strip()
    return text


# ── Checkpoint System ─────────────────────────────────────────────────

def _init_checkpoint_db() -> sqlite3.Connection:
    """Create the checkpoint table if needed."""
    _CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CHECKPOINT_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            zim_path        TEXT PRIMARY KEY,
            last_entry_idx  INTEGER NOT NULL,
            next_doc_id     INTEGER NOT NULL,
            updated_at      TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _get_checkpoint(conn: sqlite3.Connection, zim_path: str) -> tuple[int, int]:
    """Return (last_entry_idx, next_doc_id) or (0, 1) if no checkpoint."""
    row = conn.execute(
        "SELECT last_entry_idx, next_doc_id FROM checkpoints WHERE zim_path = ?",
        (zim_path,),
    ).fetchone()
    if row:
        return row[0], row[1]
    return 0, 1


def _save_checkpoint(
    conn: sqlite3.Connection, zim_path: str, entry_idx: int, doc_id: int
) -> None:
    """Upsert the checkpoint for this ZIM file."""
    conn.execute(
        """INSERT OR REPLACE INTO checkpoints (zim_path, last_entry_idx, next_doc_id, updated_at)
           VALUES (?, ?, ?, datetime('now'))""",
        (zim_path, entry_idx, doc_id),
    )
    conn.commit()


# ── RAM Guard ─────────────────────────────────────────────────────────

def _ram_guard() -> None:
    """Pause if system RAM is critically low (available < threshold).

    Also attempts posix_fadvise on Linux/WSL to hint the kernel to drop
    page cache, preventing the ZIM read from eating all available RAM.
    """
    mem = psutil.virtual_memory()
    available_mb = mem.available // (1024 * 1024)

    if available_mb < RAM_MIN_AVAILABLE_MB:
        log.warning(
            "RAM critically low: %d MB available (threshold %d MB) — "
            "collecting garbage and sleeping %ds",
            available_mb, RAM_MIN_AVAILABLE_MB, RAM_COOLDOWN_SECS,
        )
        gc.collect()

        # Try to drop page cache on Linux/WSL
        _try_drop_caches()

        time.sleep(RAM_COOLDOWN_SECS)
        gc.collect()

        # Check again
        available_mb = psutil.virtual_memory().available // (1024 * 1024)
        if available_mb < RAM_MIN_AVAILABLE_MB:
            log.warning("RAM still low after cooldown: %d MB available", available_mb)


def _try_drop_caches() -> None:
    """Attempt to hint the OS to free page cache (Linux/WSL only)."""
    try:
        # Writing to /proc/sys/vm/drop_caches requires root.
        # Try sync first to flush dirty pages, then drop caches if we can.
        if sys.platform.startswith("linux"):
            os.sync()
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("1")
            except PermissionError:
                pass  # Not root — that's fine, gc.collect() is our fallback
    except Exception:
        pass


# ── Velocity Reporting ────────────────────────────────────────────────

def _velocity_report(
    interval_start: float,
    interval_count: int,
    total_articles: int,
    current_idx: int,
    total_entries: int,
    stats: dict,
) -> None:
    """Log a speed and ETA report."""
    dt = time.time() - interval_start
    if dt <= 0:
        return

    speed = interval_count / dt
    remaining_entries = total_entries - current_idx
    # Estimate based on ratio of articles found vs entries scanned
    if total_articles > 0 and stats["total_scanned"] > 0:
        article_ratio = stats["total_scanned"] / max(current_idx - stats.get("resumed_from", 0), 1)
    else:
        article_ratio = 0.1  # conservative fallback

    eta_seconds = remaining_entries / max(speed / max(article_ratio, 0.01), 0.1)

    if eta_seconds < 3600:
        eta_str = f"{eta_seconds / 60:.0f} minutes"
    else:
        eta_str = f"{eta_seconds / 3600:.1f} hours"

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)

    progress_pct = current_idx / total_entries * 100

    log.info(
        "VELOCITY | %.1f articles/sec | %d/%d entries (%.1f%%) | "
        "accepted: %d | dupes: %d | skipped: %d | "
        "RAM available: %.1f GB | ETA: %s",
        speed, current_idx, total_entries, progress_pct,
        stats["accepted"], stats["duplicates"], stats["skipped"],
        available_gb, eta_str,
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _write_batch(writer, batch: list[dict]) -> None:
    """Write a batch of records to the JSONL file."""
    for record in batch:
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    writer.flush()


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from tools.config import setup_logging

    parser = argparse.ArgumentParser(description="Ingest Wikipedia ZIM into MoniWord")
    parser.add_argument("zim_file", type=Path, help="Path to .zim file")
    parser.add_argument("--output", type=Path, default=DEDUPED_JSONL,
                        help="Output JSONL path (default: deduped.jsonl)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    stats = ingest_zim(args.zim_file, args.output)
    print(f"\nDone: {stats}")
