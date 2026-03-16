"""
===========================================================
  MoniWord  --  Interactive Text Ingestion Pipeline
===========================================================
  A user-friendly terminal app that walks you through:

    1. Ingest flat files  (CSV / JSON / TXT  -->  unified.jsonl)
    2. Fuzzy Deduplicate   (MinHash LSH  -->  deduped.jsonl)
    3. Wikipedia ZIM        (stream + inline dedup  -->  deduped.jsonl)
    4. Vectorize            (sentence-transformers  -->  vectors.jsonl + semantic_map.db)
    5. Run Full Pipeline    (1 -> 2 -> 4, step by step)
    6. Search               (query by meaning)
    7. View Pipeline Stats
    8. Reset Pipeline       (wipe all outputs)

  Run:  python app.py
===========================================================
"""

import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from tools.config import (
    BASE_DIR,
    DEDUPED_JSONL,
    HASH_DB_PATH,
    LOG_DIR,
    OUTPUT_DIR,
    SEMANTIC_DB_PATH,
    STATE_DIR,
    UNIFIED_JSONL,
    VECTORS_JSONL,
    ensure_directories,
    setup_logging,
)

# ── Constants ──────────────────────────────────────────────────────────

BANNER = r"""
 _____ ______   ________  ________   ___  ___       __   ________  ________  ________
|\   _ \  _   \|\   __  \|\   ___  \|\  \|\  \     |\  \|\   __  \|\   __  \|\   ___ \
\ \  \\\__\ \  \ \  \|\  \ \  \\ \  \ \  \ \  \    \ \  \ \  \|\  \ \  \|\  \ \  \_|\ \
 \ \  \\|__| \  \ \  \\\  \ \  \\ \  \ \  \ \  \  __\ \  \ \  \\\  \ \   _  _\ \  \ \\ \
  \ \  \    \ \  \ \  \\\  \ \  \\ \  \ \  \ \  \|\__\_\  \ \  \\\  \ \  \\  \\ \  \_\\ \
   \ \__\    \ \__\ \_______\ \__\\ \__\ \__\ \____________\ \_______\ \__\\ _\\ \_______\
    \|__|     \|__|\|_______|\|__| \|__|\|__|\|____________|\|_______|\|__|\|__|\|_______|

         Text Ingestion, Deduplication & Semantic Search
"""

MENU = """
  +---------------------------------------------------------+
  |                    MAIN MENU                             |
  +---------------------------------------------------------+
  |  [1]  Ingest Flat Files   (CSV, JSON, TXT -> JSONL)     |
  |  [2]  Fuzzy Deduplicate   (MinHash LSH, 85% threshold)  |
  |  [3]  Wikipedia ZIM       (Stream + Inline Dedup)        |
  |  [4]  Vectorize           (Embeddings + Semantic Map)    |
  |  [5]  Run Full Pipeline   (1 -> 2 -> 4, step by step)   |
  |  [6]  Search              (Query by Meaning)             |
  |  [7]  View Pipeline Stats                                |
  |  [8]  Reset Pipeline      (wipe all outputs)             |
  |  [0]  Exit                                               |
  +---------------------------------------------------------+
"""

log = logging.getLogger("moniword.app")


# ── Helpers ────────────────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause():
    print()
    input("  Press ENTER to return to menu...")


def elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    if dt < 3600:
        return f"{dt / 60:.1f}m"
    return f"{dt / 3600:.1f}h"


def ask_path(prompt: str, must_exist: bool = True) -> Path | None:
    """Prompt user for a file/directory path. Returns None on empty input."""
    while True:
        raw = input(f"  {prompt}: ").strip().strip('"').strip("'")
        if not raw:
            return None
        p = Path(raw)
        if must_exist and not p.exists():
            print(f"  [!] Path not found: {p}")
            print(f"  [!] Please check the path and try again. (leave blank to cancel)")
            continue
        return p


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return count


def file_size_str(path: Path) -> str:
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def print_header(title: str):
    width = 57
    print()
    print(f"  {'=' * width}")
    print(f"  {title:^{width}}")
    print(f"  {'=' * width}")
    print()


# ── Stage Runners ──────────────────────────────────────────────────────

def run_ingest():
    """Stage 1: Ingest flat files into unified JSONL."""
    print_header("STAGE 1: INGEST FLAT FILES")

    if UNIFIED_JSONL.exists():
        n = count_lines(UNIFIED_JSONL)
        print(f"  [i] unified.jsonl already exists ({n:,} docs, {file_size_str(UNIFIED_JSONL)})")
        if not ask_yes_no("Overwrite and re-ingest?", default=False):
            return

    input_dir = ask_path("Enter path to your data folder (CSV/JSON/TXT files)")
    if not input_dir:
        print("  [!] Cancelled.")
        return

    if not input_dir.is_dir():
        print(f"  [!] Not a directory: {input_dir}")
        return

    print()
    print(f"  [*] Scanning: {input_dir}")
    print(f"  [*] Output:   {UNIFIED_JSONL}")
    print()

    t = time.time()
    from tools.ingestor import ingest_directory
    doc_count = ingest_directory(input_dir, UNIFIED_JSONL)

    print()
    print(f"  [OK] Ingested {doc_count:,} documents in {elapsed(t)}")
    print(f"  [OK] Saved to: {UNIFIED_JSONL}")
    print(f"  [OK] File size: {file_size_str(UNIFIED_JSONL)}")


def run_dedup():
    """Stage 2: Fuzzy deduplication on unified.jsonl."""
    print_header("STAGE 2: FUZZY DEDUPLICATION")

    if not UNIFIED_JSONL.exists():
        print("  [!] unified.jsonl not found. Run Stage 1 (Ingest) first.")
        return

    n = count_lines(UNIFIED_JSONL)
    print(f"  [i] Input: unified.jsonl ({n:,} docs, {file_size_str(UNIFIED_JSONL)})")

    if DEDUPED_JSONL.exists():
        dn = count_lines(DEDUPED_JSONL)
        print(f"  [i] deduped.jsonl already exists ({dn:,} docs)")
        if not ask_yes_no("Overwrite and re-deduplicate?", default=False):
            return

    print()
    print("  [*] Running MinHash LSH deduplication...")
    print("  [*] Threshold: 85% Jaccard similarity")
    print("  [*] Shingle size: 3 words")
    print()

    t = time.time()
    from tools.deduplicator import deduplicate
    total, dupes = deduplicate(UNIFIED_JSONL, DEDUPED_JSONL, HASH_DB_PATH)

    print()
    print(f"  [OK] Processed {total:,} documents in {elapsed(t)}")
    print(f"  [OK] Duplicates removed: {dupes:,}")
    print(f"  [OK] Unique documents:   {total - dupes:,}")
    print(f"  [OK] Saved to: {DEDUPED_JSONL}")


def run_wiki():
    """Stage 3: Wikipedia ZIM ingestion with inline dedup."""
    print_header("STAGE 3: WIKIPEDIA ZIM INGESTION")

    # Check for existing checkpoint
    ckpt_db = STATE_DIR / "wiki_checkpoint.db"
    has_checkpoint = False
    if ckpt_db.exists():
        try:
            conn = sqlite3.connect(str(ckpt_db))
            row = conn.execute(
                "SELECT zim_path, last_entry_idx, updated_at FROM checkpoints LIMIT 1"
            ).fetchone()
            conn.close()
            if row and row[1] > 0:
                has_checkpoint = True
                print(f"  [i] CHECKPOINT FOUND from previous run:")
                print(f"      ZIM: {row[0]}")
                print(f"      Progress: entry {row[1]:,}")
                print(f"      Last saved: {row[2]}")
                print()
                if ask_yes_no("Resume from this checkpoint?"):
                    zim_path = Path(row[0])
                    if not zim_path.exists():
                        # Try WSL path conversion
                        print(f"  [!] Path not found: {zim_path}")
                        zim_path = ask_path("Enter the correct path to the .zim file")
                        if not zim_path:
                            print("  [!] Cancelled.")
                            return
                else:
                    has_checkpoint = False
        except Exception:
            pass

    if not has_checkpoint:
        zim_path = ask_path("Enter path to your .zim file")
        if not zim_path:
            print("  [!] Cancelled.")
            return

        if not str(zim_path).lower().endswith(".zim"):
            print("  [!] That doesn't look like a .zim file.")
            if not ask_yes_no("Continue anyway?", default=False):
                return

    if DEDUPED_JSONL.exists():
        n = count_lines(DEDUPED_JSONL)
        print(f"  [i] deduped.jsonl has {n:,} existing docs (will append)")

    print()
    print(f"  [*] ZIM file:    {zim_path}")
    print(f"  [*] Output:      {DEDUPED_JSONL} (append)")
    print(f"  [*] Batch size:  25 articles (OOM-safe)")
    print(f"  [*] RAM guard:   pause if available < 1 GB")
    print(f"  [*] Checkpoint:  saved after EVERY batch of 25")
    print(f"  [*] Dedup:       intro-skip mode (handles similar Wikipedia intros)")
    print()
    print("  You can Ctrl+C at any time — it will resume from the last checkpoint.")
    print()

    if not ask_yes_no("Start Wikipedia ingestion?"):
        return

    print()
    t = time.time()
    from tools.wiki_ingestor import ingest_zim
    stats = ingest_zim(zim_path, output_path=DEDUPED_JSONL, db_path=HASH_DB_PATH)

    print()
    print(f"  [OK] Completed in {elapsed(t)}")
    print(f"  [OK] Entries scanned:   {stats['total_scanned']:,}")
    print(f"  [OK] Articles accepted: {stats['accepted']:,}")
    print(f"  [OK] Duplicates caught: {stats['duplicates']:,}")
    print(f"  [OK] Skipped:           {stats['skipped']:,}")
    if stats["resumed_from"] > 0:
        print(f"  [OK] Resumed from entry: {stats['resumed_from']:,}")


def run_vectorize():
    """Stage 4: Vectorization and semantic map building."""
    print_header("STAGE 4: VECTORIZE + SEMANTIC MAP")

    if not DEDUPED_JSONL.exists():
        print("  [!] deduped.jsonl not found.")
        print("  [!] Run Stage 2 (Dedup) or Stage 3 (Wikipedia) first.")
        return

    n = count_lines(DEDUPED_JSONL)
    print(f"  [i] Input: deduped.jsonl ({n:,} docs, {file_size_str(DEDUPED_JSONL)})")

    if VECTORS_JSONL.exists():
        vn = count_lines(VECTORS_JSONL)
        print(f"  [i] vectors.jsonl already exists ({vn:,} docs)")
        if not ask_yes_no("Overwrite and re-vectorize?", default=False):
            return

    print()
    print("  [*] Model: all-MiniLM-L6-v2 (384-dim, ~80MB)")
    print("  [*] Batch size: 100")
    print(f"  [*] Documents to embed: {n:,}")
    print()

    if n > 100_000:
        print(f"  [!] Large dataset ({n:,} docs). This may take a while on CPU.")
        if not ask_yes_no("Continue?"):
            return
        print()

    t = time.time()
    from tools.vectorizer import vectorize
    vec_count = vectorize(DEDUPED_JSONL, VECTORS_JSONL, SEMANTIC_DB_PATH)

    print()
    print(f"  [OK] Vectorized {vec_count:,} documents in {elapsed(t)}")
    print(f"  [OK] Vectors:      {VECTORS_JSONL}")
    print(f"  [OK] Semantic map: {SEMANTIC_DB_PATH}")


def run_full_pipeline():
    """Run stages 1 -> 2 -> 4 in sequence, prompting at each step."""
    print_header("FULL PIPELINE (Ingest -> Dedup -> Vectorize)")

    print("  This will run three stages back-to-back:")
    print("    1. Ingest flat files -> unified.jsonl")
    print("    2. Fuzzy deduplicate -> deduped.jsonl")
    print("    3. Vectorize         -> vectors.jsonl + semantic_map.db")
    print()
    print("  (For Wikipedia ZIM, use option [3] from the main menu separately)")
    print()

    input_dir = ask_path("Enter path to your data folder")
    if not input_dir:
        print("  [!] Cancelled.")
        return
    if not input_dir.is_dir():
        print(f"  [!] Not a directory: {input_dir}")
        return

    if not ask_yes_no("Start full pipeline?"):
        return

    pipeline_start = time.time()

    # ── Stage 1 ───────────────────────────────────────────────────────
    print()
    print("  " + "-" * 50)
    print("  STEP 1/3: Ingesting files...")
    print("  " + "-" * 50)
    t = time.time()
    from tools.ingestor import ingest_directory
    doc_count = ingest_directory(input_dir, UNIFIED_JSONL)
    print(f"  [OK] {doc_count:,} documents ingested ({elapsed(t)})")

    if doc_count == 0:
        print("  [!] No documents found. Pipeline stopped.")
        return

    # ── Stage 2 ───────────────────────────────────────────────────────
    print()
    print("  " + "-" * 50)
    print("  STEP 2/3: Fuzzy deduplication...")
    print("  " + "-" * 50)
    t = time.time()
    from tools.deduplicator import deduplicate
    total, dupes = deduplicate(UNIFIED_JSONL, DEDUPED_JSONL, HASH_DB_PATH)
    print(f"  [OK] {total:,} docs -> {total - dupes:,} unique ({dupes:,} dupes removed) ({elapsed(t)})")

    # ── Stage 3 ───────────────────────────────────────────────────────
    print()
    print("  " + "-" * 50)
    print("  STEP 3/3: Vectorizing...")
    print("  " + "-" * 50)
    t = time.time()
    from tools.vectorizer import vectorize
    vec_count = vectorize(DEDUPED_JSONL, VECTORS_JSONL, SEMANTIC_DB_PATH)
    print(f"  [OK] {vec_count:,} documents vectorized ({elapsed(t)})")

    # ── Done ──────────────────────────────────────────────────────────
    print()
    print(f"  {'=' * 50}")
    print(f"  PIPELINE COMPLETE  --  Total time: {elapsed(pipeline_start)}")
    print(f"  {'=' * 50}")
    print()
    print(f"  Outputs:")
    print(f"    unified.jsonl  : {file_size_str(UNIFIED_JSONL)}")
    print(f"    deduped.jsonl  : {file_size_str(DEDUPED_JSONL)}")
    print(f"    vectors.jsonl  : {file_size_str(VECTORS_JSONL)}")
    print(f"    semantic_map.db: {file_size_str(SEMANTIC_DB_PATH)}")


def run_search():
    """Search the vector database by meaning."""
    print_header("SEMANTIC SEARCH")

    if not SEMANTIC_DB_PATH.exists():
        print("  [!] semantic_map.db not found.")
        print("  [!] Run Stage 4 (Vectorize) first.")
        return

    from tools.search import load_model, search, print_results, interactive_mode

    model = load_model()
    interactive_mode(model, SEMANTIC_DB_PATH)


def show_stats():
    """Display current pipeline status and file stats."""
    print_header("PIPELINE STATUS")

    files = [
        ("unified.jsonl   (raw ingested)",     UNIFIED_JSONL),
        ("deduped.jsonl   (after dedup)",       DEDUPED_JSONL),
        ("vectors.jsonl   (final embeddings)",  VECTORS_JSONL),
    ]
    dbs = [
        ("processed_hashes.db (dedup signatures)", HASH_DB_PATH),
        ("semantic_map.db     (vector store)",      SEMANTIC_DB_PATH),
    ]

    print("  Output Files:")
    print("  " + "-" * 55)
    for label, path in files:
        if path.exists():
            lines = count_lines(path)
            size = file_size_str(path)
            print(f"  [OK] {label}")
            print(f"        {lines:>10,} docs  |  {size:>10}")
        else:
            print(f"  [ ]  {label}")
            print(f"        not created yet")

    print()
    print("  Databases:")
    print("  " + "-" * 55)
    for label, path in dbs:
        if path.exists():
            size = file_size_str(path)
            # Try to read row count
            try:
                conn = sqlite3.connect(str(path))
                if "hashes" in str(path):
                    row = conn.execute("SELECT COUNT(*) FROM signatures").fetchone()
                    extra = f"{row[0]:,} signatures"
                else:
                    row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                    extra = f"{row[0]:,} embeddings"
                    # Check for clusters
                    try:
                        crow = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()
                        extra += f", {crow[0]} clusters"
                    except sqlite3.OperationalError:
                        pass
                conn.close()
            except Exception:
                extra = ""
            print(f"  [OK] {label}")
            print(f"        {size:>10}  |  {extra}")
        else:
            print(f"  [ ]  {label}")
            print(f"        not created yet")

    # Check for wiki checkpoint
    ckpt_db = STATE_DIR / "wiki_checkpoint.db"
    if ckpt_db.exists():
        print()
        print("  Wikipedia Checkpoint:")
        print("  " + "-" * 55)
        try:
            conn = sqlite3.connect(str(ckpt_db))
            rows = conn.execute(
                "SELECT zim_path, last_entry_idx, next_doc_id, updated_at FROM checkpoints"
            ).fetchall()
            conn.close()
            for row in rows:
                zim_name = Path(row[0]).name if row[0] else "?"
                print(f"  [OK] {zim_name}")
                print(f"        Entry: {row[1]:,} | Next doc ID: {row[2]:,} | Saved: {row[3]}")
        except Exception:
            print(f"  [?]  Could not read checkpoint DB")

    print()
    print(f"  Log file: {LOG_DIR / 'pipeline.log'}")
    if (LOG_DIR / "pipeline.log").exists():
        print(f"        Size: {file_size_str(LOG_DIR / 'pipeline.log')}")


def run_reset():
    """Wipe all pipeline outputs and state."""
    print_header("RESET PIPELINE")

    print("  This will DELETE all generated files:")
    print(f"    - {OUTPUT_DIR}/ (unified.jsonl, deduped.jsonl, vectors.jsonl)")
    print(f"    - {STATE_DIR}/  (processed_hashes.db, semantic_map.db)")
    print(f"    - {LOG_DIR}/    (pipeline.log)")
    print()
    print("  Your source data will NOT be touched.")
    print()

    if not ask_yes_no("Are you sure you want to reset?", default=False):
        print("  [!] Cancelled.")
        return

    # Double-check for large datasets
    total_size = 0
    for d in (OUTPUT_DIR, STATE_DIR, LOG_DIR):
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

    if total_size > 100 * 1024 * 1024:  # > 100MB
        size_str = f"{total_size / (1024**3):.2f} GB" if total_size > 1024**3 else f"{total_size / (1024**2):.1f} MB"
        print(f"\n  [!] Total data to delete: {size_str}")
        if not ask_yes_no("Really delete?", default=False):
            print("  [!] Cancelled.")
            return

    for d in (OUTPUT_DIR, STATE_DIR, LOG_DIR):
        if d.exists():
            shutil.rmtree(d)
            print(f"  [x] Deleted {d}")

    ensure_directories()
    print()
    print("  [OK] Pipeline reset. Ready for a fresh run.")


# ── Main Loop ──────────────────────────────────────────────────────────

def main():
    setup_logging()
    ensure_directories()

    clear_screen()
    print(BANNER)
    input("  Press ENTER to start...")

    while True:
        clear_screen()
        print(BANNER)
        print(MENU)

        choice = input("  Enter your choice [0-8]: ").strip()

        try:
            if choice == "1":
                run_ingest()
            elif choice == "2":
                run_dedup()
            elif choice == "3":
                run_wiki()
            elif choice == "4":
                run_vectorize()
            elif choice == "5":
                run_full_pipeline()
            elif choice == "6":
                run_search()
            elif choice == "7":
                show_stats()
            elif choice == "8":
                run_reset()
            elif choice == "0":
                print()
                print("  Goodbye!")
                print()
                sys.exit(0)
            else:
                print(f"  [!] Invalid choice: '{choice}'")
        except KeyboardInterrupt:
            print("\n\n  [!] Interrupted by user.")
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            log.exception("Unhandled error in menu action %s", choice)

        pause()


if __name__ == "__main__":
    main()
