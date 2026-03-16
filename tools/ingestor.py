"""MoniWord Pipeline — Streaming Format Converter.

Reads CSV, JSON, JSONL, and TXT files from an input directory and emits a
single unified JSONL file where each line is:
    {"id": <int>, "source": "<filename>", "text": "<content>"}

All reading is streaming/chunked — never loads an entire file into memory.
"""

import csv
import json
import logging
from itertools import count
from pathlib import Path
from typing import IO

import pandas as pd

from tools.config import CHUNK_LINES, TXT_CHUNK_WORDS

log = logging.getLogger(__name__)

# Supported extensions (lowercase)
_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt"}


# ── Public API ─────────────────────────────────────────────────────────

def ingest_directory(input_dir: Path, output_path: Path) -> int:
    """Walk *input_dir*, convert every supported file, write to *output_path*.

    Returns the total number of documents written.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counter = count(start=1)
    files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in _EXTENSIONS
    )

    if not files:
        log.warning("No supported files found in %s", input_dir)
        return 0

    log.info("Found %d files to ingest from %s", len(files), input_dir)

    with open(output_path, "w", encoding="utf-8") as writer:
        for fpath in files:
            ext = fpath.suffix.lower()
            log.info("Ingesting %s", fpath.name)
            try:
                if ext == ".csv":
                    _ingest_csv(fpath, writer, counter)
                elif ext == ".json":
                    _ingest_json(fpath, writer, counter)
                elif ext == ".jsonl":
                    _ingest_jsonl(fpath, writer, counter)
                elif ext == ".txt":
                    _ingest_txt(fpath, writer, counter)
            except Exception:
                log.exception("Failed to ingest %s — skipping", fpath)

    total = next(counter) - 1  # counter is 1-ahead
    log.info("Ingestion complete: %d documents written to %s", total, output_path)
    return total


# ── Private helpers ────────────────────────────────────────────────────

def _write_record(writer: IO, doc_id: int, source: str, text: str) -> None:
    """Validate and write one JSONL line. Skips empty text."""
    text = text.strip()
    if not text:
        return
    record = {"id": doc_id, "source": source, "text": text}
    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ingest_csv(file_path: Path, writer: IO, counter: count) -> None:
    """Stream CSV in chunks via pandas, concatenate string columns into text."""
    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_LINES,
        dtype=str,
        on_bad_lines="skip",
        encoding_errors="replace",
    ):
        for _, row in chunk.iterrows():
            text_parts = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
            text = " ".join(text_parts)
            _write_record(writer, next(counter), file_path.name, text)


def _ingest_json(file_path: Path, writer: IO, counter: count) -> None:
    """Handle JSON files — either a JSON array or one object per line."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        first_char = f.read(1).strip()
        f.seek(0)

        if first_char == "[":
            # JSON array — try streaming with ijson, fall back to json.load chunks
            try:
                import ijson
                for obj in ijson.items(f, "item"):
                    text = _extract_text_from_obj(obj)
                    _write_record(writer, next(counter), file_path.name, text)
            except ImportError:
                log.warning("ijson not installed, loading full JSON array for %s", file_path.name)
                f.seek(0)
                for obj in json.load(f):
                    text = _extract_text_from_obj(obj)
                    _write_record(writer, next(counter), file_path.name, text)
        else:
            # Assume one JSON object per line (JSONL-like but with .json extension)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = _extract_text_from_obj(obj)
                    _write_record(writer, next(counter), file_path.name, text)
                except json.JSONDecodeError:
                    continue


def _ingest_jsonl(file_path: Path, writer: IO, counter: count) -> None:
    """Stream JSONL line-by-line."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = _extract_text_from_obj(obj)
                _write_record(writer, next(counter), file_path.name, text)
            except json.JSONDecodeError:
                continue


def _ingest_txt(file_path: Path, writer: IO, counter: count) -> None:
    """Read TXT line-by-line.

    First pass: detect if the file has blank-line paragraph breaks.
      - If yes  -> group by paragraphs (original behaviour).
      - If no   -> chunk every TXT_CHUNK_WORDS words into one document.
        This handles word-lists and wall-of-text files gracefully.
    """
    has_blanks = False
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 5000:  # sample first 5k lines
                break
            if line.strip() == "":
                has_blanks = True
                break

    if has_blanks:
        _ingest_txt_paragraphs(file_path, writer, counter)
    else:
        log.info("No blank lines detected in %s — chunking every %d words",
                 file_path.name, TXT_CHUNK_WORDS)
        _ingest_txt_chunked(file_path, writer, counter)


def _ingest_txt_paragraphs(file_path: Path, writer: IO, counter: count) -> None:
    """Group consecutive non-empty lines into paragraphs."""
    paragraph_lines: list[str] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                paragraph_lines.append(stripped)
            else:
                if paragraph_lines:
                    text = " ".join(paragraph_lines)
                    _write_record(writer, next(counter), file_path.name, text)
                    paragraph_lines = []

    # Flush remaining paragraph
    if paragraph_lines:
        text = " ".join(paragraph_lines)
        _write_record(writer, next(counter), file_path.name, text)


def _ingest_txt_chunked(file_path: Path, writer: IO, counter: count) -> None:
    """Chunk a no-blank-line TXT file every TXT_CHUNK_WORDS words."""
    word_buf: list[str] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            word_buf.append(word)
            if len(word_buf) >= TXT_CHUNK_WORDS:
                text = " ".join(word_buf)
                _write_record(writer, next(counter), file_path.name, text)
                word_buf = []

    # Flush remaining
    if word_buf:
        text = " ".join(word_buf)
        _write_record(writer, next(counter), file_path.name, text)


def _extract_text_from_obj(obj: object) -> str:
    """Extract text from a JSON object.

    Checks common field names first, then falls back to the longest string value.
    """
    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        # Check common text field names
        for key in ("text", "content", "body", "sentence", "document", "passage"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]

        # Fallback: longest string value
        best = ""
        for v in obj.values():
            if isinstance(v, str) and len(v) > len(best):
                best = v
        return best

    return str(obj)
