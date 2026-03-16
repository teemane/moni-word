# MoniWord

A text ingestion pipeline that converts raw documents and Wikipedia dumps into a searchable vector database for local RAG (Retrieval Augmented Generation).

Ingest CSVs, JSON, plain text, or Wikipedia ZIM files. Deduplicate with MinHash LSH. Embed with sentence-transformers. Search by meaning.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The interactive menu walks you through each stage:

1. **Ingest** CSV, JSON, TXT files into unified JSONL
2. **Deduplicate** with MinHash LSH (85% Jaccard threshold)
3. **Wikipedia ZIM** streaming with checkpoint resume
4. **Vectorize** with sentence-transformer embeddings + topic clustering
5. **Full Pipeline** runs 1 → 2 → 4 back-to-back
6. **Search** your database by meaning
7. **Stats** to see what's in the pipeline
8. **Reset** to wipe all outputs and start fresh

## How it works

```
Your data (CSV, JSON, TXT, Wikipedia ZIM)
     |
     v
[Ingest] --> unified.jsonl        (normalize all formats)
     |
     v
[Dedup]  --> deduped.jsonl        (remove near-duplicates via MinHash LSH)
     |
     v
[Embed]  --> semantic_map.db      (384-dim vectors + 50 topic clusters)
     |
     v
[Search] --> query by meaning     (cosine similarity over embeddings)
```

## Project Structure

```
app.py              # Single entry point, interactive terminal app
tools/
  config.py         # All tunables in one place
  ingestor.py       # CSV/JSON/TXT -> JSONL
  deduplicator.py   # MinHash LSH fuzzy dedup
  wiki_ingestor.py  # Wikipedia ZIM streaming with checkpoints
  vectorizer.py     # Sentence-transformer embeddings + clustering
  search.py         # Semantic search over the built database
```

## Output

| File | Description |
|------|-------------|
| `output/unified.jsonl` | Ingested documents (one JSON object per line) |
| `output/deduped.jsonl` | After deduplication |
| `output/vectors.jsonl` | Documents with base64-encoded vectors |
| `state/semantic_map.db` | Vector database with topic clusters (the main output) |
| `state/processed_hashes.db` | Dedup signatures |
| `state/wiki_checkpoint.db` | Wikipedia resume point |

## Configuration

Edit `tools/config.py`:

```python
SHINGLE_SIZE = 3            # Dedup shingle size
LSH_THRESHOLD = 0.85        # Similarity cutoff for duplicates
BATCH_SIZE = 100             # Vectorizer batch size
WIKI_BATCH_SIZE = 25         # Wikipedia batch size
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_CLUSTERS = 50              # Topic clusters
```

## Custom ZIM Files

You're not limited to official Wikipedia dumps. You can create your own `.zim` files from **any website** using [Zimit](https://github.com/openzim/zimit):

```bash
# Install Zimit (requires Docker)
docker pull ghcr.io/openzim/zimit

# Crawl a website into a ZIM file
docker run -v /tmp/zim:/output ghcr.io/openzim/zimit zimit --url "https://docs.example.com" --output /output/example-docs.zim

# Then ingest it with MoniWord
python app.py  # choose option [3] and point to your .zim file
```

Some ideas for custom ZIMs:
- Documentation sites (API docs, framework guides, man pages)
- Wikis (Fandom, MediaWiki-based sites, internal wikis)
- Blogs, research paper archives, course material
- Any site you want to search offline by meaning

You can also grab pre-built ZIM files from the [Kiwix library](https://library.kiwix.org/) including Wikipedia, Wiktionary, Stack Exchange, Project Gutenberg, TED Talks, and more.

## Dependencies

```
datasketch, sentence-transformers, scikit-learn,
beautifulsoup4, psutil, pandas, tqdm, libzim (Linux/WSL)
```

## License

MIT
