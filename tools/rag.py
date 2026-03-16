"""MoniWord — RAG (Retrieval Augmented Generation).

Searches the vector database for relevant articles, then sends them
as context to a local LLM via Ollama to answer the user's question.
"""

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

from tools.config import SEMANTIC_DB_PATH, EMBEDDING_MODEL
from tools.search import load_model, search

log = logging.getLogger(__name__)

DEFAULT_MODEL = "phi3:mini"
OLLAMA_URL = "http://localhost:11434"
TOP_K = 5
MAX_CONTEXT_CHARS = 4000  # keep context short enough for small models


def list_ollama_models() -> list[str]:
    """Fetch available models from Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to Ollama and return the response."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 4096,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data.get("response", "")


def build_prompt(question: str, articles: list[dict]) -> str:
    """Build a RAG prompt with retrieved articles as context."""
    context_parts = []
    chars_used = 0

    for i, article in enumerate(articles, 1):
        text = article["text"]
        # Trim individual articles if needed
        remaining = MAX_CONTEXT_CHARS - chars_used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "..."
        context_parts.append(f"[Article {i}] (Source: {article['source']})\n{text}")
        chars_used += len(text)

    context = "\n\n".join(context_parts)

    return f"""Use the following articles to answer the question. Base your answer on the provided articles. If the articles don't contain enough information, say so.

{context}

Question: {question}

Answer:"""


def rag_query(question: str, model: str = DEFAULT_MODEL, db_path: Path = SEMANTIC_DB_PATH, top_k: int = TOP_K) -> dict:
    """Run a full RAG query: search + generate.

    Returns dict with 'answer', 'sources', and 'model'.
    """
    embed_model = load_model()

    # Search for relevant articles
    results = search(question, embed_model, db_path, top_k=top_k)

    if not results:
        return {
            "answer": "No articles found in the database. Run the vectorizer first.",
            "sources": [],
            "model": model,
        }

    # Build prompt and ask LLM
    prompt = build_prompt(question, results)
    answer = ask_ollama(prompt, model=model)

    sources = [
        {
            "doc_id": r["doc_id"],
            "source": r["source"],
            "score": r["score"],
            "preview": r["text"][:150],
        }
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "model": model,
    }


def interactive_rag(model: str = DEFAULT_MODEL, db_path: Path = SEMANTIC_DB_PATH) -> None:
    """Interactive RAG loop."""
    print()
    print(f"  MoniWord RAG (using {model})")
    print("  Ask a question and get answers backed by your database.")
    print("  Type 'quit' to exit.")
    print()

    while True:
        try:
            question = input("  Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        print()
        print("  Searching database...")
        try:
            result = rag_query(question, model=model, db_path=db_path)
        except urllib.error.URLError:
            print("  [!] Could not connect to Ollama. Is it running?")
            print("  [!] Start it with: ollama serve")
            print()
            continue
        except Exception as e:
            print(f"  [!] Error: {e}")
            print()
            continue

        print()
        print(f"  {'=' * 60}")
        print(f"  Answer ({result['model']}):")
        print(f"  {'=' * 60}")
        print()

        # Word wrap the answer
        for line in result["answer"].split("\n"):
            print(f"  {line}")

        print()
        print(f"  {'-' * 60}")
        print(f"  Sources:")
        for i, src in enumerate(result["sources"], 1):
            score_pct = src["score"] * 100
            print(f"    {i}. [{score_pct:.0f}% match] {src['source']}")
            print(f"       {src['preview']}...")
        print()
