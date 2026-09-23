"""Build the knowledge index: fetch, chunk, embed, save.

    python -m copilot.ingest
    python -m copilot.ingest --refresh          # re-fetch instead of using the cache
    python -m copilot.ingest --provider hashing # offline, test-quality embeddings
"""

import argparse
import sys

from copilot.chunking import chunk_document
from copilot.embeddings import get_embedder
from copilot.fetch import fetch_source, load_sources, local_documents
from copilot.store import DEFAULT_INDEX_DIR, VectorStore


def collect_documents(refresh=False, verbose=True):
    """Project-owned markdown plus the fetched public-domain references."""
    documents = local_documents()
    if verbose:
        for document in documents:
            print(f"  local  {document['id']:24s} {len(document['text']):>7,} chars")

    for source in load_sources():
        try:
            text = fetch_source(source, refresh=refresh)
        except Exception as error:  # network failure should not lose the local docs
            print(f"  SKIP   {source['id']:24s} {type(error).__name__}: {error}",
                  file=sys.stderr)
            continue
        documents.append({**source, "text": text})
        if verbose:
            print(f"  fetch  {source['id']:24s} {len(text):>7,} chars")

    return documents


def build_index(provider="ollama", refresh=False, index_dir=DEFAULT_INDEX_DIR):
    """Fetch and embed everything, then persist the index."""
    print("Collecting documents...")
    documents = collect_documents(refresh=refresh)
    if not documents:
        raise RuntimeError("No documents collected - nothing to index.")

    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    print(f"\n{len(chunks)} chunks from {len(documents)} documents")

    embedder = get_embedder(provider)
    if not embedder.is_available():
        raise RuntimeError(
            f"Embedding provider {provider!r} is not reachable.\n"
            "Start Ollama and pull the embedding model:\n"
            "    ollama pull nomic-embed-text\n"
            "Or index with offline test-quality embeddings:\n"
            "    python -m copilot.ingest --provider hashing"
        )

    print(f"Embedding with {embedder.name}...")
    vectors = embedder.embed([chunk["text"] for chunk in chunks])

    store = VectorStore(vectors, chunks, embedder_name=embedder.name)
    path = store.save(index_dir)
    print(f"Saved index ({len(store)} chunks, dim {vectors.shape[1]}) to {path}")
    return store


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the PAD copilot knowledge index.")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "hashing"],
                        help="embedding provider (default: ollama)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch sources instead of using the cache")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args(argv)

    try:
        build_index(args.provider, args.refresh, args.index_dir)
    except RuntimeError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
