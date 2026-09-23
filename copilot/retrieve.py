"""Retrieval over the knowledge index."""

from functools import lru_cache

from copilot.embeddings import get_embedder
from copilot.store import DEFAULT_INDEX_DIR, VectorStore

# Below this cosine similarity a chunk is treated as unrelated to the question.
# The copilot refuses to answer rather than citing something irrelevant.
MIN_SCORE = 0.25


@lru_cache(maxsize=4)
def _load_store(index_dir):
    return VectorStore.load(index_dir)


class Retriever:
    """Embeds a query with the same provider the index was built with."""

    def __init__(self, index_dir=DEFAULT_INDEX_DIR, provider=None):
        self.store = _load_store(str(index_dir))
        # Mixing providers silently returns nonsense, so the index records which
        # one built it and that is what gets used unless overridden.
        self.provider = provider or self.store.embedder_name or "ollama"
        self.embedder = get_embedder(self.provider)

    def retrieve(self, query, k=6, min_score=MIN_SCORE):
        """The chunks most similar to the query, best first."""
        if not query or not query.strip():
            return []
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, k=k, min_score=min_score)


def build_query(factors, question=None):
    """Turn the model's top factors (and any user question) into a search query.

    Retrieval works on the clinical concepts behind the features, so the query
    is phrased in those terms rather than in column names.
    """
    parts = ["peripheral artery disease"]
    parts.extend(factor["label"] for factor in factors)
    if question:
        parts.append(question)
    return " ".join(parts)


def format_chunks(chunks):
    """Number the chunks for the prompt, so the model can cite them as [n]."""
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{index}] {chunk['source_title']} — {chunk['section']}\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def citation_list(chunks):
    """Reference list matching the [n] markers in an answer."""
    return [
        {
            "n": index,
            "title": chunk["source_title"],
            "section": chunk["section"],
            "url": chunk.get("url", ""),
            "publisher": chunk.get("publisher", ""),
            "score": round(chunk.get("score", 0.0), 3),
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
