"""Retrieval over the knowledge index."""

from functools import lru_cache

from copilot.embeddings import get_embedder
from copilot.store import DEFAULT_INDEX_DIR, VectorStore

# The relevance cutoff is measured per embedding model at ingest and stored in
# the index (see copilot/threshold.py). It is not a constant here: a fixed 0.25
# refused nothing at all with nomic-embed-text, which scores even an unrelated
# question around 0.42, so the refusal path was dead code.
FALLBACK_MIN_SCORE = 0.0


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

    @property
    def min_score(self):
        """Cutoff calibrated for whichever embedder built this index."""
        return self.store.min_score or FALLBACK_MIN_SCORE

    def retrieve(self, query, k=6, min_score=None):
        """The chunks most similar to the query, best first."""
        if not query or not query.strip():
            return []
        threshold = self.min_score if min_score is None else min_score
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, k=k, min_score=threshold)


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


def retrieve_for(retriever, factors, question=None, k=6):
    """Retrieve for the factors and, separately, for the user's question.

    A single concatenated query buries a specific question under five factor
    labels plus the words "peripheral artery disease", so asking about the model
    returned general clinical pages. Running both and merging by score lets a
    pointed question pull in its own evidence while the factor context is still
    represented.
    """
    factor_hits = retriever.retrieve(build_query(factors), k=k)
    if not question:
        return factor_hits[:k]

    question_hits = retriever.retrieve(question, k=k)

    merged = {}
    for chunk in list(question_hits) + list(factor_hits):
        key = chunk.get("id") or (
            chunk.get("source_id"), chunk.get("section"), chunk.get("text", "")[:80]
        )
        existing = merged.get(key)
        if existing is None or chunk["score"] > existing["score"]:
            merged[key] = chunk

    return sorted(merged.values(), key=lambda c: c["score"], reverse=True)[:k]


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
