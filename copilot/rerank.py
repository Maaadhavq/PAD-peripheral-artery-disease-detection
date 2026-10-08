"""Reranking retrieved passages with the local model.

Embedding similarity is a single dot product and cannot tell that a passage is
about the right topic but answers a different question. Reranking retrieves a
wider net, then scores each candidate against the query directly.

It costs one generation per candidate, so it is off by default and should only
be switched on if the eval shows it moving hit@1. `python -m copilot.eval.run_eval
--rerank` measures exactly that.
"""

RERANK_SYSTEM = """You rate how well a passage answers a question.

Reply with a single integer from 0 to 10 and nothing else.
10 means the passage directly answers the question.
5 means it is related but does not answer it.
0 means it is irrelevant."""

CANDIDATE_MULTIPLIER = 3
MAX_CANDIDATES = 24


def _parse_score(reply):
    """Pull an integer 0-10 out of the model's reply, defaulting to neutral."""
    digits = ""
    for char in reply.strip():
        if char.isdigit():
            digits += char
        elif digits:
            break
    if not digits:
        return 5.0
    return max(0.0, min(10.0, float(digits)))


def rerank(query, chunks, llm, k=6):
    """Reorder candidates by how well the model thinks each answers the query.

    The embedding score is kept as a tiebreaker so a model that rates everything
    the same degrades to the original ordering rather than to arbitrary order.
    """
    if not chunks:
        return []

    scored = []
    for chunk in chunks[:MAX_CANDIDATES]:
        reply = llm.chat(
            RERANK_SYSTEM,
            f"Question:\n{query}\n\nPassage:\n{chunk['text']}",
        )
        scored.append({**chunk, "rerank_score": _parse_score(reply)})

    scored.sort(key=lambda c: (c["rerank_score"], c["score"]), reverse=True)
    return scored[:k]


def retrieve_and_rerank(retriever, query, llm, k=6, multiplier=CANDIDATE_MULTIPLIER):
    """Retrieve a wider candidate set, then rerank it down to k."""
    candidates = retriever.retrieve(query, k=k * multiplier)
    return rerank(query, candidates, llm, k=k)
