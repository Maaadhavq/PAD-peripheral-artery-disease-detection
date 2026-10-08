"""Shared test doubles.

Kept out of conftest because these are classes, not fixtures - pytest injects
fixtures automatically but test modules have to import classes by name.
"""


class StubRetriever:
    """Returns a fixed passage list, so copilot tests need no index."""

    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [{
            "source_id": "model_card", "source_title": "Model card", "section": "Limits",
            "text": "This is a research model and not a diagnostic device.",
            "url": "knowledge/model_card.md", "publisher": "This project", "score": 0.8,
        }]
        self.queries = []

    def retrieve(self, query, k=6, **kwargs):
        self.queries.append(query)
        return self.chunks[:k]
