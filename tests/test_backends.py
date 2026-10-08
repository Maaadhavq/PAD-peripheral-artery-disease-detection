"""Failure modes of the local backends.

Retrieval runs against the same Ollama server as generation and fails the same
way. Before this was handled, a stopped server raised a raw
requests.ConnectionError out of explain() while the LLM path next to it
degraded cleanly.
"""

import pytest

from copilot.copilot import Answer, PadCopilot
from copilot.embeddings import EmbeddingUnavailable, OllamaEmbedder
from copilot.llm import EchoClient, OllamaUnavailable
from tests.helpers import StubRetriever


class DeadRetriever:
    """Stands in for a retriever whose embedding backend is unreachable."""

    def retrieve(self, query, k=6, **kwargs):
        raise EmbeddingUnavailable(
            "Could not reach Ollama at http://localhost:11434 to embed the query."
        )


class DeadLLM:
    def chat(self, system, user):
        raise OllamaUnavailable("Could not reach Ollama at http://localhost:11434")


class TestEmbeddingFailure:
    def test_returns_an_answer_rather_than_raising(self, trained_bundle, patient):
        llm = EchoClient()
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=DeadRetriever(), llm=llm)
        answer = copilot.explain(patient)

        assert isinstance(answer, Answer)
        assert not answer.grounded
        assert "Ollama" in answer.text

    def test_the_risk_score_survives(self, trained_bundle, patient):
        """Scoring needs neither backend, so a dead server must not cost it."""
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=DeadRetriever(), llm=EchoClient())
        answer = copilot.explain(patient)
        assert 0.0 <= answer.risk <= 1.0
        assert answer.factors

    def test_the_llm_is_not_called_without_passages(self, trained_bundle, patient):
        llm = EchoClient("I would have invented something.")
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=DeadRetriever(), llm=llm)
        copilot.explain(patient)
        assert llm.calls == []

    def test_score_works_with_every_backend_down(self, trained_bundle, patient):
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=DeadRetriever(), llm=DeadLLM())
        assert 0.0 <= copilot.score(patient) <= 1.0

    def test_embedder_raises_a_typed_error(self):
        with pytest.raises(EmbeddingUnavailable, match="ollama pull"):
            OllamaEmbedder(host="http://localhost:1").embed(["anything"])

    def test_the_error_says_how_to_fix_it(self):
        try:
            OllamaEmbedder(host="http://localhost:1").embed(["anything"])
        except EmbeddingUnavailable as error:
            message = str(error)
        assert "ollama.com/download" in message
        assert "nomic-embed-text" in message


class TestHealth:
    def test_reports_each_backend_separately(self, trained_bundle):
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=StubRetriever(), llm=EchoClient())
        health = copilot.health()
        assert set(health) == {"llm", "embedder", "index_embedder", "model_name"}

    def test_a_client_without_a_check_is_assumed_up(self, trained_bundle):
        """A test double with no is_available must not be reported as down."""
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=StubRetriever(), llm=DeadLLM())
        assert copilot.health()["llm"] is True
