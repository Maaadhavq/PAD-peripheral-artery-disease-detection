"""Tests for the RAG layer: chunking, the index, retrieval and grounding."""

import numpy as np
import pytest

from copilot.chunking import chunk_document, chunk_text, split_sections
from copilot.copilot import Answer, PadCopilot, check_citations
from copilot.embeddings import get_embedder
from copilot.fetch import html_to_text
from copilot.llm import EchoClient, OllamaUnavailable
from copilot.retrieve import build_query, citation_list, format_chunks
from copilot.store import VectorStore
from pad.config import FEATURE_COLUMNS

DOCUMENT = {
    "id": "demo",
    "title": "Demo document",
    "url": "knowledge/demo.md",
    "publisher": "This project",
    "text": (
        "# Demo document\n\n"
        "Intro paragraph about peripheral artery disease and atherosclerosis.\n\n"
        "## Diagnosis\n\n"
        "The ankle-brachial index compares ankle and arm blood pressure. "
        + "Diagnostic detail sentence. " * 40
        + "\n\n## Treatment\n\nQuit smoking and take a statin.\n"
    ),
}


class TestChunking:
    def test_sections_are_split_on_headings(self):
        sections = split_sections(DOCUMENT["text"])
        titles = [title for title, _ in sections]
        assert "Diagnosis" in titles
        assert "Treatment" in titles

    def test_long_sections_are_windowed_with_overlap(self):
        words = " ".join(f"word{i}" for i in range(600))
        chunks = chunk_text(words, chunk_words=100, overlap_words=20)
        assert len(chunks) > 1
        first_tail = chunks[0].split()[-20:]
        second_head = chunks[1].split()[:20]
        assert first_tail == second_head

    def test_short_text_is_one_chunk(self):
        assert chunk_text("a short sentence") == ["a short sentence"]

    def test_empty_text_yields_nothing(self):
        assert chunk_text("") == []

    def test_chunks_carry_source_and_section(self):
        chunks = chunk_document(DOCUMENT)
        assert chunks
        for chunk in chunks:
            assert chunk["source_id"] == "demo"
            assert chunk["section"]
            assert chunk["text"].strip()

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document(DOCUMENT)
        assert len({chunk["id"] for chunk in chunks}) == len(chunks)


class TestHtmlExtraction:
    def test_headings_become_markdown(self):
        text = html_to_text("<article><h2>Diagnosis</h2><p>ABI test.</p></article>")
        assert "## Diagnosis" in text
        assert "ABI test." in text

    def test_scripts_and_nav_are_dropped(self):
        html = (
            "<article><nav>Menu</nav><script>evil()</script>"
            "<p>Real content here.</p></article>"
        )
        text = html_to_text(html)
        assert "Real content here." in text
        assert "evil" not in text
        assert "Menu" not in text

    def test_boilerplate_is_dropped(self):
        text = html_to_text("<article><p>MENU</p><p>Actual clinical content.</p></article>")
        assert "MENU" not in text
        assert "Actual clinical content." in text


class TestVectorStore:
    @pytest.fixture
    def store(self):
        embedder = get_embedder("hashing")
        chunks = chunk_document(DOCUMENT)
        vectors = embedder.embed([chunk["text"] for chunk in chunks])
        return VectorStore(vectors, chunks, embedder_name="hashing")

    def test_search_returns_scored_chunks_in_order(self, store):
        embedder = get_embedder("hashing")
        query = embedder.embed(["ankle brachial index blood pressure"])[0]
        results = store.search(query, k=3)
        assert results
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_results(self, store):
        embedder = get_embedder("hashing")
        query = embedder.embed(["completely unrelated aardvark taxonomy"])[0]
        assert store.search(query, k=5, min_score=0.99) == []

    def test_round_trip_preserves_everything(self, store, tmp_path):
        store.save(tmp_path / "index")
        loaded = VectorStore.load(tmp_path / "index")
        assert len(loaded) == len(store)
        assert loaded.embedder_name == "hashing"
        assert loaded.chunks[0]["id"] == store.chunks[0]["id"]
        np.testing.assert_allclose(loaded.vectors, store.vectors, rtol=1e-6)

    def test_missing_index_says_how_to_build_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="copilot.ingest"):
            VectorStore.load(tmp_path / "absent")

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="corrupt"):
            VectorStore(np.zeros((3, 4)), [{"id": "only-one"}])


class TestEmbedders:
    def test_hashing_embeddings_are_unit_length(self):
        vectors = get_embedder("hashing").embed(["some text", "other text"])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_hashing_is_deterministic(self):
        first = get_embedder("hashing").embed(["stable text"])
        second = get_embedder("hashing").embed(["stable text"])
        np.testing.assert_array_equal(first, second)

    def test_unknown_provider_fails_loudly(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder("magic")


class TestCitationChecking:
    def test_valid_citations_survive(self):
        text, warnings = check_citations("Plaque narrows arteries [1] and [2].", 3)
        assert "[1]" in text and "[2]" in text
        assert warnings == []

    def test_out_of_range_citations_are_stripped(self):
        """A model citing [7] when handed 3 passages must not look grounded."""
        text, warnings = check_citations("Claim one [1]. Claim two [7].", 3)
        assert "[7]" not in text
        assert "[1]" in text
        assert warnings
        assert "7" in str(warnings)

    def test_stripping_leaves_clean_punctuation(self):
        text, _ = check_citations("A claim [9] .", 2)
        assert "  " not in text
        assert text.endswith(".")

    def test_zero_is_not_a_valid_citation(self):
        text, warnings = check_citations("Claim [0].", 3)
        assert "[0]" not in text
        assert warnings


class TestRetrieveHelpers:
    def test_query_mentions_factor_labels(self):
        factors = [{"label": "diabetes history"}, {"label": "statin therapy"}]
        query = build_query(factors, question="why so high?")
        assert "peripheral artery disease" in query
        assert "diabetes history" in query
        assert "why so high?" in query

    def test_passages_are_numbered_from_one(self):
        chunks = [
            {"source_title": "A", "section": "S1", "text": "first"},
            {"source_title": "B", "section": "S2", "text": "second"},
        ]
        formatted = format_chunks(chunks)
        assert formatted.startswith("[1] A — S1")
        assert "[2] B — S2" in formatted

    def test_citation_list_matches_passage_numbers(self):
        chunks = [{"source_title": "A", "section": "S", "url": "u",
                   "publisher": "p", "score": 0.5}]
        assert citation_list(chunks)[0]["n"] == 1


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


@pytest.fixture
def trained_bundle(dataset):
    from pad import train as train_mod

    X, y, groups = train_mod.split_xy(dataset)
    X_train, _, y_train, _, _ = train_mod.patient_split(X, y, groups)
    model = train_mod.build_models()["Logistic Regression"]
    pipeline = train_mod.fit_pipeline(train_mod.make_pipeline(model), X_train, y_train)
    return {"pipeline": pipeline, "model_name": "Logistic Regression",
            "features": FEATURE_COLUMNS}


@pytest.fixture
def patient():
    return {
        "gender": 1, "age_at_admission": 72, "cholesterol": 215.0, "glucose": 148.0,
        "creatinine": 1.4, "hemoglobin": 11.8, "platelet_count": 240.0,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 1,
        "has_stroke_history": 0, "is_on_statin": 1, "is_on_antiplatelet": 1,
    }


class TestCopilot:
    def test_explain_returns_risk_factors_and_citations(self, trained_bundle, patient):
        copilot = PadCopilot(
            bundle=trained_bundle, card={}, retriever=StubRetriever(),
            llm=EchoClient("Atherosclerosis narrows the arteries [1]."),
        )
        answer = copilot.explain(patient)
        assert isinstance(answer, Answer)
        assert 0.0 <= answer.risk <= 1.0
        assert answer.factors
        assert answer.grounded
        assert answer.citations[0]["n"] == 1

    def test_query_includes_the_top_factors(self, trained_bundle, patient):
        retriever = StubRetriever()
        copilot = PadCopilot(bundle=trained_bundle, card={}, retriever=retriever,
                             llm=EchoClient())
        copilot.explain(patient)
        assert "peripheral artery disease" in retriever.queries[0]

    def test_no_relevant_passages_means_no_generation(self, trained_bundle, patient):
        """With nothing retrieved the copilot must refuse, not improvise."""
        llm = EchoClient("I would have made something up.")
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=StubRetriever(chunks=[]), llm=llm)
        answer = copilot.explain(patient)
        assert not answer.grounded
        assert llm.calls == []
        assert "does not contain passages" in answer.text

    def test_hallucinated_citation_is_stripped_and_flagged(self, trained_bundle, patient):
        copilot = PadCopilot(
            bundle=trained_bundle, card={}, retriever=StubRetriever(),
            llm=EchoClient("Claim one [1]. Invented claim [4]."),
        )
        answer = copilot.explain(patient)
        assert "[4]" not in answer.text
        assert "[1]" in answer.text
        assert answer.warnings

    def test_uncited_answer_is_flagged_as_ungrounded(self, trained_bundle, patient):
        copilot = PadCopilot(
            bundle=trained_bundle, card={}, retriever=StubRetriever(),
            llm=EchoClient("A confident claim with no citation at all."),
        )
        answer = copilot.explain(patient)
        assert not answer.grounded
        assert any("cites no passages" in w for w in answer.warnings)

    def test_unreachable_llm_degrades_without_crashing(self, trained_bundle, patient):
        class DeadClient:
            def chat(self, system, user):
                raise OllamaUnavailable("Could not reach Ollama at http://localhost:11434")

        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=StubRetriever(), llm=DeadClient())
        answer = copilot.explain(patient)
        assert not answer.grounded
        assert "Ollama" in answer.text
        assert answer.risk >= 0.0

    def test_prompt_carries_features_and_passages(self, trained_bundle, patient):
        llm = EchoClient("Grounded [1].")
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=StubRetriever(), llm=llm)
        copilot.explain(patient)
        prompt = llm.calls[0]["user"]
        assert "age_at_admission" in prompt
        assert "[1]" in prompt
        assert "research" in llm.calls[0]["system"].lower()

    def test_score_alone_skips_retrieval_and_llm(self, trained_bundle, patient):
        retriever, llm = StubRetriever(), EchoClient()
        copilot = PadCopilot(bundle=trained_bundle, card={},
                             retriever=retriever, llm=llm)
        risk = copilot.score(patient)
        assert 0.0 <= risk <= 1.0
        assert retriever.queries == []
        assert llm.calls == []
