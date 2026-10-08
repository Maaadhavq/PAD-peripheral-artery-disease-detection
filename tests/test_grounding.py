"""Relevance thresholds, faithfulness scoring, reranking and query construction."""

import numpy as np
import pytest

from copilot import faithfulness
from copilot.chunking import chunk_document
from copilot.embeddings import get_embedder
from copilot.rerank import _parse_score, rerank
from copilot.retrieve import build_query, retrieve_for
from copilot.store import VectorStore
from copilot.threshold import calibrate_threshold, describe
from tests.helpers import StubRetriever

PASSAGE = (
    "The ankle-brachial index compares blood pressure in the ankle with blood "
    "pressure in the arm. A resting value below 0.90 suggests peripheral artery "
    "disease, and a value below 0.40 indicates severe disease."
)


class FakeEmbedder:
    """Maps a fixed set of queries to controlled similarity scores."""

    name = "fake"

    def __init__(self, scores):
        self.scores = scores

    def is_available(self):
        return True

    def embed(self, texts):
        out = []
        for text in texts:
            score = self.scores.get(text, 0.0)
            # A 2-D unit vector whose dot product with (1, 0) is `score`.
            out.append([score, float(np.sqrt(max(0.0, 1 - score ** 2)))])
        return np.asarray(out, dtype=np.float32)


def fake_store(chunk_count=1):
    chunks = [{"id": f"c{i}", "text": f"chunk {i}", "source_id": "s",
               "source_title": "S", "section": "Sec", "url": "", "publisher": ""}
              for i in range(chunk_count)]
    vectors = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (chunk_count, 1))
    return VectorStore(vectors, chunks, embedder_name="fake")


class TestThresholdCalibration:
    def test_picks_a_cutoff_between_the_two_distributions(self):
        store = fake_store()
        embedder = FakeEmbedder({"on": 0.8, "off": 0.4})
        result = calibrate_threshold(store, embedder, on_topic=["on"], off_topic=["off"])

        assert result["separated"] is True
        assert result["off_topic_max"] < result["min_score"] < result["on_topic_min"]

    def test_refuses_to_invent_a_cutoff_when_they_overlap(self):
        """No threshold separates overlapping distributions, so it must say so
        rather than pick one that silently blocks real questions."""
        store = fake_store()
        embedder = FakeEmbedder({"on": 0.5, "off": 0.7})
        result = calibrate_threshold(store, embedder, on_topic=["on"], off_topic=["off"])

        assert result["separated"] is False
        assert result["min_score"] == 0.0
        assert "not calibrated" in describe(result)

    def test_reports_the_measurements_behind_the_number(self):
        store = fake_store()
        embedder = FakeEmbedder({"on": 0.9, "off": 0.3})
        result = calibrate_threshold(store, embedder, on_topic=["on"], off_topic=["off"])

        for key in ["on_topic_min", "off_topic_max", "margin", "n_on_topic", "n_off_topic"]:
            assert key in result
        assert "margin" in describe(result)


class TestThresholdPersistence:
    def test_calibration_survives_a_round_trip(self, tmp_path):
        store = fake_store(3)
        store.calibration = {"min_score": 0.42, "separated": True}
        store.save(tmp_path / "idx")

        loaded = VectorStore.load(tmp_path / "idx")
        assert loaded.min_score == pytest.approx(0.42)
        assert loaded.calibration["separated"] is True

    def test_an_index_without_calibration_refuses_nothing(self, tmp_path):
        """Older indexes have no stored threshold; they must keep working."""
        store = fake_store(2)
        store.save(tmp_path / "idx")
        assert VectorStore.load(tmp_path / "idx").min_score == 0.0


class TestQueryConstruction:
    def test_factor_query_mentions_the_labels(self):
        query = build_query([{"label": "diabetes history"}], question="why?")
        assert "diabetes history" in query
        assert "peripheral artery disease" in query

    def test_a_question_gets_its_own_retrieval(self):
        """A specific question was previously buried under five factor labels,
        so it is now searched for separately and the results merged."""
        retriever = StubRetriever()
        retrieve_for(retriever, [{"label": "statin therapy"}],
                     question="what are the model's limits?", k=6)
        assert len(retriever.queries) == 2
        assert "what are the model's limits?" in retriever.queries

    def test_without_a_question_only_the_factors_are_searched(self):
        retriever = StubRetriever()
        retrieve_for(retriever, [{"label": "statin therapy"}], question=None, k=6)
        assert len(retriever.queries) == 1

    def test_an_unanswerable_question_returns_nothing(self):
        """The factor query is about the record and always retrieves something,
        so merging it in unconditionally meant an off-topic question was never
        refused in the explain path - 0/8 on the golden set, even though the
        question alone was refused 6/8."""
        class QuestionMisses:
            def __init__(self):
                self.queries = []

            def retrieve(self, query, k=6, **kwargs):
                self.queries.append(query)
                # The factor query hits; the user's question does not.
                if query.startswith("peripheral artery disease"):
                    return [{"id": "f", "text": "t", "score": 0.7}]
                return []

        retriever = QuestionMisses()
        assert retrieve_for(retriever, [{"label": "statin therapy"}],
                            question="who directed Jaws?", k=6) == []

    def test_factor_passages_still_returned_without_a_question(self):
        retriever = StubRetriever()
        assert retrieve_for(retriever, [{"label": "statin therapy"}], question=None, k=6)

    def test_merged_results_are_deduplicated_and_ranked(self):
        shared = {"id": "dup", "text": "t", "source_id": "s", "source_title": "S",
                  "section": "x", "score": 0.5}
        other = {"id": "other", "text": "t", "source_id": "s", "source_title": "S",
                 "section": "y", "score": 0.9}
        retriever = StubRetriever(chunks=[shared, other])

        merged = retrieve_for(retriever, [{"label": "a"}], question="q", k=6)
        assert len({c["id"] for c in merged}) == len(merged)
        assert [c["score"] for c in merged] == sorted(
            [c["score"] for c in merged], reverse=True
        )


class TestFaithfulness:
    def test_a_supported_sentence_scores_high(self):
        sentence = "A resting ankle-brachial index below 0.90 suggests PAD [1]."
        assert faithfulness.overlap_score(sentence, PASSAGE) > 0.6

    def test_an_unsupported_sentence_scores_low(self):
        """The failure this catches: a valid citation on a claim the passage
        never made. check_citations cannot see this - the number resolves."""
        sentence = "Platelet transfusions reverse arterial calcification [1]."
        assert faithfulness.overlap_score(sentence, PASSAGE) < 0.3

    def test_a_sentence_of_only_stopwords_is_not_penalised(self):
        """With no content words there is nothing to check, so it must not be
        reported as unsupported."""
        sentence = "It is that which was in the of and to [1]."
        assert faithfulness.content_words(sentence) == set()
        assert faithfulness.overlap_score(sentence, PASSAGE) == 1.0

    def test_only_cited_sentences_are_scored(self):
        text = "Uncited claim. A cited claim [1]."
        assert [s for s, _ in faithfulness.cited_sentences(text)] == ["A cited claim [1]."]

    def test_scoring_skips_out_of_range_citations(self):
        passages = [{"text": PASSAGE}]
        rows = faithfulness.score_answer("Claim [9].", passages)
        assert rows == []

    def test_summary_flags_weak_support(self):
        passages = [{"text": PASSAGE}]
        rows = faithfulness.score_answer(
            "Platelet transfusions reverse arterial calcification [1].", passages
        )
        summary = faithfulness.summarize(rows)
        assert summary["n_cited_sentences"] == 1
        assert summary["weak_fraction"] == 1.0

    def test_judge_reads_the_verdict(self):
        class Judge:
            def chat(self, system, user):
                return "SUPPORTED"

        rows = faithfulness.judge_answer("A claim [1].", [{"text": PASSAGE}], Judge())
        assert rows[0]["supported"] is True


class TestRerank:
    def test_parses_a_plain_integer(self):
        assert _parse_score("9") == 9.0
        assert _parse_score("  10  ") == 10.0

    def test_clamps_and_defaults_safely(self):
        assert _parse_score("99") == 10.0
        assert _parse_score("no idea") == 5.0

    def test_reorders_by_the_model_score(self):
        class Judge:
            def chat(self, system, user):
                return "9" if "ankle-brachial" in user else "1"

        chunks = [
            {"id": "a", "text": "unrelated filler text", "score": 0.9},
            {"id": "b", "text": "the ankle-brachial index test", "score": 0.1},
        ]
        assert rerank("q", chunks, Judge(), k=2)[0]["id"] == "b"

    def test_ties_fall_back_to_the_embedding_order(self):
        """A model that rates everything the same must degrade to the original
        ranking, not to arbitrary order."""
        class Flat:
            def chat(self, system, user):
                return "5"

        chunks = [{"id": "a", "text": "x", "score": 0.3},
                  {"id": "b", "text": "y", "score": 0.8}]
        assert [c["id"] for c in rerank("q", chunks, Flat(), k=2)] == ["b", "a"]

    def test_empty_candidates_are_handled(self):
        assert rerank("q", [], None, k=5) == []
