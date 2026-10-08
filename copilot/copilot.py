"""The copilot: score a patient record, explain it, ground the explanation.

Flow:
    features -> risk (pad.train)
             -> top factors (pad.explain, SHAP)
             -> retrieval query -> passages (copilot.retrieve)
             -> local LLM (copilot.llm) -> answer with [n] citations
             -> citation check

The citation check is not decoration. A language model will happily write [7]
when it was handed four passages, and an unresolvable citation looks exactly
like a grounded one to a reader. Any marker that does not point at a retrieved
passage is stripped before the answer is returned.
"""

import re
from dataclasses import dataclass, field

from pad.explain import (
    factors_to_text,
    format_value,
    predict_risk,
    to_frame,
    top_factors,
)
from pad.train import load_artifacts
from copilot.embeddings import EmbeddingUnavailable
from copilot.llm import OllamaClient, OllamaUnavailable
from copilot.prompts import INSUFFICIENT_EVIDENCE, SYSTEM_PROMPT, build_user_prompt
from copilot.rerank import rerank
from copilot.retrieve import (
    Retriever,
    build_query,
    citation_list,
    format_chunks,
    retrieve_for,
)

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    """What the copilot returns: a score, why, the prose, and its sources."""

    risk: float
    model_name: str
    factors: list
    text: str
    citations: list = field(default_factory=list)
    passages: list = field(default_factory=list)
    grounded: bool = True
    warnings: list = field(default_factory=list)


class StreamedExplanation:
    """A streaming explanation and its verified result.

    Iterating yields display text. Once exhausted, ``answer`` holds the
    citation-checked version, which is what should finally be shown.
    """

    def __init__(self, tokens, failure, copilot, risk, factors, passages):
        self._tokens = tokens
        self._copilot = copilot
        self._risk = risk
        self._factors = factors
        self._passages = passages
        self.answer = failure
        self.raw = "" if failure is None else failure.text

    def __iter__(self):
        if self._tokens is None:
            # A failure before generation: show the message, nothing to stream.
            yield self.raw
            return

        pieces = []
        for piece in self._tokens:
            pieces.append(piece)
            yield piece

        self.raw = "".join(pieces)
        self.answer = self._copilot.finish(
            self.raw, self._risk, self._factors, self._passages
        )

    @property
    def changed_by_verification(self):
        """True when citation checking altered what was streamed."""
        return self.answer is not None and self.answer.text.strip() != self.raw.strip()


def check_citations(text, n_passages):
    """Strip [n] markers that do not resolve to a retrieved passage.

    Returns the cleaned text and a warning for each marker removed.
    """
    removed = []

    def replace(match):
        index = int(match.group(1))
        if 1 <= index <= n_passages:
            return match.group(0)
        removed.append(index)
        return ""

    cleaned = CITATION_RE.sub(replace, text)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)

    warnings = []
    if removed:
        unique = sorted(set(removed))
        warnings.append(
            f"Removed {len(removed)} citation marker(s) pointing at passages that "
            f"were never retrieved: {unique}."
        )
    return cleaned.strip(), warnings


def features_to_text(features, columns):
    """Readable dump of every feature value, for the prompt."""
    row = to_frame(features, columns).iloc[0]
    return "\n".join(f"- {column}: {format_value(column, row[column])}"
                     for column in columns)


class PadCopilot:
    """Holds the model, the retriever and the LLM client."""

    def __init__(self, bundle=None, card=None, retriever=None, llm=None,
                 artifact_dir=None):
        if bundle is None:
            bundle, card = load_artifacts(
                **({"artifact_dir": artifact_dir} if artifact_dir else {})
            )
        self.bundle = bundle
        self.card = card or {}
        self.retriever = retriever if retriever is not None else Retriever()
        self.llm = llm if llm is not None else OllamaClient()

    @property
    def model_name(self):
        return self.bundle.get("model_name", "unknown")

    def score(self, features):
        """Risk probability alone, no retrieval and no LLM."""
        return predict_risk(self.bundle, features)

    def health(self):
        """Which backends are reachable, and what built the index.

        Scoring works with everything else down, so the app can report each
        piece separately instead of guessing from the client's type.
        """
        def reachable(component):
            check = getattr(component, "is_available", None)
            return bool(check()) if callable(check) else True

        return {
            "llm": reachable(self.llm),
            "embedder": reachable(getattr(self.retriever, "embedder", None)),
            "index_embedder": getattr(
                getattr(self.retriever, "store", None), "embedder_name", ""
            ),
            "model_name": self.model_name,
        }

    def prepare(self, features, question=None, k=6, background=None, top_k_factors=5,
                use_rerank=False):
        """Everything up to the generation step: score, factors, passages.

        Shared by explain() and stream_explain() so the two cannot drift apart.
        Returns (risk, factors, passages, failure) where failure is a ready-made
        Answer when a backend is down or nothing relevant was retrieved.
        """
        risk = predict_risk(self.bundle, features)
        if background is None:
            background = self.bundle.get("background")
        factors = top_factors(self.bundle, features, k=top_k_factors, background=background)

        try:
            # Reranking measurably helps (hit@1 0.79 -> 0.92 on the golden set)
            # but costs one generation per candidate, so the caller opts in.
            if use_rerank:
                candidates = retrieve_for(self.retriever, factors, question, k=k * 3)
                query = question or build_query(factors)
                passages = rerank(query, candidates, self.llm, k=k)
            else:
                passages = retrieve_for(self.retriever, factors, question, k=k)
        except EmbeddingUnavailable as error:
            return risk, factors, [], Answer(
                risk=risk, model_name=self.model_name, factors=factors,
                text=str(error), grounded=False,
                warnings=["The embedding backend was unreachable; no passages retrieved."],
            )

        if not passages:
            # Nothing cleared the relevance threshold, so there is nothing to
            # ground an answer in. Say so instead of letting the model improvise.
            return risk, factors, [], Answer(
                risk=risk, model_name=self.model_name, factors=factors,
                text=INSUFFICIENT_EVIDENCE, grounded=False,
                warnings=["No passage scored above the relevance threshold."],
            )

        return risk, factors, passages, None

    def build_prompt(self, features, risk, factors, passages, question=None):
        """The user turn handed to the model."""
        return build_user_prompt(
            risk=risk,
            model_name=self.model_name,
            factors_text=factors_to_text(factors),
            features_text=features_to_text(features, self.bundle["features"]),
            passages=format_chunks(passages),
            task=question,
        )

    def finish(self, raw, risk, factors, passages):
        """Verify citations on a finished answer and package the result."""
        text, warnings = check_citations(raw, len(passages))
        if not CITATION_RE.search(text):
            warnings.append("The answer cites no passages, so it may not be grounded.")

        return Answer(
            risk=risk,
            model_name=self.model_name,
            factors=factors,
            text=text,
            citations=citation_list(passages),
            passages=passages,
            grounded=bool(CITATION_RE.search(text)),
            warnings=warnings,
        )

    def stream_explain(self, features, question=None, k=6, background=None,
                       top_k_factors=5, use_rerank=False):
        """Stream the explanation, then verify it.

        Returns a StreamedExplanation: iterate it for display, then read
        ``.answer`` for the verified version. Citation checking cannot run until
        the text is complete, so what streams is unverified and the caller is
        expected to re-render ``.answer`` afterwards.
        """
        risk, factors, passages, failure = self.prepare(
            features, question, k, background, top_k_factors, use_rerank
        )
        if failure is not None:
            return StreamedExplanation(None, failure, self, risk, factors, passages)

        prompt = self.build_prompt(features, risk, factors, passages, question)
        try:
            tokens = self.llm.stream_chat(SYSTEM_PROMPT, prompt)
        except OllamaUnavailable as error:
            return StreamedExplanation(
                None,
                Answer(risk=risk, model_name=self.model_name, factors=factors,
                       text=str(error), citations=citation_list(passages),
                       passages=passages, grounded=False,
                       warnings=["The local LLM was unreachable; no explanation generated."]),
                self, risk, factors, passages,
            )
        return StreamedExplanation(tokens, None, self, risk, factors, passages)

    def explain(self, features, question=None, k=6, background=None, top_k_factors=5,
                use_rerank=False):
        """Score, explain and ground - the whole pipeline for one patient.

        Shares prepare() and finish() with stream_explain, so the streamed and
        non-streamed paths cannot drift apart in how they retrieve or verify.
        """
        risk, factors, passages, failure = self.prepare(
            features, question, k, background, top_k_factors, use_rerank
        )
        if failure is not None:
            return failure

        prompt = self.build_prompt(features, risk, factors, passages, question)
        try:
            raw = self.llm.chat(SYSTEM_PROMPT, prompt)
        except OllamaUnavailable as error:
            return Answer(
                risk=risk,
                model_name=self.model_name,
                factors=factors,
                text=str(error),
                citations=citation_list(passages),
                passages=passages,
                grounded=False,
                warnings=["The local LLM was unreachable; no explanation generated."],
            )

        return self.finish(raw, risk, factors, passages)
