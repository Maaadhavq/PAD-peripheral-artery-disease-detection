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

from pad.explain import factors_to_text, predict_risk, to_frame, top_factors
from pad.train import load_artifacts
from copilot.llm import OllamaClient, OllamaUnavailable
from copilot.prompts import INSUFFICIENT_EVIDENCE, SYSTEM_PROMPT, build_user_prompt
from copilot.retrieve import Retriever, build_query, citation_list, format_chunks

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
    lines = []
    for column in columns:
        value = row[column]
        lines.append(f"- {column}: {'missing' if value is None or value != value else value}")
    return "\n".join(lines)


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

    def explain(self, features, question=None, k=6, background=None, top_k_factors=5):
        """Score, explain and ground - the whole pipeline for one patient."""
        risk = predict_risk(self.bundle, features)
        factors = top_factors(self.bundle, features, k=top_k_factors, background=background)

        query = build_query(factors, question)
        passages = self.retriever.retrieve(query, k=k)

        if not passages:
            # Nothing relevant was retrieved, so there is nothing to ground an
            # answer in. Say so instead of letting the model improvise.
            return Answer(
                risk=risk,
                model_name=self.model_name,
                factors=factors,
                text=INSUFFICIENT_EVIDENCE,
                grounded=False,
                warnings=["No passage scored above the relevance threshold."],
            )

        prompt = build_user_prompt(
            risk=risk,
            model_name=self.model_name,
            factors_text=factors_to_text(factors),
            features_text=features_to_text(features, self.bundle["features"]),
            passages=format_chunks(passages),
            task=question,
        )

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
