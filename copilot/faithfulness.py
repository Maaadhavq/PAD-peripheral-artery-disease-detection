"""Does a cited passage actually support the sentence citing it?

`check_citations` in copilot.py only verifies that `[n]` points at a passage
that was retrieved. That catches invented reference numbers, but not the more
common failure: a real citation attached to a claim the passage never made.
Both look identical to a reader.

Two checks, deliberately separate:

* **Lexical overlap** is cheap, deterministic and always on. It cannot judge
  meaning, so it is a weak signal - a low score flags a sentence worth reading,
  it does not prove the sentence wrong.
* **An LLM judge** is closer to the real question but costs a generation per
  sentence and is itself fallible, so it is opt-in.

Neither filters the answer. They are reported, because silently dropping
sentences would hide exactly the behaviour worth measuring.
"""

import re

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9]+")

# Words that match between any two English sentences and say nothing about
# whether one supports the other.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "it", "its", "has", "have",
    "had", "can", "could", "may", "might", "will", "would", "should", "which",
    "who", "what", "when", "where", "how", "than", "also", "not", "no", "their",
    "there", "they", "them", "patient", "record", "model", "score", "value",
}

JUDGE_SYSTEM = """You check whether a passage supports a claim.

Answer with one word only: SUPPORTED if the passage states or directly implies \
the claim, or UNSUPPORTED if it does not. A passage that is merely on the same \
topic does not support the claim. Do not explain."""


def content_words(text):
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def cited_sentences(text):
    """Sentences carrying a citation, paired with the passage numbers they cite."""
    out = []
    for sentence in SENTENCE_RE.split(text.strip()):
        markers = [int(m) for m in CITATION_RE.findall(sentence)]
        if markers:
            out.append((sentence.strip(), markers))
    return out


def overlap_score(sentence, passage_text):
    """Fraction of the sentence's content words that appear in the passage.

    A blunt instrument: it rewards shared vocabulary, not shared meaning. Useful
    as a cheap screen, not as a verdict.
    """
    claim_words = content_words(CITATION_RE.sub("", sentence))
    if not claim_words:
        return 1.0
    return len(claim_words & content_words(passage_text)) / len(claim_words)


def score_answer(text, passages, threshold=0.3):
    """Per-sentence lexical support for every citation in an answer."""
    rows = []
    for sentence, markers in cited_sentences(text):
        for marker in markers:
            if not 1 <= marker <= len(passages):
                continue  # out-of-range markers are check_citations' job
            score = overlap_score(sentence, passages[marker - 1]["text"])
            rows.append({
                "sentence": sentence,
                "citation": marker,
                "overlap": round(score, 3),
                "weak": score < threshold,
            })
    return rows


def summarize(rows):
    """Aggregate lexical support across an answer."""
    if not rows:
        return {"n_cited_sentences": 0, "mean_overlap": 0.0, "weak_fraction": 0.0}
    overlaps = [r["overlap"] for r in rows]
    return {
        "n_cited_sentences": len(rows),
        "mean_overlap": round(sum(overlaps) / len(overlaps), 3),
        "weak_fraction": round(sum(r["weak"] for r in rows) / len(rows), 3),
    }


def judge_answer(text, passages, llm, limit=None):
    """Ask the local model whether each cited passage supports its sentence.

    Closer to the real question than word overlap, but it is one more fallible
    model judging another, so treat a disagreement as a prompt to go and read
    the passage rather than as a verdict.
    """
    rows = []
    pairs = [(s, m) for s, markers in cited_sentences(text) for m in markers]
    if limit:
        pairs = pairs[:limit]

    for sentence, marker in pairs:
        if not 1 <= marker <= len(passages):
            continue
        passage = passages[marker - 1]
        verdict = llm.chat(
            JUDGE_SYSTEM,
            f"Passage:\n{passage['text']}\n\n"
            f"Claim:\n{CITATION_RE.sub('', sentence).strip()}",
        )
        rows.append({
            "sentence": sentence,
            "citation": marker,
            "supported": verdict.strip().upper().startswith("SUPPORTED"),
            "verdict": verdict.strip()[:40],
        })
    return rows
