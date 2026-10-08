"""Calibrating the relevance threshold that decides when to refuse.

The threshold was previously a module constant of 0.25, chosen by eye. Measured
against the real index it refused nothing at all: `nomic-embed-text` scores an
unrelated question like "what is the capital of France?" at 0.425, so every
off-topic query sailed through and the refusal branch was dead code.

Cosine similarity scales differently for every embedding model, so the cutoff is
a property of the model that built the index, not of this code. It is measured
at ingest from two probe sets and stored alongside the vectors.

The probes here are deliberately *not* the evaluation golden set. Tuning the
threshold on the same questions used to report refusal accuracy would make that
number meaningless.
"""

import numpy as np

# Clearly in scope: PAD clinically, and the model's own documentation.
ON_TOPIC_PROBES = [
    "how does plaque build up in the leg arteries",
    "symptoms of poor circulation in the legs",
    "how is PAD diagnosed in a clinic",
    "treatment options for blocked leg arteries",
    "does smoking affect peripheral artery disease",
    "how does diabetes relate to vascular disease",
    "which features does this risk model use",
    "how was the training cohort assembled",
    "what are the limitations of this model",
    "why is a lab value included as a predictor",
]

# Clearly out of scope: no amount of retrieval should answer these.
OFF_TOPIC_PROBES = [
    "what is the capital of France",
    "how do I fix a null pointer exception in Java",
    "recipe for tomato pasta sauce",
    "who won the football world cup in 2022",
    "my cat keeps scratching the sofa",
    "explain the offside rule",
    "how do I renew my passport",
    "best hiking trails near Seattle",
    "how to change a bicycle tyre",
    "what is the plot of Hamlet",
]

# Fallback when the probes fail to separate: refuse nothing rather than refuse
# everything, since a bad threshold that blocks real questions is worse than one
# that lets a few odd ones through to a model told to say it cannot answer.
FALLBACK_MIN_SCORE = 0.0

# How far to sit above the off-topic ceiling, as a fraction of the gap between
# the two distributions. Below the midpoint, favouring recall of real questions.
GAP_FRACTION = 0.5


def top_scores(store, embedder, queries):
    """Best similarity each query achieves against the index."""
    scores = []
    for query in queries:
        vector = embedder.embed([query])[0]
        hits = store.search(vector, k=1, min_score=0.0)
        scores.append(float(hits[0]["score"]) if hits else 0.0)
    return np.asarray(scores)


def calibrate_threshold(store, embedder, on_topic=None, off_topic=None,
                        gap_fraction=GAP_FRACTION):
    """Pick a cutoff that separates in-scope from out-of-scope questions.

    Returns the threshold and the measurements behind it, so the choice is
    inspectable rather than a bare number in a config file. If the two
    distributions overlap, no threshold separates them and the function says so
    instead of inventing one.
    """
    on_topic = on_topic if on_topic is not None else ON_TOPIC_PROBES
    off_topic = off_topic if off_topic is not None else OFF_TOPIC_PROBES

    on_scores = top_scores(store, embedder, on_topic)
    off_scores = top_scores(store, embedder, off_topic)

    on_floor = float(on_scores.min())
    off_ceiling = float(off_scores.max())
    separated = on_floor > off_ceiling

    if separated:
        threshold = off_ceiling + gap_fraction * (on_floor - off_ceiling)
    else:
        threshold = FALLBACK_MIN_SCORE

    return {
        "min_score": round(float(threshold), 4),
        "separated": bool(separated),
        "on_topic_min": round(on_floor, 4),
        "on_topic_mean": round(float(on_scores.mean()), 4),
        "off_topic_max": round(off_ceiling, 4),
        "off_topic_mean": round(float(off_scores.mean()), 4),
        "margin": round(on_floor - off_ceiling, 4),
        "n_on_topic": len(on_topic),
        "n_off_topic": len(off_topic),
    }


def describe(calibration):
    """One-line summary for ingest output and the app's status line."""
    if not calibration.get("separated", False):
        return (
            "threshold not calibrated: the probe sets overlap "
            f"(on-topic min {calibration.get('on_topic_min')}, "
            f"off-topic max {calibration.get('off_topic_max')}), so nothing is refused"
        )
    return (
        f"threshold {calibration['min_score']:.3f} "
        f"(off-topic max {calibration['off_topic_max']:.3f}, "
        f"on-topic min {calibration['on_topic_min']:.3f}, "
        f"margin {calibration['margin']:.3f})"
    )
