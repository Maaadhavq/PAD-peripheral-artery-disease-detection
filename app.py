"""Streamlit front end for the PAD risk copilot.

    streamlit run app.py

Needs a trained model (run pad_model.ipynb) and a knowledge index
(python -m copilot.ingest). If either is missing, or a backend is down, the page
says what to do rather than failing with a traceback.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from copilot.copilot import PadCopilot
from copilot.retrieve import Retriever
from copilot.threshold import describe
from pad.explain import FEATURE_LABELS

st.set_page_config(page_title="PAD Risk Copilot", page_icon="🫀", layout="wide")

# Synthetic profiles. Real MIMIC-IV rows must not ship in the app under the
# PhysioNet data use agreement.
DEMO_PATIENTS = {
    "Higher risk profile": {
        "gender": 1, "age_at_admission": 76, "cholesterol": 228.0, "glucose": 168.0,
        "creatinine": 1.6, "hemoglobin": 10.9, "platelet_count": 265.0,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 1,
        "has_stroke_history": 1, "is_on_statin": 1, "is_on_antiplatelet": 1,
    },
    "Lower risk profile": {
        "gender": 0, "age_at_admission": 44, "cholesterol": 172.0, "glucose": 92.0,
        "creatinine": 0.8, "hemoglobin": 13.6, "platelet_count": 245.0,
        "has_diabetes": 0, "has_hypertension": 0, "has_heart_disease": 0,
        "has_stroke_history": 0, "is_on_statin": 0, "is_on_antiplatelet": 0,
    },
    "Mixed signals": {
        "gender": 1, "age_at_admission": 63, "cholesterol": 205.0, "glucose": 131.0,
        "creatinine": 1.1, "hemoglobin": 12.8, "platelet_count": 230.0,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 0,
        "has_stroke_history": 0, "is_on_statin": 1, "is_on_antiplatelet": 0,
    },
    "Labs not yet back": {
        "gender": 1, "age_at_admission": 69, "cholesterol": None, "glucose": None,
        "creatinine": None, "hemoglobin": None, "platelet_count": None,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 0,
        "has_stroke_history": 0, "is_on_statin": 0, "is_on_antiplatelet": 0,
    },
}

# A neutral starting point, rather than silently reusing another preset.
BLANK_PATIENT = {
    "gender": 1, "age_at_admission": 65, "cholesterol": 190.0, "glucose": 110.0,
    "creatinine": 1.0, "hemoglobin": 13.0, "platelet_count": 250.0,
    "has_diabetes": 0, "has_hypertension": 0, "has_heart_disease": 0,
    "has_stroke_history": 0, "is_on_statin": 0, "is_on_antiplatelet": 0,
}

NUMERIC_FIELDS = [
    ("age_at_admission", "Age at admission", 18.0, 100.0, 1.0, "years", False),
    ("cholesterol", "Total cholesterol", 80.0, 400.0, 1.0, "mg/dL", True),
    ("glucose", "Glucose", 40.0, 500.0, 1.0, "mg/dL", True),
    ("creatinine", "Creatinine", 0.2, 12.0, 0.1, "mg/dL", True),
    ("hemoglobin", "Hemoglobin", 5.0, 20.0, 0.1, "g/dL", True),
    ("platelet_count", "Platelet count", 20.0, 800.0, 5.0, "K/uL", True),
]

BINARY_FIELDS = [
    ("has_diabetes", "Diabetes"),
    ("has_hypertension", "Hypertension"),
    ("has_heart_disease", "Heart disease"),
    ("has_stroke_history", "Stroke history"),
    ("is_on_statin", "On a statin"),
    ("is_on_antiplatelet", "On an antiplatelet"),
]

BANDS = [(0.33, "Lower", "🟢"), (0.66, "Moderate", "🟡"), (1.01, "Higher", "🔴")]


def record_signature(features):
    """Identifies the record an explanation belongs to.

    Answers persist across reruns so they are not wiped by an unrelated widget
    change, which means a stale one can outlive the record it describes. This
    is what lets the page notice and say so."""
    return tuple(sorted((k, v) for k, v in features.items()))

DISCLAIMER = (
    "Research demo on MIMIC-IV data. Not a medical device, not validated for "
    "clinical use, and not a substitute for professional medical advice."
)


@st.cache_resource(show_spinner=False)
def load_copilot():
    """Load the model and index once per session. Returns (copilot, error)."""
    try:
        retriever = Retriever()
    except FileNotFoundError as error:
        return None, ("index", str(error))
    try:
        return PadCopilot(retriever=retriever), None
    except FileNotFoundError as error:
        return None, ("model", str(error))


def setup_page(kind, message):
    """Explain what is missing instead of showing a traceback."""
    st.title("🫀 PAD Risk Copilot")
    st.error(f"Not ready yet — the {kind} is missing.")
    st.code(message, language="text")
    st.subheader("Setup")
    if kind == "model":
        st.markdown(
            "Train a model first: put the MIMIC-IV files in `mimic_data/`, then run "
            "`pad_model.ipynb` end to end. It writes `artifacts/model.joblib`."
        )
    else:
        st.markdown("Build the knowledge index:")
        st.code("python -m copilot.ingest", language="bash")
        st.caption(
            "Needs Ollama running with `nomic-embed-text` pulled. "
            "`--provider hashing` works offline but retrieves poorly."
        )


def band_for(risk):
    for ceiling, label, icon in BANDS:
        if risk < ceiling:
            return label, icon
    return BANDS[-1][1], BANDS[-1][2]


def percentile_of(risk, distribution):
    """Where this score sits among held-out scores, or None if unavailable."""
    if distribution is None or not len(distribution):
        return None
    return float((np.asarray(distribution) < risk).mean() * 100)


def sidebar_inputs():
    """Feature form. Returns the patient dict; unmeasured labs come back None."""
    st.sidebar.header("Patient record")

    preset = st.sidebar.selectbox(
        "Start from", ["Blank"] + list(DEMO_PATIENTS),
        help="Synthetic profiles — no MIMIC-IV records ship with this app.",
    )
    defaults = DEMO_PATIENTS.get(preset, BLANK_PATIENT)

    values = {}
    values["gender"] = 1 if st.sidebar.radio(
        "Gender", ["Male", "Female"], index=0 if defaults["gender"] == 1 else 1,
        horizontal=True, key=f"{preset}_gender",
    ) == "Male" else 0

    for key, label, low, high, step, unit, optional in NUMERIC_FIELDS:
        default = defaults.get(key)
        missing = False
        if optional:
            # Roughly a fifth of lab values are absent in the real data and the
            # pipeline imputes them. Without this the form cannot express a
            # record that any real hospital would produce.
            missing = st.sidebar.checkbox(
                f"{label} not measured", value=default is None,
                key=f"{preset}_{key}_missing",
            )
        if missing:
            values[key] = None
            continue
        values[key] = st.sidebar.number_input(
            f"{label} ({unit})", min_value=low, max_value=high,
            value=float(default if default is not None else BLANK_PATIENT[key]),
            step=step, key=f"{preset}_{key}",
        )

    st.sidebar.markdown("**History and medications**")
    st.sidebar.caption("From admissions before this one.")
    for key, label in BINARY_FIELDS:
        values[key] = int(st.sidebar.toggle(
            label, value=bool(defaults[key]), key=f"{preset}_{key}"
        ))

    return values


def render_score(risk, card, distribution):
    """The score, with the context needed to read it honestly."""
    label, icon = band_for(risk)
    metrics = (card or {}).get("test_metrics", {})

    # Stacked rather than side by side: this sits in the narrower of two
    # columns, and nesting columns inside it truncated the figure itself.
    st.metric("Estimated PAD probability", f"{risk:.1%}", help=DISCLAIMER)
    st.progress(min(max(risk, 0.0), 1.0))
    st.markdown(f"### {icon} {label} relative risk")

    pct = percentile_of(risk, distribution)
    if pct is not None:
        st.caption(f"Higher than **{pct:.0f}%** of held-out records in this cohort.")

    method = metrics.get("calibration_method")
    if not method:
        st.warning("These probabilities are uncalibrated — read them as a ranking, "
                   "not a likelihood.")

    with st.expander("How to read this number"):
        lo, hi = metrics.get("auc_ci_low"), metrics.get("auc_ci_high")
        if lo is not None and hi is not None:
            st.markdown(
                f"- Model AUC **{metrics.get('test_auc', float('nan')):.3f}** "
                f"(95% CI {lo:.3f}–{hi:.3f})"
            )
        if method:
            st.markdown(
                f"- Probabilities calibrated by **{method}**; "
                f"Brier {metrics.get('calibration_brier', float('nan')):.3f}"
            )
        st.markdown(
            "- The cohort is matched 1:1 by construction, so this is a "
            "discrimination score, not a population prevalence."
        )
        st.markdown(f"- {DISCLAIMER}")


def render_factors(factors):
    """Contributing factors as a diverging bar chart."""
    if not factors:
        st.caption("No contributing factors were computed for this record.")
        return

    frame = pd.DataFrame([{
        "factor": f["label"],
        "effect": f["shap"],
        "value": "not measured" if f["value"] is None else f"{f['value']:g}",
        "direction": "pushes score up" if f["shap"] > 0 else "pushes score down",
    } for f in factors])

    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("effect:Q", title="Effect on the model's score (SHAP)"),
            y=alt.Y("factor:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["pushes score up", "pushes score down"],
                                range=["#c0392b", "#2471a3"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["factor", "value", "effect", "direction"],
        )
        .properties(height=max(140, 34 * len(frame)))
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "SHAP values describe which way a feature moved **this model's score**. "
        "They are not claims about what causes disease."
    )


def render_sources(answer, query):
    if query:
        st.caption("Retrieval query")
        st.code(query, language="text")

    if not answer.passages:
        st.info("No passages were retrieved for this question.")
        return

    for index, passage in enumerate(answer.passages, start=1):
        url = passage.get("url", "")
        header = (f"**[{index}] {passage['source_title']} — {passage['section']}** "
                  f"· score {passage['score']:.3f}")
        if url.startswith("http"):
            header += f" · [source]({url})"
        st.markdown(header)
        st.caption(passage["text"])
        st.divider()


def render_model_card(card, health):
    if not card:
        st.info("No model card was saved with this model.")
        return

    provenance = card.get("data_provenance")
    if provenance:
        st.warning(provenance)

    cohort = card.get("cohort", {})
    if cohort:
        columns = st.columns(len(cohort))
        for column, (key, value) in zip(columns, cohort.items()):
            column.metric(key.replace("_", " "), f"{value:,}")

    metrics = card.get("test_metrics", {})
    if metrics:
        st.markdown("**Held-out performance**")
        # Values are a mix of floats and strings (the calibration method), so
        # they are rendered as text rather than left for Arrow to reconcile.
        st.dataframe(pd.DataFrame([{
            k: (f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in metrics.items() if not isinstance(v, list)
        }]).T.rename(columns={0: "value"}), use_container_width=True)

    cv = card.get("cv_results")
    if cv:
        st.markdown("**Cross-validated model comparison**")
        st.dataframe(pd.DataFrame(cv).T, use_container_width=True)

    st.markdown("**Backends**")
    st.json(health)

    st.markdown(
        "Full limitations are in `copilot/knowledge/model_card.md`, which is part "
        "of the copilot's knowledge base — ask it about them directly."
    )


def main():
    copilot, error = load_copilot()
    if error:
        setup_page(*error)
        return

    health = copilot.health()
    card = copilot.card
    distribution = copilot.bundle.get("score_distribution")

    st.title("🫀 PAD Risk Copilot")
    st.caption(
        "A research demo: a model trained on MIMIC-IV billing codes, explained "
        "against public clinical references. Not a medical device."
    )

    if health["index_embedder"] == "hashing":
        st.warning(
            "This index was built with the offline stand-in embedder. Retrieval "
            "will be poor — rebuild with `python -m copilot.ingest`."
        )
    if not health["llm"] or not health["embedder"]:
        down = [name for name in ("llm", "embedder") if not health[name]]
        st.error(
            f"Ollama is not reachable, so {' and '.join(down)} are unavailable. "
            "Scoring still works; explanations do not.\n\n"
            "`ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`"
        )

    features = sidebar_inputs()

    left, right = st.columns([5, 7], gap="large")

    with left:
        render_score(copilot.score(features), card, distribution)
        st.divider()
        question = st.text_input(
            "Ask about this prediction (optional)",
            placeholder="e.g. Why does being on a statin not lower the score?",
        )
        use_rerank = st.checkbox(
            "Rerank sources with the LLM",
            help="Measured on the golden set: hit@1 rises from 0.79 to 0.92, at "
                 "the cost of one extra generation per candidate, so it is slower.",
        )
        explain = st.button("Explain this record", type="primary",
                            use_container_width=True)

    if explain:
        # Results live in session state, so changing a widget afterwards does
        # not wipe the explanation off the page.
        stream = copilot.stream_explain(features, question=question or None,
                                        use_rerank=use_rerank)
        with right:
            st.subheader("Explanation")
            st.write_stream(stream)
        st.session_state["answer"] = stream.answer
        st.session_state["verified"] = not stream.changed_by_verification
        st.session_state["query"] = question or "(derived from the top factors)"
        st.session_state["signature"] = record_signature(features)
        st.rerun()

    answer = st.session_state.get("answer")
    stale = (
        answer is not None
        and st.session_state.get("signature") != record_signature(features)
    )

    with right:
        if answer is None:
            st.info("Press **Explain this record** for a grounded explanation.")
        else:
            tabs = st.tabs(["Explanation", "Factors", "Sources", "Model card"])
            stale_note = (
                "The record has changed since this explanation was generated — it "
                f"describes a score of {answer.risk:.1%}, not the one shown. "
                "Press **Explain this record** again."
            )

            with tabs[0]:
                if stale:
                    st.warning(stale_note)
                if not answer.grounded:
                    st.warning("This answer is not grounded in retrieved references.")
                if not st.session_state.get("verified", True):
                    st.info(
                        "Citations were verified after streaming finished; the text "
                        "below is the checked version and differs from what streamed."
                    )
                st.markdown(answer.text)
                for warning in answer.warnings:
                    st.caption(f"⚠️ {warning}")

                if answer.citations:
                    st.markdown("**References**")
                    for citation in answer.citations:
                        url = citation["url"]
                        line = (f"[{citation['n']}] {citation['title']} — "
                                f"{citation['section']}")
                        if url.startswith("http"):
                            line += f" · [source]({url})"
                        st.markdown(f"{line} · score {citation['score']}")

            with tabs[1]:
                if stale:
                    st.warning(stale_note)
                render_factors(answer.factors)

            with tabs[2]:
                render_sources(answer, st.session_state.get("query"))

            with tabs[3]:
                render_model_card(card, health)

    st.divider()
    st.caption(DISCLAIMER)
    if copilot.retriever.store.calibration:
        st.caption(f"Retrieval {describe(copilot.retriever.store.calibration)}")


if __name__ == "__main__":
    main()
