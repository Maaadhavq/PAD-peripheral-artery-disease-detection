"""Streamlit front end for the PAD risk copilot.

    streamlit run app.py

Needs a trained model (run pad_model.ipynb) and a knowledge index
(python -m copilot.ingest). If either is missing, or a backend is down, the page
says what to do rather than failing with a traceback.

Presentation lives in app_ui.py; this file is the behaviour.
"""

import html

import numpy as np
import pandas as pd
import streamlit as st

import app_ui as ui
from copilot.copilot import PadCopilot
from copilot.retrieve import Retriever
from copilot.threshold import describe

st.set_page_config(page_title="PAD Risk Copilot", page_icon="◐", layout="wide")

# Synthetic profiles. Real MIMIC-IV rows must not ship in the app under the
# PhysioNet data use agreement.
DEMO_PATIENTS = {
    "Established vascular disease": {
        "gender": 1, "age_at_admission": 76, "cholesterol": 228.0, "glucose": 168.0,
        "creatinine": 1.6, "hemoglobin": 10.9, "platelet_count": 265.0,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 1,
        "has_stroke_history": 1, "is_on_statin": 1, "is_on_antiplatelet": 1,
    },
    "No cardiovascular history": {
        "gender": 0, "age_at_admission": 44, "cholesterol": 172.0, "glucose": 92.0,
        "creatinine": 0.8, "hemoglobin": 13.6, "platelet_count": 245.0,
        "has_diabetes": 0, "has_hypertension": 0, "has_heart_disease": 0,
        "has_stroke_history": 0, "is_on_statin": 0, "is_on_antiplatelet": 0,
    },
    "Treated risk, no diagnosis": {
        "gender": 1, "age_at_admission": 63, "cholesterol": 205.0, "glucose": 131.0,
        "creatinine": 1.1, "hemoglobin": 12.8, "platelet_count": 230.0,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 0,
        "has_stroke_history": 0, "is_on_statin": 1, "is_on_antiplatelet": 0,
    },
    "Labs not yet resulted": {
        "gender": 1, "age_at_admission": 69, "cholesterol": None, "glucose": None,
        "creatinine": None, "hemoglobin": None, "platelet_count": None,
        "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 0,
        "has_stroke_history": 0, "is_on_statin": 0, "is_on_antiplatelet": 0,
    },
}

BLANK_PATIENT = {
    "gender": 1, "age_at_admission": 65, "cholesterol": 190.0, "glucose": 110.0,
    "creatinine": 1.0, "hemoglobin": 13.0, "platelet_count": 250.0,
    "has_diabetes": 0, "has_hypertension": 0, "has_heart_disease": 0,
    "has_stroke_history": 0, "is_on_statin": 0, "is_on_antiplatelet": 0,
}

NUMERIC_FIELDS = [
    ("age_at_admission", "Age", 18.0, 100.0, 1.0, "yr", False),
    ("cholesterol", "Cholesterol", 80.0, 400.0, 1.0, "mg/dL", True),
    ("glucose", "Glucose", 40.0, 500.0, 1.0, "mg/dL", True),
    ("creatinine", "Creatinine", 0.2, 12.0, 0.1, "mg/dL", True),
    ("hemoglobin", "Hemoglobin", 5.0, 20.0, 0.1, "g/dL", True),
    ("platelet_count", "Platelets", 20.0, 800.0, 5.0, "K/uL", True),
]

BINARY_FIELDS = [
    ("has_diabetes", "Diabetes"),
    ("has_hypertension", "Hypertension"),
    ("has_heart_disease", "Heart disease"),
    ("has_stroke_history", "Stroke"),
    ("is_on_statin", "Statin"),
    ("is_on_antiplatelet", "Antiplatelet"),
]

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
        return None, ("knowledge index", str(error))
    try:
        return PadCopilot(retriever=retriever), None
    except FileNotFoundError as error:
        return None, ("trained model", str(error))


def record_signature(features):
    """Identifies the record an explanation belongs to.

    Answers persist across reruns so an unrelated widget change does not wipe
    them, which means a stale one can outlive the record it describes. This is
    what lets the page notice and say so.
    """
    return tuple(sorted((k, v) for k, v in features.items()))


def setup_page(kind, message):
    """Explain what is missing instead of showing a traceback."""
    st.markdown(ui.masthead("not loaded", ""), unsafe_allow_html=True)
    st.markdown(ui.notice(f"Not ready — the {kind} is missing."),
                unsafe_allow_html=True)
    st.code(message, language="text")
    st.markdown(ui.eyebrow("Setup"), unsafe_allow_html=True)
    if kind == "trained model":
        st.markdown(
            "Put the MIMIC-IV files in `mimic_data/`, then run `pad_model.ipynb` "
            "end to end. It writes `artifacts/model.joblib`."
        )
    else:
        st.code("python -m copilot.ingest", language="bash")
        st.caption(
            "Needs Ollama running with `nomic-embed-text` pulled. "
            "`--provider hashing` works offline but retrieves poorly."
        )


def sidebar_inputs():
    """Record form. Returns the patient dict; unmeasured labs come back None."""
    st.sidebar.markdown(ui.eyebrow("Record"), unsafe_allow_html=True)

    preset = st.sidebar.selectbox(
        "Start from", ["Blank"] + list(DEMO_PATIENTS),
        help="Synthetic profiles. No MIMIC-IV records ship with this app.",
    )
    defaults = DEMO_PATIENTS.get(preset, BLANK_PATIENT)

    values = {}
    values["gender"] = 1 if st.sidebar.radio(
        "Sex", ["Male", "Female"], index=0 if defaults["gender"] == 1 else 1,
        horizontal=True, key=f"{preset}_gender",
    ) == "Male" else 0

    for key, label, low, high, step, unit, optional in NUMERIC_FIELDS:
        default = defaults.get(key)
        if optional and st.sidebar.checkbox(
            f"{label} — not resulted", value=default is None,
            key=f"{preset}_{key}_missing",
        ):
            # Roughly a fifth of lab values are absent in the real data and the
            # pipeline imputes them. Without this the form could not express a
            # record that any real hospital would produce.
            values[key] = None
            continue
        values[key] = st.sidebar.number_input(
            f"{label} ({unit})", min_value=low, max_value=high,
            value=float(default if default is not None else BLANK_PATIENT[key]),
            step=step, key=f"{preset}_{key}",
        )

    st.sidebar.markdown(ui.eyebrow("Prior admissions"), unsafe_allow_html=True)
    for key, label in BINARY_FIELDS:
        values[key] = int(st.sidebar.toggle(
            label, value=bool(defaults[key]), key=f"{preset}_{key}"
        ))

    return values


def render_readout(risk, card, distribution):
    """The score, the bands, and the distribution it is measured against."""
    metrics = (card or {}).get("test_metrics", {})
    band = ui.band_for(risk)

    figure, scale = st.columns([2, 5], gap="medium")

    with figure:
        st.markdown(
            f'<p class="readout-value">{risk:.1%}</p>'
            f'<p class="readout-band" style="color:{ui.ARTERIAL}">{band} band</p>',
            unsafe_allow_html=True,
        )

    with scale:
        st.markdown(ui.readout(risk, distribution), unsafe_allow_html=True)
        if distribution is not None and len(distribution):
            pct = float((np.asarray(distribution) < risk).mean() * 100)
            st.markdown(
                f'<p class="readout-note">Higher than <strong>{pct:.0f}%</strong> '
                "of held-out records. The bars are that cohort, matched 1:1 by "
                "construction — this is a position within it, not a population "
                "prevalence.</p>",
                unsafe_allow_html=True,
            )

    method = metrics.get("calibration_method")
    if not method:
        st.markdown(
            ui.notice("Probabilities are uncalibrated. Read this as a ranking, "
                      "not a likelihood."),
            unsafe_allow_html=True,
        )
        return

    lo, hi = metrics.get("auc_ci_low"), metrics.get("auc_ci_high")
    bits = [f"calibrated · {method}"]
    if metrics.get("calibration_brier") is not None:
        bits.append(f"brier {metrics['calibration_brier']:.3f}")
    if lo is not None and hi is not None:
        bits.append(f"auc {metrics.get('test_auc', float('nan')):.3f} "
                    f"[{lo:.3f}–{hi:.3f}]")
    st.markdown(
        f'<p class="readout-note" style="font-family:{ui.MONO},monospace;'
        f'font-size:0.72rem;letter-spacing:0.06em">{" · ".join(bits)}</p>',
        unsafe_allow_html=True,
    )


def render_factors(factors):
    if not factors:
        st.markdown('<p class="readout-note">No contributing factors were '
                    "computed for this record.</p>", unsafe_allow_html=True)
        return

    frame = pd.DataFrame([{
        "factor": f["label"],
        "effect": f["shap"],
        "value": "not resulted" if f["value"] is None else f"{f['value']:g}",
        "direction": "raises the score" if f["shap"] > 0 else "lowers the score",
    } for f in factors])

    st.altair_chart(ui.factor_chart(frame, max(130, 30 * len(frame))),
                    use_container_width=True)
    st.markdown(
        '<p class="readout-note">SHAP values say which way a feature moved '
        "<strong>this model's score</strong>. They are not claims about what "
        "causes disease.</p>",
        unsafe_allow_html=True,
    )


def render_sources(answer, query):
    if query:
        st.markdown(ui.eyebrow("Retrieval query"), unsafe_allow_html=True)
        st.code(query, language="text")

    if not answer.passages:
        st.markdown('<p class="readout-note">No passages were retrieved.</p>',
                    unsafe_allow_html=True)
        return

    st.markdown(ui.eyebrow(f"{len(answer.passages)} passages"),
                unsafe_allow_html=True)
    for index, passage in enumerate(answer.passages, start=1):
        url = passage.get("url", "")
        head = (f"[{index}] {html.escape(passage['source_title'])} — "
                f"{html.escape(passage['section'])} · {passage['score']:.3f}")
        if url.startswith("http"):
            head += f' · <a href="{html.escape(url)}">source</a>'
        st.markdown(
            f'<div class="passage"><div class="passage-head">{head}</div>'
            f'<div class="passage-body">{html.escape(passage["text"])}</div></div>',
            unsafe_allow_html=True,
        )


def render_model_card(card, health):
    if not card:
        st.markdown('<p class="readout-note">No model card was saved.</p>',
                    unsafe_allow_html=True)
        return

    provenance = card.get("data_provenance")
    if provenance:
        st.markdown(ui.notice(provenance), unsafe_allow_html=True)

    cohort = card.get("cohort", {})
    if cohort:
        st.markdown(ui.eyebrow("Cohort"), unsafe_allow_html=True)
        st.markdown(
            " ".join(
                f'<span style="font-family:{ui.MONO},monospace;font-size:0.78rem;'
                f'margin-right:1.4rem"><strong>{v:,}</strong> '
                f'<span style="color:{ui.MUTED}">{k.replace("n_", "")}</span></span>'
                for k, v in cohort.items()
            ),
            unsafe_allow_html=True,
        )

    metrics = card.get("test_metrics", {})
    if metrics:
        st.markdown(ui.eyebrow("Held-out performance"), unsafe_allow_html=True)
        # Values mix floats and strings (the calibration method), so they are
        # rendered as text rather than left for Arrow to reconcile.
        st.dataframe(pd.DataFrame([{
            k: (f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in metrics.items() if not isinstance(v, list)
        }]).T.rename(columns={0: "value"}), use_container_width=True)

    cv = card.get("cv_results")
    if cv:
        st.markdown(ui.eyebrow("Cross-validated comparison"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(cv).T, use_container_width=True)

    st.markdown(ui.eyebrow("Backends"), unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-family:{ui.MONO},monospace;font-size:0.74rem">'
        + "<br>".join(f"{k} · {v}" for k, v in health.items())
        + "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="readout-note">Full limitations are in '
        "<code>copilot/knowledge/model_card.md</code>, which is in the copilot's "
        "knowledge base — ask it about them directly.</p>",
        unsafe_allow_html=True,
    )


def main():
    st.markdown(ui.stylesheet(), unsafe_allow_html=True)

    copilot, error = load_copilot()
    if error:
        setup_page(*error)
        return

    health = copilot.health()
    card = copilot.card
    distribution = copilot.bundle.get("score_distribution")

    st.markdown(ui.masthead(health["model_name"], health["index_embedder"]),
                unsafe_allow_html=True)

    if health["index_embedder"] == "hashing":
        st.markdown(
            ui.notice("Index built with the offline stand-in embedder. Retrieval "
                      "will be poor — rebuild with python -m copilot.ingest."),
            unsafe_allow_html=True,
        )
    if not health["llm"] or not health["embedder"]:
        st.markdown(
            ui.notice("Ollama unreachable. Scoring works; explanations do not. "
                      "ollama pull llama3.1:8b · ollama pull nomic-embed-text"),
            unsafe_allow_html=True,
        )

    features = sidebar_inputs()

    render_readout(copilot.score(features), card, distribution)

    st.markdown(ui.eyebrow("Ask"), unsafe_allow_html=True)
    # Three columns collapsed to one word per line on a narrow viewport, so the
    # question takes the full width and the two controls share a row beneath it.
    question = st.text_input(
        "Ask about this prediction", label_visibility="collapsed",
        placeholder="Why does a statin not lower the score? (optional)",
    )
    rerank_col, button_col = st.columns([1, 1], gap="medium")
    with rerank_col:
        use_rerank = st.checkbox(
            "Rerank sources",
            help="Measured on the golden set: hit@1 rises from 0.79 to 0.92, at "
                 "one extra generation per candidate, so it is slower.",
        )
    with button_col:
        explain = st.button("Explain this record", use_container_width=True)

    if explain:
        stream = copilot.stream_explain(features, question=question or None,
                                        use_rerank=use_rerank)
        st.markdown(ui.eyebrow("Explanation"), unsafe_allow_html=True)
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

    if answer is None:
        st.markdown(ui.eyebrow("Explanation"), unsafe_allow_html=True)
        st.markdown(
            '<p class="readout-note">The score above comes from the record in '
            "the sidebar. Press <strong>Explain this record</strong> for the "
            "reasoning behind it, with sources.</p>",
            unsafe_allow_html=True,
        )
    else:
        stale_note = (
            f"Record changed since this ran. It describes {answer.risk:.1%}, "
            "not the score shown. Explain again."
        )
        tabs = st.tabs(["Explanation", "Factors", "Sources", "Model"])

        with tabs[0]:
            if stale:
                st.markdown(ui.notice(stale_note), unsafe_allow_html=True)
            if not answer.grounded:
                st.markdown(ui.notice("Not grounded in retrieved references."),
                            unsafe_allow_html=True)
            if not st.session_state.get("verified", True):
                st.markdown(
                    ui.notice("Citations were checked once streaming finished. "
                              "The text below is the checked version.",
                              quiet=True),
                    unsafe_allow_html=True,
                )
            st.markdown(f'<div class="prose">{answer.text}</div>',
                        unsafe_allow_html=True)
            for warning in answer.warnings:
                st.markdown(ui.notice(warning, quiet=True),
                            unsafe_allow_html=True)

            if answer.citations:
                st.markdown(ui.eyebrow("References"), unsafe_allow_html=True)
                for citation in answer.citations:
                    url = citation["url"]
                    title = f"{citation['title']} — {citation['section']}"
                    if url.startswith("http"):
                        title = (f'<a href="{html.escape(url)}">'
                                 f'{html.escape(title)}</a>')
                    else:
                        title = html.escape(title)
                    st.markdown(
                        f'<div class="ref"><span class="ref-n">'
                        f'{citation["n"]}</span><span>{title} '
                        f'<span class="ref-score">{citation["score"]}</span>'
                        "</span></div>",
                        unsafe_allow_html=True,
                    )

        with tabs[1]:
            if stale:
                st.markdown(ui.notice(stale_note), unsafe_allow_html=True)
            render_factors(answer.factors)

        with tabs[2]:
            render_sources(answer, st.session_state.get("query"))

        with tabs[3]:
            render_model_card(card, health)

    st.markdown(
        f'<p class="readout-note" style="font-family:{ui.MONO},monospace;'
        f'font-size:0.68rem;letter-spacing:0.05em;margin-top:2.5rem;'
        f'border-top:1px solid {ui.RULE};padding-top:0.6rem">{DISCLAIMER}'
        + (f"<br>retrieval {describe(copilot.retriever.store.calibration)}"
           if copilot.retriever.store.calibration else "")
        + "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
