"""Streamlit front end for the PAD risk copilot.

    streamlit run app.py

Needs a trained model (run pad_model.ipynb) and a knowledge index
(python -m copilot.ingest). If either is missing the page explains how to
create it rather than failing with a traceback.
"""

import streamlit as st

from copilot.copilot import PadCopilot
from copilot.llm import OllamaClient
from copilot.retrieve import Retriever
from pad.explain import FEATURE_LABELS

st.set_page_config(page_title="PAD Risk Copilot", page_icon="🫀", layout="wide")

# Synthetic, not drawn from MIMIC-IV. Real patient rows must not be shipped in
# the app under the PhysioNet data use agreement.
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
}

NUMERIC_FIELDS = [
    ("age_at_admission", "Age at admission", 18, 100, 1.0, "years"),
    ("cholesterol", "Total cholesterol", 80, 400, 1.0, "mg/dL"),
    ("glucose", "Glucose", 40, 500, 1.0, "mg/dL"),
    ("creatinine", "Creatinine", 0.2, 12.0, 0.1, "mg/dL"),
    ("hemoglobin", "Hemoglobin", 5.0, 20.0, 0.1, "g/dL"),
    ("platelet_count", "Platelet count", 20.0, 800.0, 5.0, "K/uL"),
]

BINARY_FIELDS = [
    ("has_diabetes", "Diabetes"),
    ("has_hypertension", "Hypertension"),
    ("has_heart_disease", "Heart disease"),
    ("has_stroke_history", "Stroke history"),
    ("is_on_statin", "On a statin"),
    ("is_on_antiplatelet", "On an antiplatelet"),
]


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
            "Train a model first. Put the MIMIC-IV files in `mimic_data/`, then run "
            "`pad_model.ipynb` end to end. It writes `artifacts/model.joblib`."
        )
    else:
        st.markdown("Build the knowledge index:")
        st.code("python -m copilot.ingest", language="bash")
        st.caption(
            "Needs Ollama running with `nomic-embed-text` pulled. For an offline "
            "smoke test use `--provider hashing` — retrieval quality will be poor."
        )


def sidebar_inputs():
    """Feature form. Returns the patient dict."""
    st.sidebar.header("Patient record")

    preset = st.sidebar.selectbox(
        "Load a demo patient", ["Custom"] + list(DEMO_PATIENTS),
        help="Synthetic profiles — no MIMIC-IV records are shipped with this app.",
    )
    defaults = DEMO_PATIENTS.get(preset, DEMO_PATIENTS["Mixed signals"])

    values = {}
    values["gender"] = 1 if st.sidebar.radio(
        "Gender", ["Male", "Female"], index=0 if defaults["gender"] == 1 else 1,
        horizontal=True,
    ) == "Male" else 0

    for key, label, low, high, step, unit in NUMERIC_FIELDS:
        values[key] = st.sidebar.number_input(
            f"{label} ({unit})", min_value=float(low), max_value=float(high),
            value=float(defaults[key]), step=float(step), key=f"{preset}_{key}",
        )

    st.sidebar.markdown("**History and medications**")
    st.sidebar.caption("From admissions before this one.")
    for key, label in BINARY_FIELDS:
        values[key] = int(st.sidebar.toggle(
            label, value=bool(defaults[key]), key=f"{preset}_{key}"
        ))

    return values


def render_risk(answer):
    left, right = st.columns([1, 2])
    with left:
        st.metric("PAD probability", f"{answer.risk:.1%}")
        st.progress(min(max(answer.risk, 0.0), 1.0))
        st.caption(f"Model: {answer.model_name}")
    with right:
        st.markdown("**Top contributing factors**")
        for factor in answer.factors:
            arrow = "▲" if factor["direction"] == "increases" else "▼"
            colour = "#c0392b" if factor["direction"] == "increases" else "#2471a3"
            value = "missing" if factor["value"] is None else f"{factor['value']:g}"
            width = min(abs(factor["shap"]) / max(
                abs(answer.factors[0]["shap"]), 1e-9), 1.0) * 100
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                f"<span style='width:170px'>{factor['label']} "
                f"<code>{value}</code></span>"
                f"<span style='color:{colour}'>{arrow}</span>"
                f"<span style='background:{colour};height:10px;width:{width * 1.6}px;"
                f"display:inline-block;border-radius:2px'></span>"
                f"<span style='color:#888;font-size:0.8em'>{factor['shap']:+.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_answer(answer):
    st.subheader("Explanation")
    if not answer.grounded:
        st.warning("This answer is not grounded in retrieved references.")
    st.markdown(answer.text)

    for warning in answer.warnings:
        st.caption(f"⚠️ {warning}")

    if answer.citations:
        st.markdown("**References**")
        for citation in answer.citations:
            url = citation["url"]
            label = f"[{citation['n']}] {citation['title']} — {citation['section']}"
            if url.startswith("http"):
                st.markdown(f"{label} · [source]({url}) · score {citation['score']}")
            else:
                st.markdown(f"{label} · `{url}` · score {citation['score']}")

    if answer.passages:
        with st.expander(f"Retrieved passages ({len(answer.passages)})"):
            for index, passage in enumerate(answer.passages, start=1):
                st.markdown(
                    f"**[{index}] {passage['source_title']} — {passage['section']}** "
                    f"(score {passage['score']:.3f})"
                )
                st.caption(passage["text"])


def main():
    copilot, error = load_copilot()
    if error:
        setup_page(*error)
        return

    st.title("🫀 PAD Risk Copilot")
    st.caption(
        "A research demo: a model trained on MIMIC-IV billing codes, explained "
        "against public clinical references. Not a medical device."
    )

    features = sidebar_inputs()

    question = st.text_input(
        "Ask about this prediction (optional)",
        placeholder="e.g. Why does being on a statin not lower the score?",
    )

    if not st.button("Explain this record", type="primary"):
        risk = copilot.score(features)
        st.metric("PAD probability", f"{risk:.1%}")
        st.info("Press **Explain this record** for a grounded explanation.")
        return

    llm = copilot.llm
    if isinstance(llm, OllamaClient) and not llm.is_available():
        st.error(
            f"Ollama is not reachable at `{llm.host}`. Start it, then pull the "
            f"models:\n\n`ollama pull {llm.model}` and `ollama pull nomic-embed-text`"
        )
        st.metric("PAD probability", f"{copilot.score(features):.1%}")
        return

    with st.spinner("Scoring, retrieving references and generating..."):
        answer = copilot.explain(features, question=question or None)

    render_risk(answer)
    st.divider()
    render_answer(answer)

    st.divider()
    st.caption(
        "Research demo on MIMIC-IV data. Not a medical device, not validated for "
        "clinical use, and not a substitute for professional medical advice."
    )


if __name__ == "__main__":
    main()
