# PAD Prediction Model + Risk Copilot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maaadhavq/PAD-peripheral-artery-disease-detection/blob/main/pad_model.ipynb)

> 🚧 **Note:** This project is under active development.

Two halves of one project:

1. **A machine learning pipeline** that predicts **Peripheral Artery Disease (PAD)** from routine [MIMIC-IV](https://physionet.org/content/mimiciv/) hospital data — labs, demographics, comorbidities and medications.
2. **A RAG copilot** that turns a prediction into an explanation a reader can actually check: SHAP attributions for *why* the score came out that way, plus a natural-language answer grounded in public clinical references, with citations verified in code.

Everything runs locally. The language model is a local Ollama model, not an API — MIMIC-derived values must not be sent to a third party under the PhysioNet data use agreement.

---

## Architecture

```
MIMIC-IV tables
      │
      ▼
  pad/  ──────────────────────────────────────────────┐
   cohort.py    matched case-control cohort            │
   features.py  13 features, leakage-controlled        │
   train.py     grouped CV → best model → artifacts/   │
   explain.py   SHAP attributions                      │
      │                                                │
      ▼                                                ▼
  artifacts/model.joblib                        pad_model.ipynb
      │                                         (the driver)
      ▼
  copilot/  ───────────────────────────────────────────┐
   ingest.py    fetch → chunk → embed → index          │
   retrieve.py  cosine search over the index           │
   llm.py       local Ollama client                    │
   copilot.py   score → SHAP → retrieve → answer       │
      │              → verify every citation           │
      ▼                                                │
   app.py       Streamlit UI ◄────────────────────────┘
```

---

## Part 1 — the model

### What it does

- Pulls admissions, diagnoses, lab events and prescriptions from MIMIC-IV
- Identifies PAD patients by ICD-9/ICD-10 code and samples a matched control group
- Engineers 13 features across four categories:
  - **Demographics** — age at admission, gender
  - **Labs** — cholesterol, glucose, creatinine, hemoglobin, platelet count
  - **Comorbidities** — diabetes, hypertension, heart disease, stroke history
  - **Medications** — statin and antiplatelet use
- Compares six classifiers — logistic regression, random forest, SVM, MLP, XGBoost, LightGBM
- Picks the winner by cross-validated AUC and evaluates it once on a held-out test set

### Avoiding leakage

Most of the engineering here went into making sure the model cannot cheat. Each of these is covered by a test that was checked to fail against the earlier behaviour.

- **Patient-level cohort** — controls come from patients with *no* PAD code in any admission, so the same person never appears as both a case and a control. Each control patient contributes one admission.
- **Patient-level split** — `GroupShuffleSplit` on `subject_id`, so a patient with several admissions cannot straddle train and test.
- **Model selection on cross-validation** — the winner is chosen by grouped 5-fold CV on the training set. The test set is scored once, afterwards, so the headline number is not the best of six attempts at the same data.
- **Preprocessing inside the pipeline** — imputation and scaling are pipeline steps, refit per fold, so they never see validation or test data.
- **No future information** — comorbidities and medications come from strictly earlier admissions. ICD codes are assigned at discharge, and statins and antiplatelets are the standard PAD treatment, so counting either from the index stay would feed the model a consequence of the diagnosis it is meant to predict. Labs come from the index admission, which is what is measurable at presentation.
- **Matched controls** — 1:1 on age bracket × gender, so the model has to learn something clinical rather than "older male".
- **Lab de-duplication** — several `itemid`s map to one lab label (point-of-care vs laboratory glucose); they are averaged, not silently dropped.
- **Drug names, not substrings** — the statin pattern lists explicit drug names. A bare `statin` pattern also matches **nystatin**, an antifungal.

### Data

Requires [MIMIC-IV](https://physionet.org/content/mimiciv/) and PhysioNet credentialed access:

```
mimic_data/
├── admissions.csv.gz
├── patients.csv.gz
├── diagnoses_icd.csv.gz
├── d_labitems.csv.gz
├── labevents.csv.gz
└── prescriptions.csv.gz
```

### Running it

**Colab:** open the notebook via the badge above, clone the repo in the first cell, and put the MIMIC-IV files in `My Drive/mimic_data/`.

**Locally:**

```bash
pip install -r requirements.txt
```

Put the files in `mimic_data/` next to the notebook and run it — it detects it is not on Colab and reads from there. It writes `artifacts/model.joblib` and `artifacts/model_card.json`, which the copilot loads.

---

## Part 2 — the copilot

A prediction on its own is not much use to a reader who cannot check it. The copilot answers "why this score?" with attribution and sources.

For a given patient record it:

1. scores the record with the trained pipeline,
2. computes SHAP attributions for the top contributing features,
3. builds a retrieval query from those factors and searches the knowledge index,
4. asks a local LLM to explain the score **citing the retrieved passages**,
5. **verifies every citation** — any `[n]` that does not resolve to a retrieved passage is stripped and flagged.

That last step is the part that matters. A model will happily write `[7]` when it was handed four passages, and to a reader an unresolvable citation looks exactly like a grounded one. Asking for citations in the prompt is not the same as having them, so the check is enforced in code.

One related trap is worth naming, because it produced clinically false text before it was fixed: a SHAP value says which way a feature moved *the model's score*, not whether it causes disease. Handed "statin therapy — increases", the model wrote that statins increase the risk of PAD. The factors are now phrased as "pushed the score up/down", and the prompt separates score movement from clinical meaning.

If nothing clears the relevance threshold, the copilot says so and the LLM is never called — no improvising.

### Knowledge base

| Source | Provenance |
|---|---|
| NHLBI — PAD overview, causes, symptoms, diagnosis, treatment | NIH, public domain, fetched at ingest |
| CDC — About Peripheral Arterial Disease | CDC, public domain, fetched at ingest |
| Model card | Written for this project |
| Feature dictionary | Written for this project |

External pages are fetched at ingest rather than committed, so the repo does not redistribute someone else's text and the index tracks the live pages.

### Setup

```bash
pip install -r requirements-app.txt
```

Install [Ollama](https://ollama.com/download), then pull the two models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Build the index and start the app:

```bash
python -m copilot.ingest
streamlit run app.py
```

Both models fit together on an 8 GB GPU. Override with `PAD_LLM_MODEL`, `PAD_EMBED_MODEL` and `OLLAMA_HOST` if you want something else.

### Evaluation

```bash
python -m copilot.eval.run_eval                 # retrieval only
python -m copilot.eval.run_eval --with-answers  # also generate and check answers
```

Scores a 24-question golden set for retrieval `hit@k` and `hit@1`, and checks that every citation in a generated answer resolves to a retrieved passage.

Measured with `nomic-embed-text` embeddings and `llama3.1:8b` generation, k=6:

| Metric | Result |
|---|---|
| Retrieval hit@6 | **24/24** (1.00) |
| Retrieval hit@1 | **19/24** (0.79) |
| Mean top score | 0.749 |
| Citations valid | **24/24** — no unresolvable citation survived |
| Answers citing a source | 22/24 |
| Empty answers | 0/24 |

Two answers out of 24 still make claims without citing anything. They are flagged as ungrounded in the UI rather than quietly passed off as sourced — an 8B model does not follow a citation instruction perfectly, which is exactly why the check is in code.

There is also an offline stand-in embedder for testing without Ollama:

```bash
python -m copilot.ingest --provider hashing
```

It is requested by name and never substituted silently — a hashed bag of words retrieves far worse than a real embedding model, and a quiet fallback would make the copilot look like it works when it does not. The gap is the point: it scores `hit@6 0.79 / hit@1 0.54` against `1.00 / 0.79` for real embeddings.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

54 tests, run against a synthetic MIMIC-shaped fixture, so no credentialed data is needed. The fixture deliberately reproduces the structures that caused bugs: patients with several admissions, PAD codes only on the final admission, `itemid`s sharing a lab label, and nystatin prescriptions.

---

## Project structure

```
├── pad/                  # pipeline: cohort, features, training, SHAP
├── copilot/              # RAG: ingest, retrieve, LLM, grounding
│   ├── knowledge/        # project-written docs (model card, feature dictionary)
│   └── eval/             # golden set + eval harness
├── tests/                # synthetic fixture + 54 tests
├── pad_model.ipynb       # the training driver
├── app.py                # Streamlit copilot UI
└── requirements*.txt
```

---

## Limitations

Worth being blunt about, since the numbers look good:

- **The label is a billing code, not a diagnosis.** Undercoded PAD patients sit in the control pool; miscoded ones inflate the case group.
- **ICU population.** MIMIC-IV patients are sicker than a general population, so these probabilities are not population risk.
- **No ankle-brachial index, no smoking status, no imaging** — smoking is among the strongest PAD risk factors and is not reliably available in these tables.
- **Matched controls change the base rate**, so the output is a discrimination score, not a calibrated probability.
- **Uncalibrated.** No Platt scaling or isotonic regression yet.

`copilot/knowledge/model_card.md` covers this in full, and it is in the copilot's knowledge base, so the assistant can answer questions about its own limits.

## What's next

- Probability calibration (Platt / isotonic)
- Reranking retrieved passages before generation
- Note-derived features from MIMIC-IV-Note (smoking status, claudication mentions)

## License

Code is MIT — see [LICENSE](LICENSE). The MIMIC-IV data it processes is governed by the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/); make sure you have access before using it.

**This is a research project, not a medical device.** It has not been validated prospectively or externally and must not be used for clinical decisions.
