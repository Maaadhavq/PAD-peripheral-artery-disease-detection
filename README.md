# PAD Prediction Model + Risk Copilot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maaadhavq/PAD-peripheral-artery-disease-detection/blob/main/pad_model.ipynb)

> 🚧 **Note:** This project is under active development.

Two halves of one project:

1. **A machine learning pipeline** that predicts **Peripheral Artery Disease (PAD)** from routine [MIMIC-IV](https://physionet.org/content/mimiciv/) hospital data — labs, demographics, comorbidities and medications.
2. **A RAG copilot** that turns a prediction into an explanation a reader can check: SHAP attributions for *why* the score came out that way, plus a natural-language answer grounded in public clinical references, with citations verified in code.

Everything runs locally. The language model is a local Ollama model, not an API — MIMIC-derived values must not be sent to a third party under the PhysioNet data use agreement.

---

## Architecture

```
MIMIC-IV tables
      │
      ▼
  pad/  ───────────────────────────────────────────────┐
   cohort.py     matched case-control cohort            │
   features.py   13 features, leakage-controlled        │
   train.py      grouped CV → best model                │
   calibrate.py  isotonic/Platt on a held-out split     │
   explain.py    SHAP attributions                      │
   analysis.py   CIs, subgroups, ablation, thresholds   │
      │                                                 │
      ▼                                        pad_model.ipynb (train)
  artifacts/model.joblib                       analysis.ipynb  (analyse)
      │
      ▼
  copilot/  ────────────────────────────────────────────┐
   ingest.py        fetch → chunk → embed → index        │
   threshold.py     measure when to refuse               │
   retrieve.py      cosine search, question + factors    │
   rerank.py        LLM reranking (opt-in)               │
   llm.py           local Ollama client, streaming       │
   faithfulness.py  is the cited passage actually used?  │
   copilot.py       score → SHAP → retrieve → answer     │
      │                  → verify every citation         │
      ▼                                                  │
   app.py           Streamlit UI ◄─────────────────────┘
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
- Calibrates the winner on a held-out split and evaluates it once on a test set

### Avoiding leakage

Most of the engineering went into making sure the model cannot cheat. Each of these is covered by a test that was checked to fail against the earlier behaviour.

- **Patient-level cohort** — controls come from patients with *no* PAD code in any admission, so the same person never appears as both a case and a control. Each control patient contributes one admission.
- **Patient-level splits** — fit, calibration and test are disjoint by `subject_id`, so a patient with several admissions cannot straddle them.
- **Model selection on cross-validation** — the winner is chosen by grouped 5-fold CV. The test set is scored once, afterwards, so the headline number is not the best of six attempts at the same data.
- **Preprocessing inside the pipeline** — imputation and scaling are pipeline steps, refit per fold, so they never see validation or test data.
- **No future information** — comorbidities and medications come from strictly earlier admissions. ICD codes are assigned at discharge, and statins and antiplatelets are the standard PAD treatment, so counting either from the index stay would feed the model a consequence of the diagnosis it is meant to predict. Labs come from the index admission, which is what is measurable at presentation.
- **Matched controls** — 1:1 on age bracket × gender, so the model has to learn something clinical rather than "older male".
- **Lab de-duplication** — several `itemid`s map to one lab label (point-of-care vs laboratory glucose); they are averaged, not silently dropped.
- **Drug names, not substrings** — the statin pattern lists explicit drug names. A bare `statin` pattern also matches **nystatin**, an antifungal.

### Calibration

Discrimination and calibration are different things. The cohort is matched 1:1 by construction, so the training base rate is an artefact rather than a prevalence — an uncalibrated "0.9" was never a nine-in-ten chance, but the app shows the number to a reader who will read it that way.

The model is therefore fitted on one split, calibrated on a second it never saw (isotonic, or Platt below 200 rows), and scored once on a third. The notebook reports a reliability curve, Brier score, ECE and MCE before and after, and the calibrated pipeline is what ships.

### Analysis

`analysis.ipynb` covers what a single AUC hides:

- **Bootstrap confidence interval** on test AUC — a point estimate on one split carries no uncertainty
- **Subgroup performance** by gender and age bracket, with group sizes; subgroups too small to estimate are marked as such rather than given a number that reads as evidence
- **Feature-group ablation** by cross-validation on the training split
- **Baseline comparison** against age + gender alone, so the clinical features have to show their worth
- **Threshold table** — the sensitivity/specificity tradeoff this project invokes when it argues for AUC

### Running it

**Colab:** open the notebook via the badge above and put the MIMIC-IV files in `My Drive/mimic_data/`.

**Locally:**

```bash
pip install -r requirements.txt
```

Put the files in `mimic_data/` next to the notebook and run `pad_model.ipynb` — it detects it is not on Colab. It writes `artifacts/model.joblib` (the calibrated pipeline, a SHAP background sample and the held-out score distribution) and `artifacts/model_card.json`. Then run `analysis.ipynb`.

---

## Part 2 — the copilot

A prediction on its own is not much use to a reader who cannot check it. For a given patient record the copilot:

1. scores the record with the calibrated pipeline,
2. computes SHAP attributions for the top contributing features,
3. retrieves passages for both the factors and any question asked,
4. asks a local LLM to explain the score **citing those passages**,
5. **verifies every citation** — any `[n]` that does not resolve to a retrieved passage is stripped and flagged.

Step 5 is the part that matters. A model will happily write `[7]` when it was handed four passages, and an unresolvable citation looks exactly like a grounded one. Asking for citations in a prompt is not the same as having them, so the check is enforced in code.

### Two traps worth naming

**SHAP is not causation.** Handed a factor labelled "statin therapy — increases", the model wrote that statins increase the risk of PAD. A SHAP value says which way a feature moved *the model's score*. Factors now read "pushed the score up/down", and the prompt separates score movement from clinical meaning.

**A resolvable citation is not a supported one.** The range check cannot see a valid `[2]` attached to a claim the passage never made. `copilot/faithfulness.py` scores each cited sentence against the passage it cites — lexical overlap always, an LLM judge on request. Both are reported, never used to silently drop sentences.

### Knowing when to refuse

The relevance threshold was a hand-picked `0.25`. Measured against the real index that refused nothing at all: `nomic-embed-text` scores *"what is the capital of France?"* at **0.425**, so every off-topic question sailed through and the refusal branch was dead code.

Cosine similarity scales differently for every embedding model, so the cutoff is now measured at ingest from two probe sets and stored **in the index** alongside the vectors. On the current index it lands at **0.523**.

The probes are deliberately disjoint from the evaluation set — tuning the threshold on the questions used to report refusal accuracy would make that number meaningless.

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

Both models fit together on an 8 GB GPU. Override with `PAD_LLM_MODEL`, `PAD_EMBED_MODEL` and `OLLAMA_HOST`.

---

## Evaluation

```bash
python -m copilot.eval.run_eval                  # retrieval + refusal
python -m copilot.eval.run_eval --with-answers   # generate and check answers
python -m copilot.eval.run_eval --rerank         # does reranking help?
python -m copilot.eval.run_eval --with-answers --judge   # LLM faithfulness judge
```

The golden set holds 24 in-scope questions naming the sources that should answer them, and 8 out-of-scope ones that should be refused. Measured with `nomic-embed-text` and `llama3.1:8b`, k=6:

| Retrieval | Baseline | With reranking |
|---|---|---|
| hit@6 (in scope) | 0.96 (23/24) | **1.00** (24/24) |
| hit@1 (in scope) | 0.63 (15/24) | **0.79** (19/24) |
| False refusals | **0** | **0** |
| Out-of-scope refused | 6/8 | 6/8 |

Reranking costs a generation per candidate, so it is opt-in — but it was measured before being kept, and it recovers everything cleaning the index cost.

**Those baseline numbers used to be higher, and the drop was an improvement.** Four of sixty-six chunks were pure navigation — the breadcrumb and section menus that sit above the first heading on every NHLBI page, which were being indexed as that page's "Overview" and cited as though they were content. Stripping them moved hit@6 from 1.00 to 0.96 and hit@1 from 0.79 to 0.63, because those menus carried the page's section names ("Treatment", "Diagnosis"), so they matched topic queries and scored as hits for the right *source document* while saying nothing at all. The metric was partly measuring junk. A lower honest number beats a higher one built on citing a menu.

Generated answers, same settings:

| Answers | Result |
|---|---|
| Citations valid | **32/32** — no unresolvable citation survived |
| Empty answers | **0/32** |
| Out-of-scope refused | **6/8** |
| In-scope answers citing a source | 18/24 |
| Mean citation overlap | 0.42 |
| Weakly supported cited sentences | 49% |

Three of these deserve honesty rather than a headline.

**"Grounded" varies run to run** — 17, 18 and 22 out of 24 across three runs at temperature 0.2. An 8B model does not follow a citation instruction deterministically, which is precisely why the check lives in code and not only in the prompt.

**Half the cited sentences score weakly on overlap.** That measure is lexical, not semantic, so it over-reports: a sentence can paraphrase its source faithfully and still share few words. It is a screen that says which sentences to go and read, not a verdict. `--judge` runs the semantic version.

**The answer path had to be fixed to refuse at all.** Retrieval refused 6/8 off-topic questions, but `explain()` refused 0/8 — because the question's results were merged with a factor query that describes the patient record and therefore always retrieves something. An off-topic question arrived with a full set of passages and got answered. A question that retrieves nothing now returns nothing. The retrieval eval could not have caught this; only generating the answers did.

**The two off-topic questions that get through are the deliberately adjacent ones** — marathon training and appendicitis symptoms. No threshold separates them: "symptoms of appendicitis" scores 0.656, above several genuine questions, so raising the cutoff would start refusing real ones. The prompt catches both, which is why there are two layers. Embedding similarity alone cannot tell "medical" from "this model's subject", and pretending otherwise would mean overfitting the threshold to the eval set.

An offline stand-in embedder exists for testing without Ollama:

```bash
python -m copilot.ingest --provider hashing
```

It is requested by name and never substituted silently — a hashed bag of words retrieves far worse than a real embedding model, and a quiet fallback would make the copilot look like it works when it does not.

---

## The interface

The app is built as an instrument panel rather than a dashboard. PAD is diagnosed by reading a ratio — the ankle-brachial index — against a marked scale, so the page is organised around a calibrated readout: the score, the band boundaries, and the cohort distribution it is being compared against, all on one axis.

That last element is doing honest work. The cohort is matched 1:1 by construction, so a probability from this model only means something relative to the distribution it came from. Drawing that distribution under the score puts the caveat in front of the reader instead of in a footnote.

Everything else is kept quiet: one arterial red for every positive signal, a desaturated slate for its opposite, lab values in mono because that is how a chart renders them, and explanations set in a serif because clinical prose reads better that way. The theme is fixed rather than following the OS — an instrument has one appearance.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

104 tests against a synthetic MIMIC-shaped fixture, so no credentialed data is needed. The fixture deliberately reproduces the structures that caused bugs: patients with several admissions, PAD codes only on the final admission, `itemid`s sharing a lab label, and nystatin prescriptions.

---

## Project structure

```
├── pad/                  # pipeline: cohort, features, training, calibration, analysis
├── copilot/              # RAG: ingest, threshold, retrieve, rerank, grounding
│   ├── knowledge/        # project-written docs (model card, feature dictionary)
│   └── eval/             # golden set + eval harness
├── tests/                # synthetic fixture + 104 tests
├── pad_model.ipynb       # training driver
├── analysis.ipynb        # CIs, subgroups, ablation, thresholds
├── app.py                # Streamlit copilot UI
└── requirements*.txt
```

---

## Limitations

Worth being blunt about, since the numbers look good:

- **The label is a billing code, not a diagnosis.** Undercoded PAD patients sit in the control pool; miscoded ones inflate the case group.
- **ICU population.** MIMIC-IV patients are sicker than a general population, so these probabilities are not population risk.
- **No ankle-brachial index, no smoking status, no imaging** — smoking is among the strongest PAD risk factors and is not reliably available in these tables.
- **Matched controls change the base rate.** Even calibrated, this is a discrimination score on a constructed cohort, not a prevalence.
- **The copilot runs an 8B model.** It does not follow a citation instruction perfectly (17–22 of 24 across runs), and adjacent-domain questions slip past retrieval. Both are measured above rather than papered over.
- **Faithfulness is screened, not guaranteed.** Citations are checked for resolvability always and for support approximately. Nothing here proves a cited passage entails the claim.

`copilot/knowledge/model_card.md` covers this in full, and it is in the copilot's knowledge base — the assistant can answer questions about its own limits.

## What's next

- Re-run every number on real MIMIC-IV (see the caveat below)
- Multi-turn conversation about one record
- Note-derived features from MIMIC-IV-Note (smoking status, claudication mentions)

## A caveat on the numbers

The ML metrics in this README were produced against a **synthetic fixture**, not real MIMIC-IV — see `data_provenance` in `artifacts/model_card.json`. They demonstrate that the code runs end to end; they say nothing about clinical performance. The retrieval and refusal numbers are real, since the knowledge base is real.

## License

Code is MIT — see [LICENSE](LICENSE). The MIMIC-IV data it processes is governed by the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/).

**This is a research project, not a medical device.** It has not been validated prospectively or externally and must not be used for clinical decisions.
