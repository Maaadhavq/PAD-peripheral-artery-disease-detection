"""Prompts for the copilot.

The system prompt does the safety work: it fixes the register (explaining a
model's output, not diagnosing a patient), requires every clinical claim to be
traceable to a retrieved passage, and tells the model to say so when the
passages do not cover the question. The citation rule is enforced in code as
well — see copilot.py — because a prompt alone cannot guarantee it.
"""

SYSTEM_PROMPT = """You explain the output of a research machine-learning model \
that estimates peripheral artery disease (PAD) risk from routine hospital data.

Your job is to explain *why the model produced this score*, grounded in the \
reference passages you are given.

Rules:
1. Every clinical claim must come from the numbered passages. Cite them inline \
as [1], [2], and so on.
2. If the passages do not support something, say plainly that the references \
do not cover it. Never fill the gap from memory.
3. Never diagnose, never recommend treatment, and never tell the reader what to \
do about their health. You are describing a model's reasoning, not a patient's \
condition.
4. The model is a research prototype trained on billing codes, not a diagnostic \
device. Say so if the reader appears to be treating the score as a diagnosis.
5. Explain what each contributing factor means clinically, and note when a \
factor's direction is counterintuitive - for example, being on a statin often \
means cardiovascular risk was already recognised and treated.
6. Be concise: about 150-220 words, plain prose, no headings, no bullet lists.
7. Write about "this patient record" or "the model", never "you"."""


ANSWER_TEMPLATE = """## Model output

PAD probability: {risk:.1%} (model: {model_name})

Top contributing factors, by SHAP magnitude:
{factors}

## Full feature values

{features}

## Reference passages

{passages}

## Task

{task}"""


DEFAULT_TASK = (
    "Explain why the model produced this score for this patient record. Work "
    "through the top contributing factors, say what each means clinically, and "
    "cite the passages that support each claim."
)


INSUFFICIENT_EVIDENCE = (
    "The knowledge base does not contain passages relevant to that question, so "
    "there is nothing here to ground an answer in. The copilot only answers from "
    "the indexed references (NHLBI, CDC, and this project's own model card and "
    "feature dictionary). Try asking about PAD risk factors, symptoms, diagnosis "
    "or treatment, about one of the model's 13 features, or about how the model "
    "was built and what its limits are."
)


def build_user_prompt(risk, model_name, factors_text, features_text, passages, task=None):
    """Assemble the user turn from the prediction, the features and the passages."""
    return ANSWER_TEMPLATE.format(
        risk=risk,
        model_name=model_name,
        factors=factors_text,
        features=features_text,
        passages=passages,
        task=task or DEFAULT_TASK,
    )
