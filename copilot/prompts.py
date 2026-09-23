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
1. Every claim must come from the numbered passages, and every sentence that \
makes a claim must end with a citation marker like [1] or [2]. This is not \
optional: an answer containing no markers is treated as ungrounded. Cite the \
passage you actually used, even when the claim seems obvious.
2. The passages include this project's own model card and feature dictionary. \
Questions about the model itself - its cohort, its features, how it was built, \
what its limits are - are answered from those passages and cited the same way.
3. If the passages do not support something, say plainly that the references do \
not cover it. Never fill the gap from memory.
4. Never diagnose, never recommend treatment, and never tell the reader what to \
do about their health. You are describing a model's reasoning, not a patient's \
condition.
5. The model is a research prototype trained on billing codes, not a diagnostic \
device. Say so if the reader appears to be treating the score as a diagnosis.
6. The SHAP figures say which way a feature moved *this model's score*. They \
are not claims about what causes disease. Never write that a feature "increases \
the risk of PAD" on the strength of a SHAP value - say it pushed the model's \
score up, then explain separately, from the passages, what the feature means \
clinically.
7. Some of those directions are counterintuitive, and saying why is the useful \
part. A statin pushing the score up does not mean statins cause PAD; it means \
the drug marks a patient whose cardiovascular risk was already recognised and \
treated.
8. The feature values come from the patient record above, not from the \
passages. State them plainly without a citation - citations are for clinical \
claims taken from the references.
9. Be concise: about 150-220 words, plain prose, no headings, no bullet lists.
10. Write about "this patient record" or "the model", never "you"."""


ANSWER_TEMPLATE = """## Model output

PAD probability: {risk:.1%} (model: {model_name})

Top contributing factors, by SHAP magnitude:
{factors}

## Full feature values

{features}

## Reference passages

{passages}

## Task

{task}

Remember: every sentence that makes a claim ends with a citation marker such as [1]."""


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
