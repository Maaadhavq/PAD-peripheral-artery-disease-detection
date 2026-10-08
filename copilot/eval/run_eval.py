"""Evaluate the copilot.

    python -m copilot.eval.run_eval                  # retrieval + refusal
    python -m copilot.eval.run_eval --with-answers   # also generate and check answers
    python -m copilot.eval.run_eval --rerank         # measure whether reranking helps
    python -m copilot.eval.run_eval --with-answers --judge   # LLM faithfulness judge

The golden set holds two kinds of question. In-scope ones name the source
documents that ought to answer them, scored by:

  hit@k    the right source appeared in the top k
  hit@1    the top passage came from the right source

Out-of-scope ones expect no sources at all: the copilot should refuse rather
than answer from whatever happened to score highest. Measuring that is the point
of including them, because a threshold tuned only on questions it should answer
will happily answer everything.

Answers are checked three ways: every [n] resolves to a retrieved passage,
every cited sentence shares content with the passage it cites, and optionally an
LLM judge is asked whether the passage supports the claim.
"""

import argparse
import json
from pathlib import Path

from copilot import faithfulness
from copilot.copilot import CITATION_RE, PadCopilot
from copilot.embeddings import EmbeddingUnavailable
from copilot.llm import OllamaClient, OllamaUnavailable
from copilot.rerank import retrieve_and_rerank
from copilot.retrieve import Retriever
from copilot.threshold import describe

GOLDEN_FILE = Path(__file__).parent / "golden.jsonl"

DEMO_PATIENT = {
    "gender": 1, "age_at_admission": 72, "cholesterol": 215.0, "glucose": 148.0,
    "creatinine": 1.4, "hemoglobin": 11.8, "platelet_count": 240.0,
    "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 1,
    "has_stroke_history": 0, "is_on_statin": 1, "is_on_antiplatelet": 1,
}


def load_golden(path=GOLDEN_FILE):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_retrieval(retriever, golden, k=6, reranker_llm=None):
    """Per-question retrieval outcome, split by whether it is in scope."""
    rows = []
    for case in golden:
        question = case["question"]
        if reranker_llm is not None:
            chunks = retrieve_and_rerank(retriever, question, reranker_llm, k=k)
        else:
            chunks = retriever.retrieve(question, k=k)

        found = [chunk["source_id"] for chunk in chunks]
        expected = set(case["expect_sources"])
        in_scope = bool(expected)

        rows.append({
            "question": question,
            "in_scope": in_scope,
            "expected": sorted(expected),
            "retrieved": found,
            "refused": len(chunks) == 0,
            "hit_at_k": bool(expected & set(found)) if in_scope else None,
            "hit_at_1": bool(found and found[0] in expected) if in_scope else None,
            "top_score": round(chunks[0]["score"], 3) if chunks else 0.0,
        })
    return rows


def summarize_retrieval(rows):
    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]

    summary = {}
    if in_scope:
        summary["in_scope_questions"] = len(in_scope)
        summary["hit@k"] = sum(r["hit_at_k"] for r in in_scope) / len(in_scope)
        summary["hit@1"] = sum(r["hit_at_1"] for r in in_scope) / len(in_scope)
        # An in-scope question that gets refused is a false refusal.
        summary["false_refusals"] = sum(r["refused"] for r in in_scope)
    if out_scope:
        summary["out_of_scope_questions"] = len(out_scope)
        summary["refusal_rate"] = sum(r["refused"] for r in out_scope) / len(out_scope)
    return summary


def evaluate_answers(copilot, golden, limit=None, judge=False):
    """Generate answers and check that citations resolve and are supported."""
    cases = golden[:limit] if limit else golden
    rows = []
    for case in cases:
        answer = copilot.explain(DEMO_PATIENT, question=case["question"])
        markers = [int(m) for m in CITATION_RE.findall(answer.text)]
        n_passages = len(answer.passages)

        support = faithfulness.summarize(
            faithfulness.score_answer(answer.text, answer.passages)
        )

        row = {
            "question": case["question"],
            "in_scope": bool(case["expect_sources"]),
            "refused": not answer.passages,
            "grounded": answer.grounded,
            "n_citations": len(markers),
            "citations_valid": all(1 <= m <= n_passages for m in markers),
            "empty_answer": not answer.text.strip(),
            "mean_overlap": support["mean_overlap"],
            "weak_fraction": support["weak_fraction"],
            "chars": len(answer.text),
        }

        if judge and answer.passages:
            verdicts = faithfulness.judge_answer(
                answer.text, answer.passages, copilot.llm, limit=4
            )
            row["judged"] = len(verdicts)
            row["judged_supported"] = sum(v["supported"] for v in verdicts)
        rows.append(row)
    return rows


def print_table(rows, columns, widths):
    header = "".join(f"{c:<{w}}" for c, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("".join(f"{str(row.get(c, ''))[:w - 1]:<{w}}" for c, w in zip(columns, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the PAD copilot.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--with-answers", action="store_true",
                        help="generate answers (needs a local LLM)")
    parser.add_argument("--rerank", action="store_true",
                        help="rerank candidates with the local LLM before scoring")
    parser.add_argument("--judge", action="store_true",
                        help="ask the LLM whether each cited passage supports its claim")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    golden = load_golden()
    kwargs = {}
    if args.provider:
        kwargs["provider"] = args.provider
    if args.index_dir:
        kwargs["index_dir"] = args.index_dir

    try:
        retriever = Retriever(**kwargs)
    except FileNotFoundError as error:
        print(error)
        return 1

    print(f"Index: {len(retriever.store)} chunks, embedder={retriever.store.embedder_name}")
    print(f"Relevance: {describe(retriever.store.calibration)}")
    print(f"Golden set: {len(golden)} questions, k={args.k}"
          f"{', reranked' if args.rerank else ''}\n")

    reranker_llm = OllamaClient() if args.rerank else None

    try:
        rows = evaluate_retrieval(retriever, golden, k=args.k, reranker_llm=reranker_llm)
    except (EmbeddingUnavailable, OllamaUnavailable) as error:
        print(error)
        return 1

    display = []
    for row in rows:
        display.append({
            **row,
            "scope": "in" if row["in_scope"] else "OUT",
            "hit_at_k": "-" if row["hit_at_k"] is None else ("yes" if row["hit_at_k"] else "NO"),
            "hit_at_1": "-" if row["hit_at_1"] is None else ("yes" if row["hit_at_1"] else "no"),
            "refused": "REFUSED" if row["refused"] else "",
        })
    print_table(display, ["question", "scope", "hit_at_k", "hit_at_1", "refused", "top_score"],
                [52, 7, 10, 10, 10, 10])

    summary = summarize_retrieval(rows)
    print("\nRetrieval summary")
    for key, value in summary.items():
        print(f"  {key:24s} {value:.3f}" if isinstance(value, float)
              else f"  {key:24s} {value}")

    misses = [r for r in rows if r["in_scope"] and not r["hit_at_k"]]
    if misses:
        print(f"\n{len(misses)} in-scope miss(es):")
        for row in misses:
            print(f"  {row['question']}")
            print(f"    expected {row['expected']}, got {row['retrieved'][:3]}")

    leaked = [r for r in rows if not r["in_scope"] and not r["refused"]]
    if leaked:
        print(f"\n{len(leaked)} out-of-scope question(s) NOT refused:")
        for row in leaked:
            print(f"  {row['top_score']:.3f}  {row['question']}")

    if args.with_answers:
        print("\nGenerating answers...")
        copilot = PadCopilot(retriever=retriever)
        answer_rows = evaluate_answers(copilot, golden, limit=args.limit, judge=args.judge)

        columns = ["question", "scope", "grounded", "citations_valid", "mean_overlap"]
        widths = [44, 7, 10, 17, 14]
        if args.judge:
            columns += ["judged_supported", "judged"]
            widths += [18, 8]
        print_table(
            [{**r, "scope": "in" if r["in_scope"] else "OUT",
              "grounded": "yes" if r["grounded"] else ("refused" if r["refused"] else "NO"),
              "citations_valid": "yes" if r["citations_valid"] else "NO"}
             for r in answer_rows],
            columns, widths,
        )

        in_scope = [r for r in answer_rows if r["in_scope"]]
        total = len(answer_rows)
        print("\nAnswer summary")
        print(f"  citations valid          {sum(r['citations_valid'] for r in answer_rows)}/{total}")
        print(f"  empty answers            {sum(r['empty_answer'] for r in answer_rows)}/{total}")
        if in_scope:
            grounded = sum(r["grounded"] for r in in_scope)
            cited = [r for r in in_scope if r["n_citations"]]
            print(f"  in-scope grounded        {grounded}/{len(in_scope)}")
            if cited:
                mean_overlap = sum(r["mean_overlap"] for r in cited) / len(cited)
                weak = sum(r["weak_fraction"] for r in cited) / len(cited)
                print(f"  mean citation overlap    {mean_overlap:.3f}")
                print(f"  weakly supported         {weak:.1%} of cited sentences")
        out_scope = [r for r in answer_rows if not r["in_scope"]]
        if out_scope:
            print(f"  out-of-scope refused     {sum(r['refused'] for r in out_scope)}/{len(out_scope)}")
        if args.judge:
            judged = sum(r.get("judged", 0) for r in answer_rows)
            supported = sum(r.get("judged_supported", 0) for r in answer_rows)
            if judged:
                print(f"  judge: supported         {supported}/{judged} ({supported / judged:.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
