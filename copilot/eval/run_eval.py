"""Evaluate the copilot: does retrieval find the right source, and do answers cite?

    python -m copilot.eval.run_eval                      # retrieval only
    python -m copilot.eval.run_eval --with-answers       # also generate and check answers

Retrieval is scored against golden.jsonl, where each question names the source
documents that ought to answer it. Two numbers matter:

  hit@k      - the right source appeared anywhere in the top k
  hit@1      - the top passage came from the right source

Answer scoring checks that every [n] marker resolves to a passage that was
actually retrieved, which is the property the copilot enforces in code.
"""

import argparse
import json
from pathlib import Path

from copilot.copilot import CITATION_RE, PadCopilot
from copilot.retrieve import Retriever

GOLDEN_FILE = Path(__file__).parent / "golden.jsonl"


def load_golden(path=GOLDEN_FILE):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_retrieval(retriever, golden, k=6):
    """Per-question hit@k and hit@1 against the expected source documents."""
    rows = []
    for case in golden:
        chunks = retriever.retrieve(case["question"], k=k)
        found = [chunk["source_id"] for chunk in chunks]
        expected = set(case["expect_sources"])

        rows.append({
            "question": case["question"],
            "expected": sorted(expected),
            "retrieved": found,
            "hit_at_k": bool(expected & set(found)),
            "hit_at_1": bool(found and found[0] in expected),
            "n_retrieved": len(found),
            "top_score": round(chunks[0]["score"], 3) if chunks else 0.0,
        })
    return rows


def summarize(rows):
    total = len(rows)
    if not total:
        return {}
    return {
        "questions": total,
        "hit@k": sum(r["hit_at_k"] for r in rows) / total,
        "hit@1": sum(r["hit_at_1"] for r in rows) / total,
        "empty_retrievals": sum(r["n_retrieved"] == 0 for r in rows),
        "mean_top_score": round(sum(r["top_score"] for r in rows) / total, 3),
    }


DEMO_PATIENT = {
    "gender": 1, "age_at_admission": 72, "cholesterol": 215.0, "glucose": 148.0,
    "creatinine": 1.4, "hemoglobin": 11.8, "platelet_count": 240.0,
    "has_diabetes": 1, "has_hypertension": 1, "has_heart_disease": 1,
    "has_stroke_history": 0, "is_on_statin": 1, "is_on_antiplatelet": 1,
}


def evaluate_answers(copilot, golden, limit=None):
    """Generate answers and check that every citation resolves."""
    cases = golden[:limit] if limit else golden
    rows = []
    for case in cases:
        answer = copilot.explain(DEMO_PATIENT, question=case["question"])
        markers = [int(m) for m in CITATION_RE.findall(answer.text)]
        n_passages = len(answer.passages)
        rows.append({
            "question": case["question"],
            "grounded": answer.grounded,
            "n_citations": len(markers),
            "citations_valid": all(1 <= m <= n_passages for m in markers),
            "empty_answer": not answer.text.strip(),
            "warnings": answer.warnings,
            "chars": len(answer.text),
        })
    return rows


def print_table(rows, columns, widths):
    header = "".join(f"{c:<{w}}" for c, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("".join(f"{str(row[c])[:w - 1]:<{w}}" for c, w in zip(columns, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the PAD copilot.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--provider", default=None,
                        help="embedding provider (default: whatever built the index)")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--with-answers", action="store_true",
                        help="also generate answers (needs a local LLM)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    golden = load_golden()
    retriever_kwargs = {"provider": args.provider}
    if args.index_dir:
        retriever_kwargs["index_dir"] = args.index_dir
    retriever = Retriever(**{k: v for k, v in retriever_kwargs.items() if v is not None})

    print(f"Index: {len(retriever.store)} chunks, embedder={retriever.store.embedder_name}")
    print(f"Golden set: {len(golden)} questions, k={args.k}\n")

    rows = evaluate_retrieval(retriever, golden, k=args.k)
    print_table(
        [{**r, "hit_at_k": "yes" if r["hit_at_k"] else "NO",
          "hit_at_1": "yes" if r["hit_at_1"] else "no"} for r in rows],
        ["question", "hit_at_k", "hit_at_1", "top_score"],
        [58, 10, 10, 10],
    )

    summary = summarize(rows)
    print("\nRetrieval summary")
    for key, value in summary.items():
        print(f"  {key:18s} {value:.3f}" if isinstance(value, float) else
              f"  {key:18s} {value}")

    misses = [r for r in rows if not r["hit_at_k"]]
    if misses:
        print(f"\n{len(misses)} miss(es):")
        for row in misses:
            print(f"  {row['question']}")
            print(f"    expected {row['expected']}, got {row['retrieved'][:3]}")

    if args.with_answers:
        print("\nGenerating answers...")
        copilot = PadCopilot(retriever=retriever)
        answer_rows = evaluate_answers(copilot, golden, limit=args.limit)
        print_table(
            [{**r, "grounded": "yes" if r["grounded"] else "NO",
              "citations_valid": "yes" if r["citations_valid"] else "NO"} for r in answer_rows],
            ["question", "grounded", "citations_valid", "n_citations", "chars"],
            [50, 10, 17, 13, 8],
        )
        total = len(answer_rows)
        print("\nAnswer summary")
        print(f"  grounded           {sum(r['grounded'] for r in answer_rows)}/{total}")
        print(f"  citations valid    {sum(r['citations_valid'] for r in answer_rows)}/{total}")
        print(f"  empty answers      {sum(r['empty_answer'] for r in answer_rows)}/{total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
