"""A small, dependency-light evaluation harness.

Most portfolio RAG repos stop at "it answers questions". Adding even a simple
eval loop signals that you think about quality, not just plumbing. This measures
two things over a hand-written question set (data/eval.jsonl):

  * retrieval hit-rate: did the chunk that contains the answer get retrieved?
  * keyword groundedness: does the answer contain the expected key fact?

Swap this for RAGAS / an LLM-judge later; the interface stays the same.

Run:  python -m src.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag import RagChain

EVAL_FILE = Path("data/eval.jsonl")


def load_eval() -> list[dict]:
    if not EVAL_FILE.exists():
        raise SystemExit(
            f"{EVAL_FILE} not found. Create it with lines like:\n"
            '{"question": "...", "expected_source": "faq.md", "must_contain": "14 days"}'
        )
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


def main() -> None:
    chain = RagChain()
    cases = load_eval()
    hits, grounded = 0, 0

    for c in cases:
        ans = chain.ask(c["question"])
        got_source = c["expected_source"] in ans.sources
        got_fact = c["must_contain"].lower() in ans.text.lower()
        hits += got_source
        grounded += got_fact
        mark = "OK " if (got_source and got_fact) else "-- "
        print(f"{mark} {c['question'][:60]:60s}  src={got_source}  fact={got_fact}")

    n = len(cases)
    print("\n--- summary ---")
    print(f"retrieval hit-rate : {hits}/{n}  ({hits / n:.0%})")
    print(f"groundedness       : {grounded}/{n}  ({grounded / n:.0%})")


if __name__ == "__main__":
    main()
