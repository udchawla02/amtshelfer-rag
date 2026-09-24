"""Pure helpers with no heavy imports, so they are trivially unit-testable."""

from __future__ import annotations

import json
import re


def format_docs(docs) -> str:
    """Render retrieved documents into a numbered, source-tagged context block."""
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        tag = f"{src}" + (f" p.{page + 1}" if isinstance(page, int) else "")
        blocks.append(f"[{i}] ({tag})\n{d.page_content}")
    return "\n\n".join(blocks)


def parse_triage_json(raw: str) -> dict:
    """Extract a JSON object from an LLM reply that may add fences or prose.

    Defensive on purpose: local models sometimes wrap JSON in ```json fences or
    add a stray sentence. We grab the first {...} block and parse it; on failure
    we return a safe 'yellow' verdict so the UI never crashes.
    """
    if raw is None:
        return {"level": "yellow", "reason": "No response."}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"level": "yellow", "reason": "Could not parse the model output."}
