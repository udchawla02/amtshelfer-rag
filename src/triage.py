"""Letter triage: rate an incoming German document red / yellow / green.

Paste the text of a letter (or its OCR) and get back a structured verdict:
urgency level, the deadline if any, why, and the recommended next action -- in
the user's language. This is the feature most "chat over docs" clones skip, and
it maps directly to the real fear: "is this letter urgent and what do I do?".

The model is told to never invent facts and to treat a Mahnung (payment demand)
as urgent. Output is strict JSON, parsed defensively (see src/formatting).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from src.formatting import parse_triage_json
from src.models import get_llm

TRIAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You triage official German letters for someone who may not read German well.
Read the letter and return ONLY a JSON object, no prose, no code fences, with keys:
  "level": one of "red", "yellow", "green"
     red    = urgent, money owed or a hard legal deadline (e.g. Mahnung, fines)
     yellow = action needed but not an emergency (e.g. registration, appointment)
     green  = informational only, no action required
  "title": a 3-6 word summary of what the letter is
  "deadline": the deadline as written, or null if none is stated
  "reason": one sentence explaining the level, citing what the letter actually says
  "action": the single most important next step the reader should take
Rules: never invent a deadline or amount that is not in the text. Treat any Mahnung
as red. If no date is given but one is implied, say so in "reason". Write "title",
"reason" and "action" in {language}.""",
        ),
        ("human", "Letter text:\n\n{letter}"),
    ]
)


@dataclass
class Triage:
    level: str
    title: str
    deadline: str | None
    reason: str
    action: str

    @property
    def emoji(self) -> str:
        return {"red": "🔴", "yellow": "🟡", "green": "🟢"}.get(self.level, "⚪")


def triage_letter(letter_text: str, language: str | None = None) -> Triage:
    language = language or settings.answer_language
    chain = TRIAGE_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"letter": letter_text, "language": language})
    data = parse_triage_json(raw)
    return Triage(
        level=str(data.get("level", "yellow")).lower(),
        title=data.get("title", "Unknown document"),
        deadline=data.get("deadline"),
        reason=data.get("reason", ""),
        action=data.get("action", ""),
    )


if __name__ == "__main__":
    import sys

    text = sys.stdin.read() if not sys.stdin.isatty() else input("Paste letter text: ")
    t = triage_letter(text)
    print(f"\n{t.emoji}  {t.level.upper()} — {t.title}")
    if t.deadline:
        print(f"Deadline: {t.deadline}")
    print(f"Why: {t.reason}")
    print(f"Do this: {t.action}")
