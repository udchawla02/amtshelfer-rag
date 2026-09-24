"""FastAPI service for AmtsHelfer.

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs

Endpoints:
  POST /ask     -> grounded answer + sources (with optional language)
  POST /triage  -> red/yellow/green urgency verdict for a pasted letter
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import RagChain
from src.triage import triage_letter

app = FastAPI(title="AmtsHelfer API", version="2.0.0")
_chain: RagChain | None = None


def chain() -> RagChain:
    global _chain
    if _chain is None:
        _chain = RagChain()
    return _chain


class Query(BaseModel):
    question: str
    language: str | None = None


class Answer(BaseModel):
    answer: str
    sources: list[str]


class Letter(BaseModel):
    text: str
    language: str | None = None


class TriageResult(BaseModel):
    level: str
    title: str
    deadline: str | None
    reason: str
    action: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(query: Query) -> Answer:
    ans = chain().ask(query.question, language=query.language)
    return Answer(answer=ans.text, sources=ans.sources)


@app.post("/triage", response_model=TriageResult)
def triage(letter: Letter) -> TriageResult:
    t = triage_letter(letter.text, language=letter.language)
    return TriageResult(
        level=t.level, title=t.title, deadline=t.deadline, reason=t.reason, action=t.action
    )
