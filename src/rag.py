"""The RAG chain: hybrid retrieval -> rerank -> grounded, multilingual answer.

Loads the retrieval pipeline (see src/retriever.py), fetches the best chunks for
a question, and asks the LLM to answer ONLY from those chunks, in the requested
language, with citations. Grounded-or-refuse behaviour is deliberate: for
admin/medical questions a confident wrong answer is worse than "check with the
office".
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from src.formatting import format_docs
from src.models import get_llm
from src.retriever import build_retriever

SYSTEM_PROMPT = """You are AmtsHelfer, an assistant that helps people in Germany \
understand administrative and health-insurance documents.

Rules:
- Answer ONLY using the context below. Do not use outside knowledge.
- If the context does not contain the answer, say so plainly and suggest which \
office or document the user should check. Never invent rules, deadlines, or amounts.
- Quote the exact figure, deadline, or clause when the user asks for one.
- Keep answers short and concrete. Explain German terms in plain language.
- Write your entire answer in {language}, even though the documents are in German.
- End with a "Sources:" line listing the document names you used.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


@dataclass
class Answer:
    text: str
    sources: list[str]


class RagChain:
    """Callable wrapper so the UI/API get both the answer text and its sources."""

    def __init__(self):
        self.retriever = build_retriever()
        self.llm = get_llm()
        self.chain = (
            {
                "context": (lambda x: x["question"]) | self.retriever | format_docs,
                "question": lambda x: x["question"],
                "language": lambda x: x["language"],
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str, language: str | None = None) -> Answer:
        language = language or settings.answer_language
        payload = {"question": question, "language": language}
        docs = self.retriever.invoke(question)
        text = self.chain.invoke(payload)
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        return Answer(text=text, sources=sources)


if __name__ == "__main__":
    chain = RagChain()
    print("AmtsHelfer ready. Ask a question (Ctrl-C to quit).\n")
    try:
        while True:
            q = input("you > ").strip()
            if not q:
                continue
            ans = chain.ask(q)
            print(f"\n{ans.text}\n(retrieved from: {', '.join(ans.sources)})\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
