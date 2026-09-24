"""Ingest documents into a FAISS index (+ persist chunks for BM25).

Run:  python -m src.ingest

Loads every PDF / Markdown / text file under data/documents, splits them into
overlapping chunks, embeds them into FAISS, and also pickles the raw chunks so
the hybrid retriever can build a BM25 keyword index over the same text.
"""

from __future__ import annotations

import pickle
import sys

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.models import get_embeddings


def _markdown_loader():
    """Prefer `unstructured` for Markdown, fall back to plain text.

    `unstructured` is a heavy optional dependency: it ships only in
    requirements-local.txt, not in the default requirements.txt that Streamlit
    Community Cloud installs. Reading Markdown as plain text is perfectly
    adequate for chunking, so the app runs either way.
    """
    try:
        import unstructured  # noqa: F401
        from langchain_community.document_loaders import UnstructuredMarkdownLoader

        return UnstructuredMarkdownLoader
    # Any import failure at all should degrade to plain text, not break ingest.
    except Exception:  # noqa: BLE001
        return TextLoader


LOADERS = {
    ".pdf": PyPDFLoader,
    ".md": _markdown_loader(),
    ".txt": TextLoader,
}


def load_documents() -> list:
    docs = []
    files = sorted(p for p in settings.docs_dir.rglob("*") if p.suffix.lower() in LOADERS)
    if not files:
        raise FileNotFoundError(
            f"No documents found in {settings.docs_dir}/. "
            "Drop some PDFs or .md files there and re-run."
        )
    for path in files:
        loader_cls = LOADERS[path.suffix.lower()]
        loaded = loader_cls(str(path)).load()
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
        print(f"  loaded {len(loaded):>3} pages/blocks  <-  {path.name}")
    return docs


def build_index() -> None:
    print(f"Backend: {settings.backend}")
    print(f"Reading documents from {settings.docs_dir}/ ...")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks. Embedding ...")

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.index_dir))

    # Persist chunks so the BM25 half of the hybrid retriever can be rebuilt.
    with settings.chunks_file.open("wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index + {len(chunks)} chunks to {settings.index_dir}/.")


def ensure_index() -> bool:
    """Build the index if it is not on disk yet. Returns True if it built one.

    The index is generated, so it is gitignored and therefore absent on a fresh
    deploy (Streamlit Cloud, a new clone, a container). Building it on first use
    means the app works straight after deployment instead of erroring out.
    """
    if (settings.index_dir / "index.faiss").exists():
        return False
    print("[ingest] no index found -> building it now (first run only) ...")
    build_index()
    return True


if __name__ == "__main__":
    try:
        build_index()
    except FileNotFoundError as exc:
        sys.exit(str(exc))
