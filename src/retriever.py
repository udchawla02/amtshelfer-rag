"""Hybrid retrieval + cross-encoder reranking.

Pipeline (each stage is flag-controlled in config):

    query
      |-- BM25 (keyword)  --\
      |                      >-- EnsembleRetriever (RRF-style merge) --\
      |-- FAISS (semantic) -/                                          |
                                                                       v
                                    CrossEncoderReranker (top_k) --> chunks

Why this matters: dense-only retrieval misses exact tokens like form numbers,
paragraph refs, or German compound terms ("Wohnungsgeberbestaetigung"); BM25
catches those. The reranker then orders the merged pool by true relevance, which
is what lifts answer quality above a typical single-retriever clone.

Everything degrades gracefully: if hybrid/rerank are off (or their optional deps
are missing) it falls back to plain dense retrieval, so the app always runs.
"""

from __future__ import annotations

import pickle

from langchain_community.vectorstores import FAISS

from config import settings
from src.models import get_embeddings


def _load_faiss() -> FAISS:
    # Build the index on first use if it is missing (fresh deploy or clone).
    from src.ingest import ensure_index

    ensure_index()
    return FAISS.load_local(
        str(settings.index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,  # index is generated locally by us
    )


def _load_chunks() -> list:
    if not settings.chunks_file.exists():
        return []
    with settings.chunks_file.open("rb") as f:
        return pickle.load(f)


def _base_retriever():
    """Dense-only, or a BM25+dense ensemble if hybrid is enabled."""
    faiss = _load_faiss()
    dense = faiss.as_retriever(search_kwargs={"k": settings.dense_k})

    if not settings.use_hybrid:
        return dense

    chunks = _load_chunks()
    if not chunks:
        print("[retriever] no chunks.pkl found -> falling back to dense-only.")
        return dense

    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.bm25_k
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[settings.ensemble_weight_bm25, settings.ensemble_weight_dense],
    )


def build_retriever():
    """Return the full retrieval pipeline, adding a reranker if enabled."""
    base = _base_retriever()

    if not settings.use_reranker:
        return base

    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError:
        print("[retriever] reranker deps missing -> skipping rerank.")
        return base

    encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    reranker = CrossEncoderReranker(model=encoder, top_n=settings.top_k)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base,
    )
