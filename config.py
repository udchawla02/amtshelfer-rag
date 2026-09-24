"""Central configuration for AmtsHelfer.

Env-driven (see .env.example) so one codebase runs four backends
(openai | gemini | ollama | hf) and toggles the advanced retrieval features without edits.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- backend selection -------------------------------------------------
    backend: str = "ollama"  # openai | gemini | ollama | hf

    # --- OpenAI ------------------------------------------------------------
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # --- Google Gemini (free tier, no credit card; works on Streamlit Cloud)
    google_api_key: str | None = None
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "models/text-embedding-004"

    # --- Ollama (local, free) ---------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # --- HF local embeddings ----------------------------------------------
    hf_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- chunking ----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- retrieval (hybrid + rerank) --------------------------------------
    use_hybrid: bool = True      # BM25 (keyword) + dense (semantic) ensemble
    use_reranker: bool = True    # cross-encoder reranks the merged candidates
    dense_k: int = 8             # candidates from FAISS before rerank
    bm25_k: int = 8              # candidates from BM25 before rerank
    top_k: int = 4               # final chunks passed to the LLM
    ensemble_weight_bm25: float = 0.4
    ensemble_weight_dense: float = 0.6
    reranker_model: str = "BAAI/bge-reranker-base"  # multilingual cross-encoder

    # --- generation --------------------------------------------------------
    answer_language: str = "English"  # default; overridable per request

    # --- paths -------------------------------------------------------------
    docs_dir: Path = Path("data/documents")
    index_dir: Path = Path("data/index")
    chunks_file: Path = Path("data/index/chunks.pkl")


settings = Settings()
