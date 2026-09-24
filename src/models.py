"""Model factories: swap OpenAI / Gemini / local without touching the RAG logic.

Keeping this in one place is the small bit of engineering that turns a notebook
into something maintainable: every other module just asks for `get_llm()` /
`get_embeddings()` and never cares which provider is behind it.
"""

from __future__ import annotations

from config import settings


def get_embeddings():
    if settings.backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
        )
    if settings.backend == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embed_model,
            google_api_key=settings.google_api_key,
        )
    if settings.backend == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    if settings.backend == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_embed_model)

    raise ValueError(f"Unknown backend: {settings.backend!r}")


def get_llm():
    if settings.backend in ("openai", "hf"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
    if settings.backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
    if settings.backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )

    raise ValueError(f"Unknown backend: {settings.backend!r}")
