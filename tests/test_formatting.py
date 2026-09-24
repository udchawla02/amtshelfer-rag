"""Offline unit tests — no models, no network, so CI is fast and free."""

from __future__ import annotations

from types import SimpleNamespace

from src.formatting import format_docs, parse_triage_json


def _doc(content, source, page=None):
    meta = {"source": source}
    if page is not None:
        meta["page"] = page
    return SimpleNamespace(page_content=content, metadata=meta)


def test_format_docs_numbers_and_tags_sources():
    docs = [_doc("hello", "faq.md"), _doc("world", "brief.pdf", page=2)]
    out = format_docs(docs)
    assert "[1] (faq.md)" in out
    assert "[2] (brief.pdf p.3)" in out  # page is 0-indexed internally
    assert "hello" in out and "world" in out


def test_format_docs_handles_missing_source():
    out = format_docs([SimpleNamespace(page_content="x", metadata={})])
    assert "unknown" in out


def test_parse_triage_plain_json():
    data = parse_triage_json('{"level": "red", "action": "Pay now"}')
    assert data["level"] == "red"
    assert data["action"] == "Pay now"


def test_parse_triage_strips_code_fences_and_prose():
    raw = 'Here you go:\n```json\n{"level": "green", "title": "Info letter"}\n```'
    data = parse_triage_json(raw)
    assert data["level"] == "green"
    assert data["title"] == "Info letter"


def test_parse_triage_bad_output_is_safe():
    data = parse_triage_json("the model rambled with no json")
    assert data["level"] == "yellow"  # safe fallback, never crashes
