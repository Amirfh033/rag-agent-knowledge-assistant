from pathlib import Path

import numpy as np
import pytest

from app.rag import BM25, VectorIndex, chunk_document, split_long, split_sections, tokenize
from conftest import ROOT, FakeLLM

DOCS = ROOT / "data" / "docs"


def test_split_sections_keeps_title_and_heading_path():
    title, secs = split_sections("# Doc\nintro\n## Plans\n### Starter\nA\n### Business\nB\n## Other\nC")
    assert title == "Doc"
    assert [s for s, _ in secs] == ["", "Plans > Starter", "Plans > Business", "Other"]


def test_split_long_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    parts = split_long(text, 300, 60)
    assert len(parts) > 3
    assert all(len(p) <= 360 for p in parts)
    assert parts[0][-30:].split()[-1] in parts[1]  # overlap carries text forward


def test_every_document_is_chunked_with_citations():
    for f in sorted(DOCS.glob("*.md")):
        chunks = chunk_document(f)
        assert chunks, f.name
        assert all(c.text.strip() and c.source == f.name for c in chunks)
        assert all(len(c.text) <= 1100 for c in chunks)
    c = chunk_document(DOCS / "01-pricing-plans.md")
    assert any("Business" in x.section for x in c)
    assert c[0].title == "Nimbus Flow — Pricing and Plans"


def test_tokenizer_keeps_numbers_and_persian():
    assert "99.9" in tokenize("uptime 99.9% SLA")
    assert "مشتری" in tokenize("اطلاعات مشتری")


def test_bm25_prefers_matching_document():
    bm = BM25(["refund annual plan within 30 days", "backups every 6 hours", "phone support enterprise"])
    assert int(np.argmax(bm.scores("annual refund"))) == 0


@pytest.mark.parametrize("query,expected", [
    ("How much does the Business plan cost?", "01-pricing-plans.md"),
    ("Can I get a refund on an annual plan?", "02-refund-and-cancellation.md"),
    ("first response time for a P1 critical incident", "03-support-and-sla.md"),
    ("Where is customer data stored?", "04-security-and-privacy.md"),
    ("free onboarding call for business customers", "05-onboarding-guide.md"),
    ("API rate limit requests per minute", "06-integrations-and-limits.md"),
])
def test_retrieval_top_hit(index, fake_llm, query, expected):
    hits = index.search(fake_llm, query, k=3, min_score=0)
    assert hits[0].chunk.source == expected, [h.chunk.citation() for h in hits]


def test_min_score_filters_irrelevant_queries(index, fake_llm):
    assert index.search(fake_llm, "zebra giraffe volcano", k=3, min_score=0.25) == []


def test_save_load_roundtrip(tmp_path, index, fake_llm):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.search(fake_llm, "refund annual", k=1)[0].chunk.id == index.search(fake_llm, "refund annual", k=1)[0].chunk.id


def test_model_mismatch_is_detected(index):
    other = FakeLLM(); other.embed_model = "another-model"
    with pytest.raises(ValueError, match="Re-run ingest"):
        index.search(other, "price")


def test_missing_index_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        VectorIndex.load(tmp_path)
