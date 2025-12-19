"""Unit tests for utility helpers."""
from __future__ import annotations

from daystrom_dml import utils


def test_chunk_text_respects_token_budget():
    text = "\n\n".join([f"Paragraph {i} " + ("word " * 80) for i in range(10)])
    chunks = utils.chunk_text(text, max_tokens=120, overlap=20)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert utils.estimate_tokens(chunk) <= 120


def test_chunk_text_overlap_is_applied():
    paragraphs = [
        " ".join(["alpha"] * 30),
        " ".join(["beta"] * 30),
        " ".join(["gamma"] * 30),
    ]
    text = "\n\n".join(paragraphs)
    chunks = utils.chunk_text(text, max_tokens=60, overlap=20)
    assert len(chunks) >= 2
    assert "alpha" in chunks[0]
    # overlap keeps a slice of the first paragraph in the next chunk
    assert "alpha" in chunks[1]
    # and the subsequent chunks eventually contain the second paragraph content
    assert any("beta" in chunk for chunk in chunks[1:])
