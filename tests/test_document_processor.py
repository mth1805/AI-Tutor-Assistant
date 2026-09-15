"""Test document_processor.py: đọc file, chunking, gắn metadata nguồn, OCR fallback."""

from __future__ import annotations

import pytest

import document_processor as dp
from exceptions import DocumentProcessingError


def test_extract_text_txt(dummy_file):
    f = dummy_file("bai1.txt", "Xin chào thế giới".encode())
    assert dp.extract_text(f) == "Xin chào thế giới"


def test_extract_text_unsupported_extension_raises(dummy_file):
    f = dummy_file("bai1.xyz", b"noi dung")
    with pytest.raises(DocumentProcessingError):
        dp.extract_text(f)


def test_split_text_empty_raises():
    with pytest.raises(DocumentProcessingError):
        dp.split_text("   ")


def test_split_text_fixed_produces_multiple_chunks():
    text = "Đây là một câu ví dụ. " * 100
    chunks = dp.split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_split_text_semantic_without_embeddings_raises():
    with pytest.raises(DocumentProcessingError):
        dp.split_text("một đoạn văn bản", strategy="semantic", embeddings=None)


def test_process_files_to_chunks_attaches_source_metadata(dummy_file):
    text = ("Nội dung bài học rất quan trọng. " * 50).strip()
    f = dummy_file("chuong1.txt", text.encode("utf-8"))
    chunks = dp.process_files_to_chunks([f], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, dp.TextChunk) for c in chunks)
    assert all(c.metadata["source"] == "chuong1.txt" for c in chunks)


def test_process_files_to_chunks_all_files_fail_raises(dummy_file):
    f = dummy_file("bad.xyz", b"noi dung")
    with pytest.raises(DocumentProcessingError):
        dp.process_files_to_chunks([f])


def test_process_files_to_chunks_partial_failure_still_returns_good_chunks(dummy_file):
    good = dummy_file("ok.txt", ("Nội dung hợp lệ. " * 50).encode("utf-8"))
    bad = dummy_file("bad.xyz", b"noi dung")
    chunks = dp.process_files_to_chunks([good, bad], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(c.metadata["source"] == "ok.txt" for c in chunks)


def test_extract_text_pdf_falls_back_gracefully_without_ocr(monkeypatch, dummy_file):
    """Khi PDF không có text (nghi scan ảnh) và OCR không khả dụng, trả về
    chuỗi rỗng thay vì crash — người dùng sẽ thấy thông báo lỗi thân thiện
    ở tầng process_files_to_chunks."""
    monkeypatch.setattr(dp, "_OCR_AVAILABLE", False)

    class _DummyPage:
        def extract_text(self):
            return ""

    class _DummyReader:
        def __init__(self, file):
            self.pages = [_DummyPage()]

    monkeypatch.setattr(dp, "PdfReader", _DummyReader)

    f = dummy_file("scan.pdf", b"%PDF-fake-bytes")
    assert dp.extract_text(f) == ""
