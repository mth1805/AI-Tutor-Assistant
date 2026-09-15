"""Test tools.py: định dạng trích dẫn nguồn, fallback khi Cohere rerank lỗi."""

from __future__ import annotations

from langchain_core.documents import Document

import tools


def test_format_with_citations_includes_source_labels():
    docs = [
        Document(page_content="Nội dung 1", metadata={"source": "a.pdf"}),
        Document(page_content="Nội dung 2", metadata={}),
    ]
    formatted = tools._format_with_citations(docs)
    assert "[Nguồn: a.pdf]" in formatted
    assert "[Nguồn: tài liệu đã tải lên]" in formatted
    assert "Nội dung 1" in formatted
    assert "Nội dung 2" in formatted


def test_document_reader_tool_reports_when_nothing_found(monkeypatch):
    monkeypatch.setattr(tools, "hybrid_search", lambda *a, **kw: [])
    reader = tools.make_document_reader_tool(collection_name="ws_test")
    result = reader.invoke("câu hỏi bất kỳ")
    assert "Không tìm thấy" in result


def test_document_reader_tool_falls_back_when_cohere_rerank_fails(monkeypatch):
    docs = [Document(page_content="nội dung quan trọng", metadata={"source": "x.pdf"})]
    monkeypatch.setattr(tools, "hybrid_search", lambda *a, **kw: docs)

    class _BoomCompressor:
        def __init__(self, *args, **kwargs):
            pass

        def compress_documents(self, documents, query):
            raise RuntimeError("Cohere 401 Unauthorized")

    monkeypatch.setattr(tools, "CohereRerank", _BoomCompressor)

    reader = tools.make_document_reader_tool(collection_name="ws_test")
    result = reader.invoke("câu hỏi")

    # Vẫn trả lời được bằng kết quả hybrid search thô, kèm trích dẫn nguồn,
    # thay vì báo lỗi hoàn toàn cho người dùng.
    assert "[Nguồn: x.pdf]" in result
    assert "nội dung quan trọng" in result


def test_document_reader_tool_reports_db_error_message(monkeypatch):
    from exceptions import DatabaseConnectionError

    def _boom(*args, **kwargs):
        raise DatabaseConnectionError("Không thể kết nối Postgres")

    monkeypatch.setattr(tools, "hybrid_search", _boom)
    reader = tools.make_document_reader_tool(collection_name="ws_test")
    result = reader.invoke("câu hỏi")
    assert "Không thể truy vấn kho tài liệu" in result
