"""Test database.py: hybrid search + fallback, RRF, lịch sử chat.

Tất cả các lệnh gọi Postgres/model thật đều được mock — đây là UNIT TEST cho
logic nghiệp vụ (fallback, RRF, error-swallowing), không phải test tích hợp.
Test tích hợp thật (cần Postgres/model chạy thật) nên tách riêng, chạy có
điều kiện trong CI (xem README, mục Testing) bằng testcontainers hoặc DB dev.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import psycopg
import pytest
from langchain_core.documents import Document

import database
from exceptions import DatabaseConnectionError, EmbeddingError


def test_get_embeddings_returns_cached_instance():
    database.get_embeddings.cache_clear()
    emb1 = database.get_embeddings()
    emb2 = database.get_embeddings()
    assert emb1 is emb2  # @lru_cache hoạt động -> không nạp lại model mỗi lần gọi
    database.get_embeddings.cache_clear()


def test_get_embeddings_wraps_errors_as_embedding_error(monkeypatch):
    database.get_embeddings.cache_clear()

    def _boom(*args, **kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(database, "HuggingFaceEmbeddings", _boom)
    with pytest.raises(EmbeddingError):
        database.get_embeddings()
    database.get_embeddings.cache_clear()


def test_reciprocal_rank_fusion_ranks_and_dedupes():
    doc_a = Document(page_content="Nội dung A " * 5, metadata={"source": "a.pdf"})
    doc_b = Document(page_content="Nội dung B " * 5, metadata={"source": "b.pdf"})
    doc_a_dup = Document(page_content=doc_a.page_content, metadata={"source": "a.pdf"})

    # doc_b đứng hạng 0 (cao nhất) ở CẢ 2 danh sách -> tổng điểm RRF cao nhất.
    vector_results = [doc_b, doc_a]
    keyword_results = [doc_b, doc_a_dup]

    merged = database._reciprocal_rank_fusion([vector_results, keyword_results], k=5)

    assert merged[0].metadata["source"] == "b.pdf"
    assert len(merged) == 2  # doc_a và doc_a_dup trùng nội dung -> không nhân đôi


def test_hybrid_search_falls_back_to_vector_only_when_keyword_search_fails(monkeypatch):
    fake_store = MagicMock()
    fake_docs = [Document(page_content="đoạn 1"), Document(page_content="đoạn 2")]
    fake_store.similarity_search.return_value = fake_docs

    monkeypatch.setattr(database, "get_vector_store", lambda collection_name=None: fake_store)

    def _boom(*args, **kwargs):
        raise RuntimeError("bảng khác schema do phiên bản langchain-postgres khác nhau")

    monkeypatch.setattr(database, "_keyword_search", _boom)

    results = database.hybrid_search("câu hỏi", collection_name="ws_test", k=5)

    assert results == fake_docs
    fake_store.similarity_search.assert_called_once()


def test_hybrid_search_merges_vector_and_keyword_results(monkeypatch):
    doc_v = Document(page_content="kết quả vector", metadata={"source": "v.pdf"})
    doc_k = Document(page_content="kết quả keyword", metadata={"source": "k.pdf"})

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [doc_v]
    monkeypatch.setattr(database, "get_vector_store", lambda collection_name=None: fake_store)
    monkeypatch.setattr(database, "_keyword_search", lambda *a, **kw: [doc_k])

    results = database.hybrid_search("câu hỏi", collection_name="ws_test", k=5)

    sources = {d.metadata["source"] for d in results}
    assert sources == {"v.pdf", "k.pdf"}


def test_collection_has_documents_returns_false_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("không kết nối được DB")

    monkeypatch.setattr(database, "_connect", _boom)
    assert database.collection_has_documents("ws_x") is False


def test_save_chat_message_swallows_errors_without_raising(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    # Không được raise: lỗi lưu lịch sử không nên làm gián đoạn cuộc trò chuyện.
    database.save_chat_message("ws_x", "user", "xin chào")


def test_get_chat_history_returns_empty_list_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    assert database.get_chat_history("ws_x") == []


def test_check_database_connection_wraps_operational_error(monkeypatch):
    def _boom():
        raise psycopg.OperationalError("cannot connect")

    monkeypatch.setattr(database, "_connect", _boom)
    with pytest.raises(DatabaseConnectionError):
        database.check_database_connection()


# ---------------------------------------------------------------------------
# index_chunks: regression test cho bug "metadata_common=None -> crash"
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self):
        self.calls: list[tuple[list[str], list[dict]]] = []

    def add_texts(self, texts, metadatas):
        self.calls.append((texts, metadatas))


def test_index_chunks_does_not_crash_when_metadata_common_omitted(monkeypatch):
    """Test hồi quy: trước đây index_chunks() dùng metadata_common trước khi
    gán giá trị mặc định {} -> dict.update(None) raise TypeError khi người gọi
    không truyền metadata_common. Không được tái phát lỗi này."""
    from document_processor import TextChunk

    fake_store = _FakeStore()
    monkeypatch.setattr(database, "get_vector_store", lambda collection_name=None: fake_store)

    chunks = [TextChunk(text="nội dung 1", metadata={"source": "a.pdf"})]

    # Cố tình KHÔNG truyền metadata_common — đây chính là điều kiện gây bug.
    database.index_chunks(chunks, collection_name="ws_test")

    assert len(fake_store.calls) == 1
    texts, metadatas = fake_store.calls[0]
    assert texts == ["nội dung 1"]
    assert metadatas == [{"source": "a.pdf"}]


def test_index_chunks_merges_metadata_common_into_each_chunk(monkeypatch):
    from document_processor import TextChunk

    fake_store = _FakeStore()
    monkeypatch.setattr(database, "get_vector_store", lambda collection_name=None: fake_store)

    chunks = [
        TextChunk(text="đoạn 1", metadata={"source": "a.pdf"}),
        TextChunk(text="đoạn 2", metadata={"source": "b.pdf"}),
    ]
    database.index_chunks(chunks, collection_name="ws_test", metadata_common={"workspace_id": "ws_test"})

    _, metadatas = fake_store.calls[0]
    assert metadatas[0] == {"source": "a.pdf", "workspace_id": "ws_test"}
    assert metadatas[1] == {"source": "b.pdf", "workspace_id": "ws_test"}


def test_index_chunks_does_nothing_for_empty_list(monkeypatch):
    calls = []
    monkeypatch.setattr(
        database, "get_vector_store", lambda collection_name=None: calls.append("should not be called")
    )
    database.index_chunks([], collection_name="ws_test")
    assert calls == []  # get_vector_store không được gọi khi không có chunk nào


# ---------------------------------------------------------------------------
# Quản lý workspace
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Giả lập psycopg cursor: ghi lại các câu lệnh SQL đã chạy, cho phép
    cấu hình sẵn kết quả trả về (fetchall/fetchone) và rowcount."""

    def __init__(self, fetchall_result=None, fetchone_result=None, rowcount=0):
        self.executed: list[tuple[str, tuple]] = []
        self._fetchall_result = fetchall_result or []
        self._fetchone_result = fetchone_result
        self.rowcount = rowcount

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall_result

    def fetchone(self):
        return self._fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_all_workspaces_returns_list_of_dicts(monkeypatch):
    cursor = _FakeCursor(fetchall_result=[("ws_1", "Toán 10"), ("ws_2", "Văn 12")])
    monkeypatch.setattr(database, "_connect", lambda: _FakeConnection(cursor))

    workspaces = database.get_all_workspaces()

    assert workspaces == [
        {"id": "ws_1", "title": "Toán 10"},
        {"id": "ws_2", "title": "Văn 12"},
    ]


def test_get_all_workspaces_returns_empty_list_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    assert database.get_all_workspaces() == []


def test_ensure_workspace_exists_swallows_errors(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    # Không được raise: đây là hàm "best-effort", lỗi ghi workspace không nên
    # chặn người dùng tiếp tục dùng app.
    database.ensure_workspace_exists("ws_x", "Tên workspace")


def test_rename_workspace_swallows_errors(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    database.rename_workspace("ws_x", "Tên mới")


def test_delete_workspace_raises_database_connection_error_on_failure(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    with pytest.raises(DatabaseConnectionError):
        database.delete_workspace("ws_x")


def test_delete_workspace_executes_cleanup_in_correct_order(monkeypatch):
    cursor = _FakeCursor(rowcount=2)
    monkeypatch.setattr(database, "_connect", lambda: _FakeConnection(cursor))

    database.delete_workspace("ws_x")

    executed_sql = [sql for sql, _params in cursor.executed]
    assert len(executed_sql) == 4
    assert "DELETE FROM " + database.CHAT_TABLE in executed_sql[0]
    assert "DELETE FROM " + database.WORKSPACE_TABLE in executed_sql[1]
    assert "langchain_pg_embedding" in executed_sql[2]
    assert "langchain_pg_collection" in executed_sql[3]
    # Mọi câu lệnh đều lọc theo đúng workspace_id, tránh xóa nhầm workspace khác.
    for _sql, params in cursor.executed:
        assert params == ("ws_x",)


def test_get_indexed_files_returns_file_names(monkeypatch):
    cursor = _FakeCursor(fetchall_result=[("bai1.pdf",), ("bai2.docx",)])
    monkeypatch.setattr(database, "_connect", lambda: _FakeConnection(cursor))

    files = database.get_indexed_files("ws_x")

    assert files == ["bai1.pdf", "bai2.docx"]


def test_get_indexed_files_filters_out_none_values(monkeypatch):
    # COALESCE có thể trả None nếu 1 chunk không có metadata source hợp lệ nào
    # (dữ liệu cũ trước khi có tính năng trích dẫn nguồn) -> phải lọc bỏ.
    cursor = _FakeCursor(fetchall_result=[("bai1.pdf",), (None,)])
    monkeypatch.setattr(database, "_connect", lambda: _FakeConnection(cursor))

    files = database.get_indexed_files("ws_x")

    assert files == ["bai1.pdf"]


def test_get_indexed_files_returns_empty_list_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(database, "_connect", _boom)
    assert database.get_indexed_files("ws_x") == []
