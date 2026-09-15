"""
tests/conftest.py
------------------
Cấu hình chung cho toàn bộ test suite:
  1. Đặt sẵn các biến môi trường bắt buộc TRƯỚC khi bất kỳ module nào của app
     được import, để config.py không raise lỗi khi collect test.
  2. Stub (giả lập) module `langchain_huggingface` bằng 1 module giả.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

# --- 1. Env vars bắt buộc ---
os.environ.setdefault("PG_USER", "test_user")
os.environ.setdefault("PG_PASSWORD", "test_password")
os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_DATABASE", "test_db")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")

# --- 2. Stub langchain_huggingface ---
if "langchain_huggingface" not in sys.modules:
    _fake_module = types.ModuleType("langchain_huggingface")

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 8

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

    _fake_module.HuggingFaceEmbeddings = _FakeHuggingFaceEmbeddings
    sys.modules["langchain_huggingface"] = _fake_module


# --- 3. Stub streamlit (chưa cài trong môi trường test, và ta chỉ cần đủ
#        bề mặt API để test phần LOGIC thuần trong app.py — không test UI).
#        Định nghĩa các lớp giả ở mức module (không lồng trong if) để
#        test_app.py luôn import được _FakeStopException, kể cả khi
#        streamlit thật đã có sẵn trong môi trường (CI thật cài đủ
#        requirements.txt) — ta vẫn CHỦ ĐỘNG ghi đè bằng bản giả để test
#        logic app.py tách biệt hoàn toàn khỏi runtime Streamlit thật. ---


class _FakeStopException(Exception):
    """Giả lập đúng hành vi thật của st.stop(): dừng thực thi ngay lập tức
    (Streamlit thật raise 1 exception nội bộ mà runtime của nó tự bắt)."""


class _FakeSessionState(dict):
    """Giả lập st.session_state: hỗ trợ cả truy cập kiểu dict lẫn thuộc
    tính, giống Streamlit thật."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


class _FakeQueryParams(dict):
    pass


_fake_st = types.ModuleType("streamlit")
_fake_st.session_state = _FakeSessionState()
_fake_st.query_params = _FakeQueryParams()
_fake_st.error = lambda *a, **kw: None


def _fake_stop():
    raise _FakeStopException()


_fake_st.stop = _fake_stop
_fake_st.StopException = _FakeStopException
# Các hàm vẽ UI khác mà app.py/ui/* có gọi trong main() — không cần đúng hành
# vi thật vì các test ở đây CHỈ nhắm vào logic thuần (workspace resolution,
# bootstrap fail-fast, cache agent), không gọi main() hay test phần render
# UI/toggle (thuộc phạm vi test tích hợp, xem README).
for _name in (
    "set_page_config",
    "markdown",
    "header",
    "columns",
    "subheader",
    "button",
    "rerun",
    "container",
    "chat_message",
    "chat_input",
    "write_stream",
    "progress",
    "success",
    "warning",
    "expander",
    "divider",
    "text_input",
    "code",
    "caption",
    "text",
    "selectbox",
    "file_uploader",
    "spinner",
):
    setattr(_fake_st, _name, lambda *a, **kw: None)

sys.modules["streamlit"] = _fake_st

# ui/preview.py cần streamlit_pdf_viewer để hiển thị PDF gốc — không liên
# quan tới logic đang test (workspace resolution, bootstrap, cache agent),
# stub tối giản để import app.py không lỗi.
if "streamlit_pdf_viewer" not in sys.modules:
    _fake_pdf_viewer = types.ModuleType("streamlit_pdf_viewer")
    _fake_pdf_viewer.pdf_viewer = lambda *a, **kw: None
    sys.modules["streamlit_pdf_viewer"] = _fake_pdf_viewer


class DummyUploadedFile:
    """Giả lập interface tối thiểu của Streamlit UploadedFile
    (name + getvalue()/read()/seek()) để test document_processor mà không
    cần chạy trong Streamlit."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content

    def read(self, *args, **kwargs) -> bytes:
        return self._content

    def seek(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def dummy_file():
    return DummyUploadedFile
