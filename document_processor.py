"""
document_processor.py
----------------------
Đọc và chia nhỏ (chunk) tài liệu người dùng tải lên (PDF, DOCX, TXT).
Không phụ thuộc Streamlit hay bất kỳ hạ tầng nào khác -> có thể unit test
độc lập với các file mẫu.
"""
from __future__ import annotations

from typing import Iterable

import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import get_logger
from exceptions import DocumentProcessingError

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    text = text.replace("\x00", "")  # Lọc an toàn ký tự NUL
    if not text.strip():
        raise DocumentProcessingError(
            "Tài liệu rỗng hoặc không trích xuất được nội dung nào "
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

def _read_pdf(file) -> str:
    reader = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text.replace("\x00", "")


def _read_docx(file) -> str:
    document = docx.Document(file)
    text = "\n".join(p.text for p in document.paragraphs)
    return text.replace("\x00", "")


def _read_txt(file) -> str:
    text = file.getvalue().decode("utf-8", errors="replace")
    return text.replace("\x00", "")


def extract_text(file) -> str:
    """Trích xuất text từ 1 file upload (Streamlit UploadedFile hoặc file-like)."""
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            text = _read_pdf(file)
        elif name.endswith(".docx"):
            text = _read_docx(file)
        elif name.endswith(".txt"):
            text = _read_txt(file)
        else:
            raise DocumentProcessingError(
                f"Định dạng file không được hỗ trợ: '{file.name}'. "
                f"Chỉ hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
    except DocumentProcessingError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Lỗi khi đọc file %s", file.name)
        raise DocumentProcessingError(f"Không thể đọc file '{file.name}': {e}") from e

    if not text.strip():
        logger.warning("File '%s' không trích xuất được nội dung nào (có thể là PDF scan ảnh).", file.name)
    return text


def extract_text_from_files(files: Iterable) -> str:
    """Trích xuất và gộp text từ nhiều file. Lỗi ở 1 file không làm dừng toàn bộ
    batch — được gom lại và raise 1 lần cuối để người dùng thấy đầy đủ vấn đề."""
    parts: list[str] = []
    errors: list[str] = []
    for f in files:
        try:
            parts.append(extract_text(f))
        except DocumentProcessingError as e:
            errors.append(str(e))

    if errors and not parts:
        raise DocumentProcessingError("; ".join(errors))
    if errors:
        logger.warning("Một số file lỗi nhưng vẫn tiếp tục với các file còn lại: %s", errors)

    return "\n".join(parts)


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    if not text.strip():
        raise DocumentProcessingError(
            "Tài liệu rỗng hoặc không trích xuất được nội dung nào "
            "(kiểm tra xem PDF có phải dạng ảnh scan cần OCR không)."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)