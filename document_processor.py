"""
document_processor.py
----------------------
Đọc, OCR (khi cần) và chia nhỏ (chunk) tài liệu người dùng tải lên.
Mỗi đoạn văn bản (TextChunk) được gắn kèm metadata {"source": <tên file>}
để phục vụ trích dẫn nguồn khi trả lời (xem tools.make_document_reader_tool).

Module này không phụ thuộc Streamlit -> có thể unit test độc lập.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import docx
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import get_logger
from exceptions import DocumentProcessingError

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

# OCR là tính năng TÙY CHỌN: nếu thiếu thư viện Python (pytesseract/pdf2image)
# hoặc gói hệ thống (poppler, tesseract-ocr) chưa cài, tự động bỏ qua OCR
# thay vì làm crash toàn bộ app. Xem packages.txt để cài gói hệ thống khi
# deploy lên Streamlit Community Cloud.
try:
    import pytesseract
    from pdf2image import convert_from_bytes

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


ChunkingStrategy = Literal["fixed", "semantic"]


@dataclass
class TextChunk:
    """1 đoạn văn bản sau khi chunk, kèm metadata để trích dẫn nguồn."""

    text: str
    metadata: dict = field(default_factory=dict)


def _ocr_pdf(file) -> str:
    """OCR toàn bộ PDF thành text có xử lý ảnh (Grayscale) để đọc biểu đồ màu."""
    if not _OCR_AVAILABLE:
        logger.warning(
            "Thiếu thư viện OCR (pytesseract/pdf2image) hoặc poppler/tesseract "
            "chưa cài ở hệ thống -> bỏ qua OCR."
        )
        return ""
    try:
        file.seek(0)
        images = convert_from_bytes(file.read())
        logger.info("Bắt đầu OCR cho file '%s' (%d trang)...", getattr(file, "name", "unknown"), len(images))

        text_parts = []
        for idx, img in enumerate(images):
            gray_img = img.convert("L")
            txt = pytesseract.image_to_string(gray_img, lang="eng")
            if txt.strip():
                text_parts.append(txt)
                logger.info(
                    "-> OCR thành công trang %d/%d (Đã quét được %d ký tự).",
                    idx + 1,
                    len(images),
                    len(txt.strip()),
                )
            else:
                logger.warning("-> Trang %d/%d không quét được chữ nào qua OCR.", idx + 1, len(images))

        return "\n".join(text_parts)
    except Exception as e:  # noqa: BLE001
        logger.warning("OCR thất bại cho file '%s': %s", getattr(file, "name", "?"), e)
        return ""


def _read_pdf(file) -> str:
    """Đọc PDF kết hợp text thuần và OCR toàn trang, log kết quả ra console để theo dõi."""
    text_parts = []
    file_name = getattr(file, "name", "unknown")

    # 1. Thử trích xuất text thuần trước
    try:
        file.seek(0)
        reader = PdfReader(file)
        raw_text = "".join(page.extract_text() or "" for page in reader.pages)
        if raw_text.strip():
            logger.info(
                "Đã trích xuất %d ký tự text thuần (selectable text) từ file '%s'.",
                len(raw_text.strip()),
                file_name,
            )
            text_parts.append(raw_text)
        else:
            logger.info("File '%s' không có text thuần (hoặc hoàn toàn là file ảnh/scan).", file_name)
    except Exception as e:
        logger.warning("Không thể đọc text thuần từ PDF: %s", e)

    # 2. Chạy OCR bổ sung
    if _OCR_AVAILABLE:
        logger.info("Kích hoạt tiến trình OCR hình ảnh/biểu đồ cho file '%s'...", file_name)
        ocr_text = _ocr_pdf(file)
        if ocr_text.strip():
            text_parts.append(ocr_text)
            logger.info(
                "Đã bổ sung thêm %d ký tự từ OCR vào nội dung xử lý của file '%s'.",
                len(ocr_text.strip()),
                file_name,
            )

    combined_text = "\n".join(text_parts)
    logger.info(
        "Tổng số lượng ký tự thu được sau khi xử lý file '%s' là: %d", file_name, len(combined_text.strip())
    )
    return combined_text.replace("\x00", "")


def _read_docx(file) -> str:
    document = docx.Document(file)
    text = "\n".join(p.text for p in document.paragraphs)
    return text.replace("\x00", "")


def _read_txt(file) -> str:
    text = file.getvalue().decode("utf-8", errors="replace")
    return text.replace("\x00", "")


def extract_text(file, *, enable_ocr: bool = True) -> str:
    """Trích xuất text từ 1 file upload (Streamlit UploadedFile hoặc file-like).

    Với PDF: nếu PdfReader không trích xuất được text nào (nghi là bản scan
    ảnh) và enable_ocr=True, tự động thử OCR trước khi báo rỗng.
    """
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            text = _read_pdf(file)
            if not text.strip() and enable_ocr:
                logger.info(
                    "File '%s' không có text trích xuất được (có thể là PDF scan ảnh) -> thử OCR.",
                    file.name,
                )
                text = _ocr_pdf(file)
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
        logger.warning(
            "File '%s' vẫn không trích xuất được nội dung nào (kể cả sau OCR nếu có).",
            file.name,
        )
    return text


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def _split_semantic(text: str, embeddings) -> list[str]:
    """Chia đoạn theo điểm ngắt ngữ nghĩa thay vì độ dài cố định. Thường cho
    kết quả retrieval tốt hơn với tài liệu học thuật có mạch ý rõ ràng, nhưng
    chậm hơn vì phải encode văn bản để xác định điểm ngắt."""

    splitter = SemanticChunker(embeddings)
    return splitter.split_text(text)


def split_text(
    text: str,
    *,
    strategy: ChunkingStrategy = "fixed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embeddings=None,
) -> list[str]:
    text = text.replace("\x00", "")  # Lọc ký tự NUL (Postgres không chấp nhận trong text)
    if not text.strip():
        raise DocumentProcessingError(
            "Tài liệu rỗng hoặc không trích xuất được nội dung nào "
            "(kiểm tra xem PDF có phải dạng ảnh scan cần OCR không)."
        )
    if strategy == "semantic":
        if embeddings is None:
            raise DocumentProcessingError(
                "Cần truyền embeddings để dùng chunking theo ngữ nghĩa (strategy='semantic')."
            )
        try:
            return _split_semantic(text, embeddings)
        except Exception as e:  # noqa: BLE001
            logger.warning("Semantic chunking lỗi (%s), fallback về fixed-size chunking.", e)
            return _split_fixed(text, chunk_size, chunk_overlap)
    return _split_fixed(text, chunk_size, chunk_overlap)


def process_files_to_chunks(
    files: Iterable,
    *,
    strategy: ChunkingStrategy = "fixed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embeddings=None,
    enable_ocr: bool = True,
) -> list[TextChunk]:
    """Đọc + chunk nhiều file, trả về danh sách TextChunk có gắn kèm
    metadata {"source": tên_file} để phục vụ trích dẫn nguồn khi trả lời.

    Lỗi ở 1 file không làm dừng toàn bộ batch — được gom lại; nếu TẤT CẢ file
    đều lỗi thì mới raise, để người dùng luôn nhận được kết quả tốt nhất có thể
    từ các file hợp lệ.
    """
    all_chunks: list[TextChunk] = []
    errors: list[str] = []

    for f in files:
        try:
            text = extract_text(f, enable_ocr=enable_ocr)
        except DocumentProcessingError as e:
            errors.append(str(e))
            continue

        if not text.strip():
            errors.append(f"File '{f.name}' không có nội dung để index (có thể cần OCR thủ công).")
            continue

        chunks = split_text(
            text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embeddings=embeddings,
        )
        all_chunks.extend(TextChunk(text=c, metadata={"source": f.name}) for c in chunks)

    if errors and not all_chunks:
        raise DocumentProcessingError("; ".join(errors))
    if errors:
        logger.warning("Một số file lỗi nhưng vẫn tiếp tục với các file còn lại: %s", errors)

    return all_chunks
