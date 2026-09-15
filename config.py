"""
config.py
---------
Tập trung toàn bộ cấu hình, biến môi trường và logging cho ứng dụng.
Mọi module khác chỉ import từ đây, không tự đọc os.environ riêng lẻ.
Việc tập trung cấu hình giúp:
  - Dễ audit: biết chính xác app cần biến môi trường nào.
  - Dễ test: có thể monkeypatch 1 chỗ duy nhất.
  - Fail-fast có kiểm soát: lỗi cấu hình được bắt và hiển thị đẹp trên UI
    thay vì làm crash tiến trình với traceback thô.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import os
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise OSError(
            f"Thiếu biến môi trường bắt buộc: '{name}'. "
            f"Vui lòng khai báo trong file .env hoặc biến môi trường hệ thống "
            f"(xem .env.example)."
        )
    return value


@dataclass(frozen=True)
class DatabaseConfig:
    """Cấu hình kết nối PostgreSQL + pgvector."""

    host: str = field(default_factory=lambda: _get_env("PG_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_get_env("PG_PORT", "5432")))
    database: str = field(default_factory=lambda: _get_env("PG_DATABASE", "ai_tutor"))
    user: str = field(default_factory=lambda: _get_env("PG_USER", required=True))
    password: str = field(default_factory=lambda: _get_env("PG_PASSWORD", required=True))
    collection_name: str = field(default_factory=lambda: _get_env("PG_COLLECTION", "ai_tutor_documents"))

    @property
    def connection_string(self) -> str:
        # Dùng driver psycopg (v3) - bắt buộc cho langchain-postgres.
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class AppConfig:
    """Cấu hình các API bên ngoài (Gemini, Cohere) và tham số mô hình."""

    google_api_key: str = field(default_factory=lambda: _get_env("GOOGLE_API_KEY", required=True))
    cohere_api_key: str = field(default_factory=lambda: _get_env("COHERE_API_KEY", required=True))

    # --- Embedding: chạy LOCAL bằng sentence-transformers ---
    embedding_model: str = field(default_factory=lambda: _get_env("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_device: str = field(default_factory=lambda: _get_env("EMBEDDING_DEVICE", "cpu"))

    llm_model: str = field(default_factory=lambda: _get_env("LLM_MODEL", "gemini-3.1-flash-lite"))
    llm_temperature: float = field(default_factory=lambda: float(_get_env("LLM_TEMPERATURE", "0.3")))
    rerank_model: str = field(
        default_factory=lambda: _get_env("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")
    )
    embedding_batch_size: int = field(default_factory=lambda: int(_get_env("EMBEDDING_BATCH_SIZE", "50")))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))

    # --- Chunking ---
    # "fixed": chia theo độ dài cố định (nhanh, ổn định).
    # "semantic": chia theo điểm ngắt ngữ nghĩa (langchain_experimental.SemanticChunker),
    # thường cho kết quả retrieval tốt hơn với tài liệu học thuật, nhưng chậm
    # hơn vì phải encode văn bản để tìm điểm ngắt.
    chunking_strategy: str = field(default_factory=lambda: _get_env("CHUNKING_STRATEGY", "fixed"))

    # --- OCR cho PDF dạng ảnh scan ---
    # Yêu cầu thư viện pytesseract/pdf2image + gói hệ thống poppler/tesseract
    # (xem packages.txt). Nếu thiếu, tự động bỏ qua OCR thay vì lỗi.
    enable_ocr: bool = field(default_factory=lambda: _get_env("ENABLE_OCR", "true").strip().lower() == "true")
    # --- Cloud Object Storage (S3 / R2) ---
    # --- Cloud Object Storage (S3 / R2) ---
    s3_endpoint_url: str | None = field(default_factory=lambda: _get_env("S3_ENDPOINT_URL", ""))
    s3_bucket_name: str = field(default_factory=lambda: _get_env("S3_BUCKET_NAME", "ai-tutor-files"))
    s3_access_key: str | None = field(default_factory=lambda: _get_env("S3_ACCESS_KEY", ""))
    s3_secret_key: str | None = field(default_factory=lambda: _get_env("S3_SECRET_KEY", ""))


def get_logger(name: str) -> logging.Logger:
    """Trả về logger đã cấu hình sẵn, tránh add handler trùng lặp khi Streamlit rerun
    (Streamlit chạy lại toàn bộ script mỗi lần tương tác).

    Mỗi dòng log tự động kèm workspace_id hiện tại (xem set_log_context) — giúp
    lọc log theo đúng phiên làm việc/người dùng khi nhiều người dùng chung app,
    thay vì phải tự tay truyền workspace_id vào từng lời gọi logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | ws=%(workspace_id)s | %(message)s"
        )
        handler.setFormatter(formatter)
        handler.addFilter(_WorkspaceLogFilter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


# --- Observability: gắn workspace_id vào mọi dòng log mà không cần truyền
# thủ công qua từng hàm (dùng contextvars — an toàn theo từng luồng chạy,
# phù hợp với cách Streamlit chạy mỗi phiên trên 1 thread riêng). ---
_workspace_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("workspace_id", default="-")


def set_log_context(workspace_id: str) -> None:
    """Gọi 1 lần ở đầu mỗi lượt xử lý request (xem app.main()) để mọi log sau
    đó — kể cả từ database.py, tools.py, agent.py — tự động có workspace_id."""
    _workspace_ctx.set(workspace_id)


class _WorkspaceLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.workspace_id = _workspace_ctx.get()
        return True


def log_duration(label: str | None = None):
    """Decorator đo thời gian thực thi 1 hàm và ghi log INFO khi hoàn thành.
    Dùng cho các bước tốn thời gian (gọi LLM, gọi tool, truy vấn DB) để dễ
    phát hiện bước nào đang là điểm nghẽn hiệu năng."""

    def decorator(func):
        step_name = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("⏱ %s hoàn thành sau %.2fs", step_name, elapsed)

        return wrapper

    return decorator


# --- Khởi tạo cấu hình 1 lần khi import, nhưng KHÔNG raise ngay ---
# Lý do: nếu raise ở import-time, Streamlit sẽ hiển thị traceback thô, khó chịu
# cho người dùng cuối. Thay vào đó, app.py sẽ gọi get_config_error() và hiển thị
# lỗi qua st.error() một cách thân thiện.
try:
    db_config: DatabaseConfig | None = DatabaseConfig()
    app_config: AppConfig | None = AppConfig()
    _CONFIG_ERROR: str | None = None
except OSError as e:
    db_config = None
    app_config = None
    _CONFIG_ERROR = str(e)


def get_config_error() -> str | None:
    """Trả về thông báo lỗi cấu hình (nếu có), None nếu cấu hình hợp lệ."""
    return _CONFIG_ERROR
