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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(
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
    collection_name: str = field(
        default_factory=lambda: _get_env("PG_COLLECTION", "ai_tutor_documents")
    )

    @property
    def connection_string(self) -> str:
        # Dùng driver psycopg (v3) - bắt buộc cho langchain-postgres.
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class AppConfig:
    """Cấu hình các API bên ngoài (Gemini, Cohere) và tham số mô hình."""

    google_api_key: str = field(default_factory=lambda: _get_env("GOOGLE_API_KEY", required=True))
    cohere_api_key: str = field(default_factory=lambda: _get_env("COHERE_API_KEY", required=True))

    # --- Embedding: chạy LOCAL bằng sentence-transformers, KHÔNG gọi API Gemini ---
    # Lý do: bước index tài liệu (embedding hàng trăm/nghìn đoạn văn bản) là nơi
    # tốn quota free-tier nhanh nhất. Chuyển sang model chạy local giúp bước này
    # không còn phụ thuộc rate-limit/quota của Google nữa.
    # BAAI/bge-m3 là model đa ngôn ngữ, hỗ trợ tiếng Việt tốt, phù hợp cho RAG.
    # Nếu đổi embedding_model, PHẢI đổi luôn PG_COLLECTION (số chiều vector khác
    # nhau giữa các model -> không thể dùng chung 1 collection cũ) và index lại
    # toàn bộ tài liệu từ đầu.
    embedding_model: str = field(default_factory=lambda: _get_env("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_device: str = field(default_factory=lambda: _get_env("EMBEDDING_DEVICE", "cpu"))

    # NOTE: Tên model Gemini cho phần LLM (chat/agent) thay đổi theo thời gian.
    # Luôn kiểm tra danh sách model hiện hành tại Google AI Studio / Vertex AI
    # trước khi deploy, và ưu tiên cấu hình qua biến môi trường thay vì hard-code.
    # gemini-2.0-flash đã bị Google khai tử (03/2026) — dùng model Flash/Flash-Lite
    # còn được hỗ trợ trong free tier tại thời điểm deploy.
    llm_model: str = field(default_factory=lambda: _get_env("LLM_MODEL", "gemini-3.1-flash-lite"))
    llm_temperature: float = field(
        default_factory=lambda: float(_get_env("LLM_TEMPERATURE", "0.3"))
    )
    rerank_model: str = field(
        default_factory=lambda: _get_env("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(_get_env("EMBEDDING_BATCH_SIZE", "50"))
    )
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))


def get_logger(name: str) -> logging.Logger:
    """Trả về logger đã cấu hình sẵn, tránh add handler trùng lặp khi Streamlit rerun
    (Streamlit chạy lại toàn bộ script mỗi lần tương tác)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


# --- Khởi tạo cấu hình 1 lần khi import, nhưng KHÔNG raise ngay ---
# Lý do: nếu raise ở import-time, Streamlit sẽ hiển thị traceback thô, khó chịu
# cho người dùng cuối. Thay vào đó, app.py sẽ gọi get_config_error() và hiển thị
# lỗi qua st.error() một cách thân thiện.
try:
    db_config: Optional[DatabaseConfig] = DatabaseConfig()
    app_config: Optional[AppConfig] = AppConfig()
    _CONFIG_ERROR: Optional[str] = None
except EnvironmentError as e:
    db_config = None
    app_config = None
    _CONFIG_ERROR = str(e)


def get_config_error() -> Optional[str]:
    """Trả về thông báo lỗi cấu hình (nếu có), None nếu cấu hình hợp lệ."""
    return _CONFIG_ERROR
