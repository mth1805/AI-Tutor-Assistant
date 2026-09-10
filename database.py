"""
database.py
-----------
Quản lý toàn bộ tương tác với PostgreSQL + pgvector thông qua LangChain
(langchain-postgres). Tách biệt hoàn toàn khỏi UI và Agent để:
  - Có thể unit test độc lập (mock psycopg / PGVector).
  - Có thể tái sử dụng trong script batch-import tài liệu chạy ngoài Streamlit
    (ví dụ: một cron job index tài liệu giảng viên upload hàng loạt).

Thiết kế phân quyền (xem README):
  Mỗi lời gọi index_text_chunks / get_retriever đều nhận `collection_name`
  và `metadata` / `filter` tùy chọn. Đây là 2 cơ chế cô lập dữ liệu:
    - collection_name: cô lập ở mức bảng logic (1 lớp học / 1 user / 1 dự án).
    - metadata filter: cô lập mịn hơn trong cùng 1 collection (vd owner_id,
      role được phép xem, môn học...).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import psycopg
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import app_config, db_config, get_logger
from exceptions import DatabaseConnectionError, EmbeddingError

logger = get_logger(__name__)


def check_database_connection() -> None:
    """Kiểm tra kết nối Postgres + đảm bảo extension pgvector tồn tại.
    Nên gọi 1 lần khi app khởi động (fail-fast) thay vì để lỗi kết nối
    xuất hiện bất ngờ giữa phiên chat của người dùng."""
    try:
        with psycopg.connect(
            host=db_config.host,
            port=db_config.port,
            dbname=db_config.database,
            user=db_config.user,
            password=db_config.password,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("Kết nối PostgreSQL/pgvector thành công.")
    except psycopg.OperationalError as e:
        logger.error("Không thể kết nối PostgreSQL: %s", e)
        raise DatabaseConnectionError(
            "Không thể kết nối tới PostgreSQL. Vui lòng kiểm tra host/port/user/password "
            "trong file .env, đảm bảo server đang chạy và cho phép kết nối."
        ) from e
    except psycopg.errors.InsufficientPrivilege as e:
        raise DatabaseConnectionError(
            "Tài khoản Postgres không có quyền tạo extension 'vector'. "
            "Hãy nhờ DBA chạy: CREATE EXTENSION IF NOT EXISTS vector; với quyền superuser."
        ) from e
    except Exception as e:  # noqa: BLE001 - đây là điểm fail-fast, cần bắt mọi lỗi bất ngờ
        logger.exception("Lỗi không xác định khi kiểm tra kết nối DB.")
        raise DatabaseConnectionError(f"Lỗi không xác định khi kết nối DB: {e}") from e


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Khởi tạo model embedding chạy LOCAL (sentence-transformers), không gọi
    API Gemini -> không tốn quota/rate-limit của Google cho bước index tài liệu.

    Dùng @lru_cache vì nạp model từ đĩa/HuggingFace Hub khá tốn thời gian
    (vài giây) — Streamlit chạy lại toàn bộ script mỗi lần tương tác, nên nếu
    không cache, model sẽ bị nạp lại liên tục làm chậm UI.
    Lần chạy đầu tiên trên máy mới sẽ tự động tải model (~1-2GB tùy model) từ
    HuggingFace Hub về cache local (~/.cache/huggingface).
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=app_config.embedding_model,
            model_kwargs={"device": app_config.embedding_device},
            # bge-m3 và các model BGE khuyến nghị normalize để dùng cosine similarity.
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Không thể khởi tạo embedding model local: %s", app_config.embedding_model)
        raise EmbeddingError(
            f"Không thể khởi tạo mô hình embedding local '{app_config.embedding_model}': {e}"
        ) from e


def get_vector_store(collection_name: Optional[str] = None) -> PGVector:
    """Trả về 1 PGVector store gắn với 1 collection cụ thể (tạo mới nếu chưa có)."""
    embeddings = get_embeddings()
    try:
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name or db_config.collection_name,
            connection=db_config.connection_string,
            use_jsonb=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Không thể khởi tạo PGVector store.")
        raise DatabaseConnectionError(f"Không thể khởi tạo kho vector: {e}") from e


@retry(
    retry=retry_if_exception_type(EmbeddingError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def _add_batch(store: PGVector, texts: list[str], metadatas: list[dict]) -> None:
    """Thêm 1 batch vào store. Bọc lỗi thành EmbeddingError để tenacity retry
    (giữ lại cơ chế retry làm lưới an toàn cho lỗi I/O/OOM tạm thời khi encode
    local; không còn rate-limit API vì embedding chạy local)."""
    try:
        store.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:  # noqa: BLE001
        raise EmbeddingError(f"Lỗi khi tạo embedding cho batch: {e}") from e


def index_text_chunks(
    text_chunks: list[str],
    *,
    collection_name: Optional[str] = None,
    metadata_common: Optional[dict] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> None:
    """Nhúng (embed) và lưu các đoạn văn bản vào PGVector theo batch.

    So với bản gốc dùng FAISS + Gemini embedding + time.sleep(8) cứng, hàm này:
      - Dùng embedding LOCAL (bge-m3) -> không tốn quota Gemini, không cần
        sleep giữa các batch vì không có rate-limit API.
      - Vẫn giữ retry có backoff làm lưới an toàn cho lỗi I/O/tài nguyên tạm thời.
      - Lưu trực tiếp vào Postgres (không cần save_local/load_local thủ công).
      - Nhận metadata_common để gắn nhãn owner/lớp học cho từng đoạn văn bản.
    """
    if not text_chunks:
        logger.warning("Không có đoạn văn bản nào để index.")
        return

    store = get_vector_store(collection_name)
    batch_size = batch_size or app_config.embedding_batch_size
    metadata_common = metadata_common or {}
    total = len(text_chunks)

    for start in range(0, total, batch_size):
        batch = text_chunks[start : start + batch_size]
        metadatas = [dict(metadata_common) for _ in batch]
        _add_batch(store, batch, metadatas)  # có thể raise EmbeddingError sau khi hết retry
        if progress_callback:
            progress_callback(min((start + batch_size) / total, 1.0))

    logger.info(
        "Đã index xong %d đoạn văn bản vào collection '%s'.",
        total,
        collection_name or db_config.collection_name,
    )


def get_retriever(
    collection_name: Optional[str] = None,
    k: int = 10,
    filter: Optional[dict] = None,
):
    """Trả về retriever thô (chưa rerank).

    `filter` áp dụng lên metadata (vd {'session_id': ...}, hoặc trong tương lai
    {'owner_id': user_id} / {'allowed_roles': {'$contains': role}}) — đây là cơ
    chế row-level filtering cho phân quyền nhiều người dùng trên cùng 1 collection.
    """
    store = get_vector_store(collection_name)
    search_kwargs = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return store.as_retriever(search_kwargs=search_kwargs)
