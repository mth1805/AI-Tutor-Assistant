"""
database.py
-----------
Quản lý toàn bộ tương tác với PostgreSQL + pgvector thông qua LangChain
(langchain-postgres), cộng thêm 2 phần bổ sung chạy bằng SQL thuần (psycopg):
  - Hybrid search: kết hợp vector search (pgvector) + full-text search
    (Postgres tsvector) qua Reciprocal Rank Fusion — tốt hơn vector thuần khi
    câu hỏi chứa từ khóa/số liệu chính xác (công thức, tên riêng, số liệu).
  - Lưu trữ lịch sử chat theo workspace_id, để refresh trang không mất hội thoại.

Tách biệt hoàn toàn khỏi UI và Agent để có thể unit test độc lập, và tái sử
dụng trong script batch-import tài liệu chạy ngoài Streamlit.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import psycopg
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import app_config, db_config, get_logger, log_duration
from document_processor import TextChunk
from exceptions import DatabaseConnectionError, EmbeddingError

logger = get_logger(__name__)

CHAT_TABLE = "ai_tutor_chat_messages"
WORKSPACE_TABLE = "ai_tutor_workspaces"

def _connect() -> psycopg.Connection:
    return psycopg.connect(
        host=db_config.host,
        port=db_config.port,
        dbname=db_config.database,
        user=db_config.user,
        password=db_config.password,
        connect_timeout=5,
    )


def check_database_connection() -> None:
    """Kiểm tra kết nối Postgres, đảm bảo extension pgvector tồn tại, và tạo
    sẵn bảng lịch sử chat. Nên gọi 1 lần khi app khởi động (fail-fast)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {CHAT_TABLE} (
                        id SERIAL PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {WORKSPACE_TABLE} (
                        workspace_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT now()
                    );
                    """
                )
        logger.info("Kết nối PostgreSQL/pgvector thành công.")
    except psycopg.OperationalError as e:
        logger.error("Không thể kết nối PostgreSQL: %s", e)
        raise DatabaseConnectionError(
            "Không thể kết nối tới PostgreSQL. Vui lòng kiểm tra host/port/user/password "
            "trong file .env (hoặc Secrets trên hosting), đảm bảo server đang chạy và cho "
            "phép kết nối từ nơi app đang deploy."
        ) from e
    except psycopg.errors.InsufficientPrivilege as e:
        raise DatabaseConnectionError(
            "Tài khoản Postgres không có quyền tạo extension 'vector'. "
            "Hãy nhờ DBA chạy: CREATE EXTENSION IF NOT EXISTS vector; với quyền superuser."
        ) from e
    except Exception as e:  # noqa: BLE001 - đây là điểm fail-fast, cần bắt mọi lỗi bất ngờ
        logger.exception("Lỗi không xác định khi kiểm tra kết nối DB.")
        raise DatabaseConnectionError(f"Lỗi không xác định khi kết nối DB: {e}") from e


# ---------------------------------------------------------------------------
# Embedding & Vector store (pgvector)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Khởi tạo model embedding chạy LOCAL (sentence-transformers), không gọi
    API Gemini -> không tốn quota/rate-limit của Google cho bước index.

    Dùng @lru_cache vì nạp model từ đĩa/HuggingFace Hub khá tốn thời gian —
    Streamlit chạy lại toàn bộ script mỗi lần tương tác, nên nếu không cache,
    model sẽ bị nạp lại liên tục làm chậm UI.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=app_config.embedding_model,
            model_kwargs={"device": app_config.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Không thể khởi tạo embedding model local: %s", app_config.embedding_model)
        raise EmbeddingError(
            f"Không thể khởi tạo mô hình embedding local '{app_config.embedding_model}': {e}"
        ) from e


def get_vector_store(collection_name: str | None = None) -> PGVector:
    """Trả về 1 PGVector store gắn với 1 collection cụ thể (tạo mới nếu chưa có)."""
    if collection_name is None:
        logger.warning("CẢNH BÁO: collection_name bị None!")
        
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
    try:
        store.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:  # noqa: BLE001
        raise EmbeddingError(f"Lỗi khi tạo embedding cho batch: {e}") from e


@log_duration("index_chunks")
def index_chunks(
    chunks: list[TextChunk],
    *,
    collection_name: str | None = None,
    metadata_common: dict | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Nhúng (embed) và lưu các TextChunk vào PGVector theo batch.

    Mỗi chunk có metadata riêng (vd {"source": "bai1.pdf"}) được gộp với
    metadata_common (vd {"workspace_id": ...}) — cho phép vừa trích dẫn nguồn,
    vừa lọc theo không gian làm việc khi truy vấn.
    """
    if not chunks:
        logger.warning("Không có đoạn văn bản nào để index.")
        return

    # QUAN TRỌNG: phải gán giá trị mặc định TRƯỚC khi dùng bên dưới — gọi
    # index_chunks(chunks) mà không truyền metadata_common (None) rồi
    # dict.update(None) ngay sau đó sẽ raise TypeError. Đây từng là 1 bug
    # thật, có test hồi quy ở tests/test_database.py.
    metadata_common = metadata_common or {}

    for chunk in chunks:
        # Nếu chunk chưa có metadata, tạo mới
        if not chunk.metadata:
            chunk.metadata = {}
        
        # Chỉ lấy 'source' từ chunk, sau đó cập nhật thêm 'workspace_id' vào
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata = {"source": source}
        chunk.metadata.update(metadata_common)

    store = get_vector_store(collection_name)
    batch_size = batch_size or app_config.embedding_batch_size
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        metadatas = [c.metadata for c in batch]
        _add_batch(store, texts, metadatas)  # có thể raise EmbeddingError sau khi hết retry
        if progress_callback:
            progress_callback(min((start + batch_size) / total, 1.0))

    logger.info(
        "Đã index xong %d đoạn văn bản vào collection '%s'.",
        total,
        collection_name or db_config.collection_name,
    )


def collection_has_documents(collection_name: str) -> bool:
    """Kiểm tra 1 collection đã có dữ liệu chưa — dùng để xác định is_indexed
    khi người dùng quay lại workspace cũ qua link (không dựa vào session_state,
    vốn mất khi Streamlit khởi động lại process)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        LIMIT 1
                    );
                    """,
                    (collection_name,),
                )
                row = cur.fetchone()
                return bool(row and row[0])
    except Exception as e:  # noqa: BLE001
        logger.warning("Không kiểm tra được collection '%s' đã có dữ liệu chưa: %s", collection_name, e)
        return False


# ---------------------------------------------------------------------------
# Hybrid search: vector (pgvector) + full-text (Postgres tsvector) qua RRF
# ---------------------------------------------------------------------------


def _keyword_search(
    query: str, *, collection_name: str | None, filter: dict | None, k: int
) -> list[Document]:
    """Full-text search thô trên cột 'document' của bảng pgvector nội bộ.

    Dùng text search config 'simple' (không stemming) vì Postgres mặc định
    không có dictionary tiếng Việt — vẫn hữu ích để bắt các từ khóa/số liệu
    chính xác mà vector search đôi khi bỏ sót.
    """
    coll = collection_name or db_config.collection_name
    sql = """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND to_tsvector('simple', e.document) @@ plainto_tsquery('simple', %s)
        ORDER BY ts_rank(to_tsvector('simple', e.document), plainto_tsquery('simple', %s)) DESC
        LIMIT %s;
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (coll, query, query, k))
            rows = cur.fetchall()

    results: list[Document] = []
    for document_text, cmetadata in rows:
        meta = cmetadata or {}
        if filter and not all(meta.get(fk) == fv for fk, fv in filter.items()):
            continue
        results.append(Document(page_content=document_text, metadata=meta))
    return results


def _reciprocal_rank_fusion(
    result_lists: list[list[Document]], *, k: int, rrf_k: int = 60
) -> list[Document]:
    """Gộp nhiều danh sách kết quả xếp hạng khác nhau (vector, keyword) thành
    1 danh sách duy nhất bằng công thức RRF: score = sum(1 / (rrf_k + rank))."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for docs in result_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content[:300]  # dedupe xấp xỉ theo nội dung
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map.setdefault(key, doc)
    ranked_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [doc_map[key] for key in ranked_keys[:k]]


def hybrid_search(
    query: str,
    *,
    collection_name: str | None = None,
    filter: dict | None = None,
    k: int = 10,
) -> list[Document]:
    """Truy vấn kết hợp vector search + full-text search, gộp bằng RRF.

    Nếu full-text search lỗi (vd khác schema bảng do phiên bản langchain-postgres
    khác nhau), tự động fallback về vector-only để không làm gián đoạn trải
    nghiệm người dùng — đây là lý do lỗi được bắt và chỉ log warning, không raise.
    """
    store = get_vector_store(collection_name)
    vector_docs = store.similarity_search(query, k=k, filter=filter)

    keyword_docs: list[Document] = []
    try:
        keyword_docs = _keyword_search(query, collection_name=collection_name, filter=filter, k=k)
    except Exception as e:  # noqa: BLE001
        logger.warning("Full-text search lỗi, chỉ dùng kết quả vector search: %s", e)

    if not keyword_docs:
        return vector_docs[:k]
    return _reciprocal_rank_fusion([vector_docs, keyword_docs], k=k)


# ---------------------------------------------------------------------------
# Lịch sử chat (theo workspace_id)
# ---------------------------------------------------------------------------


def save_chat_message(workspace_id: str, role: str, content: str) -> None:
    """Lưu 1 tin nhắn vào lịch sử chat. Lỗi ở đây KHÔNG raise ra ngoài —
    mất khả năng lưu lịch sử không nên làm gián đoạn cuộc trò chuyện hiện tại."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {CHAT_TABLE} (workspace_id, role, content) VALUES (%s, %s, %s);",
                    (workspace_id, role, content),
                )
    except Exception:  # noqa: BLE001
        logger.exception("Không thể lưu tin nhắn chat cho workspace '%s'.", workspace_id)


def get_chat_history(workspace_id: str, limit: int = 200) -> list[dict]:
    """Tải lịch sử chat của 1 workspace. Trả về [] nếu lỗi (vd DB tạm thời
    không kết nối được) thay vì làm crash UI khi mới load trang."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT role, content FROM {CHAT_TABLE} "
                    f"WHERE workspace_id = %s ORDER BY created_at ASC LIMIT %s;",
                    (workspace_id, limit),
                )
                rows = cur.fetchall()
        return [{"role": r, "content": c} for r, c in rows]
    except Exception:  # noqa: BLE001
        logger.exception("Không thể tải lịch sử chat cho workspace '%s'.", workspace_id)
        return []

@log_duration("get_all_workspaces")
def get_all_workspaces() -> list[dict]:
    """Lấy danh sách tất cả các workspace đã lưu."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT workspace_id, title FROM {WORKSPACE_TABLE} ORDER BY created_at DESC;")
                rows = cur.fetchall()
        logger.debug("Tìm thấy %d workspace.", len(rows))
        return [{"id": r[0], "title": r[1]} for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.warning("Không thể lấy danh sách workspace: %s", e)
        return []


@log_duration("ensure_workspace_exists")
def ensure_workspace_exists(workspace_id: str, title: str | None = None) -> None:
    """Đảm bảo workspace luôn có trong bảng quản lý khi được truy cập.
    Log ở mức INFO CHỈ khi thực sự tạo mới (rowcount == 1 sau ON CONFLICT DO
    NOTHING), tránh làm ngập log vì hàm này được gọi ở mọi lần Streamlit rerun."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {WORKSPACE_TABLE} (workspace_id, title) "
                    f"VALUES (%s, %s) ON CONFLICT (workspace_id) DO NOTHING;",
                    (workspace_id, title or workspace_id),
                )
                if cur.rowcount == 1:
                    logger.info("Đã tạo workspace mới: '%s' (title=%r)", workspace_id, title)
    except Exception as e:  # noqa: BLE001
        logger.warning("Không thể lưu workspace vào bảng quản lý: %s", e)


@log_duration("rename_workspace")
def rename_workspace(workspace_id: str, new_title: str) -> None:
    """Đổi tên hiển thị của workspace."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {WORKSPACE_TABLE} SET title = %s WHERE workspace_id = %s;",
                    (new_title, workspace_id),
                )
        logger.info("Đã đổi tên workspace '%s' -> %r.", workspace_id, new_title)
    except Exception:  # noqa: BLE001
        logger.exception("Không thể đổi tên workspace '%s'", workspace_id)


@log_duration("delete_workspace")
def delete_workspace(workspace_id: str) -> None:
    """Xóa sạch mọi dữ liệu liên quan đến workspace: tin nhắn, embedding, và collection."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                # 1. Xóa lịch sử chat (không có ràng buộc FK nên xóa thoải mái)
                cur.execute(f"DELETE FROM {CHAT_TABLE} WHERE workspace_id = %s;", (workspace_id,))
                
                # 2. Xóa bảng quản lý workspace
                cur.execute(f"DELETE FROM {WORKSPACE_TABLE} WHERE workspace_id = %s;", (workspace_id,))
                
                # 3. Xóa các đoạn vector embedding 
                # Phải xóa embedding trước vì nó có khóa ngoại tham chiếu đến collection
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    USING langchain_pg_collection
                    WHERE langchain_pg_embedding.collection_id = langchain_pg_collection.uuid
                      AND langchain_pg_collection.name = %s;
                    """,
                    (workspace_id,),
                )
                
                # 4. Xóa collection tương ứng (sau khi embedding con đã bị xóa)
                cur.execute(
                    "DELETE FROM langchain_pg_collection WHERE name = %s;",
                    (workspace_id,),
                )
                
                conn.commit()
        logger.info("Đã xóa hoàn toàn workspace '%s' và dữ liệu liên quan.", workspace_id)
    except Exception as e:
        logger.exception("Lỗi khi xóa workspace '%s': %s", workspace_id, e)
        raise DatabaseConnectionError(f"Không thể xóa workspace: {e}") from e

@log_duration("get_indexed_files")
def get_indexed_files(workspace_id: str) -> list[str]:
    """Lấy danh sách tên file đã được index từ metadata JSONB."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                # Tìm tất cả các giá trị unique của key 'source' trong cmetadata
                logger.debug("Đang debug SQL: tìm file cho collection name = '%s'", workspace_id)
                cur.execute(
                    """
                    SELECT DISTINCT e.cmetadata->>'source'
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s AND e.cmetadata->>'source' IS NOT NULL;
                    """,
                    (workspace_id,),
                )
                rows = cur.fetchall()
        
        # Lọc bỏ giá trị None và trả về danh sách
        files = [r[0] for r in rows if r[0]]
        logger.debug("Workspace '%s' có %d file: %s", workspace_id, len(files), files)
        return files
    except Exception as e:
        logger.warning("Không thể lấy danh sách file của workspace '%s': %s", workspace_id, e)
        return []
