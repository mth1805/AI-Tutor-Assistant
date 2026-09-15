"""
tools.py
--------
Định nghĩa các Tool cho LangGraph Agent:
  - web_search_tool: tìm kiếm web (DuckDuckGo).
  - python_math_tool: chạy code Python cho tính toán.
  - document_reader_tool (factory): hybrid search (vector + full-text) trên
    tài liệu đã index, rerank bằng Cohere, kèm trích dẫn nguồn (tên file).

document_reader_tool được tạo qua factory (make_document_reader_tool) thay vì
là 1 tool tĩnh toàn cục, để mỗi phiên/agent có thể gắn với đúng collection và
metadata filter của mình (cô lập dữ liệu giữa các workspace).
"""

from __future__ import annotations

import time

from langchain.tools import tool
from langchain_cohere import CohereRerank
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL

from config import app_config, get_logger, log_duration
from database import hybrid_search
from exceptions import DatabaseConnectionError

logger = get_logger(__name__)

_web_search = DuckDuckGoSearchRun()


@tool("web_search_tool")
@log_duration("web_search_tool")
def web_search_tool(query: str) -> str:
    """Sử dụng khi cần tìm kiếm thông tin cập nhật, tin tức, hoặc các kiến thức
    không có trong tài liệu đã tải lên."""
    try:
        return _web_search.run(query)
    except Exception as e:
        logger.exception("Lỗi web search cho query: %s", query)
        return f"Không thể tìm kiếm web lúc này (lỗi tạm thời): {e}"


@tool("python_math_tool")
@log_duration("python_math_tool")
def python_math_tool(code: str) -> str:
    """Sử dụng để tính toán các phép toán phức tạp, công thức khoa học dữ liệu,
    hoặc chạy code Python. Đầu vào phải là code Python hợp lệ.

    CẢNH BÁO BẢO MẬT (xem README): PythonREPL thực thi code tùy ý. Trong môi
    trường production, tool này PHẢI được chạy trong sandbox cô lập (container
    riêng, không có quyền mạng/filesystem ra ngoài) để tránh rủi ro thực thi mã
    độc nếu người dùng cố tình prompt-inject agent.
    """
    repl = PythonREPL()
    try:
        result = repl.run(code)
        return f"Kết quả chạy code: {result}"
    except Exception as e:  # noqa: BLE001
        return f"Lỗi khi chạy code: {e}"


def _format_with_citations(docs) -> str:
    """Định dạng các đoạn tài liệu kèm tên nguồn, để LLM có thể trích dẫn
    trong câu trả lời cuối cùng thay vì trả lời mà không rõ lấy từ đâu."""
    parts = []
    for d in docs:
        source = d.metadata.get("source", "tài liệu đã tải lên")
        parts.append(f"[Nguồn: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def make_document_reader_tool(
    *,
    collection_name: str | None = None,
    metadata_filter: dict | None = None,
):
    """Factory tạo document_reader_tool gắn với 1 collection/filter cụ thể.

    Cho phép mỗi phiên chat (theo workspace) chỉ truy vấn đúng phạm vi tài
    liệu của mình — nền tảng cho việc mở rộng phân quyền đa người dùng.
    """

    @tool("document_reader_tool")
    def document_reader_tool(query: str) -> str:
        """LUÔN ƯU TIÊN sử dụng công cụ này đầu tiên để tìm kiếm câu trả lời từ
        tài liệu đã tải lên."""
        logger.info(
            "document_reader_tool GỌI: query=%r collection=%s filter=%s",
            query,
            collection_name,
            metadata_filter,
        )
        start = time.perf_counter()
        try:
            candidates = hybrid_search(query, collection_name=collection_name, filter=metadata_filter, k=10)
        except DatabaseConnectionError as e:
            logger.error("document_reader_tool LỖI DB: %s", e)
            return f"Không thể truy vấn kho tài liệu lúc này: {e}"
        except Exception as e:
            logger.exception("document_reader_tool LỖI KHÔNG XÁC ĐỊNH khi truy vấn")
            return f"Lỗi khi đọc tài liệu: {e}"
        finally:
            logger.info("⏱ hybrid_search (trong tool) mất %.2fs", time.perf_counter() - start)

        if not candidates:
            return "Không tìm thấy nội dung liên quan trong tài liệu đã tải lên."

        top_docs = candidates[:3]
        try:
            compressor = CohereRerank(
                model=app_config.rerank_model,
                cohere_api_key=app_config.cohere_api_key,
                top_n=5,
            )
            reranked = compressor.compress_documents(candidates, query)
            if reranked:
                top_docs = list(reranked)
        except Exception as e:  # noqa: BLE001
            # Cohere lỗi (hết quota, sai key, downtime...) KHÔNG được làm hỏng
            # toàn bộ câu trả lời — fallback về top-k của hybrid search (chưa
            # rerank) vẫn tốt hơn nhiều so với báo lỗi hoàn toàn cho người dùng.
            logger.warning("Cohere rerank lỗi (%s) -> dùng kết quả hybrid search không rerank.", e)

        logger.info("document_reader_tool KẾT QUẢ: %d đoạn liên quan", len(top_docs))
        return _format_with_citations(top_docs)

    return document_reader_tool
