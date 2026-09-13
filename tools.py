"""
tools.py
--------
Định nghĩa các Tool cho LangGraph Agent:
  - web_search_tool: tìm kiếm web (DuckDuckGo).
  - python_math_tool: chạy code Python cho tính toán.
  - document_reader_tool (factory): truy vấn tài liệu đã index trong pgvector,
    có rerank bằng Cohere.

document_reader_tool được tạo qua factory (make_document_reader_tool) thay vì
là 1 tool tĩnh toàn cục, để mỗi phiên/agent có thể gắn với đúng collection và
metadata filter của mình (cô lập dữ liệu giữa các người dùng).
"""
from __future__ import annotations

from typing import Optional

from langchain.tools import tool
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL

from config import app_config, get_logger
from database import get_retriever
from exceptions import DatabaseConnectionError

logger = get_logger(__name__)

_web_search = DuckDuckGoSearchRun()


@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """Sử dụng khi cần tìm kiếm thông tin cập nhật, tin tức, hoặc các kiến thức
    không có trong tài liệu đã tải lên."""
    try:
        return _web_search.run(query)
    except Exception as e:  # noqa: BLE001
        logger.exception("Lỗi web search cho query: %s", query)
        return f"Không thể tìm kiếm web lúc này (lỗi tạm thời): {e}"


@tool("python_math_tool")
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


def make_document_reader_tool(
    *,
    collection_name: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
):
    """Factory tạo document_reader_tool gắn với 1 collection/filter cụ thể.

    Cho phép mỗi phiên chat (theo user/lớp học) chỉ truy vấn đúng phạm vi tài
    liệu của mình — nền tảng cho việc mở rộng phân quyền đa người dùng.
    """

    @tool("document_reader_tool")
    def document_reader_tool(query: str) -> str:
        """LUÔN ƯU TIÊN sử dụng công cụ này đầu tiên để tìm kiếm câu trả lời từ
        tài liệu đã tải lên."""
        logger.info(
            "document_reader_tool GỌI: query=%r collection=%s filter=%s",
            query, collection_name, metadata_filter,
        )
        try:
            base_retriever = get_retriever(
                collection_name=collection_name, k=10, filter=metadata_filter
            )
            compressor = CohereRerank(
                model=app_config.rerank_model,
                cohere_api_key=app_config.cohere_api_key,
                top_n=3,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
            docs = compression_retriever.invoke(query)
            logger.info(
                "document_reader_tool KẾT QUẢ: %d đoạn liên quan sau rerank", len(docs)
            )
            if not docs:
                return "Không tìm thấy nội dung liên quan trong tài liệu đã tải lên."
            return "\n\n".join(d.page_content for d in docs)
        except DatabaseConnectionError as e:
            logger.error("document_reader_tool LỖI DB: %s", e)
            return f"Không thể truy vấn kho tài liệu lúc này: {e}"
        except Exception as e:  # noqa: BLE001
            logger.exception("document_reader_tool LỖI KHÔNG XÁC ĐỊNH")
            return f"Lỗi khi đọc tài liệu: {e}"

    return document_reader_tool