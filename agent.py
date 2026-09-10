"""
agent.py
--------
Khởi tạo LLM (Gemini) và Agent (LangGraph create_react_agent), và cung cấp
hàm ask_agent() để gửi câu hỏi và luôn trả về string an toàn cho UI.
"""
from __future__ import annotations

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import app_config, get_logger
from exceptions import AgentError
from tools import make_document_reader_tool, python_math_tool, web_search_tool

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Bạn là một Trợ lý AI Gia sư thông minh. Bạn có các công cụ để tra cứu tài liệu, "
    "tính toán và tìm kiếm web. Hãy ưu tiên dùng tài liệu trước, nếu không có mới dùng web. "
    "Khi tính toán, BẮT BUỘC dùng python_math_tool."
)


def get_llm() -> ChatGoogleGenerativeAI:
    try:
        return ChatGoogleGenerativeAI(
            model=app_config.llm_model,
            temperature=app_config.llm_temperature,
            google_api_key=app_config.google_api_key,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Không thể khởi tạo Gemini LLM")
        raise AgentError(f"Không thể khởi tạo mô hình ngôn ngữ: {e}") from e


def build_agent(
    *,
    collection_name: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
):
    """Tạo agent executor. collection_name/metadata_filter cho phép mỗi phiên
    (mỗi user/lớp học) có phạm vi tài liệu riêng — xem tools.make_document_reader_tool.
    """
    try:
        llm = get_llm()
        tool_list = [
            make_document_reader_tool(
                collection_name=collection_name, metadata_filter=metadata_filter
            ),
            python_math_tool,
            web_search_tool,
        ]
        return create_react_agent(llm, tool_list, prompt=SYSTEM_PROMPT)
    except AgentError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Không thể khởi tạo Agent")
        raise AgentError(f"Không thể khởi tạo agent: {e}") from e


def ask_agent(agent, question: str) -> str:
    """Gửi câu hỏi tới agent, chuẩn hóa các định dạng response khác nhau
    (string hoặc list[dict] content block) và bắt lỗi để UI luôn nhận được
    1 chuỗi string thân thiện, không bao giờ crash giữa phiên chat."""
    try:
        response = agent.invoke({"messages": [("user", question)]})
        last_message = response["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return content
    except Exception as e:  # noqa: BLE001
        logger.exception("Lỗi khi agent xử lý câu hỏi: %s", question)
        return (
            "⚠️ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn. "
            f"Chi tiết: {e}"
        )
