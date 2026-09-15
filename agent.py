"""
agent.py
--------
Khởi tạo LLM (Gemini) và Agent (LangGraph create_react_agent), cung cấp:
  - ask_agent(): gửi câu hỏi, trả về string đầy đủ (không streaming).
  - stream_agent(): generator sinh dần từng đoạn text — dùng cho streaming
    trong UI (st.write_stream), cải thiện cảm giác chờ khi câu trả lời dài.
"""

from __future__ import annotations

import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import app_config, get_logger, log_duration
from exceptions import AgentError
from tools import make_document_reader_tool, python_math_tool, web_search_tool

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Bạn là một Trợ lý AI Gia sư thông minh. Bạn có các công cụ để tra cứu tài liệu, "
    "tính toán và tìm kiếm web.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. Với BẤT KỲ câu hỏi nào liên quan đến nội dung tài liệu, tóm tắt, hoặc kiến thức "
    "trong bài học, BẠN BẮT BUỘC PHẢI gọi document_reader_tool TRƯỚC TIÊN, kể cả khi bạn "
    "không chắc tài liệu đã được tải lên hay chưa. KHÔNG được tự trả lời rằng chưa có tài "
    "liệu nào được tải lên nếu bạn chưa thực sự gọi công cụ này để kiểm tra.\n"
    "2. Nếu document_reader_tool trả về 'Không tìm thấy nội dung liên quan', hãy nói rõ "
    "điều đó với người dùng và hỏi họ có muốn thử tìm kiếm trên web (web_search_tool) không, "
    "thay vì suy đoán rằng chưa có tài liệu nào được tải lên.\n"
    "3. Kết quả từ document_reader_tool có kèm nhãn '[Nguồn: tên_file]' trước mỗi đoạn — "
    "khi trả lời, LUÔN trích dẫn nguồn tương ứng (vd: 'Theo tài liệu abc.pdf, ...') để "
    "người dùng biết thông tin lấy từ đâu.\n"
    "4. Khi tính toán, BẮT BUỘC dùng python_math_tool.\n"
    "5. Chỉ dùng web_search_tool khi tài liệu không có thông tin liên quan."
)


def get_llm() -> ChatGoogleGenerativeAI:
    try:
        return ChatGoogleGenerativeAI(
            model=app_config.llm_model,
            temperature=app_config.llm_temperature,
            google_api_key=app_config.google_api_key,
            max_retries=5,  # Tự động retry tối đa 5 lần khi gặp lỗi 503/429
            timeout=60,     # Thời gian timeout chờ phản hồi
        )
    except Exception as e:
        logger.exception("Không thể khởi tạo Gemini LLM")
        raise AgentError(f"Không thể khởi tạo mô hình ngôn ngữ: {e}") from e


def build_agent(
    *,
    collection_name: str | None = None,
    metadata_filter: dict | None = None,
):
    """Tạo agent executor. collection_name/metadata_filter cho phép mỗi
    workspace có phạm vi tài liệu riêng — xem tools.make_document_reader_tool.
    """
    try:
        llm = get_llm()
        tool_list = [
            make_document_reader_tool(collection_name=collection_name, metadata_filter=metadata_filter),
            python_math_tool,
            web_search_tool,
        ]
        return create_react_agent(llm, tool_list, prompt=SYSTEM_PROMPT)
    except AgentError:
        raise
    except Exception as e:
        logger.exception("Không thể khởi tạo Agent")
        raise AgentError(f"Không thể khởi tạo agent: {e}") from e


def _log_agent_steps(messages) -> None:
    """Log lại từng bước agent đã thực hiện — hữu ích để debug: agent có thực
    sự gọi tool nào không, tool trả về gì trước khi LLM tổng hợp câu trả lời."""
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            logger.info("AGENT gọi tool: %s", [tc.get("name") for tc in tool_calls])
        if getattr(m, "type", None) == "tool":
            logger.info(
                "AGENT nhận kết quả từ tool '%s': %s",
                getattr(m, "name", "?"),
                str(m.content)[:300],
            )


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return content or ""


@log_duration("ask_agent")
def ask_agent(agent, question: str) -> str:
    """Gửi câu hỏi tới agent (không streaming), luôn trả về string an toàn
    cho UI, không bao giờ ném exception ra ngoài."""
    try:
        response = agent.invoke({"messages": [("user", question)]})
        messages = response["messages"]
        _log_agent_steps(messages)
        return _extract_text(messages[-1].content)
    except Exception as e:
        logger.exception("Lỗi khi agent xử lý câu hỏi: %s", question)
        return f"⚠️ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn. Chi tiết: {e}"


def stream_agent(agent, question: str):
    """Generator sinh dần từng đoạn text câu trả lời cuối cùng của agent
    (token streaming), dùng cho st.write_stream ở UI.

    Chỉ yield phần content sinh ra ở bước LLM cuối (node 'agent' trong graph
    của create_react_agent), bỏ qua các bước tool-calling trung gian để người
    dùng không thấy dữ liệu tool lộn xộn giữa chừng.

    Nếu streaming lỗi hoặc không được hỗ trợ (phụ thuộc phiên bản langgraph),
    tự động fallback về ask_agent() không streaming.
    """
    start = time.perf_counter()
    try:
        streamed_any = False
        for chunk, metadata in agent.stream(
            {"messages": [("user", question)]},
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "agent":
                continue
            text = _extract_text(getattr(chunk, "content", None))
            if text:
                streamed_any = True
                yield text
        if not streamed_any:
            # Không có token nào được stream (vd toàn bộ là tool-call) ->
            # fallback để vẫn có câu trả lời thay vì im lặng.
            yield ask_agent(agent, question)
    except Exception:
        logger.exception("Lỗi khi streaming câu trả lời, fallback về ask_agent không streaming.")
        yield ask_agent(agent, question)
    finally:
        logger.info("⏱ stream_agent hoàn thành sau %.2fs", time.perf_counter() - start)
