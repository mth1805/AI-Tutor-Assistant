"""
app.py
------
Entry point Streamlit. CHỈ chịu trách nhiệm điều phối UI (orchestration):
  1. Kiểm tra cấu hình & kết nối DB khi khởi động (fail-fast).
  2. Gọi các module ui/* để vẽ giao diện.
  3. Quản lý vòng đời agent trong session_state.

Toàn bộ logic nghiệp vụ (DB, xử lý tài liệu, agent, tools) nằm ở các module
riêng — file này không nên vượt quá ~100 dòng khi mở rộng thêm tính năng.
"""
from __future__ import annotations

import uuid

import streamlit as st

from agent import build_agent
from config import get_config_error, get_logger
from database import check_database_connection
from exceptions import AITutorError
from ui.chat import render_chat
from ui.preview import render_preview
from ui.sidebar import render_sidebar
from ui.styles import CUSTOM_CSS

logger = get_logger(__name__)


def _get_session_id() -> str:
    """ID phiên duy nhất cho mỗi tab trình duyệt, dùng làm collection_name /
    metadata filter để cô lập dữ liệu giữa các người dùng đang dùng chung app.

    Khi tích hợp hệ thống đăng nhập thật (xem README - "Mở rộng phân quyền"),
    thay hàm này bằng việc lấy user_id đã xác thực (vd từ st.session_state
    sau khi qua middleware auth, hoặc từ SSO/OAuth token).
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:12]}"
    return st.session_state["session_id"]


def _bootstrap() -> bool:
    """Kiểm tra cấu hình & kết nối DB một lần khi khởi động (fail-fast).
    Trả về False và dừng render nếu môi trường chưa sẵn sàng, tránh để lỗi
    xuất hiện đột ngột giữa phiên làm việc của người dùng."""
    config_error = get_config_error()
    if config_error:
        st.error(f"⚠️ Lỗi cấu hình: {config_error}")
        st.stop()
        return False

    if st.session_state.get("db_checked"):
        return True

    try:
        check_database_connection()
        st.session_state["db_checked"] = True
        return True
    except AITutorError as e:
        st.error(f"⚠️ {e}")
        st.stop()
        return False


def _get_or_build_agent(session_id: str):
    """Cache agent trong session_state, chỉ rebuild khi session_id đổi
    (tránh khởi tạo lại LLM/tools mỗi lần Streamlit rerun script)."""
    if st.session_state.get("agent_session_id") == session_id and "agent" in st.session_state:
        return st.session_state["agent"]

    agent = build_agent(
        collection_name=session_id,
        metadata_filter={"session_id": session_id},
    )
    st.session_state["agent"] = agent
    st.session_state["agent_session_id"] = session_id
    return agent


def main() -> None:
    st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.header("🎓 Trợ lý Gia sư AI - Đọc tài liệu")

    if not _bootstrap():
        return

    session_id = _get_session_id()
    metadata_common = {"session_id": session_id}

    docs = render_sidebar(collection_name=session_id, metadata_common=metadata_common)

    col1, col2 = st.columns([1, 1])

    with col1:
        render_preview(docs)

    with col2:
        try:
            agent = _get_or_build_agent(session_id)
        except AITutorError as e:
            st.error(f"⚠️ Không thể khởi tạo trợ lý AI: {e}")
            return

        render_chat(agent, is_indexed=st.session_state.get("is_indexed", False))


if __name__ == "__main__":
    main()
