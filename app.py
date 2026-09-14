"""
app.py
------
Entry point Streamlit. CHỈ chịu trách nhiệm điều phối UI (orchestration):
  1. Kiểm tra cấu hình & kết nối DB khi khởi động (fail-fast).
  2. Xác định workspace hiện tại (qua URL) và tải lại lịch sử chat từ Postgres.
  3. Gọi các module ui/* để vẽ giao diện.
  4. Quản lý vòng đời agent trong session_state.
"""
from __future__ import annotations

import uuid

import streamlit as st

from agent import build_agent
from config import get_config_error, get_logger, set_log_context
from database import check_database_connection, collection_has_documents, get_chat_history, get_indexed_files
from exceptions import AITutorError
from ui.chat import render_chat
from ui.preview import render_preview
from ui.sidebar import render_sidebar
from ui.styles import CUSTOM_CSS
from storage import get_cloud_files_for_preview

logger = get_logger(__name__)


def _get_workspace_id() -> str:
    """Workspace ID lấy từ URL query param 'ws'. Nếu chưa có, tạo mới và gắn
    vào URL. Nhờ đó tài liệu + lịch sử chat còn nguyên khi refresh trang, và
    có thể chia sẻ link để dùng chung 1 bộ tài liệu (vd cho cả lớp học).

    Khi tích hợp hệ thống đăng nhập thật (xem README - "Mở rộng phân quyền"),
    thay hàm này bằng việc lấy user_id đã xác thực.
    """
    ws = st.query_params.get("ws")
    if not ws:
        ws = f"ws_{uuid.uuid4().hex[:12]}"
        st.query_params["ws"] = ws
    return ws


def _bootstrap() -> bool:
    """Kiểm tra cấu hình & kết nối DB một lần khi khởi động (fail-fast)."""
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


def _get_or_build_agent(workspace_id: str):
    """Cache agent trong session_state, chỉ rebuild khi workspace đổi."""
    if st.session_state.get("agent_workspace_id") == workspace_id and "agent" in st.session_state:
        return st.session_state["agent"]

    agent = build_agent(
        collection_name=workspace_id,
        metadata_filter={"workspace_id": workspace_id},
    )
    st.session_state["agent"] = agent
    st.session_state["agent_workspace_id"] = workspace_id
    return agent


def _load_workspace_state(workspace_id: str) -> None:
    """Tải lịch sử chat + trạng thái is_indexed từ Postgres khi vào 1
    workspace mới (kể cả sau khi Streamlit process bị khởi động lại, vì
    session_state không còn nhưng dữ liệu trong Postgres vẫn còn)."""
    if st.session_state.get("loaded_workspace") == workspace_id:
        return
    st.session_state["messages"] = get_chat_history(workspace_id)
    st.session_state["is_indexed"] = collection_has_documents(workspace_id)
    st.session_state["loaded_workspace"] = workspace_id


def main() -> None:
    st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.header("🎓 Trợ lý Gia sư AI - Đọc tài liệu")

    if not _bootstrap():
        return

    workspace_id = _get_workspace_id()
    metadata_common = {"workspace_id": workspace_id}

    # Gắn workspace_id vào mọi dòng log từ đây trở đi (xem config.set_log_context)
    # -> khi debug production nhiều người dùng chung app, có thể lọc log theo
    # đúng phiên làm việc (tạo/đổi tên/xóa workspace, index tài liệu, chat...)
    # thay vì phải đoán dòng log nào của ai.
    set_log_context(workspace_id)

    _load_workspace_state(workspace_id)

    docs = render_sidebar(workspace_id=workspace_id, metadata_common=metadata_common)
    
    # --- TÍCH HỢP CLOUD STORAGE: LẤY LẠI FILE TỪ CLOUD ĐỂ XEM TRƯỚC ---
    cloud_files = get_cloud_files_for_preview(workspace_id)
    
    # Ưu tiên file người dùng mới upload ở phiên hiện tại, nếu không có thì lấy từ Cloud Object Storage lên
    active_docs = docs if docs else cloud_files
    indexed_file_names = get_indexed_files(workspace_id)

    # Khởi tạo trạng thái ẩn/hiện cột xem trước tài liệu nếu chưa có
    if "show_preview" not in st.session_state:
        st.session_state["show_preview"] = True

    # Khởi tạo trước agent AI để dùng chung cho khung chat
    try:
        agent = _get_or_build_agent(workspace_id)
    except AITutorError as e:
        st.error(f"⚠️ Không thể khởi tạo trợ lý AI: {e}")
        return

    # Phân chia bố cục dựa vào trạng thái show_preview
    if st.session_state["show_preview"]:
        col1, col2 = st.columns([1, 1])

        with col1:
            # Tạo hàng ngang gồm Tiêu đề bên trái và Nút thu gọn (◀) bên phải
            c_title, c_btn = st.columns([0.85, 0.15])
            with c_title:
                st.subheader("📄 Xem trước tài liệu")
            with c_btn:
                if st.button("◀", help="Thu gọn xem trước", use_container_width=True):
                    st.session_state["show_preview"] = False
                    st.rerun()
            
            # TRUYỀN ACTIVE_DOCS ĐỂ HIỂN THỊ FILE TỪ CLOUD LÊN GIAO DIỆN
            render_preview(active_docs, indexed_file_names)

        with col2:
            render_chat(
                agent,
                workspace_id=workspace_id,
                is_indexed=st.session_state.get("is_indexed", False),
            )
    else:
        # Nút mở rộng (▶) khi xem trước đang bị ẩn
        c_toggle_back, _ = st.columns([0.05, 0.95])
        with c_toggle_back:
            if st.button("▶", help="Hiện xem trước tài liệu", use_container_width=True):
                st.session_state["show_preview"] = True
                st.rerun()

        render_chat(
            agent,
            workspace_id=workspace_id,
            is_indexed=st.session_state.get("is_indexed", False),
        )


if __name__ == "__main__":
    main()
