"""ui/sidebar.py - Sidebar: quản lý workspace, đổi tên, upload & xử lý tài liệu."""

from __future__ import annotations

import uuid

import streamlit as st

from config import app_config, get_logger
from database import (
    collection_has_documents,
    delete_workspace,
    ensure_workspace_exists,
    get_all_workspaces,
    get_embeddings,
    get_indexed_files,
    index_chunks,
    rename_workspace,
)
from document_processor import process_files_to_chunks
from exceptions import AITutorError
from storage import upload_file_to_cloud

logger = get_logger(__name__)


def render_sidebar(*, workspace_id: str, metadata_common: dict) -> list:
    """Hiển thị sidebar: quản lý workspace (selectbox, đổi tên, xóa) + upload/xử lý tài liệu."""
    # ensure_workspace_exists(workspace_id, f"Workspace {workspace_id[-8:]}") #ko tu tao ws moi moi lan load trang

    with st.sidebar:
        st.title("📂 Quản lý tài liệu")

        with st.expander("🔗 Workspace", expanded=True):
            workspaces = get_all_workspaces()
            ws_ids = [w["id"] for w in workspaces]
            ws_titles = {w["id"]: w["title"] for w in workspaces}

            if workspace_id not in ws_ids:
                # ensure_workspace_exists(workspace_id, f"Workspace {workspace_id[-8:]}")
                workspaces = get_all_workspaces()
                # ws_ids = [w["id"] for w in workspaces]
                # ws_titles = {w["id"]: w["title"] for w in workspaces}
                if ws_ids:
                    st.query_params["ws"] = ws_ids[0]
                    workspace_id = ws_ids[0]

            current_index = ws_ids.index(workspace_id) if workspace_id in ws_ids else 0

            # Chia layout: Ô selectbox chiếm phần lớn bên trái, nút Xóa (🗑️) nhỏ ở bên phải
            col_select, col_del = st.columns([0.8, 0.2])
            with col_select:
                selected_ws = st.selectbox(
                    "Chọn workspace:",
                    options=ws_ids,
                    index=current_index,
                    format_func=lambda x: ws_titles.get(x, x),
                    label_visibility="collapsed",
                )
            with col_del:
                if st.button("🗑️", key="del_current_ws", help="Xóa workspace đang chọn"):
                    delete_workspace(workspace_id)
                    remaining_workspaces = get_all_workspaces()
                    if remaining_workspaces:
                        next_ws = remaining_workspaces[0]["id"]
                        st.query_params["ws"] = next_ws
                    else:
                        if "ws" in st.query_params:
                            del st.query_params["ws"]

                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

            # Chuyển workspace khi chọn qua selectbox
            if selected_ws and selected_ws != workspace_id:
                st.query_params["ws"] = selected_ws
                st.session_state.pop("messages", None)
                st.session_state.pop("loaded_workspace", None)
                st.session_state.pop("agent", None)
                st.session_state.pop("agent_workspace_id", None)
                st.session_state.pop("uploaded_files_cache", None)
                st.session_state["is_indexed"] = collection_has_documents(selected_ws)
                st.rerun()

            st.divider()

            # Ô đổi tên cho workspace hiện tại
            current_title = ws_titles.get(workspace_id, workspace_id)
            new_title = st.text_input("Đổi tên workspace này:", value=current_title)
            if new_title and new_title != current_title:
                if st.button("Lưu tên mới"):
                    rename_workspace(workspace_id, new_title)
                    st.success("Đã đổi tên thành công!")
                    st.rerun()

            # st.code(workspace_id, language=None) # Hiển thị ID workspace mới nếu muốn tạo mới

            # Hiển thị dsach file đã upload trước đó
            indexed_files = get_indexed_files(workspace_id)
            if indexed_files:
                st.markdown("---")
                st.caption("📚 Tài liệu đã học trong workspace này:")
                for file_name in indexed_files:
                    st.text(f"• {file_name}")
            else:
                st.caption("📚 Tài liệu đã học: (Chưa có file nào)")

            if st.button("➕ Tạo workspace mới"):
                new_ws = f"ws_{uuid.uuid4().hex[:12]}"
                ensure_workspace_exists(new_ws, f"Workspace mới {new_ws[-8:]}")
                st.query_params["ws"] = new_ws
                st.session_state.pop("messages", None)
                st.session_state.pop("loaded_workspace", None)
                st.session_state.pop("agent", None)
                st.session_state.pop("agent_workspace_id", None)
                st.session_state.pop("uploaded_files_cache", None)
                st.session_state["is_indexed"] = False
                st.rerun()

        docs = st.file_uploader(
            "Upload tài liệu (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"file_uploader_{workspace_id}",
        )

        if st.button("Xử lý tài liệu"):
            if not docs:
                st.warning("Vui lòng upload tài liệu trước!")
                return docs or []

            progress_bar = st.progress(0.0, text="Đang đọc & chia nhỏ tài liệu...")
            try:
                # --- ĐẨY FILE GỐC LÊN CLOUD OBJECT STORAGE ---
                for doc in docs:
                    doc.seek(0)
                    upload_file_to_cloud(workspace_id, doc.name, doc.read())
                    logger.info("Đã đẩy file lên cloud: %s (workspace: %s)", doc.name, workspace_id)
                    doc.seek(0)

                # -------------------------------------------------------------
                embeddings = get_embeddings() if app_config.chunking_strategy == "semantic" else None
                chunks = process_files_to_chunks(
                    docs,
                    strategy=app_config.chunking_strategy,
                    embeddings=embeddings,
                    enable_ocr=app_config.enable_ocr,
                )
                progress_bar.progress(0.0, text=f"Đang mã hóa {len(chunks)} đoạn văn bản...")
                index_chunks(
                    chunks,
                    collection_name=workspace_id,
                    metadata_common=metadata_common,
                    progress_callback=lambda p: progress_bar.progress(
                        p, text=f"Đang mã hóa tài liệu... {int(p * 100)}%"
                    ),
                )
                st.session_state["is_indexed"] = True
                st.success(f"✅ Đã học xong {len(chunks)} đoạn từ {len(docs)} file!")
            except AITutorError as e:
                logger.error("Lỗi nghiệp vụ khi xử lý tài liệu: %s", e)
                st.error(f"❌ {e}")
            except Exception as e:
                logger.exception("Lỗi không xác định khi xử lý tài liệu")
                st.error(f"❌ Đã có lỗi không xác định xảy ra: {e}")
            finally:
                progress_bar.empty()

    return docs or []
