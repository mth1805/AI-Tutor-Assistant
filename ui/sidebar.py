"""ui/sidebar.py - Sidebar: upload & xử lý (index) tài liệu."""
from __future__ import annotations

import streamlit as st

from config import get_logger
from database import index_text_chunks
from document_processor import extract_text_from_files, split_text
from exceptions import AITutorError

logger = get_logger(__name__)


def render_sidebar(*, collection_name: str, metadata_common: dict) -> list:
    """Hiển thị sidebar upload tài liệu và xử lý (chunk + index vào pgvector).

    Trả về danh sách file đã upload trong lần chạy hiện tại (để cột preview
    dùng), không phải danh sách đã lưu trữ vĩnh viễn.
    """
    with st.sidebar:
        st.title("📂 Quản lý tài liệu")
        docs = st.file_uploader(
            "Upload tài liệu (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if st.button("Xử lý tài liệu"):
            if not docs:
                st.warning("Vui lòng upload tài liệu trước!")
                return docs or []

            progress_bar = st.progress(0.0, text="Đang xử lý tài liệu...")
            try:
                raw_text = extract_text_from_files(docs)
                chunks = split_text(raw_text)
                index_text_chunks(
                    chunks,
                    collection_name=collection_name,
                    metadata_common=metadata_common,
                    progress_callback=lambda p: progress_bar.progress(
                        p, text=f"Đang mã hóa tài liệu... {int(p * 100)}%"
                    ),
                )
                st.session_state["is_indexed"] = True
                st.success("✅ Đã học xong tài liệu! Bạn có thể bắt đầu hỏi.")
            except AITutorError as e:
                # Lỗi nghiệp vụ đã được đóng gói thông điệp thân thiện ở tầng dưới.
                logger.error("Lỗi nghiệp vụ khi xử lý tài liệu: %s", e)
                st.error(f"❌ {e}")
            except Exception as e:  # noqa: BLE001 - lưới an toàn cuối cùng cho UI
                logger.exception("Lỗi không xác định khi xử lý tài liệu")
                st.error(f"❌ Đã có lỗi không xác định xảy ra: {e}")
            finally:
                progress_bar.empty()

    return docs or []
