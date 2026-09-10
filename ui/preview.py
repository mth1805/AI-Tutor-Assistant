"""ui/preview.py - Cột xem trước tài liệu (PDF gốc hoặc text đã trích xuất)."""
from __future__ import annotations

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from document_processor import extract_text
from exceptions import DocumentProcessingError


def _display_pdf(uploaded_file) -> None:
    pdf_viewer(input=uploaded_file.getvalue(), width=600, height=600)


def render_preview(docs: list) -> None:
    st.subheader("📄 Xem trước tài liệu")

    if not docs:
        st.info("Vui lòng tải tài liệu lên từ thanh bên trái để xem trước.")
        return

    selected_name = st.selectbox("Chọn file để xem:", [d.name for d in docs])
    selected = next((d for d in docs if d.name == selected_name), None)
    if selected is None:
        return

    if selected.name.lower().endswith(".pdf"):
        _display_pdf(selected)
        return

    try:
        text = extract_text(selected)
    except DocumentProcessingError as e:
        st.error(f"❌ {e}")
        return

    st.info(
        "Trình duyệt chỉ hiển thị bản gốc của PDF. "
        "Dưới đây là nội dung văn bản được trích xuất:"
    )
    st.text_area("", text, height=550)
