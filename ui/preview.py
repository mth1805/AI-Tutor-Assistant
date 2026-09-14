"""ui/preview.py - Cột xem trước tài liệu (PDF gốc hoặc text đã trích xuất)."""
from __future__ import annotations

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from document_processor import extract_text
from exceptions import DocumentProcessingError


def _display_pdf(uploaded_file) -> None:
    pdf_viewer(input=uploaded_file.getvalue(), width=600, height=600)


# ui/preview.py
import streamlit as st
from document_processor import extract_text

def render_preview(docs: list, indexed_file_names: list) -> None:
    # Kết hợp file upload hiện tại và file đã index từ DB
    all_files = {d.name: d for d in docs}
    
    # Danh sách chọn hiển thị
    options = list(set(list(all_files.keys()) + indexed_file_names))
    
    if not options:
        st.info("Chưa có tài liệu nào.")
        return

    selected_name = st.selectbox("Chọn file để xem:", options)
    
    # Nếu là file vừa upload (có trong docs)
    if selected_name in all_files:
        selected = all_files[selected_name]
        if selected.name.lower().endswith(".pdf"):
            _display_pdf(selected)
        else:
            st.text_area("Nội dung:", extract_text(selected), height=550)
    else:
        # Nếu là file đã index trong DB nhưng không có trong docs hiện tại
        st.warning(f"File '{selected_name}' đã được index nhưng không có trong phiên upload hiện tại. Bạn có thể chat với nó, nhưng không thể xem trước nội dung gốc tại đây.")
