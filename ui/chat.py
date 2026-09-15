"""ui/chat.py - Giao diện chat với Agent: streaming + lưu lịch sử vào Postgres."""

from __future__ import annotations

import streamlit as st

from agent import stream_agent
from database import save_chat_message


def _render_message(role: str, content: str) -> None:
    tag_class = "user-tag" if role == "user" else "ai-tag"
    with st.chat_message(role):
        st.markdown(f"<span class='{tag_class}'></span> {content}", unsafe_allow_html=True)


def render_chat(agent, *, workspace_id: str, is_indexed: bool) -> None:
    st.subheader("💬 Chat với Gia sư")
    chat_container = st.container(height=600, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with chat_container:
        for message in st.session_state.messages:
            _render_message(message["role"], message["content"])

    placeholder = (
        "Hãy đặt câu hỏi chi tiết về tài liệu của bạn..."
        if is_indexed
        else "Bạn có thể hỏi kiến thức chung, hoặc upload + xử lý tài liệu trước để hỏi sâu hơn..."
    )

    prompt = st.chat_input(placeholder)
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(workspace_id, "user", prompt)

    with chat_container:
        _render_message("user", prompt)
        with st.chat_message("assistant"):
            # st.write_stream render dần từng đoạn text sinh ra từ generator,
            # và trả về toàn bộ nội dung đã ghép sau khi stream xong.
            # Lưu ý: khi đang stream, nội dung hiển thị bằng markdown mặc định
            # của Streamlit (không có class 'ai-tag' tùy chỉnh); sau khi lưu
            # vào lịch sử và render lại ở lần load sau, tin nhắn sẽ dùng đúng
            # style bong bóng chat như bình thường.
            response = st.write_stream(stream_agent(agent, prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_chat_message(workspace_id, "assistant", response)
