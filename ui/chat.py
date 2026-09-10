"""ui/chat.py - Giao diện chat với Agent."""
from __future__ import annotations

import streamlit as st

from agent import ask_agent


def _render_message(role: str, content: str) -> None:
    tag_class = "user-tag" if role == "user" else "ai-tag"
    with st.chat_message(role):
        st.markdown(f"<span class='{tag_class}'></span> {content}", unsafe_allow_html=True)


def render_chat(agent, *, is_indexed: bool) -> None:
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
    with chat_container:
        _render_message("user", prompt)
        with st.chat_message("assistant"):
            with st.spinner("Gia sư đang suy nghĩ..."):
                response = ask_agent(agent, prompt)
            st.markdown(f"<span class='ai-tag'></span> {response}", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
