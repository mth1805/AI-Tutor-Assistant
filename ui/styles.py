"""ui/styles.py - CSS tùy chỉnh giao diện, tách khỏi logic để dễ chỉnh sửa theme."""

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }

    /* --- TÙY CHỈNH KHUNG CHAT TRÁI/PHẢI --- */
    div[data-testid="stChatMessage"]:has(.user-tag) {
        flex-direction: row-reverse !important;
        text-align: right !important;
    }

    div[data-testid="stChatMessage"]:has(.user-tag) div[data-testid="stChatMessageContent"] {
        background-color: #d6eaf8 !important;
        color: black !important;
        border-radius: 15px !important;
        padding: 10px 15px !important;
    }

    div[data-testid="stChatMessage"]:has(.ai-tag) div[data-testid="stChatMessageContent"] {
        background-color: #f1f3f4 !important;
        color: black !important;
        border-radius: 15px !important;
        padding: 10px 15px !important;
    }
</style>
"""
