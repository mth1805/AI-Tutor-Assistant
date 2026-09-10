"""
exceptions.py
-------------
Hệ thống exception riêng cho ứng dụng. Việc định nghĩa exception theo lớp
(hierarchy) giúp UI (Streamlit) bắt lỗi có mục tiêu rõ ràng — ví dụ có thể
hiển thị thông báo khác nhau cho lỗi kết nối DB so với lỗi cấu hình — thay vì
bắt `Exception` chung chung ở khắp nơi.
"""


class AITutorError(Exception):
    """Lớp exception gốc cho toàn bộ ứng dụng. Bắt lớp này ở UI để đảm bảo
    mọi lỗi nghiệp vụ đã được xử lý/đóng gói thông điệp thân thiện."""


class ConfigError(AITutorError):
    """Lỗi thiếu/sai cấu hình (biến môi trường, API key...)."""


class DatabaseConnectionError(AITutorError):
    """Không thể kết nối / truy vấn PostgreSQL - pgvector."""


class DocumentProcessingError(AITutorError):
    """Lỗi khi đọc, giải mã hoặc chia nhỏ (chunk) tài liệu người dùng tải lên."""


class EmbeddingError(AITutorError):
    """Lỗi khi gọi API embedding (Gemini) — thường transient (rate limit, timeout)."""


class AgentError(AITutorError):
    """Lỗi khi khởi tạo hoặc thực thi Agent (LLM, tool call, LangGraph...)."""
