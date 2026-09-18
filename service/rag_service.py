import time

from utils.metrics import measure_latency


@measure_latency
def run_hybrid_search(query: str, top_k: int = 5):
    """Thực hiện Hybrid Search (pgvector + RRF)"""
    # Truy vấn PostgreSQL, Vector Search, kết hợp RRF
    time.sleep(0.15)
    return [{"chunk_id": 1, "text": "Nội dung mẫu..."}]


@measure_latency
def generate_llm_response(prompt: str):
    """Gọi LLM sinh câu trả lời cuối cùng"""
    # Gọi LangChain / Gemini API
    time.sleep(0.7)
    return "Đây là câu trả lời được sinh ra từ AI Tutor."
