import json
import os
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness


def run_evaluation():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    model_name = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    print(f"Đang sử dụng LLM model cho Ragas: {model_name}")

    data = {
    "question": [
        "Thuật toán RRF trong hệ thống Hybrid Search hoạt động thế nào?",
        "Làm cách nào để viết Recursive CTE trong PostgreSQL để xử lý dữ liệu phân cấp?",
        "Sự khác biệt chính giữa giao thức TCP và UDP trong mạng máy tính là gì?",
        "Mô hình Word2Vec với kiến trúc Skip-gram hoạt động ra sao?"
    ],
    "answer": [
        "Thuật toán Reciprocal Rank Fusion (RRF) kết hợp kết quả xếp hạng từ Vector Search và Full-Text Search bằng cách tính điểm nghịch đảo của thứ hạng mà không cần chuẩn hóa điểm số gốc.",
        "Recursive CTEs trong PostgreSQL sử dụng cấu trúc WITH RECURSIVE, bao gồm một truy vấn cơ sở (anchor member) kết hợp với một truy vấn đệ quy (recursive member) liên kết qua mệnh đề UNION ALL.",
        "TCP là giao thức có thiết lập kết nối, đảm bảo độ tin cậy và truyền dữ liệu theo thứ tự, trong khi UDP là giao thức phi kết nối, tốc độ nhanh nhưng không đảm bảo độ tin cậy.",
        "Kiến trúc Skip-gram của Word2Vec nhận đầu vào là một từ trung tâm (center word) để dự đoán các từ ngữ cảnh (context words) xung quanh trong một cửa sổ văn bản."
    ],
    "contexts": [
        [
            "Hybrid Search kết hợp Vector Search và PostgreSQL Full-Text Search thông qua Reciprocal Rank Fusion (RRF). RRF gộp các kết quả bằng công thức tính điểm dựa trên thứ hạng xếp hạng của từng phương pháp tìm kiếm."
        ],
        [
            "Recursive Common Table Expressions (CTEs) cho phép truy vấn dữ liệu phân cấp hoặc dạng cây trong PostgreSQL bằng cách lặp lại tập kết quả cho đến khi không còn bản ghi nào thỏa mãn điều kiện đệ quy."
        ],
        [
            "Giao thức truyền tải Transport Layer phân biệt rõ giữa TCP (Transmission Control Protocol) đảm bảo truyền dữ liệu tin cậy, kiểm soát tắc nghẽn và UDP (User Datagram Protocol) tối ưu cho tốc độ truyền tải thời gian thực."
        ],
        [
            "Word2Vec cung cấp hai kiến trúc học biểu diễn từ là CBOW và Skip-gram. Skip-gram dự đoán các từ ngữ cảnh từ từ khóa trung tâm, phù hợp rất tốt với các tập dữ liệu huấn luyện có kích thước lớn."
        ]
    ],
    "ground_truth": [
        "RRF kết hợp kết quả xếp hạng từ Vector Search và Full-Text Search dựa trên công thức nghịch đảo thứ hạng.",
        "Sử dụng cấu trúc WITH RECURSIVE với anchor member và recursive member nối bằng UNION ALL trong PostgreSQL.",
        "TCP có thiết lập kết nối và đảm bảo độ tin cậy, còn UDP là giao thức phi kết nối tốc độ nhanh.",
        "Skip-gram trong Word2Vec dùng từ trung tâm để dự đoán các từ ngữ cảnh xung quanh."
    ],
}

    dataset = Dataset.from_dict(data)

    # 1. Khởi tạo LLM giám khảo lấy từ biến môi trường
    evaluator_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=gemini_key
    )

    # 2. Sử dụng LangchainEmbeddingsWrapper chuẩn xác theo phiên bản Ragas mới
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        embeddings=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_key
        )
    )

    # 3. Chạy đánh giá Ragas
    print("Đang chạy đánh giá RAG tự động...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 4. Xuất kết quả ra file JSON cho CI/CD artifact
    output_path = Path("evaluation/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_pandas().to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu báo cáo đánh giá vào {output_path}")

if __name__ == "__main__":
    run_evaluation()
