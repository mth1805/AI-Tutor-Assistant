from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import answer_relevance, context_precision, faithfulness


def run_evaluation():
    # 1. Chuẩn bị tập dữ liệu test mẫu (Golden Dataset)
    data = {
        "question": ["Thuật toán RRF trong hệ thống hoạt động thế nào?"],
        "answer": [
            "Thuật toán Reciprocal Rank Fusion kết hợp kết quả xếp hạng từ Vector Search và Full-Text Search..."
        ],
        "contexts": [
            [
                "Hybrid Search kết hợp Vector Search và PostgreSQL Full-Text Search thông qua Reciprocal Rank Fusion (RRF)..."
            ]
        ],
        "ground_truth": [
            "RRF kết hợp xếp hạng từ Vector Search và Full-Text Search để tối ưu kết quả tìm kiếm."
        ],
    }

    dataset = Dataset.from_dict(data)

    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    # 3. Chạy đánh giá
    print("Đang chạy đánh giá RAG tự động...")
    result = evaluate(
        dataset=dataset, metrics=[faithfulness, answer_relevance, context_precision], llm=evaluator_llm
    )

    print("Kết quả đánh giá:", result)

    # Có thể thêm logic kiểm tra ngưỡng điểm ở đây (Ví dụ: nếu điểm dưới 0.8 thì raise exception)


if __name__ == "__main__":
    run_evaluation()
