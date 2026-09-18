import json
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness


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

    # 2. Cấu hình LLM làm giám khảo
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    # 3. Chạy đánh giá RAG tự động
    print("Đang chạy đánh giá RAG tự động...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
    )

    print("Kết quả đánh giá:", result)

    # 4. Xuất kết quả thành file JSON để CI/CD upload artifact
    output_path = Path("evaluation/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result.to_pandas().to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu báo cáo đánh giá vào {output_path}")

if __name__ == "__main__":
    run_evaluation()
