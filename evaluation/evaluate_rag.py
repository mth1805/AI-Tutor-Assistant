import json
import os
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddings
from ragas.metrics import answer_relevancy, context_precision, faithfulness


def run_evaluation():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    # Đọc tên model từ biến môi trường .env, mặc định là gemini-3.1-flash-lite nếu không tìm thấy
    model_name = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    print(f"Đang sử dụng LLM model cho Ragas: {model_name}")

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

    # 1. Khởi tạo LLM giám khảo lấy từ .env
    evaluator_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=gemini_key
    )

    # 2. Khởi tạo Embeddings bằng Gemini
    evaluator_embeddings = LangchainEmbeddings(
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

    # 4. Lưu kết quả ra file JSON cho CI/CD artifact
    output_path = Path("evaluation/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_pandas().to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu báo cáo đánh giá vào {output_path}")

if __name__ == "__main__":
    run_evaluation()
