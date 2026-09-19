"""
evaluation/evaluate_rag.py
---------------------------
Nhóm 1: đánh giá chất lượng RAG (retrieval + answer) bằng RAGAS — CHẠY THẬT
qua pipeline của app (`database.hybrid_search` + `agent.ask_agent`) thay vì
dùng answer/context viết tay.

Yêu cầu trước khi chạy:
1. Đã index sẵn tài liệu tương ứng bộ câu hỏi bên dưới vào 1 collection cụ thể.
Chạy: python -m evaluation.evaluate_rag
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from agent import ask_agent, build_agent
from config import app_config, get_config_error
from database import get_embeddings, hybrid_search

EVAL_COLLECTION = os.getenv("EVAL_COLLECTION", "eval_collection")
MIN_SCORE = float(os.getenv("RAGAS_MIN_SCORE", "0.6"))

# câu hỏi + đáp án chuẩn (ground truth).
EVAL_QUESTIONS = [
    {
        "question": "Thuật toán RRF trong hệ thống Hybrid Search hoạt động thế nào?",
        "ground_truth": "RRF kết hợp kết quả xếp hạng từ Vector Search và Full-Text "
        "Search dựa trên công thức nghịch đảo thứ hạng.",
    },
    {
        "question": "Làm cách nào để viết Recursive CTE trong PostgreSQL để xử lý dữ liệu phân cấp?",
        "ground_truth": "Sử dụng cấu trúc WITH RECURSIVE với anchor member và "
        "recursive member nối bằng UNION ALL trong PostgreSQL.",
    },
    {
        "question": "Sự khác biệt chính giữa giao thức TCP và UDP trong mạng máy tính là gì?",
        "ground_truth": "TCP có thiết lập kết nối và đảm bảo độ tin cậy, còn UDP là "
        "giao thức phi kết nối tốc độ nhanh.",
    },
    {
        "question": "Mô hình Word2Vec với kiến trúc Skip-gram hoạt động ra sao?",
        "ground_truth": "Skip-gram trong Word2Vec dùng từ trung tâm để dự đoán các "
        "từ ngữ cảnh xung quanh.",
    },
]


def _run_pipeline_for_question(agent, question: str) -> tuple[str, list[str]]:
    """Chạy THẬT qua retrieval + agent của app — đây là phần bản trước bỏ
    qua, khiến kết quả đánh giá không phản ánh hệ thống thật."""
    docs = hybrid_search(question, collection_name=EVAL_COLLECTION, k=5)
    contexts = [d.page_content for d in docs] or ["(không truy xuất được context nào)"]
    answer = ask_agent(agent, question)
    return answer, contexts


def run_evaluation() -> None:
    config_error = get_config_error()
    if config_error:
        raise RuntimeError(f"Lỗi cấu hình: {config_error}")

    print(f"Collection dùng để eval: '{EVAL_COLLECTION}' (đổi qua biến EVAL_COLLECTION)")
    print(f"Model LLM: {app_config.llm_model}")
    print("Đang chạy pipeline thật (hybrid_search + agent) cho từng câu hỏi...")

    agent = build_agent(collection_name=EVAL_COLLECTION)

    questions, answers, contexts_list, ground_truths = [], [], [], []
    for item in EVAL_QUESTIONS:
        print(f"  - {item['question']}")
        answer, contexts = _run_pipeline_for_question(agent, item["question"])
        questions.append(item["question"])
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    evaluator_llm = ChatGoogleGenerativeAI(
        model=app_config.llm_model,
        temperature=0,
        google_api_key=app_config.google_api_key,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=get_embeddings())

    print("\nĐang chạy đánh giá RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    output_path = Path(__file__).parent / "evaluation_results.json"
    output_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = df[["faithfulness", "answer_relevancy", "context_precision"]].mean().to_dict()
    print("\n=== ĐIỂM TRUNG BÌNH (chạy trên pipeline thật) ===")
    for metric, score in summary.items():
        print(f"  {metric:<20}{score:.3f}")
    print(f"\nChi tiết từng câu hỏi đã lưu vào {output_path}")

    # Cổng chất lượng cho CI: fail build nếu điểm dưới ngưỡng
    failed = {m: s for m, s in summary.items() if s == s and s < MIN_SCORE}  # s==s loại NaN
    if failed:
        print(f"\n❌ Các chỉ số dưới ngưỡng tối thiểu {MIN_SCORE} (đặt qua RAGAS_MIN_SCORE): {failed}")
        sys.exit(1)
    print(f"\n✅ Tất cả chỉ số đạt ngưỡng tối thiểu {MIN_SCORE}.")


if __name__ == "__main__":
    run_evaluation()
