"""
evaluation/evaluate_rag.py
---------------------------
Nhóm 1: đánh giá chất lượng RAG (retrieval + answer) bằng RAGAS — CHẠY THẬT
qua pipeline của app (`database.hybrid_search` + `agent.ask_agent`) thay vì
dùng answer/context viết tay.

QUAN TRỌNG: bản trước của script này viết tay cả "answer" lẫn "contexts",
khiến RAGAS chỉ đo độ nhất quán giữa 2 đoạn text tự soạn với nhau — không hề
phản ánh chất lượng hệ thống thật. Bản này CHỈ viết tay "question" và
"ground_truth" (bộ câu hỏi + đáp án chuẩn); "answer" và "contexts" LUÔN được
lấy từ việc gọi thật vào pipeline.

Yêu cầu trước khi chạy:
  1. .env đã có GOOGLE_API_KEY, COHERE_API_KEY, PG_* trỏ đúng Postgres.
  2. Đã index sẵn tài liệu tương ứng bộ câu hỏi bên dưới vào 1 collection cụ
     thể (mặc định 'eval_collection', đổi qua biến EVAL_COLLECTION). Ví dụ:
     upload đúng tài liệu chứa nội dung về RRF/CTE/TCP-UDP/Word2Vec vào
     workspace có ID trùng EVAL_COLLECTION trước khi chạy script này.

Chạy: python -m evaluation.evaluate_rag
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, faithfulness  # noqa: E402

from agent import ask_agent, build_agent  # noqa: E402
from config import app_config, get_config_error  # noqa: E402
from database import get_embeddings, hybrid_search  # noqa: E402

EVAL_COLLECTION = os.getenv("EVAL_COLLECTION", "eval_collection")
MIN_SCORE = float(os.getenv("RAGAS_MIN_SCORE", "0.6"))

# CHỈ 2 trường này được viết tay: câu hỏi + đáp án chuẩn (ground truth).
# "answer" và "contexts" KHÔNG được viết tay ở đây — luôn lấy từ pipeline
# thật (xem _run_pipeline_for_question), nếu không kết quả đánh giá vô nghĩa.
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

    # Dùng CÙNG LLM app đang dùng thật (đo đúng chất lượng model thật đang
    # chạy), và embedding LOCAL (bge-m3, giống hệt app) thay vì gọi thêm 1
    # model embedding Gemini riêng cho việc chấm điểm — lý do:
    #   1. Không tốn thêm quota Gemini chỉ để evaluate.
    #   2. "models/embedding-001" ở bản trước là model cũ/dễ deprecated —
    #      đây chính là nguyên nhân answer_relevancy=NaN toàn bộ ở kết quả cũ.
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

    # Cổng chất lượng cho CI: fail build nếu điểm dưới ngưỡng, thay vì chỉ
    # xuất báo cáo rồi không ai đọc.
    failed = {m: s for m, s in summary.items() if s == s and s < MIN_SCORE}  # s==s loại NaN
    if failed:
        print(f"\n❌ Các chỉ số dưới ngưỡng tối thiểu {MIN_SCORE} (đặt qua RAGAS_MIN_SCORE): {failed}")
        sys.exit(1)
    print(f"\n✅ Tất cả chỉ số đạt ngưỡng tối thiểu {MIN_SCORE}.")


if __name__ == "__main__":
    run_evaluation()
