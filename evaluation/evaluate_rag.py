# Ví dụ cấu trúc phác thảo script đánh giá bằng Ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision

# 1. Chuẩn bị tập dữ liệu test (Golden Test Dataset)
data = {
    "question": ["Thuật toán RRF trong project hoạt động thế nào?"],
    "answer": [
        "Thuật toán Reciprocal Rank Fusion kết hợp kết quả từ Vector Search và Full-Text Search..."
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

# 2. Chạy đánh giá tự động các chỉ số
# (Cần cấu hình LLM giám khảo, ví dụ ChatGoogleGenerativeAI)
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevance, context_precision],
)

# 3. Xuất kết quả báo cáo
print(result)