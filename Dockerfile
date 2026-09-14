# syntax=docker/dockerfile:1
FROM python:3.11-slim

# libpq-dev: cần cho psycopg build; build-essential: cần để build 1 số wheel
# của sentence-transformers/torch trên vài kiến trúc CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải sẵn embedding model NGAY LÚC BUILD image, thay vì để app tự tải lúc
# chạy lần đầu. Lý do: nhiều platform hosting (Railway, Render, Fly.io...)
# dùng filesystem tạm thời (ephemeral) -> nếu không bake vào image, mỗi lần
# container khởi động lại sẽ phải tải lại model ~1-2GB, làm cold-start rất chậm.
ARG EMBEDDING_MODEL=BAAI/bge-m3
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
