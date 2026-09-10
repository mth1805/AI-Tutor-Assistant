# AI Tutor Assistant (LangChain + LangGraph + Streamlit + PostgreSQL/pgvector)


## 1. Cấu trúc thư mục

```
ai_tutor/
├── app.py                  # Entry point Streamlit — CHỈ điều phối UI (~100 dòng)
├── config.py                # Đọc .env, dataclass cấu hình, logger dùng chung
├── exceptions.py             # Hệ thống exception riêng (AITutorError và các lớp con)
├── database.py               # Kết nối PostgreSQL + pgvector (qua langchain-postgres)
├── document_processor.py     # Đọc PDF/DOCX/TXT, chunk text — không phụ thuộc Streamlit
├── tools.py                   # web_search_tool, python_math_tool, document_reader_tool
├── agent.py                  # Khởi tạo LLM Gemini + LangGraph agent
├── ui/
│   ├── __init__.py
│   ├── styles.py              # CSS
│   ├── sidebar.py             # Upload & xử lý tài liệu
│   ├── preview.py             # Xem trước tài liệu
│   └── chat.py                # Giao diện chat
├── requirements.txt
├── .env.example
└── README.md
```

**Nguyên tắc phân lớp:** `app.py` và `ui/*` chỉ được gọi hàm từ
`database.py` / `document_processor.py` / `agent.py` / `tools.py` — không
bao giờ tự mở kết nối DB hay gọi API trực tiếp. Điều này cho phép:
- Viết unit test cho `document_processor.py`, `database.py` mà không cần
  Streamlit chạy.
- Viết một script CLI/cron riêng để batch-index tài liệu, tái sử dụng
  `document_processor` + `database` mà không đụng vào UI.

## 2. Embedding chạy LOCAL 

Dùng `HuggingFaceEmbeddings` (`BAAI/bge-m3` hoặc `intfloat/multilingual-e5-small`, chạy CPU bằng
`sentence-transformers`) thay vì `GoogleGenerativeAIEmbeddings`. 
- `bge-m3` là model đa ngôn ngữ, hỗ trợ tiếng Việt tốt, phù hợp cho RAG.
- Gemini chỉ còn được dùng cho phần LLM/agent trả lời câu hỏi — vốn có số
  lượng lệnh gọi ít hơn nhiều so với embedding.

**Lưu ý khi migrate:**
- `get_embeddings()` được cache bằng `@lru_cache` vì nạp model từ đĩa mất vài
  giây — tránh nạp lại mỗi lần Streamlit rerun script.
- Lần chạy đầu tiên trên máy mới sẽ tự động tải model (~1-2GB) từ
  HuggingFace Hub về `~/.cache/huggingface` — cần mạng ổn định lần đầu, các
  lần sau chạy hoàn toàn offline.
- **Số chiều vector khác nhau** giữa Gemini embedding và `bge-m3` (1024 chiều).
  Nếu trước đó đã index dữ liệu bằng Gemini, **phải đổi `PG_COLLECTION` sang
  tên mới** và index lại toàn bộ tài liệu — không thể trộn 2 loại vector
  trong cùng 1 collection.
- Muốn đổi sang model embedding local khác (ví dụ
  `intfloat/multilingual-e5-large` nếu cần độ chính xác cao hơn, đổi lại nặng
  hơn), chỉ cần sửa `EMBEDDING_MODEL` trong `.env`, không cần sửa code.
- Nếu deploy trên máy có GPU, đổi `EMBEDDING_DEVICE=cuda` trong `.env` và cài
  `torch` bản CUDA phù hợp để encode nhanh hơn đáng kể.

## 3. Chuyển từ FAISS sang PostgreSQL + pgvector

- **Persistent & multi-instance**: FAISS lưu ở đĩa cục bộ của container —
  không dùng được khi scale ngang (nhiều Streamlit instance) hoặc khi
  container bị redeploy. pgvector lưu trong Postgres, mọi instance đọc chung
  một nguồn.
- **Phân vùng dữ liệu tự nhiên**: mỗi collection trong pgvector có thể ứng
  với 1 lớp học / 1 user / 1 môn học, và mỗi vector còn có `metadata` (JSONB)
  để lọc chi tiết hơn (xem phần phân quyền bên dưới).
- **Vận hành chuẩn**: backup, replication, monitoring dùng lại toàn bộ tooling
  Postgres sẵn có, không cần quản lý riêng file `faiss_index/`.

### Chạy Postgres + pgvector cục bộ (Docker)

```yaml
# docker-compose.yml (ví dụ, đặt cạnh app)
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ai_tutor
      POSTGRES_USER: ai_tutor_app
      POSTGRES_PASSWORD: change_me
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

`database.check_database_connection()` tự động chạy
`CREATE EXTENSION IF NOT EXISTS vector;` khi khởi động — chỉ cần user có
quyền tạo extension (hoặc DBA tạo trước).

## 4. Xử lý lỗi (error handling)

- **Hệ thống exception riêng** (`exceptions.py`): `DatabaseConnectionError`,
  `EmbeddingError`, `DocumentProcessingError`, `AgentError`... Mỗi tầng chỉ
  raise đúng loại lỗi của mình, UI (`ui/*`, `app.py`) bắt `AITutorError` (lớp
  cha) để hiển thị thông báo thân thiện, và bắt thêm `Exception` chung làm
  lưới an toàn cuối cùng.
- **Fail-fast khi khởi động**: `app._bootstrap()` kiểm tra cấu hình
  (`get_config_error()`) và kết nối DB (`check_database_connection()`) một
  lần duy nhất trước khi vẽ UI, tránh lỗi xuất hiện đột ngột giữa phiên chat.
- **Retry có backoff cho embedding**: `database._add_batch` dùng `tenacity`
  để retry tối đa 3 lần với backoff hàm mũ khi gọi Gemini embedding lỗi
  (thường là rate-limit/timeout tạm thời) — thay cho `time.sleep(8)` cứng
  trong bản gốc.
- **Agent không bao giờ crash UI**: `agent.ask_agent()` bọc toàn bộ lời gọi
  agent trong `try/except`, luôn trả về string (kể cả khi lỗi) để
  `ui/chat.py` không cần xử lý ngoại lệ.

## 5. Kiến trúc production-ready — góp ý mở rộng

### 5.1. Mở rộng phân quyền (RBAC / multi-tenant)

1. `collection_name` truyền xuyên suốt `database.py` → `tools.py` →
   `agent.py` → `app.py` (hiện đang dùng `session_id` ngẫu nhiên per-tab).
2. `metadata_filter` / `metadata_common` gắn nhãn từng đoạn văn bản khi
   index, và lọc lại khi truy vấn.

Để lên production đa người dùng thật, đề xuất:

```
users            (id, email, hashed_password, role)          -- hoặc qua SSO/OAuth
roles            (id, name)             -- 'student', 'teacher', 'admin'
documents        (id, owner_id, title, collection_name, created_at)
document_access  (document_id, subject_type, subject_id)      -- user hoặc role được phép xem
```

- Thay `_get_session_id()` trong `app.py` bằng `user_id` lấy từ middleware
  xác thực thật (vd `streamlit-authenticator`, hoặc reverse-proxy OAuth2
  đứng trước Streamlit, hoặc tách hẳn thành API FastAPI + frontend riêng nếu
  cần OAuth chuẩn).
- Khi index tài liệu (`ui/sidebar.py` → `database.index_text_chunks`), gắn
  `metadata_common = {"owner_id": user_id, "allowed_roles": [...]}`.
- Khi truy vấn (`tools.make_document_reader_tool`), build `metadata_filter`
  từ user hiện tại: cho phép xem tài liệu của chính mình + tài liệu được
  giáo viên share theo `document_access`.
- Với yêu cầu bảo mật cao hơn (không tin tưởng filter ở tầng ứng dụng), cân
  nhắc bật **Row-Level Security (RLS)** trực tiếp trên bảng
  `langchain_pg_embedding` của Postgres, gắn với `current_setting('app.user_id')`
  set qua mỗi connection.

### 5.2. Bảo mật `python_math_tool`

`PythonREPL` thực thi code Python tùy ý do LLM sinh ra. Trong production:
- Chạy tool này trong container/sandbox riêng, không có quyền truy cập
  mạng/filesystem ra ngoài (vd `gVisor`, `nsjail`, hoặc AWS Lambda tách biệt).
- Giới hạn thời gian chạy (timeout) và tài nguyên (CPU/RAM).
- Không expose kết quả stack trace chi tiết ra người dùng cuối.

### 5.3. Khả năng mở rộng khác

- **Caching**: cache câu trả lời cho các câu hỏi lặp lại theo
  `(collection_name, question_hash)` bằng Redis, giảm chi phí gọi LLM.
- **Observability**: thêm request ID xuyên suốt log (`config.get_logger`),
  tích hợp LangSmith/OpenTelemetry để trace từng bước của agent.
- **Testing**: `document_processor.py` và `database.py` không phụ thuộc
  Streamlit — viết `pytest` với DB test (`testcontainers-python` +
  `pgvector/pgvector` image) và file mẫu PDF/DOCX/TXT.
- **CI/CD**: thêm `ruff`/`mypy` cho lint & type-check, chạy `py_compile` /
  test suite trong pipeline trước khi deploy.
- **Config theo môi trường**: tách `.env.development` / `.env.production`,
  không commit file `.env` thật (chỉ commit `.env.example`).

## 7. Cài đặt & chạy

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# điền GOOGLE_API_KEY, COHERE_API_KEY, thông tin Postgres...

# (nếu chưa có Postgres) chạy docker-compose ở mục 3, hoặc trỏ tới DB có sẵn

streamlit run app.py
```

## 8. Những điểm cần bạn tự xác nhận trước khi deploy

- **Tên model Gemini** (`LLM_MODEL` trong `.env`, mặc định `gemini-3.1-flash-lite`):
  Google thường xuyên đổi tên/khai tử model (vd `gemini-2.0-flash` đã ngừng
  hoạt động từ 03/2026, các model Pro không còn free từ 04/2026) — hãy kiểm
  tra danh sách model + quota hiện hành tại Google AI Studio trước khi chạy,
  và chỉ cần đổi trong `.env`, không phải sửa code.
- **Quota vẫn có giới hạn dù embedding đã chạy local**: các lệnh gọi LLM khi
  chat (agent) và Cohere rerank vẫn tính theo quota API tương ứng. Nếu vẫn
  hết quota nhanh, cân nhắc giới hạn số bước tool-calling của agent (mục 6.3),
  cache câu trả lời, hoặc chuyển sang Vertex AI để có quota cao hơn.
- Đảm bảo license/quyền dùng `DuckDuckGoSearchRun` phù hợp với volume truy
  vấn dự kiến của bạn (free tier có thể bị rate-limit ở quy mô lớn).
