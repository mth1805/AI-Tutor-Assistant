# AI Tutor — Refactored (LangChain + LangGraph + Streamlit + PostgreSQL/pgvector)

## 1. Vì sao refactor

File `app.py` gốc gộp chung: đọc file, chunk, embedding, DB (FAISS local),
khởi tạo agent, định nghĩa tool, và toàn bộ UI Streamlit trong **một file**.
Vấn đề với cách này ở quy mô production:

| Vấn đề trong bản gốc | Hệ quả |
|---|---|
| FAISS lưu file local (`faiss_index/`) | Không scale nhiều instance, mất dữ liệu khi container restart, không phân quyền theo user |
| `time.sleep(8)` cứng giữa các batch embedding | Chậm không cần thiết khi API ổn định, vẫn có thể fail khi bị rate-limit thật |
| Không có `try/except` quanh kết nối DB / gọi Gemini | Lỗi mạng/API sẽ crash toàn bộ Streamlit script, người dùng thấy traceback thô |
| Toàn bộ logic + UI trong 1 file 250+ dòng | Khó test, khó review, khó thêm tính năng mà không đụng chỗ khác |
| Không có khái niệm user/collection | Tất cả người dùng chia sẻ chung 1 kho tài liệu — không thể phân quyền |

## 2. Cấu trúc thư mục sau refactor

```
ai_tutor/
├── app.py                  # Entry point Streamlit — CHỈ điều phối UI (~100 dòng)
├── config.py                # Đọc .env, dataclass cấu hình, logger + observability
├── exceptions.py             # Hệ thống exception riêng (AITutorError và các lớp con)
├── database.py               # PostgreSQL + pgvector, hybrid search, lịch sử chat
├── document_processor.py     # Đọc PDF/DOCX/TXT, OCR, chunking (fixed/semantic)
├── tools.py                   # web_search_tool, python_math_tool, document_reader_tool
├── agent.py                  # LLM Gemini + LangGraph agent, ask/stream
├── ui/
│   ├── __init__.py
│   ├── styles.py              # CSS
│   ├── sidebar.py             # Workspace + upload & xử lý tài liệu
│   ├── preview.py             # Xem trước tài liệu
│   └── chat.py                # Giao diện chat (streaming + lưu lịch sử)
├── tests/                     # pytest, mock toàn bộ DB/API thật (xem mục 9)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_document_processor.py
│   ├── test_database.py
│   ├── test_tools.py
│   └── test_agent.py
├── .github/workflows/ci.yml   # Lint (ruff) + test (pytest) tự động trên mỗi push/PR
├── requirements.txt
├── requirements-dev.txt       # pytest, ruff — chỉ cần cho dev/CI
├── pyproject.toml             # Cấu hình ruff + pytest
├── packages.txt               # Gói hệ thống (apt) cho OCR — Streamlit Cloud tự đọc
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
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

## 3. Embedding chạy LOCAL thay vì gọi API Gemini

Bản này dùng `HuggingFaceEmbeddings` (`BAAI/bge-m3`, chạy CPU bằng
`sentence-transformers`) thay vì `GoogleGenerativeAIEmbeddings`. Lý do:
free-tier Gemini rất dễ hết quota, và bước **index tài liệu** (embed hàng
trăm/nghìn đoạn văn bản) là nơi tốn quota nhanh nhất trong app — nhiều hơn
hẳn so với các lệnh gọi LLM khi chat. Chuyển bước này sang chạy local giúp:

- Không còn phụ thuộc rate-limit/quota của Google cho việc index.
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

## 3.1. Nâng cao chất lượng RAG

**Trích dẫn nguồn**: mỗi đoạn văn bản khi index được gắn `metadata.source =
tên_file`. `document_reader_tool` trả kết quả kèm nhãn `[Nguồn: tên_file]`,
và system prompt của agent (`agent.SYSTEM_PROMPT`) yêu cầu LLM trích dẫn
nguồn tương ứng trong câu trả lời — người dùng biết thông tin lấy từ file nào.
*(Tài liệu đã index trước khi có tính năng này sẽ không có `source` — cần
index lại để có trích dẫn.)*

**Chunking theo ngữ nghĩa**: đặt `CHUNKING_STRATEGY=semantic` trong `.env` để
dùng `SemanticChunker` (chia đoạn theo điểm ngắt ngữ nghĩa) thay vì chia theo
độ dài cố định. Cho kết quả retrieval tốt hơn với tài liệu học thuật có mạch
ý rõ ràng, nhưng chậm hơn khi index vì phải encode để tìm điểm ngắt. Mặc định
vẫn là `fixed` (nhanh, ổn định).

**OCR cho PDF scan ảnh**: nếu `PdfReader` không trích xuất được text (PDF là
ảnh scan), hệ thống tự động thử OCR bằng `pytesseract` + `pdf2image` (yêu cầu
gói hệ thống `poppler-utils`, `tesseract-ocr`, `tesseract-ocr-vie` — đã khai
báo sẵn trong `packages.txt`, Streamlit Community Cloud tự đọc file này). Nếu
thiếu thư viện/gói hệ thống, tự động bỏ qua OCR (không crash), chỉ log
warning. Tắt tính năng này bằng `ENABLE_OCR=false` nếu không cần.

**Hybrid search**: `document_reader_tool` không chỉ dùng vector search mà còn
kết hợp full-text search của Postgres (`to_tsvector`/`plainto_tsquery`) qua
Reciprocal Rank Fusion (`database.hybrid_search`). Giúp bắt tốt hơn các câu
hỏi chứa từ khóa/số liệu chính xác (công thức, tên riêng, số liệu) mà vector
thuần đôi khi bỏ sót. Nếu full-text search lỗi (khác schema bảng do phiên bản
`langchain-postgres` khác nhau), tự động fallback về vector-only, không làm
gián đoạn trải nghiệm.

**Rerank có fallback**: nếu Cohere rerank lỗi (hết quota, sai key, downtime),
`document_reader_tool` tự động dùng top-k của hybrid search (chưa rerank)
thay vì báo lỗi hoàn toàn cho người dùng.

## 3.2. Trải nghiệm người dùng (UX)

**Streaming câu trả lời**: `agent.stream_agent()` sinh dần từng đoạn text từ
LLM (qua `agent.stream(..., stream_mode="messages")`), UI dùng
`st.write_stream()` để hiển thị ngay khi có token thay vì đợi toàn bộ câu trả
lời. Nếu streaming lỗi hoặc không được hỗ trợ (phụ thuộc phiên bản
`langgraph`), tự động fallback về `ask_agent()` không streaming.

**Không gian làm việc qua URL (workspace)**: thay vì `session_id` ngẫu nhiên
mỗi tab (mất khi refresh trang), workspace ID được lưu trong URL
(`?ws=ws_xxxxx`). Refresh trang, đóng mở lại tab, hay chia sẻ link cho người
khác đều giữ nguyên đúng bộ tài liệu + lịch sử chat. Nút "Tạo không gian làm
việc mới" trong sidebar sinh 1 workspace trống để bắt đầu bộ tài liệu khác
(vd môn học khác).

**Lịch sử chat lưu vào Postgres**: mỗi tin nhắn (user & assistant) được lưu
vào bảng `ai_tutor_chat_messages` theo `workspace_id` (`database.save_chat_message`
/ `get_chat_history`). Khi quay lại workspace (kể cả sau khi app bị khởi động
lại — Streamlit Cloud có thể restart process bất cứ lúc nào), lịch sử được
tải lại đầy đủ thay vì mất trắng như bản dùng `st.session_state` thuần.

**Quản lý nhiều workspace (bảng `ai_tutor_workspaces`)**: thay vì chỉ 1
workspace/link, sidebar giờ có selectbox liệt kê TẤT CẢ workspace đã tạo
(`database.get_all_workspaces`), cho phép đổi tên (`rename_workspace`) và xóa
hoàn toàn (`delete_workspace` — dọn sạch cả lịch sử chat, vector embedding,
và bản ghi collection, không để lại rác trong Postgres). Sidebar cũng hiển
thị danh sách file đã index trong workspace hiện tại
(`database.get_indexed_files`, đọc từ `cmetadata->>'source'` — hỗ trợ cả vài
tên khóa cũ như `file_name`/`filename`/`original_file` để tương thích ngược
nếu schema metadata từng đổi tên).

Khi xóa workspace đang mở, sidebar tự động chuyển sang workspace còn lại gần
nhất (hoặc tạo mới nếu không còn workspace nào), tránh để người dùng rơi vào
trạng thái "không có workspace nào" gây lỗi.

## 4. Vì sao chuyển từ FAISS sang PostgreSQL + pgvector

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

## 5. Xử lý lỗi (error handling)

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

## 6. Kiến trúc production-ready — góp ý mở rộng

### 5.1. Mở rộng phân quyền (RBAC / multi-tenant)

Bản refactor này đã đặt sẵn 2 "móc" (hook) cho phân quyền:

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

# (nếu chưa có Postgres) chạy docker-compose ở mục 4, hoặc trỏ tới DB có sẵn

streamlit run app.py
```

## 8. Deploy lên production

### 8.1. Chạy bằng Docker (khuyến nghị)

Đã có sẵn `Dockerfile`, `.dockerignore`, và `docker-compose.yml` (app +
Postgres/pgvector) trong repo:

```bash
cp .env.example .env   # điền API key + mật khẩu DB thật
docker compose up --build -d
```

`Dockerfile` **tải sẵn model embedding vào image lúc build** (không phải lúc
chạy) — quan trọng vì nhiều nền tảng hosting dùng filesystem tạm thời, nếu
không bake sẵn thì mỗi lần container khởi động lại sẽ tải lại model
~1-2GB, làm cold-start rất chậm.

### 8.2. Chọn nền tảng hosting

| Nền tảng | Phù hợp không? | Lý do |
|---|---|---|
| **Streamlit Community Cloud** (free) | ⚠️ Rủi ro | Giới hạn ~1GB RAM/CPU thấp — `torch` + `bge-m3` load vào RAM có thể vượt hạn mức; cũng không có Postgres đi kèm, phải trỏ ra DB ngoài. Nếu vẫn muốn dùng, cân nhắc đổi sang model embedding nhỏ hơn (vd `paraphrase-multilingual-MiniLM-L12-v2`, nhẹ hơn nhiều so với `bge-m3`) |
| **Railway / Render / Fly.io** | ✅ Phù hợp | Hỗ trợ Docker trực tiếp, có thể chọn gói RAM đủ cho `torch`; Railway/Render còn có addon Postgres (kiểm tra có bật được extension `pgvector` không) |
| **VPS riêng (DigitalOcean, Hetzner...) + Docker** | ✅ Phù hợp nhất để kiểm soát | Toàn quyền chọn RAM/CPU, chạy `docker-compose.yml` y hệt local |
| **Supabase / Neon (Postgres)** | ✅ Khuyến nghị cho DB | Có sẵn extension `pgvector`, có gói free — dùng làm `PG_HOST` thay vì tự host Postgres, giảm 1 phần việc vận hành |

### 8.3. Trước khi đẩy code lên Git

- **Đã thêm `.gitignore`** — đảm bảo `.env` (chứa API key thật, mật khẩu DB)
  **không bao giờ được commit**. Chỉ commit `.env.example`.
- Trên nền tảng hosting, khai báo `GOOGLE_API_KEY`, `COHERE_API_KEY`,
  `PG_*` qua cơ chế **secrets/environment variables** của platform đó
  (Railway/Render đều có mục "Environment Variables" riêng), không hard-code
  trong code hay commit vào repo.
- Nếu trước đó từng chạy thử và có thư mục `faiss_index/` (bản FAISS cũ) hoặc
  `.venv/` trong repo, xoá khỏi Git history nếu đã lỡ commit
  (`git rm -r --cached faiss_index .venv`) trước khi push.
- CI/CD tối thiểu nên có: chạy `python -m py_compile` hoặc `ruff check` trên
  mọi pull request trước khi merge, để bắt lỗi cú pháp sớm.

## 9. Kiểm thử tự động (Testing) & CI/CD

**Bộ test** nằm ở `tests/`, dùng `pytest`, chạy được **không cần Postgres/API
thật** — mọi lời gọi ra ngoài (DB, Cohere, Gemini, model embedding) đều được
mock (`unittest.mock`/`monkeypatch`), đúng nguyên tắc unit test: nhanh, không
phụ thuộc mạng, không cần secrets thật.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -v          # chạy toàn bộ test
ruff check .        # lint
```

Phạm vi test hiện có:

| File | Kiểm tra |
|---|---|
| `test_config.py` | Đọc biến môi trường, validate bắt buộc, giá trị mặc định |
| `test_document_processor.py` | Trích xuất text, chunking, gắn metadata nguồn, fallback khi thiếu OCR, lỗi 1 file không chặn cả batch |
| `test_database.py` | Thuật toán RRF (hybrid search), fallback vector-only khi full-text search lỗi, lưu/tải lịch sử chat không raise khi DB lỗi, **`index_chunks` với `metadata_common` bị bỏ trống (test hồi quy)**, **quản lý workspace** (`get_all_workspaces`, `ensure_workspace_exists`, `rename_workspace`, `delete_workspace` — đúng thứ tự dọn dẹp & không xóa nhầm workspace khác, `get_indexed_files` — lọc bỏ giá trị `None`) |
| `test_tools.py` | Định dạng trích dẫn nguồn, fallback khi Cohere rerank lỗi, thông báo khi không tìm thấy tài liệu |
| `test_agent.py` | Chuẩn hóa content (string/list), `ask_agent` không bao giờ raise, `stream_agent` chỉ lấy đúng token của node LLM cuối và fallback đúng khi streaming lỗi |
| `test_app.py` | Phần LOGIC thuần trong `app.py` (không test UI/layout): resolve `workspace_id` từ URL, `_bootstrap` dừng đúng cách khi lỗi cấu hình/DB (`st.stop()`), cache agent theo workspace, tải lại lịch sử chat đúng lúc |

`tests/conftest.py` stub package `langchain_huggingface` (kéo theo
`sentence-transformers`/`torch`, rất nặng) và `streamlit`/`streamlit_pdf_viewer`
(chưa cần cài trong môi trường test) bằng module giả — unit test logic
nghiệp vụ không cần tải model thật hay chạy trong runtime Streamlit thật.
Nhờ vậy `test_app.py` test được phần logic của `app.py` (resolve workspace,
fail-fast, cache agent) mà không cần `streamlit` cài đặt — nhưng KHÔNG test
phần vẽ UI (layout cột, nút toggle ẩn/hiện xem trước tài liệu), vì đó thuộc
phạm vi test tích hợp (chạy `streamlit run app.py` thật và kiểm tra bằng tay,
hoặc `streamlit.testing.v1.AppTest` nếu muốn tự động hóa sau này).
**Test tích hợp thật** (cần Postgres + model thật chạy, xác nhận toàn bộ pipeline hoạt động end-to-end) nên viết
riêng ở `tests/integration/`, dùng `testcontainers-python` với image
`pgvector/pgvector:pg16`, và chỉ chạy khi cần (không bắt buộc trong CI mỗi
lần push, vì chậm hơn nhiều).

**CI/CD**: `.github/workflows/ci.yml` tự động chạy `ruff check` + `pytest`
trên mọi push/PR vào nhánh `main`. Có lỗi lint hoặc test fail sẽ chặn merge
nếu bạn bật branch protection rule yêu cầu check này pass trên GitHub. Mở
rộng thêm khi cần: build & push Docker image lên registry sau khi CI pass,
hoặc tự động deploy lên staging.

## 10. Quan sát hệ thống (Observability)

**Log tự động gắn `workspace_id`**: `config.set_log_context(workspace_id)`
được gọi 1 lần ở đầu `app.main()`, dùng `contextvars` (an toàn theo từng luồng
chạy). Từ đó, MỌI dòng log ở `database.py`, `tools.py`, `agent.py` tự động
có `ws=<workspace_id>` mà không cần truyền tay qua từng hàm — khi debug
production nhiều người dùng chung app, lọc log theo đúng người dùng bằng
`grep "ws=ws_abc123"` thay vì phải đoán dòng nào của ai.

**Đo thời gian các bước tốn tài nguyên**: decorator `config.log_duration()`
áp cho `ask_agent`, `web_search_tool`, `python_math_tool`, `index_chunks`,
`hybrid_search`, và toàn bộ nhóm hàm quản lý workspace (`get_all_workspaces`,
`ensure_workspace_exists`, `rename_workspace`, `delete_workspace`,
`get_indexed_files`) — tự động log `⏱ <tên bước> hoàn thành sau X.XXs`. Giúp phát
hiện điểm nghẽn hiệu năng (vd embedding chậm vì máy yếu, hay Cohere rerank
chậm bất thường) mà không cần thêm code đo thời gian thủ công ở từng nơi.
`document_reader_tool` (được tạo qua `@tool` decorator nên không áp
`log_duration` trực tiếp được) đo thời gian thủ công bằng `time.perf_counter()`.

**Log lifecycle của workspace ở đúng mức độ**: các thao tác người dùng chủ
động thực hiện — tạo mới, đổi tên, xóa workspace — log ở mức **INFO** (đáng
để thấy ngay trong log mặc định). Các hàm bị gọi ở MỌI lần Streamlit rerun
script (`get_all_workspaces`, `ensure_workspace_exists` khi không có gì thay
đổi, `get_indexed_files`) log ở mức **DEBUG**, tránh làm ngập log sản xuất
bằng những dòng lặp lại vô nghĩa mỗi lần người dùng gõ 1 ký tự.

**LangSmith tracing (tuỳ chọn, không cần sửa code)**: đặt 3 biến môi trường
trong `.env` (đã có mẫu, comment sẵn):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ai-tutor
```
LangChain/LangGraph tự động đọc các biến này và gửi trace chi tiết (tool nào
được gọi theo thứ tự nào, prompt/response đầy đủ ở từng bước, latency từng
bước) lên dashboard tại smith.langchain.com — hữu ích hơn nhiều so với đọc
log thô khi cần debug hành vi của agent (vd vì sao agent không gọi tool như
kỳ vọng). Có gói free đủ dùng cho quy mô nhỏ; kiểm tra hạn mức hiện tại tại
trang chủ LangSmith trước khi dùng cho production.

**Hướng mở rộng tiếp theo** (chưa implement, gợi ý cho tương lai):
- Structured logging dạng JSON (thay vì text) để đưa vào hệ thống log tập
  trung (Datadog, Grafana Loki, CloudWatch...).
- Metrics dạng số (Prometheus) cho: số câu hỏi/phút, tỷ lệ lỗi từng tool, độ
  trễ trung bình — hiện chỉ có log dạng text, cần parse mới ra được số liệu
  tổng hợp.
- Alert tự động khi tỷ lệ lỗi Cohere/Gemini vượt ngưỡng trong khoảng thời
  gian ngắn (dấu hiệu hết quota hoặc key hết hạn).

## 11. Những điểm cần bạn tự xác nhận trước khi deploy

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
