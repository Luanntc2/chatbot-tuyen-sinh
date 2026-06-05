# Chatbot Tư Vấn Tuyển Sinh FPT School

Hệ thống chatbot tư vấn tuyển sinh cho **Trường Tiểu học, THCS và THPT FPT** sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** chạy hoàn toàn local, tích hợp thêm tính năng **Voice I/O** (nhập/xuất bằng giọng nói).

> Bài tập cuối kỳ môn **AI Tạo sinh** – Chương trình Thạc sĩ FPT

---

## Kiến trúc hệ thống

```
  🎤 Giọng nói người dùng
        │
        ▼ (Google Speech-to-Text API)
  📝 Text prompt
        │
        ▼
┌───────────────┐
│   Retriever   │  ← FAISS + multilingual-e5-small
│  (tìm context)│
└──────┬────────┘
       │  Top-K documents
       ▼
┌───────────────┐
│  Prompt Builder│ ← Template tiếng Việt
└──────┬────────┘
       │  Prompt đầy đủ
       ▼
┌───────────────┐
│   Generator   │  ← Qwen2.5-1.5B-Instruct
│  (sinh câu TL)│
└──────┬────────┘
       │  Text câu trả lời
       ▼ (Google Text-to-Speech API)
  🔊 Giọng nói phản hồi
```

**Luồng xử lý chính:**
`Input Voice → Text Prompt → AI xử lý (RAG) → Output Text → Voice`

---

## Stack công nghệ

| Thành phần | Công nghệ |
|---|---|
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` (fallback: `TinyLlama-1.1B-Chat`) |
| Embedding | `intfloat/multilingual-e5-small` (hỗ trợ tiếng Việt tốt) |
| Vector DB | FAISS (local, không cần server) |
| Framework | LangChain + HuggingFace Transformers |
| UI | Gradio (trực quan, dễ demo) |
| Speech-to-Text | Google Speech-to-Text API |
| Text-to-Speech | Google Text-to-Speech API |
| Ngôn ngữ | Python 3.10+ |

---

## Cấu trúc thư mục

```
chatbot-tuyen-sinh/
├── config.py              # Cấu hình tập trung
├── app.py                 # Giao diện Gradio (text + voice)
├── run.py                 # CLI test nhanh
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/               # Tài liệu nguồn (.txt, .pdf)
│       ├── tuyen_sinh_fpt.txt
│       ├── chuong_trinh_hoc.txt
│       └── hoc_phi_chi_tiet.txt
│
├── vectorstore/           # FAISS index (tự động tạo)
│   └── faiss_index/
│
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Đọc data → chunk → embed → lưu FAISS
│   ├── retriever.py       # Load FAISS, retrieve(query, top_k)
│   ├── generator.py       # Load LLM, generate(prompt)
│   ├── pipeline.py        # RAG pipeline = retriever + generator
│   └── evaluate.py        # Đánh giá hệ thống
│
├── notebooks/
│   └── demo.ipynb         # Demo step-by-step
│
└── reports/               # Báo cáo đánh giá JSON
```

---

## Cài đặt & Chạy

### Yêu cầu hệ thống
- Python 3.10+
- RAM: 8 GB trở lên
- GPU: Tùy chọn (CPU cũng chạy được, chậm hơn)
- Dung lượng: ~4 GB (model weights + index)

### Bước 1 – Cài dependencies

```bash
pip install -r requirements.txt
```

### Bước 2 – Ingest dữ liệu

Đặt file `.txt` hoặc `.pdf` vào thư mục `data/raw/`, sau đó chạy:

```bash
python src/ingest.py
```

Output mẫu:
```
=== Bắt đầu ingest dữ liệu từ 'data/raw' ===

1. Đọc tài liệu...
  [OK] tuyen_sinh_fpt.txt — 1 document(s)
  [OK] chuong_trinh_hoc.txt — 1 document(s)

   Tổng: 2 file (2 .txt, 0 .pdf) → 2 document(s)

2. Chia chunk...
   Tạo ra 48 chunk(s)

3. Tạo embeddings và lưu FAISS index...
   Đã lưu vào 'vectorstore/faiss_index'

=== Thống kê ===
  Số file đọc   : 2
  Số chunk tạo  : 48
  FAISS index   : vectorstore/faiss_index
```

### Bước 3 – Chạy giao diện Gradio

```bash
python app.py
```

Mở trình duyệt: **http://localhost:7860**

### Bước 4 – Cấu hình Voice API (tùy chọn)

Để dùng tính năng nhập/xuất giọng nói, cần Google Cloud credentials:

```bash
# Tạo file .env hoặc set biến môi trường
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

Hoặc tạo file `.env` trong thư mục gốc:
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

> Nếu không cấu hình, giao diện vẫn hoạt động bình thường ở chế độ **text-only**.

### Bước 5 – Test nhanh qua CLI

```bash
python run.py "Học phí của trường FPT là bao nhiêu?"
python run.py  # chế độ interactive
```

### Bước 6 – Đánh giá hệ thống (tùy chọn)

```bash
python src/evaluate.py --quick   # 3 câu hỏi (nhanh)
python src/evaluate.py           # 10 câu hỏi (đầy đủ)
```

---

## Cấu hình

Chỉnh sửa `config.py` để thay đổi các thông số:

```python
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K           = 5
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.3
```

---

## Luồng xử lý chi tiết

### 1. Ingest (src/ingest.py)
1. Quét `data/raw/` tìm file `.txt` và `.pdf`
2. Load từng file bằng `TextLoader` / `PyPDFLoader`
3. Chia nhỏ bằng `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=64)
4. Tạo vector embedding bằng `multilingual-e5-small`
5. Lưu FAISS index xuống disk

### 2. Retrieval (src/retriever.py)
1. Load FAISS index từ disk
2. `retrieve(query, top_k=5)` → similarity search → trả về (Document, score)
3. `format_context()` → ghép các đoạn thành chuỗi context có nhãn nguồn

### 3. Generation (src/generator.py)
1. Load `Qwen2.5-1.5B-Instruct` với `device_map="auto"`
2. Tự động chọn float16 (GPU≥8GB) hoặc int8 (GPU<8GB) hoặc float32 (CPU)
3. Áp dụng Qwen chat template với system prompt tiếng Việt
4. Generate với `max_new_tokens=512`, `temperature=0.3`

### 4. RAG Pipeline (src/pipeline.py)
```
query → retrieve() → format_context() → build_prompt() → generate() → answer
```

Prompt template:
```
Dựa trên các thông tin tham khảo dưới đây về tuyển sinh của Trường FPT,
hãy trả lời câu hỏi của phụ huynh/học sinh một cách chính xác và đầy đủ.

===== THÔNG TIN THAM KHẢO =====
{context}
================================

Câu hỏi: {question}
Câu trả lời:
```

---

## Tính Năng Voice (Speech I/O)

Hệ thống hỗ trợ giao tiếp bằng giọng nói theo luồng:

```
🎤 Input Voice  →  📝 Text Prompt  →  🤖 AI xử lý (RAG)  →  📝 Output Text  →  🔊 Voice
```

| Tính năng | Công nghệ | Mô tả |
|---|---|---|
| **Speech-to-Text** | Google Speech-to-Text API | Chuyển giọng nói người dùng → text prompt |
| **Text-to-Speech** | Google Text-to-Speech API | Chuyển câu trả lời AI → giọng nói phát lại |

**Ghi chú:**
- Phần Voice dùng Google Cloud API (cần internet + credentials)
- Toàn bộ xử lý RAG (retrieve + generate) vẫn chạy **hoàn toàn local**
- Nếu không cấu hình API, giao diện hoạt động bình thường ở chế độ text-only

---

## Kết quả đánh giá (mẫu)

| Metric | Giá trị |
|---|---|
| Avg retrieval score (L2) | ~0.35 |
| Avg retrieval time | ~0.05s |
| Context coverage | ~90% |
| Avg answer length | ~250 ký tự |
| Avg generation time (CPU) | ~30–120s |
| Avg generation time (GPU) | ~3–8s |

---

## Tác giả
- Học viên cao học – Môn AI Tạo sinh
- Trường Đại học FPT
