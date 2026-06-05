import os

# Embedding model (hỗ trợ tiếng Việt)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# LLM model
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Text splitting
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 5

# Ngưỡng liên quan (L2 distance, normalize_embeddings=True)
# Phạm vi: 0.0 (giống hệt) → 2.0 (hoàn toàn khác)
# Nếu score của document GẦN NHẤT vẫn > ngưỡng này
# → câu hỏi nằm ngoài phạm vi tài liệu → từ chối trả lời
# Gợi ý: 0.8 (chặt) | 1.0 (cân bằng) | 1.2 (lỏng)
RELEVANCE_THRESHOLD = 1.0

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3

# Paths
FAISS_INDEX_PATH = "vectorstore/faiss_index"
DATA_DIR = "data/raw"
REPORTS_DIR = "reports"
