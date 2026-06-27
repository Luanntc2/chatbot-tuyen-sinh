import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K

logger = logging.getLogger(__name__)


class Retriever:
        def __init__(self):
                    # FIX: Auto-detect device thay vì hardcode "cpu"
                    # → Embedding chạy trên GPU T4 nếu có, nhanh hơn đáng kể
                    if torch.cuda.is_available():
                                    _device = "cuda"
elif torch.backends.mps.is_available():
            _device = "mps"
else:
            _device = "cpu"

        print(f"  Tải embedding model (device: {_device})...")
        self.embeddings = HuggingFaceEmbeddings(
                        model_name=EMBEDDING_MODEL,
                        model_kwargs={"device": _device},
                        encode_kwargs={"normalize_embeddings": True},
        )

        if not os.path.exists(FAISS_INDEX_PATH):
                        raise FileNotFoundError(
                                            f"Không tìm thấy FAISS index tại '{FAISS_INDEX_PATH}'.\n"
                                            "Hãy chạy 'python src/ingest.py' trước."
                        )

        print(f"  Tải FAISS index từ '{FAISS_INDEX_PATH}'...")
        self.vectorstore = FAISS.load_local(
                        FAISS_INDEX_PATH,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
        )
        print("  Retriever sẵn sàng.\n")

    def retrieve(self, query: str, top_k: int = TOP_K):
                """
                        Tìm kiếm các document liên quan đến query.
                                Trả về list[(Document, score)] — score càng thấp càng liên quan (L2 distance).
                                        """
                results = self.vectorstore.similarity_search_with_score(query, k=top_k)
                return results

    def format_context(self, results) -> str:
                """Ghép các document tìm được thành chuỗi context cho LLM."""
                parts = []
                for i, (doc, score) in enumerate(results, 1):
                                source = os.path.basename(doc.metadata.get("source", "unknown"))
                                parts.append(
                                    f"[Đoạn {i} | Nguồn: {source} | Score: {score:.4f}]\n{doc.page_content.strip()}"
                                )
                            return "\n\n---\n\n".join(parts)
