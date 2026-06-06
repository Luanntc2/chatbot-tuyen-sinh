import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import Retriever
from src.generator import Generator
from config import TOP_K, RELEVANCE_THRESHOLD

# ─── Prompt template ─────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Dựa trên các thông tin tham khảo dưới đây về tuyển sinh của Trường FPT, \
hãy trả lời câu hỏi của phụ huynh/học sinh một cách chính xác và đầy đủ.

Quan trọng: Chỉ được trả lời dựa trên THÔNG TIN THAM KHẢO được cung cấp. \
Không được tự suy diễn hoặc lấy thông tin từ nguồn khác ngoài tài liệu này.

Nếu thông tin không có trong tài liệu tham khảo, hãy nói rõ là bạn không có \
thông tin về vấn đề đó và đề nghị phụ huynh liên hệ trực tiếp với nhà trường \
để được tư vấn cụ thể hơn.

===== THÔNG TIN THAM KHẢO =====
{context}
================================

Câu hỏi: {question}

Câu trả lời:"""

# Câu trả lời mặc định khi câu hỏi nằm ngoài phạm vi tài liệu
OUT_OF_SCOPE_REPLY = (
    "Xin lỗi, câu hỏi này nằm ngoài phạm vi thông tin tuyển sinh "
    "mà tôi được cung cấp.\n\n"
    "Để được tư vấn chính xác, vui lòng liên hệ trực tiếp với Trường FPT:\n"
    "📞 Hotline: 1900 6600\n"
    "📧 Email: tuyensinh@fptschool.edu.vn\n"
    "🌐 Website: https://fptschool.edu.vn"
)


# ─── Kiểm tra độ liên quan ───────────────────────────────────────────────────

def is_out_of_scope(results: list, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """
    Kiểm tra xem câu hỏi có nằm ngoài phạm vi tài liệu không.

    Logic: Nếu document GẦN NHẤT (score nhỏ nhất = liên quan nhất)
    vẫn có score > threshold → không tìm được tài liệu liên quan nào
    → câu hỏi nằm ngoài phạm vi → từ chối trả lời.

    Args:
        results:   Danh sách (Document, score) từ FAISS
        threshold: Ngưỡng L2 distance (0.0 = giống hệt, 2.0 = hoàn toàn khác)

    Returns:
        True nếu câu hỏi ngoài phạm vi, False nếu trong phạm vi
    """
    if not results:
        return True

    best_score = min(score for _, score in results)
    return best_score > threshold


def relevance_summary(results: list, threshold: float = RELEVANCE_THRESHOLD) -> str:
    """Tóm tắt độ liên quan để log/debug."""
    if not results:
        return "Không có kết quả"
    best = min(score for _, score in results)
    status = "✅ Trong phạm vi" if best <= threshold else "❌ Ngoài phạm vi"
    return f"{status} | Best score: {best:.4f} | Threshold: {threshold}"


# ─── RAG Pipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        print("=== Khởi động RAG Pipeline ===\n")
        self.retriever = Retriever()
        self.generator = Generator()
        print("=== Pipeline sẵn sàng ===\n")

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Xử lý một câu hỏi qua pipeline RAG đầy đủ.

        Luồng:
          1. Retrieve: tìm top-K document từ FAISS
          2. Kiểm tra ngưỡng liên quan → từ chối nếu ngoài phạm vi
          3. Build prompt (context + question)
          4. Generate: LLM sinh câu trả lời dựa trên context

        Trả về dict:
          - question       : câu hỏi gốc
          - answer         : câu trả lời
          - context        : context ghép từ retriever
          - sources        : list doc nguồn kèm score
          - in_scope       : True/False – câu hỏi có trong tài liệu không
          - best_score     : score thấp nhất (liên quan nhất) trong kết quả
        """
        # ── Bước 1: Retrieve ──────────────────────────────────────────────
        results = self.retriever.retrieve(question, top_k=top_k)
        best_score = min((score for _, score in results), default=999.0)

        rel_summary = relevance_summary(results)
        print(f"  [Relevance] {rel_summary}")

        # ── Bước 2: Kiểm tra ngưỡng liên quan ────────────────────────────
        if is_out_of_scope(results):
            print(f"  [Pipeline] Câu hỏi nằm NGOÀI phạm vi tài liệu → từ chối trả lời")
            return {
                "question":   question,
                "answer":     OUT_OF_SCOPE_REPLY,
                "context":    "",
                "sources":    [],
                "in_scope":   False,
                "best_score": round(best_score, 4),
            }

        # ── Bước 3: Build prompt ──────────────────────────────────────────
        context = self.retriever.format_context(results)
        prompt  = PROMPT_TEMPLATE.format(context=context, question=question)

        # ── Bước 4: Generate ──────────────────────────────────────────────
        print(f"  [Pipeline] Câu hỏi TRONG phạm vi → sinh câu trả lời...")
        answer = self.generator.generate(prompt)

        sources = [
            {
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "score":  round(float(score), 4),
            }
            for doc, score in results
        ]

        return {
            "question":   question,
            "answer":     answer,
            "context":    context,
            "sources":    sources,
            "in_scope":   True,
            "best_score": round(best_score, 4),
        }

    def stream(self, question: str, top_k: int = TOP_K):
        """
        Generator streaming: yield ("meta", dict) rồi yield ("token", str) từng token.

        Dùng cho Gradio streaming — hiện chữ ngay thay vì đợi toàn bộ câu trả lời.
        """
        results   = self.retriever.retrieve(question, top_k=top_k)
        best_score = min((score for _, score in results), default=999.0)
        in_scope  = not is_out_of_scope(results)

        sources = [
            {
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200 else doc.page_content
                ),
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "score":  round(float(score), 4),
            }
            for doc, score in results
        ] if in_scope else []

        yield "meta", {
            "in_scope":   in_scope,
            "best_score": round(best_score, 4),
            "sources":    sources,
        }

        if not in_scope:
            print(f"  [Pipeline] Câu hỏi NGOÀI phạm vi → từ chối")
            yield "answer", OUT_OF_SCOPE_REPLY
            return

        context = self.retriever.format_context(results)
        prompt  = PROMPT_TEMPLATE.format(context=context, question=question)
        print(f"  [Pipeline] Câu hỏi TRONG phạm vi → streaming...")

        for token in self.generator.stream(prompt):
            yield "token", token

    def __call__(self, question: str) -> str:
        """Shortcut: chỉ trả về chuỗi câu trả lời."""
        return self.answer(question)["answer"]
