"""
So sánh Qwen2.5-1.5B-Instruct vs TinyLlama-1.1B trên cùng RAG pipeline.

Chạy:
  python src/compare_models.py            # 5 câu hỏi
  python src/compare_models.py --quick    # 3 câu hỏi
  python src/compare_models.py --all      # 10 câu hỏi
"""

import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import Retriever
from src.generator import Generator
from src.pipeline import PROMPT_TEMPLATE, OUT_OF_SCOPE_REPLY, is_out_of_scope
from config import TOP_K, RELEVANCE_THRESHOLD, REPORTS_DIR

# ─── Cấu hình 2 model cần so sánh ───────────────────────────────────────────

MODELS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

TEST_QUESTIONS = [
    "Trường FPT tuyển sinh các cấp học nào?",
    "Học phí của trường FPT là bao nhiêu?",
    "Hồ sơ đăng ký tuyển sinh cần những giấy tờ gì?",
    "Trường FPT có học bổng không? Điều kiện nhận học bổng?",
    "Phương thức xét tuyển vào lớp 10 THPT FPT như thế nào?",
    "Chương trình học tại trường FPT có gì đặc biệt?",
    "Điều kiện tuyển sinh vào lớp 1 là gì?",
    "Thời gian nộp hồ sơ tuyển sinh là khi nào?",
    "Học sinh có thể đăng ký tuyển sinh online không?",
    "Chính sách hoàn trả học phí của trường như thế nào?",
]


# ─── Hàm đánh giá chất lượng đơn giản ───────────────────────────────────────

def score_answer(answer: str, question: str) -> dict:
    """Chấm điểm câu trả lời theo các tiêu chí cơ bản."""
    length = len(answer)
    has_number = any(c.isdigit() for c in answer)
    has_vnd = "VNĐ" in answer or "đồng" in answer.lower()
    is_refusal = answer.strip().startswith("Xin lỗi")
    too_short = length < 80
    too_long = length > 2000

    score = 0
    if not is_refusal:      score += 30
    if not too_short:       score += 20
    if not too_long:        score += 10
    if has_number:          score += 20
    if "FPT" in answer:     score += 20

    return {
        "score_100": score,
        "length": length,
        "has_numbers": has_number,
        "has_vnd_amounts": has_vnd,
        "is_refusal": is_refusal,
    }


# ─── Pipeline thủ công để dùng generator bất kỳ ─────────────────────────────

def run_question(retriever: Retriever, generator: Generator, question: str) -> dict:
    results = retriever.retrieve(question, top_k=TOP_K)
    best_score = min((s for _, s in results), default=999.0)

    if is_out_of_scope(results, RELEVANCE_THRESHOLD):
        answer = OUT_OF_SCOPE_REPLY
    else:
        context = retriever.format_context(results)
        prompt  = PROMPT_TEMPLATE.format(context=context, question=question)
        answer  = generator.generate(prompt)

    return {
        "answer": answer,
        "best_score": round(best_score, 4),
        "in_scope": best_score <= RELEVANCE_THRESHOLD,
    }


# ─── So sánh chính ───────────────────────────────────────────────────────────

def compare(questions: list):
    sep = "=" * 65

    print(sep)
    print("  SO SÁNH MODEL: Qwen2.5-1.5B  vs  TinyLlama-1.1B")
    print(f"  Số câu hỏi: {len(questions)}")
    print(sep)

    # Dùng chung 1 retriever
    print("\nKhởi động Retriever (dùng chung)...")
    retriever = Retriever()

    results_by_model = {}

    for label, model_name in MODELS.items():
        print(f"\n{'─'*65}")
        print(f"  Đang tải model: [{label}] {model_name}")
        print(f"{'─'*65}")
        generator = Generator(model_name=model_name)

        per_q = []
        for i, q in enumerate(questions, 1):
            print(f"\n  [{i}/{len(questions)}] {q}")
            t0 = time.time()
            out = run_question(retriever, generator, q)
            elapsed = round(time.time() - t0, 2)

            quality = score_answer(out["answer"], q)
            record = {
                "question": q,
                "answer": out["answer"],
                "best_score": out["best_score"],
                "in_scope": out["in_scope"],
                "gen_time_s": elapsed,
                **quality,
            }
            per_q.append(record)

            preview = out["answer"][:100].replace("\n", " ")
            print(f"  → [{elapsed}s | score={quality['score_100']}/100] {preview}...")

        results_by_model[label] = per_q

        # Giải phóng RAM trước khi tải model tiếp theo
        del generator
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return results_by_model


def print_comparison_table(results_by_model: dict, questions: list):
    labels = list(results_by_model.keys())
    sep = "─" * 65

    print(f"\n\n{'='*65}")
    print("  BẢNG SO SÁNH KẾT QUẢ")
    print(f"{'='*65}")

    # Tóm tắt tổng thể
    print(f"\n{'Tiêu chí':<30} | {labels[0]:<16} | {labels[1]:<16}")
    print(sep)

    for label in labels:
        rows = results_by_model[label]
        avg_time  = sum(r["gen_time_s"] for r in rows) / len(rows)
        avg_len   = sum(r["length"] for r in rows) / len(rows)
        avg_score = sum(r["score_100"] for r in rows) / len(rows)
        in_scope  = sum(1 for r in rows if r["in_scope"])
        refusals  = sum(1 for r in rows if r["is_refusal"])

        if label == labels[0]:
            col_a = (avg_time, avg_len, avg_score, in_scope, refusals)
        else:
            col_b = (avg_time, avg_len, avg_score, in_scope, refusals)

    metrics = [
        ("Thời gian TB (giây)", f"{col_a[0]:.1f}s", f"{col_b[0]:.1f}s"),
        ("Độ dài câu TL TB (ký tự)", f"{col_a[1]:.0f}", f"{col_b[1]:.0f}"),
        ("Điểm chất lượng TB (/100)", f"{col_a[2]:.1f}", f"{col_b[2]:.1f}"),
        ("Câu hỏi trong phạm vi", f"{col_a[3]}/{len(questions)}", f"{col_b[3]}/{len(questions)}"),
        ("Số lần từ chối TL", f"{col_a[4]}", f"{col_b[4]}"),
    ]

    for name, va, vb in metrics:
        print(f"{name:<30} | {va:<16} | {vb:<16}")

    # Chi tiết từng câu
    print(f"\n\n{'='*65}")
    print("  CHI TIẾT TỪNG CÂU HỎI")
    print(f"{'='*65}")

    for i, q in enumerate(questions):
        print(f"\n{'─'*65}")
        print(f"  Câu {i+1}: {q}")
        print(f"{'─'*65}")
        for label in labels:
            r = results_by_model[label][i]
            preview = r["answer"][:200].replace("\n", " ")
            print(f"\n  [{label}] (thời gian: {r['gen_time_s']}s | điểm: {r['score_100']}/100)")
            print(f"  {preview}{'...' if len(r['answer']) > 200 else ''}")


def save_report(results_by_model: dict, questions: list):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "model_comparison.json")

    summary = {}
    for label, rows in results_by_model.items():
        summary[label] = {
            "avg_gen_time_s":    round(sum(r["gen_time_s"] for r in rows) / len(rows), 2),
            "avg_answer_length": round(sum(r["length"] for r in rows) / len(rows)),
            "avg_quality_score": round(sum(r["score_100"] for r in rows) / len(rows), 1),
            "in_scope_count":    sum(1 for r in rows if r["in_scope"]),
            "refusal_count":     sum(1 for r in rows if r["is_refusal"]),
            "per_question":      rows,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"num_questions": len(questions), "models": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n\n  Báo cáo đầy đủ lưu tại: {path}")
    return path


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Chỉ dùng 3 câu hỏi đầu")
    parser.add_argument("--all",   action="store_true", help="Dùng tất cả 10 câu hỏi")
    args = parser.parse_args()

    if args.quick:
        questions = TEST_QUESTIONS[:3]
    elif args.all:
        questions = TEST_QUESTIONS
    else:
        questions = TEST_QUESTIONS[:5]

    results = compare(questions)
    print_comparison_table(results, questions)
    save_report(results, questions)
