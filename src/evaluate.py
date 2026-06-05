"""
Đánh giá hệ thống RAG chatbot tư vấn tuyển sinh FPT.

Metrics:
  - Retrieval: avg top-score, avg retrieval time
  - Generation: avg answer length, avg generation time
  - Context coverage: tỉ lệ câu hỏi có ít nhất 1 doc liên quan (score < ngưỡng)
"""

import os
import sys
import json
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
from config import REPORTS_DIR

TEST_QUESTIONS = [
    "Trường FPT tuyển sinh các cấp học nào?",
    "Học phí của trường FPT là bao nhiêu?",
    "Hồ sơ đăng ký tuyển sinh cần những giấy tờ gì?",
    "Thời gian tuyển sinh của trường FPT là khi nào?",
    "Trường FPT có những cơ sở nào và ở đâu?",
    "Chương trình học tại trường FPT có gì đặc biệt?",
    "Điều kiện tuyển sinh vào lớp 1 là gì?",
    "Trường FPT có học bổng không? Điều kiện nhận học bổng?",
    "Phương thức xét tuyển vào THPT FPT như thế nào?",
    "Học sinh có thể đăng ký tuyển sinh online không?",
]

RELEVANCE_THRESHOLD = 1.0  # L2 distance — score < ngưỡng này coi là có liên quan


def evaluate_retrieval(pipeline: RAGPipeline, questions: list) -> dict:
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ RETRIEVAL")
    print("=" * 50)

    per_q = []
    for q in questions:
        t0 = time.time()
        results = pipeline.retriever.retrieve(q)
        elapsed = time.time() - t0

        top_score = float(results[0][1]) if results else None
        relevant = top_score is not None and top_score < RELEVANCE_THRESHOLD

        per_q.append(
            {
                "question": q,
                "num_retrieved": len(results),
                "top_score": round(top_score, 4) if top_score else None,
                "relevant": relevant,
                "retrieval_time_s": round(elapsed, 3),
            }
        )
        tag = "[OK]" if relevant else "[LOW]"
        print(f"{tag} Q: {q[:60]}")
        print(f"     docs={len(results)} | top_score={top_score:.4f} | time={elapsed:.3f}s\n")

    valid = [r for r in per_q if r["top_score"] is not None]
    avg_score = sum(r["top_score"] for r in valid) / len(valid) if valid else 0
    avg_time = sum(r["retrieval_time_s"] for r in per_q) / len(per_q)
    coverage = sum(1 for r in per_q if r["relevant"]) / len(per_q) * 100

    summary = {
        "avg_top_score": round(avg_score, 4),
        "avg_retrieval_time_s": round(avg_time, 3),
        "context_coverage_pct": round(coverage, 1),
        "per_question": per_q,
    }

    print(f"  Avg top score     : {avg_score:.4f} (thấp = liên quan hơn)")
    print(f"  Avg retrieval time: {avg_time:.3f}s")
    print(f"  Context coverage  : {coverage:.1f}%")

    return summary


def evaluate_generation(pipeline: RAGPipeline, questions: list) -> dict:
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ GENERATION")
    print("=" * 50)

    per_q = []
    for q in questions:
        t0 = time.time()
        result = pipeline.answer(q)
        elapsed = time.time() - t0

        answer = result["answer"]
        per_q.append(
            {
                "question": q,
                "answer": answer,
                "answer_length": len(answer),
                "generation_time_s": round(elapsed, 3),
            }
        )
        print(f"Q: {q}")
        print(f"A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"   length={len(answer)} ký tự | time={elapsed:.3f}s\n")

    avg_len = sum(r["answer_length"] for r in per_q) / len(per_q)
    avg_time = sum(r["generation_time_s"] for r in per_q) / len(per_q)

    summary = {
        "avg_answer_length": round(avg_len),
        "avg_generation_time_s": round(avg_time, 3),
        "per_question": per_q,
    }

    print(f"  Avg answer length: {avg_len:.0f} ký tự")
    print(f"  Avg gen time     : {avg_time:.3f}s")

    return summary


def run_evaluation(quick: bool = False):
    questions = TEST_QUESTIONS[:3] if quick else TEST_QUESTIONS
    print(f"\nBắt đầu đánh giá với {len(questions)} câu hỏi {'(quick mode)' if quick else ''}...\n")

    pipeline = RAGPipeline()

    retrieval_metrics = evaluate_retrieval(pipeline, questions)
    generation_metrics = evaluate_generation(pipeline, questions)

    report = {
        "num_questions": len(questions),
        "quick_mode": quick,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("TÓM TẮT KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 50)
    print(f"  Số câu hỏi           : {len(questions)}")
    print(f"  Avg retrieval score  : {retrieval_metrics['avg_top_score']}")
    print(f"  Avg retrieval time   : {retrieval_metrics['avg_retrieval_time_s']}s")
    print(f"  Context coverage     : {retrieval_metrics['context_coverage_pct']}%")
    print(f"  Avg answer length    : {generation_metrics['avg_answer_length']} ký tự")
    print(f"  Avg generation time  : {generation_metrics['avg_generation_time_s']}s")
    print(f"\n  Báo cáo lưu tại: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá RAG chatbot tuyển sinh FPT")
    parser.add_argument("--quick", action="store_true", help="Chỉ chạy 3 câu hỏi đầu")
    args = parser.parse_args()
    run_evaluation(quick=args.quick)
