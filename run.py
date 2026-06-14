"""
CLI để test nhanh RAG pipeline mà không cần chạy Gradio.

Sử dụng:
  python run.py "Câu hỏi của bạn"   → trả lời một câu
  python run.py                      → chế độ hội thoại interactive
  python run.py --debug "câu hỏi"   → hiển thị cả context
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import RAGPipeline


def print_separator(char="─", width=60):
    print(char * width)


def ask_one(pipeline: RAGPipeline, question: str, debug: bool = False):
    print_separator()
    print(f"Câu hỏi: {question}")
    print_separator()

    meta = None
    answer_parts = []
    print("Câu trả lời:\n", end="", flush=True)

    for kind, value in pipeline.stream(question):
        if kind == "meta":
            meta = value
            if debug and meta["in_scope"]:
                # In context sau khi in xong câu trả lời
                pass
        elif kind == "token":
            print(value, end="", flush=True)
            answer_parts.append(value)
        elif kind == "answer":
            # Out-of-scope reply (không stream)
            print(value, end="", flush=True)

    print("\n")

    if meta and meta["sources"]:
        print("Nguồn tham khảo:")
        for i, src in enumerate(meta["sources"], 1):
            print(f"  {i}. {src['source']} (score: {src['score']:.4f})")
            print(f"     {src['content'][:100].strip()}...")
    print_separator()


def interactive_mode(pipeline: RAGPipeline, debug: bool = False):
    print_separator("═")
    print("  Chatbot Tư Vấn Tuyển Sinh FPT School")
    print("  Gõ 'quit' hoặc 'thoát' để thoát")
    print_separator("═")

    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "thoát", "q"):
            print("Tạm biệt!")
            break

        ask_one(pipeline, question, debug=debug)


def main():
    parser = argparse.ArgumentParser(
        description="CLI test RAG chatbot tuyển sinh FPT"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Câu hỏi cần trả lời (bỏ trống để vào chế độ interactive)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Hiển thị context được truy xuất",
    )
    args = parser.parse_args()

    print("Đang tải pipeline...\n")
    pipeline = RAGPipeline()

    if args.question:
        ask_one(pipeline, args.question, debug=args.debug)
    else:
        interactive_mode(pipeline, debug=args.debug)


if __name__ == "__main__":
    main()
