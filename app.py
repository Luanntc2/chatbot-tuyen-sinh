"""
Giao diện Gradio – Chatbot Tư Vấn Tuyển Sinh FPT School
=========================================================
Tab 1: 💬 Chat văn bản (text → RAG → text)
Tab 2: 🎙️ Chat giọng nói (voice → STT → RAG → TTS → voice)

Chạy: python app.py
Demo: http://localhost:7860
Colab: thêm share=True trong demo.launch()
"""

import os
import sys
import logging

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import RAGPipeline
from src.voice import (
    speech_to_text,
    text_to_speech,
    save_audio,
    convert_to_wav,
    get_voice_tier,
    GOOGLE_CLOUD_VOICES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ─── Singleton pipeline ───────────────────────────────────────────────────────

_pipeline: RAGPipeline = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 – TEXT CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def text_chat(message: str, history: list):
    if not message.strip():
        return history

    result = get_pipeline().answer(message)
    answer     = result["answer"]
    in_scope   = result.get("in_scope", True)
    best_score = result.get("best_score", 0)

    if in_scope and result["sources"]:
        # ✅ Câu hỏi trong phạm vi tài liệu
        scope_badge = (
            f"\n\n---\n"
            f"🟢 **Trong phạm vi tài liệu** *(best score: {best_score:.4f})*\n\n"
            f"**📚 Nguồn tham khảo:**\n"
        ) + "\n".join(
            f"&nbsp;&nbsp;{i}. `{s['source']}` &nbsp;*(score: {s['score']:.4f})*"
            for i, s in enumerate(result["sources"], 1)
        )
    else:
        # 🔴 Câu hỏi nằm ngoài phạm vi tài liệu
        scope_badge = (
            f"\n\n---\n"
            f"🔴 **Ngoài phạm vi tài liệu** *(best score: {best_score:.4f} > threshold)*\n"
            f"*Câu trả lời không được tạo ra từ dữ liệu huấn luyện*"
        )

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": answer + scope_badge})
    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 – VOICE CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def voice_chat(audio_path: str, voice_name: str, speaking_rate: float):
    """
    Luồng: Audio → STT → RAG Pipeline → TTS → Audio output

    Returns:
        (audio_out_path, question_text, answer_text, status_text)
    """
    if audio_path is None:
        return None, "", "", "⚠️ Chưa có âm thanh. Hãy ghi âm hoặc upload file."

    status_parts = []

    # ── Bước 1: Chuyển đổi format nếu cần ──
    converted_path = convert_to_wav(audio_path)

    # ── Bước 2: Speech → Text ──
    try:
        question = speech_to_text(converted_path)
        status_parts.append(f"✅ STT ({get_voice_tier()}): nhận dạng xong")
    except Exception as e:
        return None, "", "", f"❌ Lỗi STT: {e}"

    if not question.strip():
        return None, "", "", "⚠️ Không nhận dạng được giọng nói. Hãy nói to và rõ hơn."

    # ── Bước 3: RAG Pipeline ──
    try:
        result     = get_pipeline().answer(question)
        answer     = result["answer"]
        sources    = result["sources"]
        in_scope   = result.get("in_scope", True)
        best_score = result.get("best_score", 0)

        if in_scope:
            status_parts.append(f"✅ RAG: Trong phạm vi tài liệu (best score: {best_score:.4f})")
        else:
            status_parts.append(f"🔴 RAG: Ngoài phạm vi tài liệu (best score: {best_score:.4f}) → từ chối sinh câu TL")
    except Exception as e:
        return None, question, "", f"❌ Lỗi RAG: {e}"

    # ── Bước 4: Text → Speech ──
    try:
        audio_bytes = text_to_speech(answer, voice_name=voice_name, speaking_rate=speaking_rate)
        out_path    = save_audio(audio_bytes, suffix=".mp3")
        status_parts.append(f"✅ TTS ({get_voice_tier()}): tạo giọng nói xong")
    except Exception as e:
        # Trả về text dù TTS lỗi
        return None, question, answer, f"⚠️ TTS lỗi: {e}\n(Xem câu trả lời dạng text bên dưới)"

    # ── Bước 5: Tổng kết ──
    sources_info = " | ".join(s["source"] for s in sources[:3])
    status_parts.append(f"📚 Nguồn: {sources_info}")

    return out_path, question, answer, "\n".join(status_parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

EXAMPLE_QUESTIONS = [
    "Trường FPT tuyển sinh các cấp học nào?",
    "Học phí của trường FPT là bao nhiêu?",
    "Hồ sơ đăng ký tuyển sinh cần những gì?",
    "Trường FPT có học bổng không?",
    "Điều kiện tuyển sinh vào lớp 10 là gì?",
    "Chương trình lập trình và AI tại FPT như thế nào?",
]

VOICE_OPTIONS = list(GOOGLE_CLOUD_VOICES.keys()) + ["gTTS (miễn phí)"]

with gr.Blocks(
    title="Chatbot Tư Vấn Tuyển Sinh FPT",
) as demo:

    # ── Header ──
    gr.HTML("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <h1 style="margin:0; font-size:28px;">🎓 Chatbot Tư Vấn Tuyển Sinh</h1>
        <h3 style="margin:4px 0; color:#444; font-weight:normal;">
            Trường Tiểu học, THCS và THPT FPT
        </h3>
        <p style="margin:4px 0; color:#888; font-size:13px;">
            RAG · Qwen2.5-1.5B · FAISS · Google Voice API
        </p>
    </div>
    """)

    with gr.Tabs():

        # ════════════════════════════════════════════════
        #  TAB 1: TEXT CHAT
        # ════════════════════════════════════════════════
        with gr.TabItem("💬  Chat văn bản", elem_classes="tab-header"):
            chatbot_ui = gr.Chatbot(
                height=440,
                show_label=False,
                placeholder="Bắt đầu hỏi về tuyển sinh FPT...",
                avatar_images=(None, None),
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Nhập câu hỏi về tuyển sinh...",
                    show_label=False,
                    lines=2,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Gửi ➤", variant="primary", scale=1, min_width=80)

            clear_btn = gr.Button("🗑 Xóa hội thoại", variant="secondary", size="sm")

            gr.Markdown("**💡 Câu hỏi gợi ý:**")
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[:3]:
                    gr.Button(q, size="sm", variant="secondary").click(
                        fn=lambda x=q: x, outputs=msg_input
                    )
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[3:]:
                    gr.Button(q, size="sm", variant="secondary").click(
                        fn=lambda x=q: x, outputs=msg_input
                    )

            send_btn.click(text_chat, [msg_input, chatbot_ui], chatbot_ui).then(
                fn=lambda: "", outputs=msg_input
            )
            msg_input.submit(text_chat, [msg_input, chatbot_ui], chatbot_ui).then(
                fn=lambda: "", outputs=msg_input
            )
            clear_btn.click(fn=lambda: [], outputs=chatbot_ui)

        # ════════════════════════════════════════════════
        #  TAB 2: VOICE CHAT
        # ════════════════════════════════════════════════
        with gr.TabItem("🎙️  Chat giọng nói", elem_classes="tab-header"):
            gr.Markdown("""
            **Quy trình:** 🎙️ Ghi âm → 📝 Nhận dạng tiếng nói (STT) → 🤖 AI xử lý (RAG) → 🔊 Phát giọng nói (TTS)
            """)

            with gr.Row():
                # ── Cột trái: Input ──
                with gr.Column(scale=1):
                    gr.Markdown("### 🎙️ Input")
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Ghi âm câu hỏi (tiếng Việt)",
                        waveform_options={"waveform_color": "#4F86C6"},
                    )

                    with gr.Accordion("⚙️ Cài đặt giọng nói", open=False):
                        voice_selector = gr.Dropdown(
                            choices=[
                                "vi-VN-Neural2-A (Nữ, tự nhiên nhất)",
                                "vi-VN-Neural2-D (Nam)",
                                "vi-VN-Wavenet-A (Nữ, chất lượng cao)",
                                "vi-VN-Wavenet-D (Nam)",
                                "gTTS – miễn phí",
                            ],
                            value="vi-VN-Neural2-A (Nữ, tự nhiên nhất)",
                            label="Giọng TTS",
                        )
                        rate_slider = gr.Slider(
                            minimum=0.5, maximum=1.8, value=1.0, step=0.1,
                            label="Tốc độ đọc (0.5 = chậm, 1.0 = bình thường, 1.8 = nhanh)",
                        )

                    process_btn = gr.Button(
                        "🔄 Xử lý (STT → AI → TTS)", variant="primary", size="lg"
                    )

                # ── Cột phải: Output ──
                with gr.Column(scale=1):
                    gr.Markdown("### 🔊 Output")
                    audio_output = gr.Audio(
                        label="Câu trả lời (giọng nói)",
                        type="filepath",
                        autoplay=True,
                    )
                    status_box = gr.Textbox(
                        label="📊 Trạng thái xử lý",
                        interactive=False,
                        lines=4,
                    )

            with gr.Row():
                question_box = gr.Textbox(
                    label="📝 Câu hỏi nhận dạng được (STT output)",
                    interactive=False,
                    scale=1,
                )
                answer_box = gr.Textbox(
                    label="💬 Câu trả lời văn bản (RAG output)",
                    interactive=False,
                    lines=5,
                    scale=2,
                )

            # Xử lý voice_name từ dropdown
            def parse_voice_name(voice_display: str) -> str:
                if "gTTS" in voice_display:
                    return "vi-VN-Neural2-A"  # fallback, gTTS không dùng voice_name
                return voice_display.split(" ")[0]

            def process_voice(audio_path, voice_display, speaking_rate):
                voice_name = parse_voice_name(voice_display)
                return voice_chat(audio_path, voice_name, speaking_rate)

            process_btn.click(
                fn=process_voice,
                inputs=[audio_input, voice_selector, rate_slider],
                outputs=[audio_output, question_box, answer_box, status_box],
            )

            gr.Markdown("""
            ---
            > **Tier tự động:**
            > - Có `GOOGLE_APPLICATION_CREDENTIALS` → **Google Cloud Neural Voice** (chất lượng tốt nhất)
            > - Không có credentials → **gTTS + Google Web Speech API** (miễn phí, đủ dùng)
            """)

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center; padding:12px; color:#999; font-size:12px; border-top:1px solid #eee; margin-top:8px;">
        Trường FPT School &nbsp;·&nbsp; Bài tập cuối kỳ môn AI Tạo sinh &nbsp;·&nbsp;
        RAG + Qwen2.5 + Google Voice
    </div>
    """)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Tạo link public (dùng cho Colab)")
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()

    print("=" * 55)
    print("  Chatbot Tư Vấn Tuyển Sinh FPT School")
    print("  Voice tier:", get_voice_tier())
    print("=" * 55)
    print("Đang tải pipeline...\n")
    get_pipeline()

    print(f"\nKhởi động Gradio trên port {args.port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css="""
            .tab-header { font-size: 16px; font-weight: bold; }
            footer { display: none !important; }
        """,
    )
