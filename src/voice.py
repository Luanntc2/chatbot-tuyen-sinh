"""
Voice Module – Speech-to-Text & Text-to-Speech
==============================================
Hỗ trợ 2 tier tự động:

  Tier 1 (Ưu tiên): Google Cloud Speech-to-Text + Text-to-Speech
    → Cần GOOGLE_APPLICATION_CREDENTIALS hoặc GOOGLE_API_KEY
    → Chất lượng cao, giọng Neural tiếng Việt chuẩn

  Tier 2 (Fallback miễn phí):
    → STT: SpeechRecognition + Google Web Speech API (không cần key)
    → TTS: gTTS (Google Translate TTS, miễn phí)

Cách cấu hình Google Cloud:
  1. Tạo project tại https://console.cloud.google.com
  2. Bật Speech-to-Text API và Text-to-Speech API
  3. Tạo Service Account key (JSON)
  4. Set env: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
     Hoặc trên Colab: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/key.json"
"""

import os
import io
import tempfile
import logging

logger = logging.getLogger(__name__)

# ─── Kiểm tra credentials ────────────────────────────────────────────────────

def has_google_cloud_credentials() -> bool:
    """Kiểm tra xem có Google Cloud credentials không."""
    return bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("GOOGLE_API_KEY")
    )


def get_voice_tier() -> str:
    return "google_cloud" if has_google_cloud_credentials() else "free"


# ═══════════════════════════════════════════════════════════════════════════════
#  SPEECH-TO-TEXT
# ═══════════════════════════════════════════════════════════════════════════════

def stt_google_cloud(audio_path: str, language: str = "vi-VN") -> str:
    """
    Google Cloud Speech-to-Text API.
    Hỗ trợ file WAV, FLAC, MP3.
    Độ chính xác cao với tiếng Việt.
    """
    from google.cloud import speech

    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    # Tự động detect encoding từ đuôi file
    ext = os.path.splitext(audio_path)[1].lower()
    encoding_map = {
        ".wav":  speech.RecognitionConfig.AudioEncoding.LINEAR16,
        ".flac": speech.RecognitionConfig.AudioEncoding.FLAC,
        ".mp3":  speech.RecognitionConfig.AudioEncoding.MP3,
        ".ogg":  speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        ".webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
    }
    encoding = encoding_map.get(ext, speech.RecognitionConfig.AudioEncoding.LINEAR16)

    config = speech.RecognitionConfig(
        encoding=encoding,
        language_code=language,
        enable_automatic_punctuation=True,
        model="latest_long",
        use_enhanced=True,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return ""

    return " ".join(
        result.alternatives[0].transcript
        for result in response.results
        if result.alternatives
    )


def stt_free(audio_path: str, language: str = "vi-VN") -> str:
    """
    Google Web Speech API (miễn phí) qua thư viện SpeechRecognition.
    Không cần API key, nhưng giới hạn request/phút.
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.AudioFile(audio_path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Lỗi kết nối Google Web Speech API: {e}")


def speech_to_text(audio_path: str, language: str = "vi-VN") -> str:
    """
    Tự động chọn tier STT phù hợp.

    Args:
        audio_path: Đường dẫn file âm thanh (WAV, MP3, FLAC, WEBM...)
        language:   Mã ngôn ngữ, mặc định "vi-VN"

    Returns:
        Chuỗi văn bản nhận dạng được
    """
    tier = get_voice_tier()

    if tier == "google_cloud":
        logger.info("[STT] Dùng Google Cloud Speech-to-Text")
        return stt_google_cloud(audio_path, language)
    else:
        logger.info("[STT] Dùng Google Web Speech API (miễn phí)")
        return stt_free(audio_path, language)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT-TO-SPEECH
# ═══════════════════════════════════════════════════════════════════════════════

# Danh sách giọng Neural tiếng Việt của Google Cloud
GOOGLE_CLOUD_VOICES = {
    "vi-VN-female-1": "vi-VN-Neural2-A",   # Nữ, tự nhiên nhất
    "vi-VN-female-2": "vi-VN-Wavenet-A",   # Nữ, chất lượng cao
    "vi-VN-male-1":   "vi-VN-Neural2-D",   # Nam
    "vi-VN-male-2":   "vi-VN-Wavenet-D",   # Nam
}


def tts_google_cloud(
    text: str,
    language: str = "vi-VN",
    voice_name: str = "vi-VN-Neural2-A",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> bytes:
    """
    Google Cloud Text-to-Speech – Neural voice.
    Giọng cực kỳ tự nhiên, phát âm chuẩn tiếng Việt.
    """
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language,
        name=voice_name,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch,
        effects_profile_id=["telephony-class-application"],
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content


def tts_free(text: str, language: str = "vi", slow: bool = False) -> bytes:
    """
    gTTS – Google Translate Text-to-Speech (miễn phí).
    Chất lượng tốt, không cần API key.
    """
    from gtts import gTTS

    # Giới hạn độ dài text để tránh timeout
    MAX_LEN = 5000
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN] + "..."

    tts = gTTS(text=text, lang=language, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def text_to_speech(
    text: str,
    language: str = "vi-VN",
    voice_name: str = "vi-VN-Neural2-A",
    speaking_rate: float = 1.0,
) -> bytes:
    """
    Tự động chọn tier TTS phù hợp.

    Args:
        text:          Văn bản cần chuyển thành giọng nói
        language:      Mã ngôn ngữ, mặc định "vi-VN"
        voice_name:    Tên giọng (chỉ dùng với Google Cloud)
        speaking_rate: Tốc độ đọc (0.25 – 4.0, mặc định 1.0)

    Returns:
        bytes âm thanh MP3
    """
    tier = get_voice_tier()

    if tier == "google_cloud":
        logger.info("[TTS] Dùng Google Cloud Text-to-Speech Neural")
        return tts_google_cloud(text, language, voice_name, speaking_rate)
    else:
        logger.info("[TTS] Dùng gTTS (miễn phí)")
        lang_short = language.split("-")[0]  # "vi-VN" → "vi"
        return tts_free(text, lang_short)


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_audio(audio_bytes: bytes, suffix: str = ".mp3") -> str:
    """Lưu bytes âm thanh vào file tạm, trả về đường dẫn file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def convert_to_wav(input_path: str) -> str:
    """
    Chuyển đổi file âm thanh bất kỳ sang WAV (16kHz, mono).
    Dùng pydub. Hữu ích khi Gradio trả về định dạng WEBM/OGG.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception:
        return input_path  # Trả về file gốc nếu convert thất bại
