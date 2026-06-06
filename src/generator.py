import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LLM_MODEL, FALLBACK_MODEL, MAX_NEW_TOKENS, TEMPERATURE

SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn tuyển sinh thân thiện và chuyên nghiệp của "
    "Trường Tiểu học, THCS và THPT FPT. "
    "Hãy trả lời bằng tiếng Việt, ngắn gọn, chính xác và hữu ích. "
    "Chỉ trả lời dựa trên thông tin được cung cấp trong ngữ cảnh."
)


class Generator:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        print(f"  Tải model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = self._load_with_best_config()
            self.model.eval()
            print(f"  Model '{self.model_name}' sẵn sàng.\n")
        except Exception as e:
            print(f"  [WARN] Không tải được '{self.model_name}': {e}")
            print(f"  Thử fallback: {FALLBACK_MODEL}")
            self.model_name = FALLBACK_MODEL
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self.model = self._load_with_best_config()
            self.model.eval()
            print(f"  Model fallback '{FALLBACK_MODEL}' sẵn sàng.\n")

    def _load_with_best_config(self):
        load_kwargs = {"device_map": "auto", "trust_remote_code": True}

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
            if vram_gb < 8:
                print("  VRAM < 8 GB → dùng load_in_8bit")
                load_kwargs["load_in_8bit"] = True
            else:
                print("  Dùng float16")
                load_kwargs["torch_dtype"] = torch.float16
        else:
            print("  Không có GPU → dùng CPU (float32, chậm hơn)")
            load_kwargs["torch_dtype"] = torch.float32

        return AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

    def generate(self, prompt: str) -> str:
        """
        Nhận prompt (chuỗi thô hoặc đã có context) và trả về câu trả lời.
        Áp dụng chat template của Qwen/TinyLlama tự động.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Áp dụng chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt")

        # Truncate nếu vượt quá max context của model (TinyLlama: 2048 tokens)
        max_ctx = getattr(self.model.config, "max_position_embeddings", 4096)
        max_input = max_ctx - MAX_NEW_TOKENS
        n_tokens = inputs["input_ids"].shape[1]
        if n_tokens > max_input:
            # Giữ phần cuối (chứa câu hỏi), bỏ phần đầu context nếu quá dài
            inputs = {k: v[:, -max_input:] for k, v in inputs.items()}

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Chỉ decode phần mới sinh (bỏ phần prompt đầu vào)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()
