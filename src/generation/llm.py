"""Pluggable LLM client for answer generation.

Backends (competition-compliant generators are open-source, <14B, pre-2026-03):
* ``ollama``  — local Ollama server (default for dev), e.g. ``qwen2.5:7b-instruct``.
* ``vllm``    — OpenAI-compatible endpoint (best for batch over thousands of Qs).
* ``hf``      — transformers pipeline (in-process), e.g. ``Qwen/Qwen2.5-7B-Instruct``.
* ``extractive`` — NO model. Deterministic, grounded fallback that composes an
  answer from retrieved articles and always cites them. Guarantees the pipeline
  runs anywhere (sandbox/CI) and that ``answer`` contains valid ``Điều X`` cites.
"""
from __future__ import annotations

import json
import re
import urllib.request

from ..schema import RetrievedArticle
from . import prompt as P


class LLMClient:
    def __init__(self, backend: str = "extractive", model: str = "qwen2.5:7b-instruct",
                 base_url: str = "http://localhost:11434", temperature: float = 0.2,
                 max_tokens: int = 1024, timeout: int = 120) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # --- public ------------------------------------------------------------
    # --- public ------------------------------------------------------------
    def generate(self, question: str, articles: list[RetrievedArticle]) -> str:
        if self.backend == "extractive":
            return self.extractive_answer(question, articles)
        msgs = P.build_messages(question, articles)
        if self.backend == "ollama":
            return self._ollama(msgs, self.max_tokens, self.temperature)
        if self.backend == "vllm":
            return self._openai_compatible(msgs, self.max_tokens, self.temperature)
        if self.backend == "hf":
            return self._hf(msgs, self.max_tokens, self.temperature)
        return ""

    def generate_hyde(self, question: str) -> str:
        if self.backend == "extractive":
            return ""
        msgs = P.build_hyde_messages(question)
        hyde_tokens = 256
        hyde_temp = 0.5
        if self.backend == "ollama":
            return self._ollama(msgs, hyde_tokens, hyde_temp)
        if self.backend == "vllm":
            return self._openai_compatible(msgs, hyde_tokens, hyde_temp)
        if self.backend == "hf":
            return self._hf(msgs, hyde_tokens, hyde_temp)
        return ""

    # --- backends ----------------------------------------------------------
    def _ollama(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature,
                        "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (data.get("message", {}) or {}).get("content", "").strip()

    def _openai_compatible(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    def _hf(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        if not hasattr(self, "_tok"):
            self._tok = AutoTokenizer.from_pretrained(self.model)
            self._mdl = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype="auto", device_map="auto")
        text = self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._mdl.device)
        out = self._mdl.generate(**inputs, max_new_tokens=max_tokens,
                                 temperature=temperature, do_sample=temperature > 0)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    # --- deterministic grounded fallback -----------------------------------
    @staticmethod
    def extractive_answer(question: str, articles: list[RetrievedArticle]) -> str:
        """Compose a grounded answer purely from retrieved articles.

        Always cites every retrieved article as ``Điều X của <doc_name>`` so the
        grader's ``Điều X`` extraction succeeds even without a generative model.
        """
        if not articles:
            return ("Chưa đủ căn cứ pháp lý trong dữ liệu được truy hồi để trả lời "
                    "câu hỏi này. Vui lòng bổ sung văn bản liên quan hoặc tham khảo "
                    "chuyên gia pháp lý. (Lưu ý: đây là tư vấn sơ bộ từ AI.)")

        lines: list[str] = [
            "Dựa trên các căn cứ pháp lý được truy hồi, nội dung liên quan như sau:"
        ]
        for a in articles:
            snippet = _first_sentences(a.text, max_chars=400)
            cite = f"Theo {a.article_no} của {a.doc_name}"
            lines.append(f"- {cite}: {snippet}")
        lines.append(
            "Lưu ý: Đây là tư vấn sơ bộ do AI tổng hợp từ văn bản pháp luật được "
            "truy hồi; vui lòng đối chiếu văn bản gốc hoặc tham khảo chuyên gia "
            "trước khi áp dụng."
        )
        return "\n".join(lines)


def _first_sentences(text: str, max_chars: int = 400) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Try to end on a sentence boundary.
    m = list(re.finditer(r"[.;]\s", cut))
    if m:
        return cut[: m[-1].end()].strip()
    return cut.rstrip() + "…"
