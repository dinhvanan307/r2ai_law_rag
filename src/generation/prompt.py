"""Prompt construction for grounded Vietnamese legal QA.

Two scoring facts drive this prompt:
1. The grader extracts ``Điều X`` from the ``answer`` itself → the answer MUST
   name the articles and their document codes explicitly.
2. QA criteria reward accuracy, completeness, practicality, and clarity → the
   prompt asks for a structured, grounded, plain-Vietnamese answer with a
   short AI-limitation disclaimer.
"""
from __future__ import annotations

from ..schema import RetrievedArticle

SYSTEM_PROMPT = (
    "Bạn là Trợ lý Pháp lý AI cho doanh nghiệp nhỏ và vừa (SME) tại Việt Nam. "
    "Bạn trả lời CHỈ dựa trên các điều luật được cung cấp trong phần NGỮ CẢNH. "
    "Tuyệt đối KHÔNG bịa điều luật, số hiệu văn bản hay nội dung không có trong ngữ cảnh. "
    "Nếu ngữ cảnh không đủ căn cứ, hãy nói rõ là chưa đủ căn cứ."
)

ANSWER_INSTRUCTIONS = (
    "Yêu cầu câu trả lời:\n"
    "1. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu cho người không chuyên.\n"
    "2. BẮT BUỘC trích dẫn căn cứ theo dạng: \"theo Điều X của <Loại + Mã văn bản>\" "
    "(ví dụ: theo Điều 4 của Luật 04/2017/QH14). Chỉ trích các Điều có trong NGỮ CẢNH.\n"
    "3. Nêu đầy đủ các khía cạnh liên quan (điều kiện, nghĩa vụ, ngoại lệ nếu có).\n"
    "4. Nêu lưu ý áp dụng thực tiễn cho SME khi phù hợp.\n"
    "5. Kết thúc bằng một câu lưu ý ngắn: đây là tư vấn sơ bộ từ AI, "
    "cần đối chiếu văn bản gốc hoặc chuyên gia khi áp dụng."
)


def format_context(articles: list[RetrievedArticle]) -> str:
    """Render retrieved articles as a numbered, citation-ready context block."""
    blocks: list[str] = []
    for i, a in enumerate(articles, 1):
        header = f"[{i}] {a.article_no} — {a.doc_name}"
        if a.article_title:
            header += f"\nTiêu đề: {a.article_title}"
        blocks.append(f"{header}\nNội dung: {a.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, articles: list[RetrievedArticle]) -> str:
    context = format_context(articles) if articles else "(Không có điều luật nào được truy hồi.)"
    return (
        f"NGỮ CẢNH (các điều luật liên quan):\n{context}\n\n"
        f"CÂU HỎI: {question}\n\n"
        f"{ANSWER_INSTRUCTIONS}\n\nTRẢ LỜI:"
    )


def build_messages(question: str, articles: list[RetrievedArticle]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, articles)},
    ]


HYDE_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia pháp lý tại Việt Nam. "
    "Nhiệm vụ của bạn là viết MỘT đoạn văn bản đóng vai trò như một Điều luật giả định hoặc một lời giải thích chính thức "
    "để trả lời câu hỏi của người dùng. "
    "Hãy sử dụng văn phong pháp lý, từ vựng chuyên ngành và cấu trúc của một văn bản quy phạm pháp luật (như Luật, Nghị định). "
    "Tuyệt đối KHÔNG sử dụng mào đầu hay giải thích, hãy vào thẳng nội dung pháp lý."
)


def build_hyde_messages(question: str) -> list[dict]:
    user_content = f"Vui lòng viết một đoạn văn bản pháp lý giả định trả lời cho câu hỏi sau:\n{question}"
    return [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
