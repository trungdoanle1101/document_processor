"""How answers are worded, per kind of document collection: configuration, not framework code."""

import re
from dataclasses import dataclass

from document_processor.core.registry import Registry
from document_processor.retrieval.answer import AnswerTemplate
from document_processor.retrieval.conversation import CondenseTemplate


@dataclass(frozen=True, kw_only=True)
class AnswerStyle:
    """How answers read. A cited source is shown short: the title, the anchor heading
    if any, then the last source_depth headings, each cut to source_heading_chars."""

    template: AnswerTemplate
    condense: CondenseTemplate  # rewrites a follow-up as a standalone question
    sources_heading: str  # above the list of cited sources in a reply
    source_depth: int = 2
    # regex; the last heading it matches is always shown
    source_anchor: str | None = None
    source_heading_chars: int = 60

    def __post_init__(self) -> None:
        if self.source_depth < 0:
            raise ValueError(
                f"source_depth must not be negative, got {self.source_depth}"
            )
        if self.source_heading_chars < 10:
            raise ValueError(
                f"source_heading_chars must be at least 10, got {self.source_heading_chars}"
            )
        if self.source_anchor is not None:
            re.compile(self.source_anchor)


VN_REGULATION = AnswerStyle(
    template=AnswerTemplate(
        instructions=(
            "Bạn là trợ lý tra cứu văn bản quy định nội bộ.\n"
            "- Chỉ trả lời dựa trên các đoạn trích được cung cấp; không dùng kiến thức bên ngoài.\n"
            "- Sau mỗi ý, ghi nguồn bằng số thứ tự đoạn trích trong ngoặc vuông, ví dụ [1] hoặc [2][3].\n"
            "- Nêu rõ điều, khoản liên quan khi có.\n"
            "- Nếu các đoạn trích không chứa câu trả lời, nói rõ là không tìm thấy thông tin trong văn bản.\n"
            "- Không tự viết danh sách nguồn ở cuối câu trả lời; hệ thống sẽ tự thêm.\n"
            "- Trả lời bằng tiếng Việt, ngắn gọn, chính xác."
        ),
        not_found="Không tìm thấy thông tin liên quan trong các văn bản hiện có.",
        passage="[{number}] {source}\n{text}",
        question="Các đoạn trích:\n\n{passages}\n\nCâu hỏi: {question}",
    ),
    condense=CondenseTemplate(
        instructions=(
            "Bạn viết lại câu hỏi cho hệ thống tra cứu văn bản quy định.\n"
            "- Dựa vào đoạn hội thoại, viết lại tin nhắn cuối của người hỏi thành một câu hỏi "
            "đầy đủ, hiểu được mà không cần đọc hội thoại.\n"
            "- Thay các từ như 'nó', 'cái đó', 'còn … thì sao' bằng đối tượng cụ thể đã được nhắc đến.\n"
            "- Nếu tin nhắn đã đầy đủ, giữ nguyên.\n"
            "- Không trả lời câu hỏi. Chỉ viết đúng một câu hỏi, bằng tiếng Việt."
        ),
        request="Hội thoại:\n{history}\n\nTin nhắn cuối: {message}\n\nCâu hỏi đầy đủ:",
        user="Người hỏi",
        assistant="Trợ lý",
    ),
    sources_heading="Nguồn",
    source_anchor=r"^Điều\s+\d+",  # the article a passage belongs to
)

GENERAL_EN = AnswerStyle(
    template=AnswerTemplate(
        instructions=(
            "You answer questions about the organisation's documents.\n"
            "- Use only the passages provided; never outside knowledge.\n"
            "- After each statement, cite the passages it rests on by number, like [1] or [2][3].\n"
            "- If the passages don't contain the answer, say so plainly.\n"
            "- Don't add a list of sources at the end; it is added for you.\n"
            "- Be brief and precise."
        ),
        not_found="Nothing relevant was found in the available documents.",
        passage="[{number}] {source}\n{text}",
        question="Passages:\n\n{passages}\n\nQuestion: {question}",
    ),
    condense=CondenseTemplate(
        instructions=(
            "You rewrite questions for a document search.\n"
            "- Using the conversation, rewrite the user's last message as a complete question "
            "that makes sense without the conversation.\n"
            "- Replace words like 'it', 'that one' or 'what about …' with what they refer to.\n"
            "- If the message is already complete, keep it as it is.\n"
            "- Don't answer it. Write exactly one question."
        ),
        request="Conversation:\n{history}\n\nLast message: {message}\n\nComplete question:",
    ),
    sources_heading="Sources",
)

ANSWER_STYLES = Registry[AnswerStyle](
    "answer style", {"vn_regulation": VN_REGULATION, "general_en": GENERAL_EN}
)