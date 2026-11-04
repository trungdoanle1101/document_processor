"""The knowledge base as its readers see it: search and ask, limited to what they may read."""

import re
from collections.abc import Sequence
from dataclasses import replace
from functools import partial

from document_processor.retrieval.answer import Answer, answer_question, find_citations
from document_processor.retrieval.chat_model import ChatModel, Message
from document_processor.retrieval.conversation import condense_question, recent_turns
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkIndex, IndexedChunk
from document_processor.retrieval.reranker import Reranker
from document_processor.retrieval.search import RetrievedChunk, search_chunks
from document_processor.retrieval.sections import complete_sections
from knowledge_base.access import readable_by
from knowledge_base.registry import DocumentRecord, DocumentRegistry
from knowledge_base.templates import AnswerStyle


class KnowledgeBase:
    def __init__(
        self,
        *,
        embedder: Embedder,
        index: ChunkIndex,
        registry: DocumentRegistry,
        chat_model: ChatModel,
        style: AnswerStyle,
        reranker: Reranker | None = None,
        rerank_depth: int = 20,
        passages: int = 5,
        min_score: float | None = None,
        min_passage_score: float | None = None,
        max_tokens: int = 1024,
        section_chars: int = 0,  # complete shared sections up to this size; 0: off
        history_turns: int = 0,  # earlier questions used to understand a follow-up; 0: none
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._registry = registry
        self._chat_model = chat_model
        self._style = style
        # a line only listing sources, as a model may add: "Nguồn: [2][3]", "**Nguồn:** [1], [4]"
        self._source_line = re.compile(
            rf"\s*[*_]*\s*{re.escape(style.sources_heading)}\s*[*_]*\s*:\s*[*_]*\s*"
            r"(?:\[\d+\][\s,;.]*)+",
            re.IGNORECASE,
        )
        self._reranker = reranker
        self._rerank_depth = rerank_depth
        self._passages = passages
        self._min_score = min_score
        self._min_passage_score = min_passage_score
        self._max_tokens = max_tokens
        self._section_chars = section_chars
        self._history_turns = history_turns

    @property
    def style(self) -> AnswerStyle:
        return self._style

    def search(
        self, question: str, *, groups: Sequence[str], limit: int = 5
    ) -> list[RetrievedChunk]:
        """The passages that best answer the question, among those these groups may read."""
        return search_chunks(
            question,
            embedder=self._embedder,
            index=self._index,
            chunk_filter=readable_by(groups),
            reranker=self._reranker,
            rerank_depth=self._rerank_depth,
            limit=limit,
        )

    def ask(
        self, question: str, *, groups: Sequence[str], history: Sequence[Message] = ()
    ) -> Answer:
        """An answer written from passages these groups may read, citing them.

        history: the conversation before the question. A follow-up is first rewritten
        as a standalone question, which is what gets searched and answered; the
        answer's question is that rewrite.
        """
        turns = [
            self._without_sources(turn)
            for turn in recent_turns(history, self._history_turns)
        ]
        question = condense_question(
            question, turns, chat_model=self._chat_model, template=self._style.condense
        )
        titles = {
            record.document_id: record.title for record in self.documents(groups=groups)
        }

        def source_of(chunk: IndexedChunk) -> str:
            return " › ".join(
                (titles.get(chunk.document_id, chunk.document_id), *chunk.headings)
            )

        answer = answer_question(
            question,
            # added chunks obey the reader's access too
            search=complete_sections(
                partial(self.search, groups=groups),
                index=self._index,
                chunk_filter=readable_by(groups),
                max_chars=self._section_chars,
            ),
            chat_model=self._chat_model,
            template=self._style.template,
            passages=self._passages,
            min_score=self._min_score,
            min_passage_score=self._min_passage_score,
            max_tokens=self._max_tokens,
            source_of=source_of,
            history=turns,
        )
        if not answer.found:
            return answer
        # the reply lists the sources itself; citations come from the answer's own [n]
        text = self._without_source_lines(answer.text)
        return replace(
            answer, text=text, citations=find_citations(text, answer.passages)
        )

    def _without_sources(self, turn: Message) -> Message:
        """A reply without its sources: the list format_reply added and any line the
        model wrote itself. Their numbers were for that turn's passages, and would
        mislead the model about this one's."""
        if turn.role != "assistant":
            return turn
        text, _, _ = turn.content.partition(f"\n\n{self._style.sources_heading}:\n")
        return Message(role="assistant", content=self._without_source_lines(text))

    def _without_source_lines(self, text: str) -> str:
        """The text without lines that only list sources; other lines are kept as they are."""
        kept = [
            line for line in text.splitlines() if not self._source_line.fullmatch(line)
        ]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

    def documents(self, *, groups: Sequence[str]) -> list[DocumentRecord]:
        return self._registry.list_documents(access_groups=groups)

    def title_of(self, document_id: str) -> str:
        record = self._registry.get(document_id)
        return record.title if record is not None else document_id


def format_reply(answer: Answer, knowledge_base: KnowledgeBase) -> str:
    """The answer, then the sources it cites, shown short: what a chat reader sees."""
    if not answer.citations:
        return answer.text
    style = knowledge_base.style
    lines = [answer.text, "", f"{style.sources_heading}:"]
    for citation in answer.citations:
        chunk = citation.hit.chunk
        title = knowledge_base.title_of(chunk.document_id)
        lines.append(
            f"[{citation.number}] {short_source(title, chunk.headings, style)}"
        )
    return "\n".join(lines)


def short_source(title: str, headings: Sequence[str], style: AnswerStyle) -> str:
    """title › anchor › … › last headings: enough to find the passage, short enough to read.

    The model still sees the full heading path; this is only how a reply shows it.
    """
    names = [_short_heading(h, style.source_heading_chars) for h in headings]
    names = [name for name in names if name]  # marker-only headings, such as "-"
    shown = set(range(max(0, len(names) - style.source_depth), len(names)))
    if style.source_anchor is not None:
        anchors = [
            i for i, name in enumerate(names) if re.search(style.source_anchor, name)
        ]
        if anchors:
            shown.add(anchors[-1])

    parts: list[str] = [title]
    previous: int | None = None
    for i in sorted(shown):
        if previous is not None and i > previous + 1:
            parts.append("…")  # headings left out between two shown ones
        parts.append(names[i])
        previous = i
    return " › ".join(parts)


def _short_heading(heading: str, limit: int) -> str:
    """One heading cut to about limit characters; empty when it holds no words."""
    text = " ".join(heading.split()).rstrip(":").strip()
    if not any(character.isalnum() for character in text):
        return ""
    if len(text) > limit:
        # "Name: a sentence that follows" keeps the name
        name, separator, _ = text.partition(": ")
        text = name if separator else text
    if len(text) > limit:
        text = text[: limit + 1].rsplit(" ", 1)[0] + "…"  # a whole last word
    return text