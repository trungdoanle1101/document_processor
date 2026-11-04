"""An HTTP API: OpenAI-compatible chat for Open WebUI and similar front ends, plus search and documents.

To a chat front end the knowledge base is one model, MODEL_ID: each question is the
last user message, and the reply is the answer followed by its sources. The reader
is named by a header (READER_HEADERS); settings map readers to access groups.

Adding and removing documents reuses the models the server already loaded. It needs
the admin token from the settings as a bearer token, and is off when none is set.
"""

import hmac
import json
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from document_processor.retrieval.answer import Answer
from document_processor.retrieval.chat_model import (
    ChatModelError,
    ChatModelTimeout,
    Message,
)
from knowledge_base.assistant import format_reply
from knowledge_base.ingestion import parse_attribute
from knowledge_base.settings import AppSettings
from knowledge_base.wiring import Services

MODEL_ID = "knowledge-base"

# who is asking, by the first header present: our own, then Open WebUI's forwarded user
READER_HEADERS = ("x-reader", "x-openwebui-user-email")


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False


class SearchRequest(BaseModel):
    question: str
    limit: int = 5


class AnswerRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []  # earlier turns, oldest first, as in a chat request


class AddDocumentRequest(BaseModel):
    path: str  # a file on the machine running the server
    group: str
    document_id: str | None = None
    title: str | None = None
    attributes: dict[str, str | bool | int | float | list[str] | list[int]] = {}
    force: bool = False


def create_api(services: Services, settings: AppSettings) -> FastAPI:
    api = FastAPI(title="Knowledge base", version="0.1.0")
    knowledge_base, indexer = services.knowledge_base, services.indexer
    writing = threading.Lock()  # one add or remove at a time, so versions never race

    @api.exception_handler(ChatModelError)
    async def chat_model_failed(request: Request, error: Exception) -> JSONResponse:
        """The model is down, missing or too slow: say so, instead of a bare 500."""
        timed_out = isinstance(error, ChatModelTimeout)
        message = f"the chat model could not answer: {error}"
        return JSONResponse(
            status_code=504 if timed_out else 503,
            # detail for plain clients, error for OpenAI-style ones such as Open WebUI
            content={
                "detail": message,
                "error": {
                    "message": message,
                    "type": "chat_model_timeout"
                    if timed_out
                    else "chat_model_unavailable",
                },
            },
        )

    def groups_of(request: Request) -> tuple[str, ...]:
        reader = next(
            (
                request.headers[name]
                for name in READER_HEADERS
                if name in request.headers
            ),
            None,
        )
        return settings.groups_of(reader)

    def check_admin(authorization: str | None) -> None:
        if settings.admin_token is None:
            raise HTTPException(
                status_code=403,
                detail="indexing over the API is off: set admin_token in the settings",
            )
        expected = f"Bearer {settings.admin_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401, detail="a valid admin token is required"
            )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "knowledge-base",
                }
            ],
        }

    @api.post("/v1/chat/completions", response_model=None)
    def chat(body: ChatRequest, request: Request) -> dict[str, Any] | StreamingResponse:
        question, history = _conversation(body.messages)
        answer = knowledge_base.ask(
            question, groups=groups_of(request), history=history
        )
        reply = format_reply(answer, knowledge_base)
        finish: Literal["stop", "length"] = (
            "length" if answer.stop == "length" else "stop"
        )
        if body.stream:
            return StreamingResponse(
                _stream(reply, finish), media_type="text/event-stream"
            )
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @api.post("/v1/answer")
    def answer(body: AnswerRequest, request: Request) -> dict[str, Any]:
        """The answer with its structure kept: passages, citations, whether anything was found,
        and the standalone question that was searched for a follow-up."""
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="question has no text")
        answer = knowledge_base.ask(
            body.question, groups=groups_of(request), history=_turns(body.history)
        )
        return _answer_json(answer, asked=body.question)

    @api.post("/v1/search")
    def search(body: SearchRequest, request: Request) -> dict[str, Any]:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="question has no text")
        if not 1 <= body.limit <= 50:
            raise HTTPException(
                status_code=400, detail="limit must be between 1 and 50"
            )
        hits = knowledge_base.search(
            body.question, groups=groups_of(request), limit=body.limit
        )
        return {"hits": [_hit_json(hit) for hit in hits]}

    @api.get("/v1/documents")
    def documents(request: Request) -> dict[str, Any]:
        records = knowledge_base.documents(groups=groups_of(request))
        return {
            "documents": [
                {
                    "document_id": r.document_id,
                    "title": r.title,
                    "access_group": r.access_group,
                    "version": r.version,
                    "chunk_count": r.chunk_count,
                    "indexed_at": r.indexed_at.isoformat(),
                }
                for r in records
            ]
        }

    @api.post("/v1/documents")
    def add_document(
        body: AddDocumentRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """Index a file with the server's models, or refresh it if it changed."""
        check_admin(authorization)
        try:
            attributes = {
                name: parse_attribute(value) for name, value in body.attributes.items()
            }
            with writing:
                result = indexer.add_document(
                    body.path,
                    access_group=body.group,
                    document_id=body.document_id,
                    title=body.title,
                    attributes=attributes,
                    force=body.force,
                )
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=404, detail=f"no file at {body.path!r} on the server"
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        record = result.record
        return {
            "document_id": record.document_id,
            "action": result.action,
            "version": record.version,
            "chunk_count": record.chunk_count,
        }

    @api.delete("/v1/documents/{document_id:path}")
    def remove_document(
        document_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        check_admin(authorization)
        try:
            with writing:
                indexer.remove_document(document_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail=f"no document {document_id!r}"
            ) from error
        return {"document_id": document_id, "removed": True}

    return api


def _conversation(messages: Sequence[ChatMessage]) -> tuple[str, list[Message]]:
    """The question (the last user message) and the turns before it."""
    for position in range(len(messages) - 1, -1, -1):
        if messages[position].role == "user":
            text = _text_of(messages[position].content).strip()
            if text:
                return text, _turns(messages[:position])
            break
    raise HTTPException(status_code=400, detail="the last user message has no text")


def _turns(messages: Sequence[ChatMessage]) -> list[Message]:
    """The questions and replies, as text; system and tool messages left out."""
    turns: list[Message] = []
    for message in messages:
        text = _text_of(message.content).strip()
        if text and message.role == "user":
            turns.append(Message(role="user", content=text))
        elif text and message.role == "assistant":
            turns.append(Message(role="assistant", content=text))
    return turns


def _text_of(content: str | list[dict[str, Any]]) -> str:
    """Plain text, or the text parts of a multi-part message."""
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", "")) for part in content if part.get("type") == "text"
    )


def _stream(reply: str, finish: Literal["stop", "length"]) -> Iterator[str]:
    """The whole reply as one chunk, then the finish: enough for clients that ask to stream."""
    completion_id, created = _completion_id(), int(time.time())

    def event(delta: dict[str, str], finish_reason: str | None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    yield event({"role": "assistant", "content": reply}, None)
    yield event({}, finish)
    yield "data: [DONE]\n\n"


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _answer_json(answer: Answer, *, asked: str) -> dict[str, Any]:
    return {
        "question": asked,
        "searched_for": answer.question,  # differs from question for a rewritten follow-up
        "text": answer.text,
        "found": answer.found,
        "stop": answer.stop,
        "chat_model": answer.chat_model,
        "passages": [_hit_json(hit) for hit in answer.passages],
        "citations": [
            {
                "number": c.number,
                "document_id": c.hit.chunk.document_id,
                "chunk_id": c.hit.chunk.chunk_id,
            }
            for c in answer.citations
        ],
    }


def _hit_json(hit: Any) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "headings": list(chunk.headings),
        "text": chunk.text,
        "index_score": hit.index_score,
        "rerank_score": hit.rerank_score,
        # set when the chunk was added to complete this section, not found by the search
        "completed_section": hit.completed_section,
    }