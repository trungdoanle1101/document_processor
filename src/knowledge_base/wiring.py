"""Build the real services from settings: models, Qdrant, Ollama, the registry.

The only module that loads models or opens connections, so everything else can be
tested with fakes. Each part is built the first time it is used, so a command loads
only what it needs: list reads the registry alone, and indexing never loads the
reranker or the chat model.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from document_processor.retrieval.chat_model import ChatModel
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkIndex
from document_processor.retrieval.reranker import Reranker
from knowledge_base.assistant import KnowledgeBase
from knowledge_base.ingestion import Indexer
from knowledge_base.registry import DocumentRegistry
from knowledge_base.settings import AppSettings
from knowledge_base.templates import ANSWER_STYLES


class Services(Protocol):
    @property
    def registry(self) -> DocumentRegistry: ...
    @property
    def indexer(self) -> Indexer: ...
    @property
    def knowledge_base(self) -> KnowledgeBase: ...


@dataclass(frozen=True, kw_only=True)
class ReadyServices:
    """Services whose parts are already built, e.g. from fakes."""

    registry: DocumentRegistry
    indexer: Indexer
    knowledge_base: KnowledgeBase


class LazyServices:
    """Services built from settings, each part on first use and then kept."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @cached_property
    def registry(self) -> DocumentRegistry:
        from knowledge_base.registry import SqliteDocumentRegistry

        return SqliteDocumentRegistry(self._settings.registry)

    @cached_property
    def embedder(self) -> Embedder:
        from document_processor.integrations.huggingface import HUGGINGFACE_MODEL_SPECS
        from document_processor.integrations.huggingface.download import download_model
        from document_processor.integrations.huggingface.embedder import (
            HuggingFaceEmbedder,
        )

        settings = self._settings.embedder
        spec = HUGGINGFACE_MODEL_SPECS.get(settings.spec)
        if not settings.folder.exists():
            download_model(spec.reference, settings.folder)
        return HuggingFaceEmbedder(
            spec=spec, folder=settings.folder, device=settings.device
        )

    @cached_property
    def reranker(self) -> Reranker | None:
        settings = self._settings.reranker
        if settings is None:
            return None
        from document_processor.integrations.huggingface import (
            HUGGINGFACE_RERANKER_SPECS,
        )
        from document_processor.integrations.huggingface.download import download_model
        from document_processor.integrations.huggingface.reranker import (
            HuggingFaceReranker,
        )

        spec = HUGGINGFACE_RERANKER_SPECS.get(settings.spec)
        if not settings.folder.exists():
            download_model(spec.reference, settings.folder)
        return HuggingFaceReranker(
            spec=spec, folder=settings.folder, device=settings.device
        )

    @cached_property
    def index(self) -> ChunkIndex:
        # opening the collection checks it against the embedder, so this loads the embedder
        from qdrant_client import QdrantClient

        from document_processor.integrations.qdrant.index import QdrantChunkIndex

        return QdrantChunkIndex(
            QdrantClient(url=self._settings.qdrant_url),
            collection=self._settings.collection,
            embedder_identity=self.embedder.identity,
            vector_specs=self.embedder.vector_specs,
            indexed_attributes={"part": "text", "access_group": "text"},
        )

    @cached_property
    def chat_model(self) -> ChatModel:
        from document_processor.integrations.ollama import OllamaChatModel

        chat = self._settings.chat
        return OllamaChatModel(
            chat.model,
            base_url=chat.url,
            temperature=chat.temperature,
            context_size=chat.context_size,
            think=chat.think,
            timeout=chat.timeout,
        )

    @cached_property
    def knowledge_base(self) -> KnowledgeBase:
        settings = self._settings
        return KnowledgeBase(
            embedder=self.embedder,
            index=self.index,
            registry=self.registry,
            chat_model=self.chat_model,
            style=ANSWER_STYLES.get(settings.answer.template),
            reranker=self.reranker,
            rerank_depth=settings.reranker.depth if settings.reranker else 20,
            passages=settings.answer.passages,
            min_score=settings.answer.min_score,
            min_passage_score=settings.answer.min_passage_score,
            section_chars=settings.answer.section_chars,
            history_turns=settings.answer.history_turns,
            max_tokens=settings.chat.max_tokens,
        )

    @cached_property
    def indexer(self) -> Indexer:
        from document_processor.chunking import OutlineChunker
        from document_processor.logical import PROFILES

        settings = self._settings
        tokenizer = self.embedder.tokenizer
        if tokenizer is None:
            raise ValueError("the embedder has no tokenizer to size chunks with")
        return Indexer(
            profile=PROFILES.get(settings.profile),
            chunker=OutlineChunker(
                tokenizer=tokenizer,
                max_tokens=settings.chunk_tokens,
                include_breadcrumb_in_size=True,
            ),
            chunker_name=f"outline/{settings.chunk_tokens}",
            embedder=self.embedder,
            index=self.index,
            registry=self.registry,
            collection=settings.collection,
        )


def build_services(settings: AppSettings) -> Services:
    return LazyServices(settings)