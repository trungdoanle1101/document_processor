"""Services built from fakes: a word-hash embedder, the in-memory index, an in-memory registry
and a canned chat model, so app tests run without models, Qdrant or Ollama."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.logical import PROFILES
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import ChunkFilter, IndexedChunk
from knowledge_base.assistant import KnowledgeBase
from knowledge_base.ingestion import Indexer
from knowledge_base.registry import SqliteDocumentRegistry
from knowledge_base.templates import VN_REGULATION
from knowledge_base.wiring import ReadyServices
from tests.retrieval.fake_chat_model import FakeChatModel
from tests.retrieval.stub_embedder import StubEmbedder

DATA_REGULATION = """QUY ĐỊNH QUẢN LÝ DỮ LIỆU
Điều 1. Phạm vi điều chỉnh
1. Quy định này áp dụng cho dữ liệu khách hàng tại ngân hàng.
Điều 2. Thời hạn lưu trữ
1. Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày.
2. Dữ liệu lưu trữ dài hạn được lưu trên Tape.
"""

LEAVE_REGULATION = """QUY ĐỊNH NGHỈ PHÉP
Điều 1. Nghỉ phép năm
1. Nhân viên được nghỉ phép 12 ngày mỗi năm.
Điều 2. Nghỉ ốm
1. Nhân viên nghỉ ốm cần có giấy xác nhận của cơ sở y tế.
"""

NOW = datetime(2026, 11, 3, 9, 0, tzinfo=UTC)


@dataclass
class Fakes:
    embedder: StubEmbedder
    index: InMemoryChunkIndex
    registry: SqliteDocumentRegistry
    chat_model: FakeChatModel
    services: ReadyServices

    @property
    def indexer(self) -> Indexer:
        return self.services.indexer

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self.services.knowledge_base


def make_fakes(
    *,
    embedder: StubEmbedder | None = None,
    registry: SqliteDocumentRegistry | None = None,
    min_score: float | None = None,
    section_chars: int = 0,
    history_turns: int = 0,
) -> Fakes:
    embedder = embedder or StubEmbedder(tokenizer=WordTokenizer())
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    registry = registry or SqliteDocumentRegistry(":memory:")
    chat_model = FakeChatModel("Dữ liệu được lưu tối đa 07 ngày [1].")
    indexer = Indexer(
        profile=PROFILES.get("vn_regulation"),
        chunker=OutlineChunker(
            tokenizer=WordTokenizer(), max_tokens=200, include_breadcrumb_in_size=True
        ),
        chunker_name="outline/200",
        embedder=embedder,
        index=index,
        registry=registry,
        collection="test",
        now=lambda: NOW,
    )
    knowledge_base = KnowledgeBase(
        embedder=embedder,
        index=index,
        registry=registry,
        chat_model=chat_model,
        style=VN_REGULATION,
        passages=3,
        min_score=min_score,
        section_chars=section_chars,
        history_turns=history_turns,
    )
    return Fakes(
        embedder=embedder,
        index=index,
        registry=registry,
        chat_model=chat_model,
        services=ReadyServices(
            registry=registry, knowledge_base=knowledge_base, indexer=indexer
        ),
    )


def chunks_of(fakes: Fakes, document_id: str) -> list[IndexedChunk]:
    return fakes.index.list_chunks(ChunkFilter(document_ids=(document_id,)), limit=1000)


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path