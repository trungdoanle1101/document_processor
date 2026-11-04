"""LazyServices: each part is built on first use, so a command loads only what it needs."""

from functools import cached_property
from pathlib import Path

import pytest

from document_processor.chunking import WordTokenizer
from document_processor.retrieval.chat_model import ChatModel
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import ChunkIndex
from document_processor.retrieval.reranker import Reranker
from knowledge_base.cli import main
from knowledge_base.settings import AppSettings, parse_settings
from knowledge_base.wiring import LazyServices, Services
from tests.knowledge_base.helpers import DATA_REGULATION, write
from tests.retrieval.fake_chat_model import FakeChatModel
from tests.retrieval.stub_embedder import StubEmbedder

HEAVY = {"embedder", "reranker", "index", "chat_model"}


class StandInServices(LazyServices):
    """The real wiring, with stand-ins for the parts that load models or connect."""

    @cached_property
    def embedder(self) -> Embedder:
        return StubEmbedder(tokenizer=WordTokenizer())

    @cached_property
    def reranker(self) -> Reranker | None:
        return None

    @cached_property
    def index(self) -> ChunkIndex:
        return InMemoryChunkIndex(
            embedder_identity=self.embedder.identity,
            vector_specs=self.embedder.vector_specs,
        )

    @cached_property
    def chat_model(self) -> ChatModel:
        return FakeChatModel()


def built(services: LazyServices) -> set[str]:
    # cached_property keeps each built part in the instance's __dict__
    return set(vars(services)) - {"_settings"}


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    return parse_settings({"registry": str(tmp_path / "kb.sqlite3")})


def test_nothing_is_built_up_front(settings: AppSettings) -> None:
    assert built(StandInServices(settings)) == set()


def test_the_registry_needs_nothing_else(settings: AppSettings) -> None:
    services = StandInServices(settings)
    services.registry.list_documents()
    assert built(services) == {"registry"}


def test_indexing_never_builds_the_reranker_or_the_chat_model(
    settings: AppSettings,
) -> None:
    services = StandInServices(settings)
    assert services.indexer is not None
    assert built(services) == {"indexer", "embedder", "index", "registry"}


def test_answering_builds_everything_and_shares_the_parts(
    settings: AppSettings,
) -> None:
    services = StandInServices(settings)
    assert services.indexer is not None and services.knowledge_base is not None
    assert built(services) == {"indexer", "knowledge_base", "registry"} | HEAVY
    assert services.indexer._embedder is services.knowledge_base._embedder
    assert services.indexer._registry is services.knowledge_base._registry


def test_a_disabled_reranker_is_none_without_loading_anything() -> None:
    services = LazyServices(parse_settings({"reranker": {"enabled": False}}))
    assert services.reranker is None


def test_list_reads_only_the_registry(
    settings: AppSettings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    services = StandInServices(settings)
    services.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="public"
    )

    fresh = StandInServices(settings)  # a new command, the same registry file

    def build(_: AppSettings) -> Services:
        return fresh

    config = tmp_path / "kb.toml"
    config.write_text(
        f'registry = "{(tmp_path / "kb.sqlite3").as_posix()}"\n', encoding="utf-8"
    )
    assert main(["--config", str(config), "list"], build=build) == 0
    assert capsys.readouterr().out.split()[0] == "a"
    assert built(fresh) == {"registry"}