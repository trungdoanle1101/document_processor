from collections.abc import Mapping

from document_processor.retrieval.embedding import VectorSpec
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import ChunkIndex
from tests.retrieval.index_contract import (
    DENSE,
    EMBEDDER,
    ChunkIndexContract,
    ChunkIndexSearchContract,
)


def make_in_memory_index(vector_specs: Mapping[str, VectorSpec]) -> ChunkIndex:
    return InMemoryChunkIndex(embedder_identity=EMBEDDER, vector_specs=vector_specs)


class TestInMemoryChunkIndex(ChunkIndexContract):
    def make_index(self, vector_specs: Mapping[str, VectorSpec]) -> ChunkIndex:
        return make_in_memory_index(vector_specs)

    def test_it_reports_what_it_was_made_with(self) -> None:
        index = InMemoryChunkIndex(embedder_identity=EMBEDDER, vector_specs=DENSE)
        assert index.embedder_identity == EMBEDDER
        assert index.vector_specs == DENSE
        assert index.supported_conditions is None


class TestInMemoryChunkIndexSearch(ChunkIndexSearchContract):
    def make_index(self, vector_specs: Mapping[str, VectorSpec]) -> ChunkIndex:
        return make_in_memory_index(vector_specs)
