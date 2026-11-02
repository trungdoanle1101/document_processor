"""QdrantChunkIndex: the shared contract, plus what only the Qdrant layout can get wrong.

The contract runs on local Qdrant (":memory:") always, and on a server when QDRANT_URL is
set, e.g. QDRANT_URL=http://localhost:6333. Server runs make and delete their own collections.
"""

import os
import uuid
from collections.abc import Mapping
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

import document_processor.integrations.qdrant.index as qdrant_index
from document_processor.integrations.qdrant._payload import encode_attributes
from document_processor.integrations.qdrant.index import QdrantChunkIndex
from document_processor.retrieval.embedding import DenseVectorSpec, VectorSpec
from document_processor.retrieval.index import ChunkFilter, ChunkIndex, ChunkKey
from tests.retrieval.index_contract import (
    DENSE,
    EMBEDDER,
    ChunkIndexContract,
    ChunkIndexSearchContract,
    chunk,
    dense,
    entry,
)

QDRANT_URL = os.environ.get("QDRANT_URL")
needs_server = pytest.mark.skipif(
    QDRANT_URL is None, reason="set QDRANT_URL to run against a Qdrant server"
)


@pytest.fixture(autouse=True)
def small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pages and upload batches of two, so even small tests cross a boundary."""
    monkeypatch.setattr(qdrant_index, "_SCROLL_PAGE_SIZE", 2)
    monkeypatch.setattr(qdrant_index, "_UPLOAD_BATCH_SIZE", 2)


def open_index(
    client: QdrantClient,
    collection: str = "chunks",
    vector_specs: Mapping[str, VectorSpec] = DENSE,
    embedder_identity: str = EMBEDDER,
) -> QdrantChunkIndex:
    return QdrantChunkIndex(
        client,
        collection=collection,
        embedder_identity=embedder_identity,
        vector_specs=vector_specs,
    )


def payloads(
    client: QdrantClient, kind: str, collection: str = "chunks"
) -> list[dict[str, Any]]:
    """Every stored payload of this kind, as Qdrant holds it."""
    records, _ = client.scroll(
        collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind))
            ]
        ),
        limit=1000,
    )
    return [record.payload or {} for record in records]


# The contract --------------------------------------------------------------------------------


class _Local:
    def make_index(self, vector_specs: Mapping[str, VectorSpec]) -> ChunkIndex:
        return open_index(QdrantClient(":memory:"), vector_specs=vector_specs)


class TestLocal(_Local, ChunkIndexContract):
    pass


class TestLocalSearch(_Local, ChunkIndexSearchContract):
    pass


class _Server:
    def setup_method(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)
        self.collections: list[str] = []

    def teardown_method(self) -> None:
        for collection in self.collections:
            self.client.delete_collection(collection)
        self.client.close()

    def make_index(self, vector_specs: Mapping[str, VectorSpec]) -> ChunkIndex:
        collection = f"test-{uuid.uuid4().hex}"
        self.collections.append(collection)
        return open_index(self.client, collection, vector_specs)


@needs_server
class TestServer(_Server, ChunkIndexContract):
    pass


@needs_server
class TestServerSearch(_Server, ChunkIndexSearchContract):
    pass


@needs_server
class TestServerPayloadIndexes(_Server):
    def test_keyword_fields_and_indexed_attributes_get_indexes(self) -> None:
        collection = f"test-{uuid.uuid4().hex}"
        self.collections.append(collection)
        QdrantChunkIndex(
            self.client,
            collection=collection,
            embedder_identity=EMBEDDER,
            vector_specs=DENSE,
            indexed_attributes={"year": "number", "title": "text"},
        )
        schema = self.client.get_collection(collection).payload_schema
        assert set(schema) == {
            "kind",
            "chunk_key",
            "document_id",
            "generator",
            "reference_keys",
            "attributes.integer.year",
            "attributes.float.year",
            "attributes.text.title",
        }


# Opening a collection --------------------------------------------------------------------------


def test_it_reports_what_it_was_made_with() -> None:
    index = open_index(QdrantClient(":memory:"))
    assert index.embedder_identity == EMBEDDER
    assert index.vector_specs == DENSE
    assert index.supported_conditions == qdrant_index._SUPPORTED_CONDITIONS


def test_reopening_a_collection_keeps_its_contents() -> None:
    client = QdrantClient(":memory:")
    open_index(client).replace_document(
        "law", [chunk("a", position=0), chunk("b", position=1)], [dense(1, 0)] * 2
    )
    assert open_index(client).list_chunks(ChunkFilter(), limit=10) == [
        chunk("a", position=0),
        chunk("b", position=1),
    ]


def test_reopening_with_another_embedder_is_refused() -> None:
    client = QdrantClient(":memory:")
    open_index(client)
    with pytest.raises(ValueError, match="holds vectors from 'stub', not 'other'"):
        open_index(client, embedder_identity="other")


def test_reopening_with_other_vector_specs_is_refused() -> None:
    client = QdrantClient(":memory:")
    open_index(client)
    with pytest.raises(ValueError, match="holds vector specs"):
        open_index(client, vector_specs={"dense": DenseVectorSpec(dimension=3)})


def test_a_collection_made_elsewhere_is_refused() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("chunks", vectors_config={})
    with pytest.raises(ValueError, match="no embedder identity"):
        open_index(client)


def test_an_indexed_attribute_needs_a_valid_name() -> None:
    with pytest.raises(ValueError, match="lower-case"):
        QdrantChunkIndex(
            QdrantClient(":memory:"),
            collection="chunks",
            embedder_identity=EMBEDDER,
            vector_specs=DENSE,
            indexed_attributes={"Năm": "integer"},
        )


# Copies kept in step: the stored payloads, not just what comes back ---------------------------


def test_an_update_leaves_nothing_of_the_old_values_on_chunks_or_entry_copies() -> None:
    client = QdrantClient(":memory:")
    index = open_index(client)
    index.replace_document(
        "law", [chunk("a", code="5", tags=(), note="x", kept=True)], [dense(1, 0)]
    )
    index.replace_generated("law", "questions-v1", [entry("q", "a")], [dense(1, 0)])

    index.update_attributes(
        ChunkFilter(document_ids=("law",)), {"code": 5, "tags": (2, 3), "note": None}
    )

    expected = encode_attributes({"code": 5, "tags": (2, 3), "kept": True})
    assert [p["attributes"] for p in payloads(client, "chunk")] == [expected]
    assert [p["attributes"] for p in payloads(client, "entry")] == [expected]


def test_a_partial_delete_rewrites_the_surviving_copies() -> None:
    client = QdrantClient(":memory:")
    index = open_index(client)
    index.replace_document(
        "law", [chunk(c, position=i) for i, c in enumerate("abc")], [dense(1, 0)] * 3
    )
    index.replace_generated(
        "law",
        "questions-v1",
        [entry("q", "a", "b", "c"), entry("only_b", "b")],
        [dense(1, 0)] * 2,
    )

    index.delete_chunks(
        ChunkFilter(chunk_keys=(ChunkKey(document_id="law", chunk_id="b"),))
    )

    copies = {
        (p["entry_id"], p["chunk_id"], tuple(p["entry_chunk_ids"]))
        for p in payloads(client, "entry")
    }
    assert copies == {("q", "a", ("a", "c")), ("q", "c", ("a", "c"))}


def test_entry_copies_take_their_chunks_routing_fields() -> None:
    client = QdrantClient(":memory:")
    index = open_index(client)
    chunks = [
        chunk("a", position=0, reference_keys=("dieu:1",), year=2024),
        chunk("b", position=1, year=2020),
    ]
    index.replace_document("law", chunks, [dense(1, 0)] * 2)
    index.replace_generated(
        "law", "questions-v1", [entry("q", "a", "b")], [dense(1, 0)]
    )

    by_chunk = {p["chunk_id"]: p for p in payloads(client, "entry")}
    assert by_chunk["a"]["reference_keys"] == ["dieu:1"]
    assert by_chunk["a"]["attributes"] == encode_attributes({"year": 2024})
    assert by_chunk["b"]["reference_keys"] == []
    assert by_chunk["b"]["attributes"] == encode_attributes({"year": 2020})
