"""What `QdrantIndex.open` creates and what it refuses to open.

The checks on stored points run on both Qdrant backends. The payload index tests need
a server (KB_TEST_QDRANT_URL): the local client keeps no payload indexes.
"""

import re
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from salemdex_retrieval.chunks import Chunk, SchemaMismatchError
from salemdex_retrieval.index.qdrant import QdrantIndex
from salemdex_retrieval.index.qdrant.payload import encode_payload
from salemdex_retrieval.schema import Many, One, Schema

from .conftest import QDRANT_URL


class Docs(Schema):
    document_id = One(uuid.UUID, indexed=True)
    version = One(int)
    groups = Many(uuid.UUID, indexed=True)


DOCUMENT = uuid.UUID("00000000-0000-0000-0000-000000000001")
HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")

needs_server = pytest.mark.skipif(
    QDRANT_URL is None, reason="KB_TEST_QDRANT_URL is not set"
)


@pytest.fixture(
    params=[
        pytest.param("local", id="qdrant-local"),
        pytest.param("server", id="qdrant-server", marks=needs_server),
    ]
)
def client(request: pytest.FixtureRequest) -> Iterator[QdrantClient]:
    """Return a client of each Qdrant backend."""
    opened = (
        QdrantClient(":memory:")
        if request.param == "local"
        else QdrantClient(url=QDRANT_URL)
    )
    yield opened
    opened.close()


@pytest.fixture
def server_client() -> Iterator[QdrantClient]:
    """Return a client of the Qdrant server; the test is skipped without one."""
    if QDRANT_URL is None:
        pytest.skip("KB_TEST_QDRANT_URL is not set")
    opened = QdrantClient(url=QDRANT_URL)
    yield opened
    opened.close()


@pytest.fixture
def collection(client: QdrantClient) -> Iterator[str]:
    """Return a new collection name, dropped after the test."""
    yield from _dropped_afterwards(client)


@pytest.fixture
def server_collection(server_client: QdrantClient) -> Iterator[str]:
    """Return a new collection name on the server, dropped after the test."""
    yield from _dropped_afterwards(server_client)


def _dropped_afterwards(client: QdrantClient) -> Iterator[str]:
    name = f"test_{uuid.uuid4().hex}"
    yield name
    client.delete_collection(name)


def open_docs(client: QdrantClient, collection: str) -> QdrantIndex:
    return QdrantIndex.open(schema=Docs, client=client, collection=collection)


def chunk(*, groups: list[uuid.UUID]) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        text="Article 1. Scope",
        field_values=[
            Docs.document_id.value(DOCUMENT),
            Docs.version.value(2),
            Docs.groups.value(groups),
        ],
    )


def store_raw(client: QdrantClient, collection: str, payload: dict[str, Any]) -> None:
    """Write a point straight to Qdrant, bypassing the index's checks."""
    client.upsert(
        collection,
        points=[models.PointStruct(id=str(uuid.uuid4()), payload=payload, vector={})],
        wait=True,
    )


def payload_with_written_fields(entries: list[str]) -> dict[str, Any]:
    payload = encode_payload(chunk(groups=[HR]))
    payload["written_fields"] = entries
    return payload


EXPECTED_ENTRIES = ["document_id:uuid:one", "groups:uuid:many", "version:int:one"]


# Opening a healthy collection


def test_opening_a_missing_collection_creates_it(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection)

    assert client.collection_exists(collection)


def test_points_written_by_the_index_open_again(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection).put([chunk(groups=[HR]), chunk(groups=[])])

    open_docs(client, collection)


def test_a_many_with_no_values_does_not_count_as_missing(
    client: QdrantClient, collection: str
) -> None:
    # Qdrant can't tell [] from a missing list; written_fields can.
    open_docs(client, collection).put([chunk(groups=[])])

    open_docs(client, collection)


# Refusing points that don't fit the schema


def test_points_lacking_a_field_refuse_to_open_counting_them(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection)
    for _ in range(2):
        store_raw(
            client,
            collection,
            payload_with_written_fields(["document_id:uuid:one", "groups:uuid:many"]),
        )

    with pytest.raises(SchemaMismatchError, match="version:int:one: 2 points lack it"):
        open_docs(client, collection)


def test_a_point_with_no_written_fields_lacks_every_field(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection)
    store_raw(client, collection, {"text": "Article 1. Scope"})

    with pytest.raises(SchemaMismatchError) as raised:
        open_docs(client, collection)

    for entry in EXPECTED_ENTRIES:
        assert f"{entry}: 1 points lack it" in str(raised.value)


def test_a_field_written_with_another_type_is_reported_as_missing_and_extra(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection)
    store_raw(
        client,
        collection,
        payload_with_written_fields(
            ["document_id:uuid:one", "groups:uuid:many", "version:str:one"]
        ),
    )

    with pytest.raises(SchemaMismatchError) as raised:
        open_docs(client, collection)

    assert "version:int:one: 1 points lack it" in str(raised.value)
    assert "version:str:one: 1 points have it" in str(raised.value)


def test_the_error_names_the_collection_and_the_schema(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection)
    store_raw(client, collection, {"text": "Article 1. Scope"})

    with pytest.raises(SchemaMismatchError, match=f"{collection}.*Docs"):
        open_docs(client, collection)


def test_too_many_unknown_entries_refuse_to_open_without_listing_them(
    client: QdrantClient, collection: str
) -> None:
    open_docs(client, collection)
    unknown = [f"field_{n}:str:one" for n in range(100)]
    store_raw(
        client, collection, payload_with_written_fields(EXPECTED_ENTRIES + unknown)
    )

    with pytest.raises(SchemaMismatchError, match="over 100 written_fields entries"):
        open_docs(client, collection)


# Payload indexes (server only)


def test_opening_creates_an_index_for_each_indexed_field_and_written_fields(
    server_client: QdrantClient, server_collection: str
) -> None:
    open_docs(server_client, server_collection)

    assert _index_types(server_client, server_collection) == {
        "field_values.document_id": models.PayloadSchemaType.UUID,
        "field_values.groups": models.PayloadSchemaType.UUID,
        "written_fields": models.PayloadSchemaType.KEYWORD,
    }


def test_opening_again_keeps_the_indexes(
    server_client: QdrantClient, server_collection: str
) -> None:
    open_docs(server_client, server_collection)
    first = _index_types(server_client, server_collection)

    open_docs(server_client, server_collection)

    assert _index_types(server_client, server_collection) == first


def test_an_index_of_the_wrong_type_refuses_to_open_and_creates_nothing(
    server_client: QdrantClient, server_collection: str
) -> None:
    server_client.create_collection(server_collection, vectors_config={})
    server_client.create_payload_index(
        server_collection,
        field_name="field_values.document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

    with pytest.raises(
        SchemaMismatchError,
        match=re.escape("field_values.document_id: expected uuid, actual: keyword"),
    ):
        open_docs(server_client, server_collection)

    assert _index_types(server_client, server_collection) == {
        "field_values.document_id": models.PayloadSchemaType.KEYWORD
    }


def test_an_index_the_schema_does_not_ask_for_is_left_alone(
    server_client: QdrantClient, server_collection: str
) -> None:
    server_client.create_collection(server_collection, vectors_config={})
    server_client.create_payload_index(
        server_collection,
        field_name="field_values.version",
        field_schema=models.PayloadSchemaType.INTEGER,
        wait=True,
    )

    open_docs(server_client, server_collection)

    assert (
        _index_types(server_client, server_collection)["field_values.version"]
        == models.PayloadSchemaType.INTEGER
    )


def _index_types(
    client: QdrantClient, collection: str
) -> dict[str, models.PayloadSchemaType | None]:
    return {
        key: info.data_type
        for key, info in client.get_collection(collection).payload_schema.items()
    }
