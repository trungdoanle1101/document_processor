"""Fixtures shared by the index tests: the backends every contract test runs against.

Qdrant always runs in the client's in-memory mode, and against a server too when
KB_TEST_QDRANT_URL is set (e.g. http://localhost:6333 after
`docker compose -f docker/docker-compose.yml up -d qdrant`).
"""

import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from qdrant_client import QdrantClient

from salemdex_retrieval.index import Index, InMemoryIndex
from salemdex_retrieval.index.qdrant import QdrantIndex
from salemdex_retrieval.schema import Schema

type OpenIndex = Callable[[type[Schema]], Index]

QDRANT_URL = os.environ.get("KB_TEST_QDRANT_URL")


def pytest_configure(config: pytest.Config) -> None:
    """Silence the local client's warning that payload indexes do nothing there.

    `QdrantIndex.open` creates them on every backend; only the server uses them.
    """
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Payload indexes have no effect in the local Qdrant:UserWarning",
    )


# One entry per backend.
BACKENDS = [
    pytest.param("in-memory", id="in-memory"),
    pytest.param("qdrant-local", id="qdrant-local"),
    pytest.param(
        "qdrant-server",
        id="qdrant-server",
        marks=pytest.mark.skipif(
            QDRANT_URL is None, reason="KB_TEST_QDRANT_URL is not set"
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def new_index(request: pytest.FixtureRequest) -> Iterator[OpenIndex]:
    """Return a function opening an empty index of a schema, once per backend."""
    match request.param:
        case "in-memory":
            yield lambda schema: InMemoryIndex(schema=schema)
        case "qdrant-local":
            with _opening_qdrant_indexes(QdrantClient(":memory:")) as open_index:
                yield open_index
        case "qdrant-server":
            with _opening_qdrant_indexes(QdrantClient(url=QDRANT_URL)) as open_index:
                yield open_index
        case other:
            raise ValueError(f"unknown backend {other!r}")


@contextmanager
def _opening_qdrant_indexes(client: QdrantClient) -> Iterator[OpenIndex]:
    """Yield a function opening Qdrant indexes, each in a new collection.

    The collections are dropped and the client closed afterwards, so tests sharing a
    server never see each other's chunks.
    """
    collections: list[str] = []

    def open_index(schema: type[Schema]) -> Index:
        collection = f"test_{uuid.uuid4().hex}"
        collections.append(collection)
        return QdrantIndex.open(schema=schema, client=client, collection=collection)

    try:
        yield open_index
    finally:
        for collection in collections:
            client.delete_collection(collection)
        client.close()
