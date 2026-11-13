"""Fixtures shared by the index tests: the backends every contract test runs against."""

from collections.abc import Callable

import pytest

from salemdex_retrieval.index import Index, InMemoryIndex
from salemdex_retrieval.schema import Schema


def _in_memory(schema: type[Schema]) -> Index:
    return InMemoryIndex(schema=schema)


# One entry per backend. A backend needing a server or cleanup (Qdrant) turns
# `new_index` into a yielding fixture that opens and drops its collections.
BACKENDS = [pytest.param(_in_memory, id="in-memory")]


@pytest.fixture(params=BACKENDS)
def new_index(request: pytest.FixtureRequest) -> Callable[[type[Schema]], Index]:
    """Return a function opening an empty index of a schema, once per backend."""
    open_index: Callable[[type[Schema]], Index] = request.param
    return open_index
