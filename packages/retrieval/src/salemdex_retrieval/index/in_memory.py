"""An index kept in memory: the reference behaviour for filters and writes.

Use it in tests and single-process local runs. Other backends (Qdrant) must match how it
selects and writes chunks.
"""

import uuid
from collections.abc import Callable, Iterable
from typing import Final, assert_never

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.filters import And, AnyOf, Equals, Filter, Not, Or
from salemdex_retrieval.index.protocol import check_batch, check_filter
from salemdex_retrieval.schema import Schema


class InMemoryIndex:
    """An `Index` holding its chunks in memory: the reference behaviour.

    Other backends must select and write chunks as it does. It promises no more than
    they do: `replace` isn't made atomic here either.

    Not thread-safe, and not shared between processes: use the Qdrant backend when
    another process writes or reads.

    Attributes:
        schema: The schema every chunk and every filter field must belong to.
    """

    def __init__(self, *, schema: type[Schema]) -> None:
        self.schema: Final = schema
        self._chunks: dict[uuid.UUID, Chunk] = {}

    def put(self, chunks: Iterable[Chunk]) -> None:
        """Store `chunks`, overwriting any stored chunk with the same id (`Index.put`)."""
        for chunk in check_batch(chunks, schema=self.schema):
            self._chunks[chunk.id] = chunk

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches (`Index.find`)."""
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        return [chunk for chunk in self._chunks.values() if matches(chunk)]

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches (`Index.delete`)."""
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if not matches(chunk)
        }

    def replace(self, *, where: Filter, with_: Iterable[Chunk]) -> None:
        """Swap the chunks `where` matches for `with_` (`Index.replace`)."""
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        chunks = check_batch(with_, schema=self.schema)
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if not matches(chunk)
        }

        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def _build_matcher(self, where: Filter) -> Callable[[Chunk], bool]:
        """Return a function telling whether a chunk matches `where`.

        `where` must already be checked by `check_filter`: a field outside the schema
        would raise `KeyError` on the first chunk.
        """
        match where:
            case Equals(field=field, value=value):
                return lambda chunk: chunk.field_values[field] == value
            case AnyOf(field=field, values=values):
                return lambda chunk: not values.isdisjoint(chunk.field_values[field])
            case And(filters=filters):
                parts = [self._build_matcher(f) for f in filters]
                return lambda chunk: all(part(chunk) for part in parts)
            case Or(filters=filters):
                parts = [self._build_matcher(f) for f in filters]
                return lambda chunk: any(part(chunk) for part in parts)
            case Not(filter=f):
                part = self._build_matcher(f)
                return lambda chunk: not part(chunk)
            case _:
                assert_never(where)
