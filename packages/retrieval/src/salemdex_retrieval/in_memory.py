"""An index kept in memory: the reference behaviour for filters and writes.

Use it in tests and single-process local runs. Other backends (Qdrant) must match how it
selects and writes chunks.
"""

import uuid
from collections.abc import Callable, Iterable
from typing import Any, Final, assert_never

from salemdex_retrieval.chunks import Chunk, SchemaMismatchError
from salemdex_retrieval.filters import And, AnyOf, Equals, Filter, Not, Or
from salemdex_retrieval.schema import Field, Schema


class InMemoryIndex:
    """An index holding the chunks of one schema in memory, selected by filters.

    Every filter is checked against the schema before any chunk is read or changed,
    and every write is checked in full before any chunk is stored: a write either
    happens whole or not at all.

    Not thread-safe, and not shared between processes: use the Qdrant backend when
    another process writes or reads.

    Attributes:
        schema: The schema every chunk and every filter field must belong to.
    """

    def __init__(self, *, schema: type[Schema]) -> None:
        self.schema: Final = schema
        self._chunks: dict[uuid.UUID, Chunk] = {}

    def put(self, chunks: Iterable[Chunk]) -> None:
        """Store `chunks`, overwriting any stored chunk with the same id.

        Writing the same chunks again changes nothing, so a retried write is safe.

        Args:
            chunks: The chunks to store.

        Raises:
            SchemaMismatchError: A chunk doesn't fit the schema. Nothing is stored.
            ValueError: Two chunks share an id. Nothing is stored.
        """
        for chunk in self._checked_chunk_batch(chunks):
            self._chunks[chunk.id] = chunk

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches, in no promised order.

        This reads without any asker's access filter. Only indexing code, tests and
        the engine may call it; anything serving a user searches through a session.

        Args:
            where: Which chunks to return.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
        """
        matches = self._build_matcher(where)
        return [chunk for chunk in self._chunks.values() if matches(chunk)]

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches.

        Args:
            where: Which chunks to remove.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
                Nothing is removed.
        """
        matches = self._build_matcher(where)
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if not matches(chunk)
        }

    def replace(self, *, where: Filter, with_: Iterable[Chunk]) -> None:
        """Remove the chunks that `where` matches and store `with_` in their place.

        Like SQL's DELETE … WHERE then INSERT: the new chunks needn't match `where`.
        A stored chunk with a new chunk's id is overwritten, as in `put`.

        Example:
            index.replace(where=KbChunks.document_id.eq(doc_id), with_=new_chunks)

        Args:
            where: Which chunks to remove.
            with_: The chunks to store instead.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema, or a
                new chunk doesn't fit it. Nothing is changed.
            ValueError: Two new chunks share an id. Nothing is changed.
        """
        matches = self._build_matcher(where)
        chunks = self._checked_chunk_batch(with_)
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if not matches(chunk)
        }

        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def _checked_chunk_batch(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        """Return `chunks` as a list, after checking every one of them.

        Raises:
            SchemaMismatchError: A chunk doesn't fit the schema.
            ValueError: Two chunks share an id.
        """
        batch = list(chunks)
        seen: set[uuid.UUID] = set()
        for chunk in batch:
            if chunk.id in seen:
                raise ValueError(f"duplicate chunk {chunk.id} in batch")
            chunk.check_fits(self.schema)
            seen.add(chunk.id)
        return batch

    def _build_matcher(self, where: Filter) -> Callable[[Chunk], bool]:
        """Return a function telling whether a chunk matches `where`.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
        """
        match where:
            case Equals(field=field, value=value):
                self._check_in_schema(field)
                return lambda chunk: chunk.field_values[field] == value
            case AnyOf(field=field, values=values):
                self._check_in_schema(field)
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

    def _check_in_schema(self, field: Field[Any]) -> None:
        """Check that `field` is one of the schema's fields, as an object, not a name.

        Raises:
            SchemaMismatchError: `field` isn't in the schema, or is another schema's
                field with the same name.
        """
        schema_fields = self.schema.fields()
        if schema_fields.get(field.name) is field:
            return
        in_another = " (another schema's field)" if field.name in schema_fields else ""
        raise SchemaMismatchError(
            f"field {field.name!r}{in_another} not in schema {self.schema.__name__}"
        )
