"""The interface every index backend implements, and the checks they share.

Each backend (in memory, Qdrant) must behave as `Index` describes; the shared tests in
`tests/index` hold every backend to it.
"""

import uuid
from collections.abc import Iterable
from typing import Protocol

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.schema import Schema


class Index(Protocol):
    """Chunks of one schema, stored by id and selected by filters.

    Every filter is checked against the schema before any chunk is read or changed,
    and every write is checked in full before any chunk is stored: a write either
    happens whole or not at all.
    """

    @property
    def schema(self) -> type[Schema]:
        """The schema every chunk and every filter field must belong to; never changes."""
        ...

    def put(self, chunks: Iterable[Chunk]) -> None:
        """Store `chunks`, overwriting any stored chunk with the same id.

        Writing the same chunks again changes nothing, so a retried write is safe.

        Args:
            chunks: The chunks to store.

        Raises:
            SchemaMismatchError: A chunk doesn't fit the schema. Nothing is stored.
            ValueError: Two chunks share an id. Nothing is stored.
        """
        ...

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches, in no promised order.

        This reads without any asker's access filter. Only indexing code, tests and
        the engine may call it; anything serving a user searches through a session.

        Args:
            where: Which chunks to return.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
        """
        ...

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches.

        Args:
            where: Which chunks to remove.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
                Nothing is removed.
        """
        ...

    def replace(self, *, where: Filter, with_: Iterable[Chunk]) -> None:
        """Remove the chunks that `where` matches and store `with_` in their place.

        Like SQL's DELETE … WHERE then INSERT: the new chunks needn't match `where`.
        A stored chunk with a new chunk's id is overwritten, as in `put`. A backend
        that can't make the two steps atomic (Qdrant) may let a reader see the gap.

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
        ...


def check_batch(chunks: Iterable[Chunk], *, schema: type[Schema]) -> list[Chunk]:
    """Return `chunks` as a list, after checking every one of them against `schema`.

    Every backend calls it before storing anything, so a write is checked in full
    first and nothing is stored when one chunk is wrong.

    Args:
        chunks: The chunks about to be written.
        schema: The schema of the index they're written to.

    Returns:
        The same chunks, in the same order.

    Raises:
        SchemaMismatchError: A chunk doesn't fit the schema.
        ValueError: Two chunks share an id.
    """
    batch = list(chunks)
    seen: set[uuid.UUID] = set()
    for chunk in batch:
        if chunk.id in seen:
            raise ValueError(f"duplicate chunk {chunk.id} in batch")
        chunk.check_fits(schema)
        seen.add(chunk.id)
    return batch
