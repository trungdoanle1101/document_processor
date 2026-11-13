"""An index stored in a Qdrant collection.

Each chunk is one point: the chunk id is the point id, and the payload is laid out as
`payload.py` describes. Filters run inside Qdrant (`filters.py`). Points have no
vectors yet: the embedder comes with search.
"""

from collections.abc import Iterable
from typing import Final

from qdrant_client import QdrantClient, models

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index.protocol import check_batch, check_filter
from salemdex_retrieval.index.qdrant.filters import build_qdrant_filter
from salemdex_retrieval.index.qdrant.payload import decode_payload, encode_payload
from salemdex_retrieval.schema import Schema


class QdrantIndex:
    """An `Index` storing its chunks as the points of one Qdrant collection.

    Behaves as `InMemoryIndex` does; the shared contract tests hold it to that. Every
    write waits until Qdrant has applied it, so a `find` right after sees it.

    Attributes:
        schema: The schema every chunk and every filter field must belong to.
    """

    def __init__(
        self, *, schema: type[Schema], client: QdrantClient, collection: str
    ) -> None:
        """Open the index stored in `collection`.

        Args:
            schema: The schema every chunk and every filter field must belong to.
            client: An open client. The index doesn't close it; its owner does.
            collection: The collection's name. It must already exist.
        """
        self.schema: Final = schema
        self._client = client
        self._collection = collection

    def put(self, chunks: Iterable[Chunk]) -> None:
        """Store `chunks`, overwriting any stored chunk with the same id (`Index.put`)."""
        self._store(check_batch(chunks, schema=self.schema))

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches (`Index.find`).

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema, or a
                matched point doesn't hold a chunk of the schema.
        """
        check_filter(where, schema=self.schema)
        records = self._scroll_all(build_qdrant_filter(where))
        return [self._to_chunk(record) for record in records]

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches (`Index.delete`)."""
        check_filter(where, schema=self.schema)
        self._delete(build_qdrant_filter(where))

    def replace(self, *, where: Filter, with_: Iterable[Chunk]) -> None:
        """Swap the chunks `where` matches for `with_` (`Index.replace`).

        Qdrant can't make the two steps atomic, so the new chunks are written first and
        the matched chunks they didn't overwrite are removed after: a reader may see
        old and new chunks together for a moment, but never neither. If the removal
        fails, both stay stored; repeating the call finishes it.
        """
        check_filter(where, schema=self.schema)
        batch = check_batch(with_, schema=self.schema)
        self._store(batch)
        self._delete(_left_over(where, kept=batch))

    def _store(self, batch: list[Chunk]) -> None:
        """Write chunks already checked against the schema, in one request."""
        # Qdrant refuses an update with no points.
        if not batch:
            return
        points = [
            models.PointStruct(id=chunk.id, payload=encode_payload(chunk), vector={})
            for chunk in batch
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def _scroll_all(self, qdrant_filter: models.Filter) -> list[models.Record]:
        """Return every point `qdrant_filter` matches, reading page by page."""
        records: list[models.Record] = []
        offset: models.ExtendedPointId | None = None
        while True:
            page, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qdrant_filter,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                limit=1000,
            )
            records.extend(page)
            if offset is None:
                return records

    def _to_chunk(self, record: models.Record) -> Chunk:
        """Return the chunk a point holds.

        Raises:
            SchemaMismatchError: The point doesn't hold a chunk of the schema.
        """
        # No payload reads as an empty one, so the error names the missing keys.
        return decode_payload(
            point_id=record.id, payload=record.payload or {}, schema=self.schema
        )

    def _delete(self, qdrant_filter: models.Filter) -> None:
        """Remove the points `qdrant_filter` matches."""
        self._client.delete(
            collection_name=self._collection, points_selector=qdrant_filter, wait=True
        )


def _left_over(where: Filter, *, kept: list[Chunk]) -> models.Filter:
    """Return a filter matching what `where` matches, except the chunks in `kept`."""
    return models.Filter(
        must=[build_qdrant_filter(where)],
        must_not=[models.HasIdCondition(has_id=[chunk.id for chunk in kept])],
    )
