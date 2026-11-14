"""An index stored in a Qdrant collection.

Each chunk is one point: the chunk id is the point id, and the payload is laid out as
`payload.py` describes. Filters run inside Qdrant (`filters.py`). Points have no
vectors yet: the embedder comes with search.

Open an index with `QdrantIndex.open`, which checks the collection first: it creates the
collection and its payload indexes when they're missing, and refuses to open when what
Qdrant holds contradicts the schema, whether a payload index or the stored points.
"""

from collections.abc import Iterable
from typing import Final, Self

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from salemdex_retrieval.chunks import Chunk, SchemaMismatchError
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index.protocol import check_batch, check_filter
from salemdex_retrieval.index.qdrant.filters import build_qdrant_filter
from salemdex_retrieval.index.qdrant.payload import (
    WRITTEN_FIELDS,
    decode_payload,
    encode_payload,
    field_key,
    payload_schema_type,
    written_fields_of,
)
from salemdex_retrieval.schema import Schema

# Room for every expected `written_fields` entry and this many others; more means the
# collection holds something else entirely, and listing them wouldn't help.
_MAX_EXTRA_ENTRIES = 100


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
        """Wrap `collection` as an index, without checking it.

        Use `open`, which checks the collection first; this skips those checks.

        Args:
            schema: The schema every chunk and every filter field must belong to.
            client: An open client. The index doesn't close it; its owner does.
            collection: The collection's name, already checked by `open`.
        """
        self.schema: Final = schema
        self._client = client
        self._collection = collection

    @classmethod
    def open(
        cls, *, schema: type[Schema], client: QdrantClient, collection: str
    ) -> Self:
        """Open the index stored in `collection`, after checking it against `schema`.

        Creates the collection if it's missing, and a payload index for each
        `indexed=True` field and for `written_fields` that has none. An index the schema
        doesn't ask for is left alone. Then checks every stored point was written with
        exactly the schema's fields, by counting `written_fields` entries in Qdrant
        rather than reading the points. Safe to call again: a collection that's already
        right is unchanged.

        Args:
            schema: The schema every chunk and every filter field must belong to.
            client: An open client. The index doesn't close it; its owner does.
            collection: The collection's name.

        Returns:
            The index, ready for `put`, `find`, `delete` and `replace`.

        Raises:
            SchemaMismatchError: A field's payload index has another type than the
                schema needs; nothing is created, so drop that index and open again.
                Or some stored points lack a field of the schema, or hold one it
                doesn't define; the message counts them per field. Re-index into a
                new collection.
        """
        _ensure_collection(client=client, collection=collection)
        _ensure_payload_indexes(client=client, collection=collection, schema=schema)
        _check_points_fit_schema(client=client, collection=collection, schema=schema)
        return cls(schema=schema, client=client, collection=collection)

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


def _ensure_collection(*, client: QdrantClient, collection: str) -> None:
    """Create `collection`, with no vectors, unless it already exists."""
    if client.collection_exists(collection):
        return

    try:
        client.create_collection(
            collection_name=collection,
            vectors_config={},
        )
    except UnexpectedResponse:
        if client.collection_exists(collection):
            return
        raise


def _ensure_payload_indexes(
    *, client: QdrantClient, collection: str, schema: type[Schema]
) -> None:
    """Create the payload index each indexed field and `written_fields` lacks.

    Checks the existing ones first, and creates nothing if any has the wrong type.

    Raises:
        SchemaMismatchError: An indexed field's payload index has the wrong type. The
            message names every such field; no index is created.
    """
    expected_types = {
        field_key(field): payload_schema_type(field)
        for field in schema.fields().values()
        if field.indexed
    }
    expected_types[WRITTEN_FIELDS] = models.PayloadSchemaType.KEYWORD
    actual_types = {
        field_name: index_info.data_type
        for field_name, index_info in client.get_collection(
            collection
        ).payload_schema.items()
    }
    to_create: list[tuple[str, models.PayloadSchemaType]] = []
    wrong: list[tuple[str, models.PayloadSchemaType, models.PayloadSchemaType]] = []
    for key, expected in expected_types.items():
        actual = actual_types.get(key)
        if actual is None:
            to_create.append((key, expected))
        elif actual != expected:
            wrong.append((key, expected, actual))

    if wrong:
        message = (
            f"collection {collection}: mismatch between Schema {schema.__name__} and Qdrant payload:\n"
            + "\n".join(
                [
                    f"field {key}: expected {expected}, actual: {actual}"
                    for key, expected, actual in wrong
                ]
            )
        )
        raise SchemaMismatchError(message)

    for key, field_type in to_create:
        client.create_payload_index(
            collection_name=collection,
            field_name=key,
            field_schema=field_type,
            wait=True,
        )


def _check_points_fit_schema(
    *, client: QdrantClient, collection: str, schema: type[Schema]
) -> None:
    """Check every point in `collection` was written with exactly `schema`'s fields.

    Counts the points, then each `written_fields` entry in one facet request, so no
    point is read. Counting first means a write between the two requests can't make a
    healthy collection look broken; a delete between them still can.

    Raises:
        SchemaMismatchError: Some points lack an entry the schema expects, or hold one
            it doesn't define, or there are too many unknown entries to list. The
            message counts the points for each entry.
    """
    total = client.count(
        collection_name=collection,
        exact=True,
    ).count
    expected = written_fields_of(schema)
    limit = len(expected) + _MAX_EXTRA_ENTRIES
    points_by_written_field = _count_points_by_written_field(
        client=client, collection=collection, limit=limit
    )
    if len(points_by_written_field) == limit:
        raise SchemaMismatchError(
            f"collection {collection} has over {_MAX_EXTRA_ENTRIES} {WRITTEN_FIELDS} entries "
            f"schema {schema.__name__} doesn't define; "
            "it holds another schema's points or wasn't written by this index"
        )

    missing: list[tuple[str, int]] = []
    extra: list[tuple[str, int]] = []
    for entry in expected:
        if (count := total - points_by_written_field.get(entry, 0)) > 0:
            missing.append((entry, count))

    for entry, count in points_by_written_field.items():
        if entry not in expected:
            extra.append((entry, count))

    if extra or missing:
        message = f"collection {collection}: {WRITTEN_FIELDS} don't match against schema {schema.__name__}"
        for entry, count in sorted(missing):
            message += "\n" + f"{entry}: {count} points lack it"
        for entry, count in sorted(extra):
            message += (
                "\n"
                + f"{entry}: {count} points have it, but the schema doesn't define it"
            )
        raise SchemaMismatchError(message)


def _count_points_by_written_field(
    *, client: QdrantClient, collection: str, limit: int
) -> dict[str, int]:
    """Return how many points hold each `written_fields` entry, for `limit` entries."""
    hits = client.facet(collection, key=WRITTEN_FIELDS, limit=limit, exact=True).hits
    return {str(hit.value): hit.count for hit in hits}


def _left_over(where: Filter, *, kept: list[Chunk]) -> models.Filter:
    """Return a filter matching what `where` matches, except the chunks in `kept`."""
    return models.Filter(
        must=[build_qdrant_filter(where)],
        must_not=[models.HasIdCondition(has_id=[chunk.id for chunk in kept])],
    )
