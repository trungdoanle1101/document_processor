from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import MappingProxyType
from typing import Any, Literal, assert_never

from qdrant_client import QdrantClient, models

from document_processor.integrations.qdrant._filters import translate_filter
from document_processor.integrations.qdrant._payload import (
    CHUNK,
    CHUNK_ID,
    CHUNK_KEY,
    DOCUMENT_ID,
    ENTRY,
    ENTRY_CHUNK_IDS,
    ENTRY_ID,
    GENERATOR,
    KIND,
    REFERENCE_KEYS,
    STORAGE_KINDS,
    StorageKind,
    attribute_group_key,
    attribute_key,
    chunk_from_payload,
    chunk_payload,
    chunk_point_id,
    encode_attributes,
    entry_copy_payload,
    entry_point_id,
    point_vectors,
)
from document_processor.retrieval.attributes import (
    AllOf,
    AnyOf,
    AttributeChanges,
    BooleanEquals,
    Condition,
    DateBetween,
    EveryElement,
    IntegerIn,
    NoneOf,
    NumberBetween,
    TextIn,
    check_attribute_name,
)
from document_processor.retrieval.embedding import (
    DenseVectorSpec,
    Embedding,
    SparseVectorSpec,
    VectorSpec,
)
from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkHit,
    ChunkIndex,
    ChunkKey,
    GeneratedEntry,
    GeneratedEntryKey,
    IndexedChunk,
    SearchQuery,
)

type AttributeKind = Literal["text", "integer", "number", "boolean", "date"]
type _PayloadIndex = tuple[str, models.PayloadSchemaType]

_EMBEDDER_IDENTITY = "embedder_identity"
_VECTOR_SPECS = "vector_specs"

_SUPPORTED_CONDITIONS: frozenset[type[Condition]] = frozenset(
    {
        AllOf,
        AnyOf,
        BooleanEquals,
        DateBetween,
        EveryElement,
        IntegerIn,
        NoneOf,
        NumberBetween,
        TextIn,
    }
)

_KEYWORD_FIELDS = (KIND, CHUNK_KEY, DOCUMENT_ID, GENERATOR, REFERENCE_KEYS)

_UPLOAD_BATCH_SIZE = 64
_SCROLL_PAGE_SIZE = 256

# What the caller asks to index -> where those values are stored, and how to index them
_SCHEMAS: Mapping[
    AttributeKind, tuple[tuple[StorageKind, models.PayloadSchemaType], ...]
] = {
    "text": (("text", models.PayloadSchemaType.KEYWORD),),
    "integer": (("integer", models.PayloadSchemaType.INTEGER),),
    "number": (
        ("integer", models.PayloadSchemaType.INTEGER),
        ("float", models.PayloadSchemaType.FLOAT),
    ),
    "boolean": (("boolean", models.PayloadSchemaType.BOOL),),
    "date": (("date", models.PayloadSchemaType.DATETIME),),
}


class QdrantChunkIndex(ChunkIndex):
    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str,
        embedder_identity: str,
        vector_specs: Mapping[str, VectorSpec],
        indexed_attributes: Mapping[str, AttributeKind] = {},
    ) -> None:
        super().__init__()
        self._client = client
        self._collection = collection
        self._embedder_identity = embedder_identity
        self._vector_specs = MappingProxyType(dict(vector_specs))
        self._indexed_attributes = MappingProxyType(dict(indexed_attributes))

        _create_or_verify_collection(
            client=self._client,
            collection=self._collection,
            embedder_identity=self._embedder_identity,
            vector_specs=self._vector_specs,
        )
        _create_payload_indexes(
            self._client, self._collection, _payload_indexes(self._indexed_attributes)
        )

    @property
    def embedder_identity(self) -> str:
        return self._embedder_identity

    @property
    def vector_specs(self) -> Mapping[str, VectorSpec]:
        return self._vector_specs

    @property
    def supported_conditions(self) -> AbstractSet[type[Condition]] | None:
        return _SUPPORTED_CONDITIONS

    def _replace_document(
        self,
        document_id: str,
        chunks: Sequence[IndexedChunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        points = [
            models.PointStruct(
                id=chunk_point_id(chunk.key),
                vector=point_vectors(embedding),
                payload=chunk_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self._client.upload_points(
            collection_name=self._collection,
            points=points,
            batch_size=_UPLOAD_BATCH_SIZE,
            wait=True,
        )

        self._client.delete(
            collection_name=self._collection,
            points_selector=models.Filter(
                must=[
                    translate_filter(
                        ChunkFilter(document_ids=(document_id,)), kind=None
                    )
                ],
                must_not=[models.HasIdCondition(has_id=[point.id for point in points])],
            ),
            wait=True,
        )

    def _replace_generated(
        self,
        document_id: str,
        generator: str,
        entries: Sequence[GeneratedEntry],
        embeddings: Sequence[Embedding],
    ) -> None:
        needed = {chunk_id for entry in entries for chunk_id in entry.chunk_ids}

        records = self._client.retrieve(
            collection_name=self._collection,
            ids=[
                chunk_point_id(ChunkKey(document_id=document_id, chunk_id=chunk_id))
                for chunk_id in needed
            ],
            with_payload=True,
            with_vectors=False,
        )

        chunks: dict[str, IndexedChunk] = {}
        for record in records:
            assert record.payload is not None
            chunk = chunk_from_payload(record.payload)
            chunks[chunk.chunk_id] = chunk

        points = [
            models.PointStruct(
                id=entry_point_id(entry.key, chunk_id),
                vector=point_vectors(embedding),
                payload=entry_copy_payload(entry, chunks[chunk_id]),
            )
            for entry, embedding in zip(entries, embeddings, strict=True)
            for chunk_id in entry.chunk_ids
        ]

        self._client.upload_points(
            collection_name=self._collection,
            points=points,
            batch_size=_UPLOAD_BATCH_SIZE,
            wait=True,
        )

        self._client.delete(
            collection_name=self._collection,
            points_selector=models.Filter(
                must=[
                    translate_filter(
                        ChunkFilter(document_ids=(document_id,)), kind=ENTRY
                    ),
                    models.FieldCondition(
                        key=GENERATOR, match=models.MatchValue(value=generator)
                    ),
                ],
                must_not=[models.HasIdCondition(has_id=[point.id for point in points])],
            ),
            wait=True,
        )

    def _delete_chunks(self, chunk_filter: ChunkFilter) -> None:
        doomed_copies = _scroll_all(
            self._client,
            self._collection,
            translate_filter(chunk_filter, kind=ENTRY),
            with_payload=[DOCUMENT_ID, GENERATOR, ENTRY_ID, CHUNK_ID, ENTRY_CHUNK_IDS],
        )
        leads_to: dict[GeneratedEntryKey, list[str]] = {}
        doomed: dict[GeneratedEntryKey, set[str]] = {}
        for record in doomed_copies:
            payload = record.payload
            assert payload is not None
            key = GeneratedEntryKey(
                document_id=payload[DOCUMENT_ID],
                generator=payload[GENERATOR],
                entry_id=payload[ENTRY_ID],
            )
            leads_to[key] = payload[ENTRY_CHUNK_IDS]
            doomed.setdefault(key, set()).add(payload[CHUNK_ID])

        rewrites: list[models.UpdateOperation] = []
        for key, chunk_ids in leads_to.items():
            remaining = [
                chunk_id for chunk_id in chunk_ids if chunk_id not in doomed[key]
            ]
            if remaining:
                rewrites.append(
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(
                            payload={ENTRY_CHUNK_IDS: remaining},
                            points=[
                                entry_point_id(key, chunk_id) for chunk_id in remaining
                            ],
                        )
                    )
                )

        self._client.batch_update_points(
            collection_name=self._collection,
            update_operations=rewrites,
            wait=True,
        )

        self._client.delete(
            collection_name=self._collection,
            points_selector=translate_filter(chunk_filter, kind=None),
            wait=True,
        )

    def _update_attributes(
        self, chunk_filter: ChunkFilter, changes: AttributeChanges
    ) -> None:
        ids = [
            record.id
            for record in _scroll_all(
                self._client,
                self._collection,
                translate_filter(chunk_filter, kind=None),
                with_payload=False,
            )
        ]
        if not ids:
            return

        operations: list[models.UpdateOperation] = []
        # remove old values of every changed name
        operations.append(
            models.DeletePayloadOperation(
                delete_payload=models.DeletePayload(
                    keys=[
                        attribute_key(kind, name)
                        for name in changes
                        for kind in STORAGE_KINDS
                    ],
                    points=ids,
                )
            )
        )

        # set the new values
        to_set = {name: value for name, value in changes.items() if value is not None}
        for kind, values in encode_attributes(to_set).items():
            if values:
                operations.append(
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(
                            payload=values, key=attribute_group_key(kind), points=ids
                        ),
                    )
                )

        self._client.batch_update_points(
            collection_name=self._collection,
            update_operations=operations,
            wait=True,
        )

    def _list_chunks(
        self, chunk_filter: ChunkFilter, limit: int | None
    ) -> list[IndexedChunk]:
        scroll_filter = translate_filter(chunk_filter, kind=CHUNK)
        records = _scroll_all(self._client, self._collection, scroll_filter, True)
        chunks: list[IndexedChunk] = []
        for record in records:
            assert record.payload is not None
            chunks.append(chunk_from_payload(record.payload))
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.position))
        return chunks if limit is None else chunks[:limit]

    def _search(self, query: SearchQuery) -> list[ChunkHit]:
        pass


def _scroll_all(
    client: QdrantClient,
    collection: str,
    scroll_filter: models.Filter,
    with_payload: models.WithPayloadInterface,
) -> list[models.Record]:
    records: list[models.Record] = []
    offset: models.ExtendedPointId | None = None
    while True:
        page, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            return records


def _dict_to_vector_spec(d: dict[str, Any]) -> VectorSpec:
    kind = d.get("kind")
    if kind == "dense":
        return DenseVectorSpec(dimension=d["dimension"])
    elif kind == "sparse":
        return SparseVectorSpec(needs_idf=d["needs_idf"])
    raise ValueError(f"unknown vector spec kind {kind}")


def _vector_spec_to_dict(spec: VectorSpec) -> dict[str, Any]:
    match spec:
        case DenseVectorSpec(dimension=dimension):
            return {"kind": "dense", "dimension": dimension}
        case SparseVectorSpec(needs_idf=needs_idf):
            return {"kind": "sparse", "needs_idf": needs_idf}
        case _:
            assert_never(spec)


def _create_or_verify_collection(
    *,
    client: QdrantClient,
    collection: str,
    embedder_identity: str,
    vector_specs: Mapping[str, VectorSpec],
) -> None:
    if not client.collection_exists(collection_name=collection):
        vectors_config: dict[str, models.VectorParams] = {}
        sparse_vectors_config: dict[str, models.SparseVectorParams] = {}

        for vector_name, vector_spec in vector_specs.items():
            match vector_spec:
                case DenseVectorSpec(dimension=dimension):
                    vectors_config[vector_name] = models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    )
                case SparseVectorSpec(needs_idf=needs_idf):
                    sparse_vectors_config[vector_name] = models.SparseVectorParams(
                        modifier=models.Modifier.IDF if needs_idf else None
                    )
                case _:
                    assert_never(vector_spec)

        client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            metadata={
                _EMBEDDER_IDENTITY: embedder_identity,
                _VECTOR_SPECS: {
                    name: _vector_spec_to_dict(spec)
                    for name, spec in vector_specs.items()
                },
            },
        )
    else:
        collection_info = client.get_collection(collection_name=collection)
        collection_metadata = collection_info.config.metadata or {}
        collection_identity: str | None = collection_metadata.get(
            _EMBEDDER_IDENTITY, None
        )

        if collection_identity is None:
            raise ValueError(
                f"collection {collection!r}: no embedder identity found in metadata, wasn't created by QdrantChunkIndex"
            )

        if collection_identity != embedder_identity:
            raise ValueError(
                f"collection {collection!r} holds vectors from {collection_identity!r}, not {embedder_identity!r}"
            )

        collection_vector_specs: dict[str, Any] | None = collection_metadata.get(
            _VECTOR_SPECS, None
        )
        if collection_vector_specs is None:
            raise ValueError(
                f"collection {collection!r}: no vector specs found in metadata, wasn't created by QdrantChunkIndex"
            )
        if (
            col_specs := {
                name: _dict_to_vector_spec(spec_dict)
                for name, spec_dict in collection_vector_specs.items()
            }
        ) != vector_specs:
            raise ValueError(
                f"collection {collection!r} holds vector specs {col_specs}, not {dict(vector_specs)}"
            )


def _create_payload_indexes(
    client: QdrantClient,
    collection: str,
    indexes: Sequence[_PayloadIndex],
) -> None:
    for field, schema in indexes:
        client.create_payload_index(
            collection_name=collection, field_name=field, field_schema=schema, wait=True
        )


def _payload_indexes(
    indexed_attributes: Mapping[str, AttributeKind],
) -> list[_PayloadIndex]:
    indexes: list[_PayloadIndex] = [
        (field, models.PayloadSchemaType.KEYWORD) for field in _KEYWORD_FIELDS
    ]
    for name, kind in indexed_attributes.items():
        check_attribute_name(name)
        indexes.extend(
            [
                (attribute_key(storage_kind, name), schema_type)
                for storage_kind, schema_type in _SCHEMAS[kind]
            ]
        )
    return indexes
