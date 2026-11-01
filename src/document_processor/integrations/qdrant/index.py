from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import MappingProxyType
from typing import Any, Literal, assert_never

from qdrant_client import QdrantClient, models

from document_processor.retrieval.attributes import (
    AllOf,
    AnyOf,
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
    SparseVectorSpec,
    VectorSpec,
)
from document_processor.retrieval.index import ChunkIndex

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

_KEYWORD_FIELDS = ("kind", "chunk_key", "document_id", "generator", "reference_keys")


_SCHEMAS: Mapping[AttributeKind, tuple[tuple[str, models.PayloadSchemaType], ...]] = {
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
                (f"attributes.{storage_kind}.{name}", schema_type)
                for storage_kind, schema_type in _SCHEMAS[kind]
            ]
        )
    return indexes
