import json
import uuid
from datetime import date

import pytest
from qdrant_client import QdrantClient, models

from document_processor.integrations.qdrant._payload import (
    chunk_from_payload,
    chunk_payload,
    chunk_point_id,
    decode_attributes,
    encode_attributes,
    encode_chunk_key,
    entry_copy_payload,
    entry_from_payload,
    entry_point_id,
    point_vectors,
)
from document_processor.retrieval.attributes import Attributes
from document_processor.retrieval.embedding import DenseVector, Embedding, SparseVector
from document_processor.retrieval.index import (
    ChunkKey,
    GeneratedEntry,
    GeneratedEntryKey,
    IndexedChunk,
)

EVERY_KIND: Attributes = {
    "title": "Luật Đất đai",
    "year": 2024,
    "weight": 2.0,
    "in_force": True,
    "repealed": False,
    "issued_on": date(2024, 1, 18),
    "knowledge_base_ids": ("kb-2", "kb-1"),
    "pages": (3, 1),
    "tags": (),
}


CHUNK = IndexedChunk(
    document_id="luat-dat-dai",
    chunk_id="dieu-14",
    position=3,
    text="Điều 14. Phạm vi điều chỉnh",
    embedded_text="Chương II\nĐiều 14. Phạm vi điều chỉnh",
    headings=("Chương II", "Điều 14"),
    reference_keys=("chuong:2/dieu:14", "dieu:14"),
    attributes=EVERY_KIND,
)
OTHER_CHUNK = IndexedChunk(
    document_id="luat-dat-dai",
    chunk_id="dieu-15",
    position=4,
    text="Điều 15.",
    embedded_text="Điều 15.",
    reference_keys=("dieu:15",),
    attributes={"role": "main"},
)
ENTRY = GeneratedEntry(
    document_id="luat-dat-dai",
    generator="questions-v1",
    entry_id="q1",
    chunk_ids=("dieu-14", "dieu-15"),
    text="Điều kiện chuyển nhượng là gì?",
)


def through_json(stored: object) -> object:
    """What comes back from Qdrant: the payload travels as JSON."""
    return json.loads(json.dumps(stored))


# Attributes -------------------------------------------------------------------------------


def test_attributes_come_back_as_stored() -> None:
    back = decode_attributes(encode_attributes(EVERY_KIND))
    assert back == EVERY_KIND


def test_attributes_come_back_as_stored_after_travelling_as_json() -> None:
    back = decode_attributes(through_json(encode_attributes(EVERY_KIND)))  # type: ignore[arg-type]
    assert back == EVERY_KIND
    assert type(back["weight"]) is float
    assert type(back["in_force"]) is bool
    assert type(back["issued_on"]) is date
    assert back["knowledge_base_ids"] == ("kb-2", "kb-1")


def test_each_value_is_stored_under_its_kind() -> None:
    stored = encode_attributes(EVERY_KIND)
    assert stored["text"] == {
        "title": "Luật Đất đai",
        "knowledge_base_ids": ["kb-2", "kb-1"],
    }
    assert stored["integer"] == {"year": 2024, "pages": [3, 1]}
    assert stored["float"] == {"weight": 2.0}
    assert stored["boolean"] == {"in_force": True, "repealed": False}
    assert stored["date"] == {"issued_on": "2024-01-18T00:00:00Z"}
    assert stored["empty"] == {"tags": []}


def test_booleans_are_never_stored_as_integers() -> None:
    stored = encode_attributes({"flag": True, "count": 1})
    assert stored["boolean"] == {"flag": True}
    assert stored["integer"] == {"count": 1}


def test_every_kind_group_is_always_present() -> None:
    """So a later set_payload can always write into attributes.<kind>."""
    assert set(encode_attributes({})) == {
        "text",
        "integer",
        "float",
        "boolean",
        "date",
        "empty",
    }


def test_text_is_never_split_into_characters() -> None:
    assert decode_attributes({"text": {"title": "Luật"}}) == {"title": "Luật"}


def test_a_float_stored_as_a_whole_number_comes_back_as_a_float() -> None:
    back = decode_attributes({"float": {"weight": 2}})
    assert back == {"weight": 2.0}
    assert type(back["weight"]) is float


def test_an_empty_list_comes_back_as_an_empty_tuple() -> None:
    assert decode_attributes({"empty": {"tags": []}}) == {"tags": ()}


def test_no_attributes_round_trip_to_none() -> None:
    assert decode_attributes(encode_attributes({})) == {}


# Keys and point ids -----------------------------------------------------------------------


def test_chunk_keys_that_look_alike_when_joined_stay_distinct() -> None:
    first = encode_chunk_key(ChunkKey(document_id="a/b", chunk_id="c"))
    second = encode_chunk_key(ChunkKey(document_id="a", chunk_id="b/c"))
    assert first != second


def test_a_chunk_key_keeps_vietnamese_readable() -> None:
    assert (
        encode_chunk_key(ChunkKey(document_id="luật", chunk_id="điều-14"))
        == '["luật", "điều-14"]'
    )


def test_point_ids_are_uuids() -> None:
    uuid.UUID(chunk_point_id(ChunkKey(document_id="law", chunk_id="a")))
    uuid.UUID(
        entry_point_id(
            GeneratedEntryKey(document_id="law", generator="g", entry_id="q"), "a"
        )
    )


def test_the_same_key_always_gives_the_same_id() -> None:
    key = ChunkKey(document_id="law", chunk_id="a")
    assert chunk_point_id(key) == chunk_point_id(
        ChunkKey(document_id="law", chunk_id="a")
    )
    entry = GeneratedEntryKey(document_id="law", generator="g", entry_id="q")
    assert entry_point_id(entry, "a") == entry_point_id(entry, "a")


def test_a_chunk_and_an_entry_copy_never_share_an_id() -> None:
    chunk = chunk_point_id(ChunkKey(document_id="law", chunk_id="a"))
    copy = entry_point_id(
        GeneratedEntryKey(document_id="law", generator="chunk", entry_id="a"), "a"
    )
    assert chunk != copy


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("law", "g", "q", "a"), ("law", "g", "q", "b")),
        (("law", "g", "q", "a"), ("law", "h", "q", "a")),
        (("law", "g", "q", "a"), ("law", "g", "r", "a")),
        (("law", "g", "q", "a"), ("other", "g", "q", "a")),
    ],
    ids=["chunk", "generator", "entry", "document"],
)
def test_entry_copies_differing_in_any_part_have_different_ids(
    first: tuple[str, str, str, str], second: tuple[str, str, str, str]
) -> None:
    def copy_id(parts: tuple[str, str, str, str]) -> str:
        document_id, generator, entry_id, chunk_id = parts
        return entry_point_id(
            GeneratedEntryKey(
                document_id=document_id, generator=generator, entry_id=entry_id
            ),
            chunk_id,
        )

    assert copy_id(first) != copy_id(second)


# Chunks and entry copies ------------------------------------------------------------------


def test_a_chunk_comes_back_as_stored() -> None:
    assert chunk_from_payload(chunk_payload(CHUNK)) == CHUNK


def test_a_chunk_comes_back_as_stored_after_travelling_as_json() -> None:
    back = chunk_from_payload(through_json(chunk_payload(CHUNK)))  # type: ignore[arg-type]
    assert back == CHUNK
    assert type(back.headings) is tuple
    assert type(back.reference_keys) is tuple


def test_a_chunk_payload_carries_its_kind_and_encoded_key() -> None:
    payload = chunk_payload(CHUNK)
    assert payload["kind"] == "chunk"
    assert payload["chunk_key"] == encode_chunk_key(CHUNK.key)


def test_an_entry_copy_carries_its_chunks_routing_fields() -> None:
    payload = entry_copy_payload(ENTRY, OTHER_CHUNK)
    assert payload["kind"] == "entry"
    assert payload["chunk_key"] == encode_chunk_key(OTHER_CHUNK.key)
    assert payload["chunk_id"] == "dieu-15"
    assert payload["reference_keys"] == ["dieu:15"]
    assert decode_attributes(payload["attributes"]) == {"role": "main"}


def test_every_copy_of_an_entry_decodes_to_the_same_entry() -> None:
    for chunk in (CHUNK, OTHER_CHUNK):
        payload = through_json(entry_copy_payload(ENTRY, chunk))
        assert entry_from_payload(payload) == ENTRY  # type: ignore[arg-type]


# Vectors ----------------------------------------------------------------------------------


def test_point_vectors_name_each_vector() -> None:
    embedding = Embedding(
        embedder_identity="stub",
        vectors={
            "dense": DenseVector(values=(0.5, -1.0)),
            "sparse": SparseVector(indices=(3, 7), values=(1.0, 2.0)),
        },
    )
    vectors = point_vectors(embedding)
    assert vectors["dense"] == [0.5, -1.0]
    assert vectors["sparse"] == models.SparseVector(indices=[3, 7], values=[1.0, 2.0])


# Through Qdrant itself --------------------------------------------------------------------


def test_a_chunk_comes_back_as_stored_from_qdrant() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "space",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    embedding = Embedding(
        embedder_identity="stub",
        vectors={
            "dense": DenseVector(values=(1.0, 0.0)),
            "sparse": SparseVector(indices=(3,), values=(1.0,)),
        },
    )
    point_id = chunk_point_id(CHUNK.key)
    client.upsert(
        "space",
        [
            models.PointStruct(
                id=point_id,
                vector=point_vectors(embedding),
                payload=chunk_payload(CHUNK),
            )
        ],
        wait=True,
    )
    (point,) = client.retrieve("space", ids=[point_id], with_payload=True)
    assert point.payload is not None
    assert chunk_from_payload(point.payload) == CHUNK
