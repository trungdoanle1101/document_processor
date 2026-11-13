"""How a chunk is stored as a Qdrant payload and read back. No server needed."""

import copy
import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk, SchemaMismatchError
from salemdex_retrieval.index.qdrant.payload import decode_payload, encode_payload
from salemdex_retrieval.schema import Many, One, Schema


class Chunks(Schema):
    document_id = One(uuid.UUID)
    version = One(int)
    score = One(float)
    is_draft = One(bool)
    indexed_at = One(datetime)
    title = One(str)
    headings = Many(str)
    groups = Many(uuid.UUID)
    seen_at = Many(datetime)


class Entries(Schema):
    document_id = One(uuid.UUID)


CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
DOCUMENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")
NEW_YEAR = datetime(2026, 1, 1, tzinfo=UTC)
HANOI = timezone(timedelta(hours=7))


def build(
    *, headings: list[str] | None = None, embedded_text: str | None = None
) -> Chunk:
    """Return a chunk of Chunks holding a value of every supported type."""
    return Chunk(
        id=CHUNK_ID,
        text="Article 1. Scope",
        embedded_text=embedded_text,
        field_values=[
            Chunks.document_id.value(DOCUMENT_ID),
            Chunks.version.value(2),
            Chunks.score.value(0.5),
            Chunks.is_draft.value(False),
            Chunks.indexed_at.value(NEW_YEAR),
            Chunks.title.value("Scope"),
            Chunks.headings.value(
                ["Part 1", "Scope"] if headings is None else headings
            ),
            Chunks.groups.value([HR]),
            Chunks.seen_at.value([datetime(2026, 1, 2, 9, 30, tzinfo=HANOI)]),
        ],
    )


def stored(chunk: Chunk) -> dict[str, Any]:
    """Return the payload as Qdrant gives it back: plain JSON, nothing shared."""
    loaded: dict[str, Any] = json.loads(json.dumps(encode_payload(chunk)))
    return loaded


def read(payload: dict[str, Any], *, point_id: str = str(CHUNK_ID)) -> Chunk:
    return decode_payload(point_id=point_id, payload=payload, schema=Chunks)


# Writing


def test_the_payload_has_the_stored_layout() -> None:
    # Changing this layout breaks existing indexes: re-index them, and update the
    # layout in docs/graph-workflow-design.md ("The Qdrant payload layout").
    assert encode_payload(build(embedded_text="Part 1\nArticle 1. Scope")) == {
        "text": "Article 1. Scope",
        "embedded_text": "Part 1\nArticle 1. Scope",
        "field_values": {
            "document_id": "00000000-0000-0000-0000-000000000001",
            "version": 2,
            "score": 0.5,
            "is_draft": False,
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "title": "Scope",
            "headings": ["Part 1", "Scope"],
            "groups": ["00000000-0000-0000-0000-00000000000a"],
            "seen_at": ["2026-01-02T09:30:00+07:00"],
        },
    }


def test_the_payload_is_plain_json() -> None:
    payload = encode_payload(build())

    assert json.loads(json.dumps(payload)) == payload


def test_the_embedded_text_is_stored_even_when_it_is_the_text() -> None:
    payload = encode_payload(build(embedded_text=None))

    assert payload["embedded_text"] == payload["text"]


def test_a_many_with_no_values_is_stored_as_an_empty_list() -> None:
    assert encode_payload(build(headings=[]))["field_values"]["headings"] == []


# Reading back


def test_a_chunk_reads_back_equal_to_the_one_written() -> None:
    chunk = build(embedded_text="Part 1\nArticle 1. Scope")

    assert read(stored(chunk)) == chunk


def test_values_read_back_with_their_types() -> None:
    values = {f.name: v for f, v in read(stored(build())).field_values.items()}

    assert type(values["document_id"]) is uuid.UUID
    assert type(values["indexed_at"]) is datetime
    assert values["seen_at"][0].utcoffset() == timedelta(hours=7)
    assert type(values["is_draft"]) is bool
    assert values["headings"] == ("Part 1", "Scope")


def test_a_many_with_no_values_reads_back_empty() -> None:
    chunk = build(headings=[])

    assert read(stored(chunk)) == chunk


def test_a_float_stored_as_a_whole_number_reads_back_as_a_float() -> None:
    payload = stored(build())
    payload["field_values"]["score"] = 1

    values = {f.name: v for f, v in read(payload).field_values.items()}

    assert type(values["score"]) is float


# Reading a damaged payload


def damage(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    payload = copy.deepcopy(stored(build()))
    change(payload)
    return payload


def set_field(key: str, raw: Any) -> Callable[[dict[str, Any]], None]:
    return lambda payload: payload["field_values"].__setitem__(key, raw)


DAMAGED = [
    pytest.param(lambda p: p.pop("text"), "missing text", id="no text"),
    pytest.param(
        lambda p: p.pop("embedded_text"), "missing embedded_text", id="no embedded text"
    ),
    pytest.param(
        lambda p: p.pop("field_values"), "missing field_values", id="no field values"
    ),
    pytest.param(
        lambda p: p.__setitem__("text", 5), "text is not a string", id="text not str"
    ),
    pytest.param(
        lambda p: p.__setitem__("embedded_text", None),
        "embedded_text is not a string",
        id="embedded text None",
    ),
    pytest.param(
        lambda p: p.__setitem__("field_values", "abc"),
        "field_values is not a mapping",
        id="field values a string",
    ),
    pytest.param(
        lambda p: p["field_values"].__setitem__("extra", 1),
        "keys not in schema Chunks: extra",
        id="unknown key",
    ),
    pytest.param(
        lambda p: p["field_values"].pop("version"), "missing: version", id="no value"
    ),
    pytest.param(
        set_field("headings", "Part 1"), "headings: expected a list", id="many as str"
    ),
    pytest.param(
        set_field("document_id", "not-a-uuid"), "document_id:", id="malformed uuid"
    ),
    pytest.param(
        set_field("document_id", 7),
        "document_id: expected a string, got int",
        id="uuid as int",
    ),
    pytest.param(set_field("groups", ["x"]), "groups:", id="malformed uuid in many"),
    pytest.param(set_field("version", "2"), "version:", id="int as str"),
    pytest.param(set_field("version", True), "version:", id="bool for int"),
    pytest.param(set_field("title", None), "title:", id="null value"),
    pytest.param(
        set_field("indexed_at", "2026-01-01T00:00:00"),
        "indexed_at: indexed_at is missing a timezone",
        id="naive datetime",
    ),
    pytest.param(
        set_field("indexed_at", 1767225600),
        "indexed_at: expected a string, got int",
        id="datetime int",
    ),
]


@pytest.mark.parametrize(("change", "message"), DAMAGED)
def test_a_damaged_payload_raises_one_error_naming_the_point(
    change: Callable[[dict[str, Any]], None], message: str
) -> None:
    with pytest.raises(SchemaMismatchError, match=re.escape(message)) as raised:
        read(damage(change))

    assert str(CHUNK_ID) in str(raised.value)


def test_a_malformed_point_id_raises_naming_it() -> None:
    with pytest.raises(SchemaMismatchError, match="point not-a-uuid: id is not"):
        read(stored(build()), point_id="not-a-uuid")


def test_a_point_of_another_schema_does_not_read_as_this_one() -> None:
    entry = Chunk(
        id=CHUNK_ID,
        text="Scope",
        field_values=[Entries.document_id.value(DOCUMENT_ID)],
    )

    with pytest.raises(SchemaMismatchError, match="missing"):
        read(stored(entry))
