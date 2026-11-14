"""How a chunk is stored as a Qdrant point's payload, and read back.

The layout (the chunk id is the point id, not part of the payload):

    {
        "text": "…",
        "embedded_text": "…",
        "field_values": {"document_id": "3f2a…", "version": 2, "groups": ["…"]},
        "written_fields": ["document_id:uuid:one", "groups:uuid:many", "version:int:one"],
    }

Field values sit under their own key so that a field named `text` can't overwrite the
chunk's text. A `UUID` is stored as its string, a `datetime` as ISO 8601 with its
offset, a `Many` as a list.

`written_fields` records, sorted, each field the point was written with, as
`name:type:one|many`. Qdrant can't tell a missing list from an empty one, so this is
how `QdrantIndex.open` counts the points that lack a field without reading them.

Changing any of this breaks existing indexes: re-index them.
"""

import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from qdrant_client import models

from salemdex_retrieval.chunks import Chunk, SchemaMismatchError
from salemdex_retrieval.schema import Field, FieldValue, Many, One, Schema

TEXT = "text"
EMBEDDED_TEXT = "embedded_text"
FIELD_VALUES = "field_values"
WRITTEN_FIELDS = "written_fields"

# What one stored value becomes: a UUID or a datetime is stored as a string.
type _JsonScalar = str | int | float | bool

# Each top-level key, the type its value must have, and that type as a message says it.
_PAYLOAD_KEYS: tuple[tuple[str, type, str], ...] = (
    (TEXT, str, "a string"),
    (EMBEDDED_TEXT, str, "a string"),
    (FIELD_VALUES, Mapping, "a mapping"),
    (WRITTEN_FIELDS, list, "a list"),
)

# The payload index for each value type, matching how `_encode_value` stores it: a UUID
# as its string in a `uuid` index, a datetime as ISO 8601 in a `datetime` index.
_INDEX_TYPES: Mapping[type, models.PayloadSchemaType] = {
    uuid.UUID: models.PayloadSchemaType.UUID,
    str: models.PayloadSchemaType.KEYWORD,
    bool: models.PayloadSchemaType.BOOL,
    int: models.PayloadSchemaType.INTEGER,
    float: models.PayloadSchemaType.FLOAT,
    datetime: models.PayloadSchemaType.DATETIME,
}

# The type part of each `written_fields` entry. It's stored in every point, so renaming
# one means re-indexing; it doesn't follow Python's class names for that reason.
_STORED_TYPE_NAMES: Mapping[type, str] = {
    uuid.UUID: "uuid",
    str: "str",
    bool: "bool",
    int: "int",
    float: "float",
    datetime: "datetime",
}


def encode_payload(chunk: Chunk) -> dict[str, Any]:
    """Return the payload storing `chunk`, made only of JSON values.

    Args:
        chunk: A chunk already checked against the index's schema.

    Returns:
        The text, the embedded text, the field values keyed by field name, and the
        sorted `written_fields` entries of the chunk's fields.
    """

    return {
        TEXT: chunk.text,
        EMBEDDED_TEXT: chunk.embedded_text,
        FIELD_VALUES: {
            field.name: _encode_field_value(field, value)
            for field, value in chunk.field_values.items()
        },
        WRITTEN_FIELDS: sorted(written_field(field) for field in chunk.field_values),
    }


def decode_payload(
    *,
    point_id: models.ExtendedPointId,
    payload: Mapping[str, Any],
    schema: type[Schema],
) -> Chunk:
    """Return the chunk a stored point holds, checked as strictly as a new write.

    Every value goes back through its field's `value()`, so the write-time checks run
    again, and the chunk must have exactly the schema's fields. A point that fails
    is corrupt or from another schema; it's never half-read or filled with defaults.

    Args:
        point_id: The point's id: the chunk id, as Qdrant returns it.
        payload: The point's payload, as `encode_payload` wrote it.
        schema: The schema of the index the point was read from.

    Returns:
        The chunk, equal to the one that was written.

    Raises:
        SchemaMismatchError: A key is missing or unknown, a value has the wrong
            type or form, or `written_fields` doesn't record exactly the schema's
            fields. The message names the point id and the key.
    """
    id_as_uuid = _decode_point_id(point_id)

    missing = [key for key, _, _ in _PAYLOAD_KEYS if key not in payload]
    if missing:
        raise SchemaMismatchError(
            f"point {point_id}: payload is missing {', '.join(missing)}"
        )

    for key, expected_type, described_as in _PAYLOAD_KEYS:
        if not isinstance(payload[key], expected_type):
            raise SchemaMismatchError(
                f"point {point_id}: {key} is not {described_as}, "
                f"got {type(payload[key]).__name__}"
            )

    _check_written_fields(payload[WRITTEN_FIELDS], schema=schema, point_id=id_as_uuid)

    field_values = _decode_field_values(payload, schema=schema, point_id=id_as_uuid)

    chunk = Chunk(
        id=id_as_uuid,
        text=payload[TEXT],
        embedded_text=payload[EMBEDDED_TEXT],
        field_values=field_values,
    )
    chunk.check_fits(schema)
    return chunk


def field_key(field: Field[Any]) -> str:
    """Return the payload key holding `field`'s values, as filters and indexes name it.

    Args:
        field: A field of the index's schema.

    Returns:
        The dotted key under `field_values`, e.g. `field_values.document_id`.
    """
    return f"{FIELD_VALUES}.{field.name}"


def written_field(field: Field[Any]) -> str:
    """Return the entry recording in `written_fields` that a point holds `field`.

    The entry names the field, its stored type and whether it's a `One` or a `Many`,
    so a point written before any of those changed no longer matches.

    Args:
        field: A field of the index's schema.

    Returns:
        The entry, e.g. `groups:uuid:many`.

    Raises:
        TypeError: `field` is neither a `One` nor a `Many`.
    """
    name = field.name
    stored_type = _STORED_TYPE_NAMES[field.value_type]
    match field:
        case One():
            field_type = "one"
        case Many():
            field_type = "many"
        case _:
            raise TypeError(
                f"unknown field type {type(field).__name__!r}; must be One or Many"
            )

    return f"{name}:{stored_type}:{field_type}"


def written_fields_of(schema: type[Schema]) -> set[str]:
    """Return the `written_fields` entries every point of `schema` must record.

    Args:
        schema: The schema of the index.

    Returns:
        One entry per field of `schema`, as `written_field` spells it.
    """
    return {written_field(f) for f in schema.fields().values()}


def payload_schema_type(field: Field[Any]) -> models.PayloadSchemaType:
    """Return the type of payload index that fits how `field`'s values are stored.

    A `Many` gets the same type as a `One` of its value type: Qdrant indexes each value
    of a list.

    Args:
        field: A field of the index's schema.

    Returns:
        The payload index type, e.g. `uuid` for a `UUID` field.
    """
    return _INDEX_TYPES[field.value_type]


def _check_written_fields(
    written_fields: list[object], *, schema: type[Schema], point_id: uuid.UUID
) -> None:
    """Check a point's `written_fields` record exactly the fields of `schema`.

    Raises:
        SchemaMismatchError: An entry isn't a string, or an entry is missing or
            extra; the message names the point id and every such entry.
    """
    actual: set[str] = set()
    for entry in written_fields:
        if not isinstance(entry, str):
            raise SchemaMismatchError(
                f"point {point_id}: "
                f"{WRITTEN_FIELDS} contains a non-string: {written_fields}"
            )
        actual.add(entry)
    expected = written_fields_of(schema)
    missing = expected - actual
    extra = actual - expected
    if extra or missing:
        message = f"point {point_id}: {WRITTEN_FIELDS} don't match against schema {schema.__name__}"
        if extra:
            message += "\n" + "extra: " + ", ".join(sorted(extra))
        if missing:
            message += "\n" + "missing: " + ", ".join(sorted(missing))
        raise SchemaMismatchError(message)


def _encode_field_value(
    field: Field[Any], value: Any
) -> _JsonScalar | list[_JsonScalar]:
    """Return a field's value as JSON: one value for a `One`, a list for a `Many`.

    Raises:
        TypeError: `field` is neither a `One` nor a `Many`.
    """
    match field:
        case One():
            return _encode_value(value)
        case Many():
            return [_encode_value(v) for v in value]
        case _:
            raise TypeError(
                f"unknown field type {type(field).__name__!r}; must be One or Many"
            )


def _encode_value(value: object) -> _JsonScalar:
    """Return one value as JSON: a `UUID` as its string, a `datetime` as ISO 8601.

    Raises:
        TypeError: `value` isn't of a type a field can hold.
    """
    match value:
        case uuid.UUID():
            return str(value)
        case datetime():
            return value.isoformat()
        # A bool is an int, and is stored as itself.
        case str() | int() | float():
            return value
        case _:
            raise TypeError(f"no JSON form for {type(value).__name__}")


def _decode_field_values(
    payload: Mapping[str, Any], *, schema: type[Schema], point_id: uuid.UUID
) -> list[FieldValue]:
    """Return the field values stored under `field_values`, each checked by its field.

    Raises:
        SchemaMismatchError: A key isn't one of the schema's fields, or a value can't
            be decoded; the message names the point id and the key.
    """
    field_values_dict: Mapping[str, object] = payload[FIELD_VALUES]
    schema_fields = schema.fields()
    field_values: list[FieldValue] = []
    mismatched = [key for key in field_values_dict if schema_fields.get(key) is None]
    if mismatched:
        message = (
            f"point {point_id}: keys not in schema {schema.__name__}: "
            + ", ".join(mismatched)
        )
        raise SchemaMismatchError(message)

    for key, raw in field_values_dict.items():
        try:
            field_values.append(_decode_field_value(schema_fields[key], raw))
        except (TypeError, ValueError) as error:
            raise SchemaMismatchError(f"point {point_id}: {key}: {error}") from error
    return field_values


def _decode_field_value(field: Field[Any], raw: object) -> FieldValue:
    """Return `field`'s value built from its stored JSON, through `field.value()`.

    Raises:
        TypeError: A `Many` isn't stored as a list, a value has the wrong type, or
            `field` is neither a `One` nor a `Many`.
        ValueError: A value has the right type but the wrong form.
    """
    match field:
        case One():
            return field.value(_decode_value(raw, value_type=field.value_type))

        case Many():
            if not isinstance(raw, list):
                raise TypeError(f"expected a list, got {type(raw).__name__}")
            return field.value(
                [_decode_value(value, value_type=field.value_type) for value in raw]
            )

        case _:
            raise TypeError(
                f"unknown field type {type(field).__name__!r}; must be One or Many"
            )


def _decode_value(raw: object, *, value_type: type) -> object:
    """Return one stored JSON value as `value_type`: the reverse of `_encode_value`.

    Other types come back as stored; the field's `value()` checks them.

    Raises:
        TypeError: A `UUID` or `datetime` isn't stored as a string.
        ValueError: A `UUID` or `datetime` string is malformed.
    """
    if value_type not in (uuid.UUID, datetime):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"expected a string, got {type(raw).__name__}")
    if value_type is uuid.UUID:
        return uuid.UUID(raw)
    return datetime.fromisoformat(raw)


def _decode_point_id(point_id: models.ExtendedPointId) -> uuid.UUID:
    """Return the chunk id a point id holds

    Raises:
        SchemaMismatchError: The id is an integer or a malformed UUID string
    """
    match point_id:
        case uuid.UUID():
            return point_id
        case str():
            try:
                return uuid.UUID(point_id)
            except ValueError as error:
                raise SchemaMismatchError(
                    f"point {point_id}: id is not a valid UUID"
                ) from error
        case int():
            raise SchemaMismatchError(f"point {point_id}: id is an integer, not a UUID")
