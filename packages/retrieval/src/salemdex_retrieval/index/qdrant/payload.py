"""How a chunk is stored as a Qdrant point's payload, and read back.

The layout (the chunk id is the point id, not part of the payload):

    {
        "text": "…",
        "embedded_text": "…",
        "field_values": {"document_id": "3f2a…", "version": 2, "groups": ["…"]},
    }

Field values sit under their own key so that a field named `text` can't overwrite the
chunk's text. A `UUID` is stored as its string, a `datetime` as ISO 8601 with its
offset, a `Many` as a list. Changing any of this breaks existing indexes: re-index them.
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

# What one stored value becomes: a UUID or a datetime is stored as a string.
type _JsonScalar = str | int | float | bool

# Each top-level key, the type its value must have, and that type as a message says it.
_PAYLOAD_KEYS: tuple[tuple[str, type, str], ...] = (
    (TEXT, str, "a string"),
    (EMBEDDED_TEXT, str, "a string"),
    (FIELD_VALUES, Mapping, "a mapping"),
)


def encode_payload(chunk: Chunk) -> dict[str, Any]:
    """Return the payload storing `chunk`, made only of JSON values.

    Args:
        chunk: A chunk already checked against the index's schema.

    Returns:
        The text, the embedded text and the field values, keyed by field name.
    """
    return {
        TEXT: chunk.text,
        EMBEDDED_TEXT: chunk.embedded_text,
        FIELD_VALUES: {
            field.name: _encode_field_value(field, value)
            for field, value in chunk.field_values.items()
        },
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
        SchemaMismatchError: A key is missing or unknown, or a value has the wrong
            type or form. The message names the point id and the key.
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

    field_values = _decode_field_values(payload, schema=schema, point_id=id_as_uuid)

    chunk = Chunk(
        id=id_as_uuid,
        text=payload[TEXT],
        embedded_text=payload[EMBEDDED_TEXT],
        field_values=field_values,
    )
    chunk.check_fits(schema)
    return chunk


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
