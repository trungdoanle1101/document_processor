"""How a filter becomes a Qdrant filter over the stored payload.

Each node becomes a Qdrant `Filter`: `And` its `must`, `Or` its `should`, `Not` its
`must_not`. A condition on a `Many` matches when any of its values does, which is what
`AnyOf` means. Two cases don't map one to one:

- Floats and datetimes are matched by a range from the value to itself: Qdrant can't
  match a float exactly, and a datetime stored as a string would miss the same instant
  written with another offset.
- `AnyOf` with no values matches nothing. It can't become an empty `should`: Qdrant
  reads a filter with no conditions as no constraint and matches every chunk.

The fields aren't checked against a schema here; the index does that first.
"""

import uuid
from datetime import datetime
from typing import Any, assert_never

from qdrant_client import models

from salemdex_retrieval.filters import And, AnyOf, Equals, Filter, Not, Or
from salemdex_retrieval.index.qdrant.payload import FIELD_VALUES
from salemdex_retrieval.schema import Field

# Not everything: a filter with no conditions matches every chunk, so negate it.
_MATCHES_NOTHING = models.Filter(must_not=[models.Filter()])


def build_qdrant_filter(where: Filter) -> models.Filter:
    """Return the Qdrant filter matching the chunks `where` matches.

    Args:
        where: The filter to convert, its fields already checked against the schema.

    Returns:
        A filter over payloads laid out as `encode_payload` writes them.
    """
    match where:
        case Equals(field=field, value=value):
            return models.Filter(must=[_build_equals_condition(field, value)])
        case AnyOf():
            return _build_any_of(where)
        case And(filters=filters):
            return models.Filter(must=[build_qdrant_filter(f) for f in filters])
        case Or(filters=filters):
            return models.Filter(should=[build_qdrant_filter(f) for f in filters])
        case Not(filter=f):
            return models.Filter(must_not=[build_qdrant_filter(f)])
        case _:
            assert_never(where)


def _build_any_of(where: AnyOf[Any]) -> models.Filter:
    """Return a filter matching chunks with any of the values, or nothing if none."""
    if not where.values:
        return _MATCHES_NOTHING
    conditions = [_build_equals_condition(where.field, v) for v in where.values]
    return models.Filter(should=conditions)


def _build_equals_condition(field: Field[Any], value: Any) -> models.FieldCondition:
    """Return a condition matching a stored value of `field` equal to `value`.

    Raises:
        TypeError: The field's value type has no condition here.
    """
    key = f"{FIELD_VALUES}.{field.name}"
    value_type = field.value_type
    if value_type is uuid.UUID:
        return models.FieldCondition(key=key, match=models.MatchValue(value=str(value)))
    if value_type in (str, int, bool):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))
    if value_type is float:
        return models.FieldCondition(key=key, range=models.Range(gte=value, lte=value))
    if value_type is datetime:
        return models.FieldCondition(
            key=key, range=models.DatetimeRange(gte=value, lte=value)
        )
    raise TypeError(f"no Qdrant condition for value type {value_type.__name__}")
