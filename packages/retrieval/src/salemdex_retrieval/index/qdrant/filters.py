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
from typing import assert_never

from qdrant_client import models

from salemdex_retrieval.filters import And, AnyOf, Equals, Filter, Not, Or
from salemdex_retrieval.index.qdrant.payload import field_key
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


def _build_any_of[T](where: AnyOf[T]) -> models.Filter:
    """Return a filter matching chunks with any of the values, or nothing if none."""
    if not where.values:
        return _MATCHES_NOTHING
    conditions = [_build_equals_condition(where.field, v) for v in where.values]
    return models.Filter(should=conditions)


def _build_equals_condition[T](field: Field[T], value: T) -> models.FieldCondition:
    """Return a condition matching a stored value of `field` equal to `value`.

    Raises:
        TypeError: The value's type has no condition here.
    """
    key = field_key(field)
    match value:
        case uuid.UUID():
            match_uuid = models.MatchValue(value=str(value))
            return models.FieldCondition(key=key, match=match_uuid)
        # int() also matches a bool (a subclass); MatchValue keeps it a bool, so it
        # matches the stored JSON true or false.
        case str() | int():
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))
        # A float field never holds an int: the field turns ints into floats.
        case float():
            same_float = models.Range(gte=value, lte=value)
            return models.FieldCondition(key=key, range=same_float)
        case datetime():
            same_instant = models.DatetimeRange(gte=value, lte=value)
            return models.FieldCondition(key=key, range=same_instant)
        case _:
            raise TypeError(f"no Qdrant condition for {type(value).__name__}")
