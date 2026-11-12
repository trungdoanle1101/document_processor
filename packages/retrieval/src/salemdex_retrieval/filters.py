"""Filters choosing which chunks a search or a write applies to.

Build them from a schema's fields (`One.eq`, `Many.any_of`), which check the values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from salemdex_retrieval.schema import Many, One

# Any filter, whatever its field's type: what searches, writes and backends accept.
type Filter = Equals[Any] | AnyOf[Any]


@dataclass(frozen=True, kw_only=True)
class Equals[T]:
    """A filter matching chunks whose value for a `One` field equals `value`.

    Build it with `One.eq`, which checks the value against the field.

    Attributes:
        field: The field to compare.
        value: The value to match.
    """

    field: One[T]
    value: T


@dataclass(frozen=True, kw_only=True)
class AnyOf[T]:
    """A filter matching chunks holding at least one of `values` in a `Many` field.

    Build it with `Many.any_of`, which checks each value against the field.

    Attributes:
        field: The field to look in.
        values: The values to match; none matches no chunk.
    """

    field: Many[T]
    values: frozenset[T]
