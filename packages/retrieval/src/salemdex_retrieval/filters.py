"""Filters: which chunks a search or a write applies to, built from a schema's fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from salemdex_retrieval.schema import Many, One

# Any filter, whatever its field's type: what searches, writes and backends accept.
type Filter = Equals[Any] | AnyOf[Any]


@dataclass(frozen=True, kw_only=True)
class Equals[T]:
    """Matches chunks whose value for a One field equals `value`.

    Built by `One.eq`, which checks the value against the field's type.
    """

    field: One[T]
    value: T


@dataclass(frozen=True, kw_only=True)
class AnyOf[T]:
    """Matches chunks holding at least one of `values` in a Many field; none matches nothing.

    Built by `Many.any_of`, which checks each value against the field's type.
    """

    field: Many[T]
    values: frozenset[T]
