from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from salemdex_retrieval.schema import Many, One

type Filter = Equals | AnyOf


@dataclass(frozen=True, kw_only=True)
class Equals:
    field: One[Any]
    value: object


@dataclass(frozen=True, kw_only=True)
class AnyOf:
    field: Many[Any]
    values: frozenset[object]
