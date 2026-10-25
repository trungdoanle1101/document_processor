from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


class Registry[T]:
    def __init__(self, kind: str, items: Mapping[str, T]) -> None:
        self._kind = kind
        self._items = MappingProxyType(dict(items))

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self._kind} {name!r}; known: {', '.join(sorted(self._items))}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def extend(self, items: Mapping[str, T]) -> Registry[T]:
        clashes = sorted(self._items.keys() & items.keys())
        if clashes:
            raise ValueError(f"{self._kind} already defined: {', '.join(clashes)}")
        return Registry(kind=self._kind, items={**self._items, **items})

    def replace(self, items: Mapping[str, T]) -> Registry[T]:
        """Swapping existing entries. Every name must already be registered"""
        unknown = sorted(items.keys() - self._items.keys())
        if unknown:
            raise KeyError(f"unknown {self._kind}: {', '.join(unknown)}")
        return Registry(kind=self._kind, items={**self._items, **items})

    def __contains__(self, name: str) -> bool:
        return name in self._items
