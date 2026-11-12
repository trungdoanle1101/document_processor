import uuid
from datetime import datetime
from typing import Any, Final

from salemdex_retrieval.filters import Equals, Filter

_VALUE_TYPES = (str, int, float, bool, uuid.UUID, datetime)


class SchemaError(Exception):
    """Schema definition rule violation."""


class One[T]:
    """A field holding exactly one value per chunk."""

    def __init__(self, value_type: type[T], *, indexed: bool = False) -> None:
        if value_type not in _VALUE_TYPES:
            supported_types = ", ".join(t.__name__ for t in _VALUE_TYPES)
            raise TypeError(
                f"type {value_type!r} is not supported; expected {supported_types}"
            )
        self.value_type: Final = value_type
        self.indexed: Final = indexed
        self._name: str | None = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        if self._name is not None:
            raise SchemaError(
                f"this One(...) is already the field {self._name!r}; "
                f"declare a new One(...) for {name!r} in {owner.__name__}"
            )
        self._name = name

    @property
    def name(self) -> str:
        if self._name is None:
            raise SchemaError(
                "field has no name; declare it in a schema class body, "
                "e.g. document_id = One(UUID)"
            )
        return self._name

    def eq(self, value: T) -> Filter:
        """Return a filter matching chunks whose value for this field equals `value`."""
        return Equals(field=self, value=value)
