import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final

from salemdex_retrieval.filters import AnyOf, Equals, Filter

_VALUE_TYPES = (str, int, float, bool, uuid.UUID, datetime)


class SchemaError(Exception):
    """Schema definition rule violation."""


class Field[T]:
    """A typed, named field of a schema; One and Many add the filters each kind allows."""

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
                f"this {type(self).__name__}(...) is already the field {self._name!r}; "
                f"declare a new {type(self).__name__}(...) for {name!r} "
                f"in {owner.__name__}"
            )
        self._name = name

    @property
    def name(self) -> str:
        if self._name is None:
            declaration = f"{type(self).__name__}({self.value_type.__name__})"
            raise SchemaError(
                f"field has no name; declare it in a schema class body, "
                f"e.g. my_field = {declaration}"
            )
        return self._name


class One[T](Field[T]):
    """A field holding exactly one value per chunk."""

    def eq(self, value: T) -> Filter:
        """Return a filter matching chunks whose value for this field equals `value`."""
        return Equals(field=self, value=value)


class Many[T](Field[T]):
    """A field holding any number of values per chunk, possibly none."""

    def any_of(self, values: Iterable[T]) -> Filter:
        """Return a filter matching chunks with at least one value in `values`.

        An empty `values` matches no chunk.
        """
        if isinstance(values, str):
            raise TypeError(
                f"any_of needs a collection of values, not one string; "
                f"use any_of([{values!r}])"
            )
        return AnyOf(field=self, values=frozenset(values))


class Schema:
    """Base class for a schema: declare One or Many fields in the class body."""

    _fields: Mapping[str, Field[Any]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field[Any]] = {}
        for base in cls.__bases__:
            if not issubclass(base, Schema):
                continue
            for name, field in base._fields.items():
                if name in fields and fields[name] is not field:
                    raise SchemaError(
                        f"{cls.__name__} inherits two different fields named {name!r}; rename one in its preset"
                    )
                fields[name] = field

        for name, value in vars(cls).items():
            if name in fields:
                raise SchemaError(
                    f"{cls.__name__}.{name} hides an inherited field; inherited fields can't be redefined or hidden"
                )
            if isinstance(value, Field):
                if name in dir(Schema):
                    raise SchemaError(f"{name!r} is a reserved name; use another name")
                fields[name] = value

        cls._fields = MappingProxyType(fields)

    @classmethod
    def fields(cls) -> Mapping[str, Field]:
        """Return the schema's fields by name, inherited ones first, in declaration order."""
        return cls._fields
