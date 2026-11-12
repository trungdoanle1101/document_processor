"""The typed fields a chunk holds, and the checked values written for them.

A schema subclasses `Schema` and declares `One` and `Many` fields. Each field checks
values where they enter: for writes (`value`) and for filters (`eq`, `any_of`).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final

from salemdex_retrieval.filters import AnyOf, Equals

_VALUE_TYPES = (str, int, float, bool, uuid.UUID, datetime)

# Backends store integers in 64 bits (Qdrant: int64); Python's int has no limit.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class SchemaError(Exception):
    """Raised when a schema or a field is declared against the rules."""


class Field[T]:
    """A typed, named field of a schema.

    `One` and `Many` add what differs by kind: how a chunk's value is given, and which
    filters apply. A field takes its name from the schema class body it's declared in.

    Every value given to a field, for a write or a filter, is checked: its type must be
    exactly `value_type` (an `int` is widened for a `float` field), a `datetime` must
    have a timezone, a `float` must be finite and an `int` must fit in 64 bits.

    Attributes:
        value_type: The type every value must have exactly.
        indexed: Whether backends build an index on the field for filtering.
    """

    def __init__(self, value_type: type[T], *, indexed: bool = False) -> None:
        """Declare a field whose values are of `value_type`.

        Args:
            value_type: One of `str`, `int`, `float`, `bool`, `UUID`, `datetime`.
                Subclasses, such as `bool` for `int` or a `StrEnum` for `str`, are
                refused.
            indexed: Whether backends build an index on the field for filtering.

        Raises:
            TypeError: `value_type` isn't one of the supported types.
        """
        if value_type not in _VALUE_TYPES:
            supported_types = ", ".join(t.__name__ for t in _VALUE_TYPES)
            raise TypeError(
                f"type {value_type!r} is not supported; expected {supported_types}"
            )
        self.value_type: Final = value_type
        self.indexed: Final = indexed
        self._name: str | None = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Take the attribute's name as the field's name.

        Raises:
            SchemaError: The field already has a name, in this class or another.
        """
        if self._name is not None:
            raise SchemaError(
                f"this {type(self).__name__}(...) is already the field {self._name!r}; "
                f"declare a new {type(self).__name__}(...) for {name!r} "
                f"in {owner.__name__}"
            )
        self._name = name

    def __repr__(self) -> str:
        type_name = type(self).__name__
        arguments = (
            f"{self.value_type.__name__}, name={self._name!r}, indexed={self.indexed}"
        )
        return f"{type_name}({arguments})"

    @property
    def name(self) -> str:
        """The name the field was declared under.

        Raises:
            SchemaError: The field wasn't declared in a class body.
        """
        if self._name is None:
            declaration = f"{type(self).__name__}({self.value_type.__name__})"
            raise SchemaError(
                f"field has no name; declare it in a schema class body, "
                f"e.g. my_field = {declaration}"
            )
        return self._name

    def _check(self, value: object, *, position: int | None = None) -> T:
        """Return `value` checked against this field, an `int` widened for a `float`.

        Args:
            value: The value to check.
            position: The value's position within a `Many`, named in errors.

        Returns:
            The value, as a `float` when an `int` was given for a `float` field.

        Raises:
            TypeError: The value isn't exactly of the field's type, or is a
                `datetime` without a timezone.
            ValueError: A `float` isn't finite, or an `int` is outside 64 bits.
        """
        if type(value) is int and self.value_type is float:
            try:
                value = float(value)
            except OverflowError:
                raise ValueError(
                    f"{self._label(position)} is an int too large for a float; "
                    f"expected a finite number"
                ) from None
        if type(value) is not self.value_type:
            raise TypeError(
                f"{self._label(position)} is {type(value).__name__!r}, "
                f"expected {self.value_type.__name__}"
            )
        if isinstance(value, int) and not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(
                f"{self._label(position)} is outside 64 bits; "
                f"expected {_INT64_MIN} to {_INT64_MAX}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"{self._label(position)} is {value!r}; expected a finite number"
            )

        if isinstance(value, datetime) and value.utcoffset() is None:
            raise TypeError(f"{self._label(position)} is missing a timezone")
        return value

    def _label(self, position: int | None) -> str:
        """Return the field's name for messages, with `[position]` within a `Many`."""
        return self.name + (f"[{position}]" if position is not None else "")


class One[T](Field[T]):
    """A field holding exactly one value per chunk."""

    def value(self, value: T) -> OneValue[T]:
        """Return a chunk's value for this field.

        Args:
            value: The chunk's value.

        Returns:
            The checked value paired with this field, for `Chunk(field_values=...)`.

        Raises:
            TypeError: The value is of the wrong type (see `Field`).
            ValueError: The value can't be stored (see `Field`).
        """
        return OneValue(field=self, value=self._check(value))

    def eq(self, value: T) -> Equals[T]:
        """Return a filter matching chunks whose value for this field equals `value`.

        Args:
            value: The value to match.

        Returns:
            An `Equals` filter.

        Raises:
            TypeError: The value is of the wrong type (see `Field`).
            ValueError: The value can't be stored (see `Field`).
        """
        return Equals(field=self, value=self._check(value))


class Many[T](Field[T]):
    """A field holding any number of values per chunk, possibly none."""

    def value(self, values: Iterable[T]) -> ManyValues[T]:
        """Return a chunk's values for this field.

        Order and repeats are kept. No values is allowed, and is how a chunk says it
        has none.

        Args:
            values: The chunk's values, in any iterable but a `str`, `bytes` or
                mapping.

        Returns:
            The checked values, as a tuple paired with this field, for
            `Chunk(field_values=...)`.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping, or a value is of the
                wrong type (see `Field`).
            ValueError: A value can't be stored (see `Field`).
        """
        self._check_iterable(values, owner="value(...)")

        return ManyValues(
            field=self,
            values=tuple(
                self._check(value, position=i) for i, value in enumerate(values)
            ),
        )

    def any_of(self, values: Iterable[T]) -> AnyOf[T]:
        """Return a filter matching chunks with at least one value in `values`.

        Args:
            values: The values to match, in any iterable but a `str`, `bytes` or
                mapping. Order and repeats don't matter; no values matches no chunk.

        Returns:
            An `AnyOf` filter.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping, or a value is of the
                wrong type (see `Field`).
            ValueError: A value can't be stored (see `Field`).
        """
        self._check_iterable(values, owner="any_of(...)")

        return AnyOf(
            field=self,
            values=frozenset(
                self._check(value, position=i) for i, value in enumerate(values)
            ),
        )

    def _check_iterable(self, values: Iterable[T], *, owner: str) -> None:
        """Refuse an iterable that would be taken apart into characters, ints or keys.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping.
        """
        if isinstance(values, (str, bytes, Mapping)):
            message = (
                f"{owner} needs a collection of values, not {type(values).__name__!r}"
            )
            raise TypeError(message)


@dataclass(frozen=True, kw_only=True)
class OneValue[T]:
    """A chunk's value for a `One` field, built by `One.value`.

    Attributes:
        field: The field the value is for.
        value: The value, checked against the field.
    """

    field: One[T]
    value: T


@dataclass(frozen=True, kw_only=True)
class ManyValues[T]:
    """A chunk's values for a `Many` field, built by `Many.value`.

    Attributes:
        field: The field the values are for.
        values: The values, each checked against the field, in order; possibly none.
    """

    field: Many[T]
    values: tuple[T, ...]


# A chunk holds values for fields of different types, so T is Any here; a backend
# matches on the two classes to tell one value from several.
type FieldValue = OneValue[Any] | ManyValues[Any]


class Schema:
    """Base class for a schema: the fields every chunk of an index holds.

    Declare `One` and `Many` fields in the class body. Fields come only from the class
    body and from `Schema` bases. A field in a plain mixin isn't part of the schema, so
    shared fields belong in a `Schema` preset.

    Example:
        class KbChunks(Schema):
            document_id = One(UUID, indexed=True)
            headings = Many(str)
            groups = Many(UUID, indexed=True)
    """

    _fields: Mapping[str, Field[Any]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect the subclass's fields, inherited ones first.

        Raises:
            SchemaError: Two bases give different fields one name, the class body
                hides an inherited field, or a field takes a name `Schema` uses.
        """
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field[Any]] = {}
        for base in cls.__bases__:
            if not issubclass(base, Schema):
                continue
            for name, field in base._fields.items():
                if name in fields and fields[name] is not field:
                    raise SchemaError(
                        f"{cls.__name__} inherits two different fields named {name!r}; "
                        "rename one in its preset"
                    )
                fields[name] = field

        for name, value in vars(cls).items():
            if name in fields:
                raise SchemaError(
                    f"{cls.__name__}.{name} hides an inherited field; "
                    "inherited fields can't be redefined or hidden"
                )
            if isinstance(value, Field):
                if name in dir(Schema):
                    raise SchemaError(f"{name!r} is a reserved name; use another name")
                fields[name] = value

        cls._fields = MappingProxyType(fields)

    @classmethod
    def fields(cls) -> Mapping[str, Field[Any]]:
        """Return the schema's fields by name.

        Returns:
            A read-only mapping, inherited fields first, each in declaration order.
        """
        return cls._fields
