import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import final

type Scalar = str | int | float | bool | date
type AttributeValue = Scalar | tuple[str, ...] | tuple[int, ...]
type Attributes = Mapping[str, AttributeValue]

# None removes the attribute
type AttributeChanges = Mapping[str, AttributeValue | None]


class Condition(ABC):
    """What an attribute's whole value must meet; None when the chunk lacks the attribute"""

    @abstractmethod
    def match(self, value: AttributeValue | None) -> bool: ...

    @property
    def subconditions(self) -> tuple["Condition", ...]:
        """Conditions this one is built from, so an index can check it supports them all."""
        return ()


class ElementCondition(Condition):
    """Check if one Scalar value matches the condition.
    If a list is given, it matches if at least one element does.
    Missing or empty value never matches.
    """

    @abstractmethod
    def match_one(self, value: Scalar) -> bool: ...

    @final
    def match(self, value: AttributeValue | None) -> bool:
        elements = _list_elements(value)
        return bool(elements) and any(self.match_one(element) for element in elements)


# --- Condition wrapper


@dataclass(frozen=True, kw_only=True)
class EveryElement(Condition):
    """If the given value is a list, check if every element of it matches the condition.
    Missing or empty value never matches.
    """

    condition: ElementCondition

    @property
    def subconditions(self) -> tuple[Condition, ...]:
        return (self.condition,)

    def match(self, value: AttributeValue | None) -> bool:
        elements = _list_elements(value)
        return bool(elements) and all(
            self.condition.match_one(element) for element in elements
        )


# --- Condition combinators


@dataclass(frozen=True, kw_only=True)
class AllOf(Condition):
    """Require value to match all conditions."""

    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("AllOf requires at least one condition")

    @property
    def subconditions(self) -> tuple[Condition, ...]:
        return self.conditions

    def match(self, value: AttributeValue | None) -> bool:
        return all(condition.match(value) for condition in self.conditions)


@dataclass(frozen=True, kw_only=True)
class AnyOf(Condition):
    """Require value to match at least one condition."""

    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("AnyOf requires at least one condition")

    @property
    def subconditions(self) -> tuple[Condition, ...]:
        return self.conditions

    def match(self, value: AttributeValue | None) -> bool:
        return any(condition.match(value) for condition in self.conditions)


@dataclass(frozen=True, kw_only=True)
class NoneOf(Condition):
    """Require value to match none of the conditions.
    When used for filtering chunks, will match chunks without the attribute.
    """

    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("NoneOf requires at least one condition")

    @property
    def subconditions(self) -> tuple[Condition, ...]:
        return self.conditions

    def match(self, value: AttributeValue | None) -> bool:
        return all(not condition.match(value) for condition in self.conditions)


# --- Built-in conditions
@dataclass(frozen=True, kw_only=True)
class TextIn(ElementCondition):
    """Require text to be one of the possible values."""

    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("TextIn needs at least one value")

    def match_one(self, value: Scalar) -> bool:
        return isinstance(value, str) and value in self.values


@dataclass(frozen=True, kw_only=True)
class IntegerIn(ElementCondition):
    """Require integer to be one of the possible values.
    Booleans never match.
    """

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("IntegerIn needs at least one value")

    def match_one(self, value: Scalar) -> bool:
        return type(value) is int and value in self.values


@dataclass(frozen=True, kw_only=True)
class BooleanEquals(ElementCondition):
    """Require element to be the required boolean"""

    value: bool

    def match_one(self, element: Scalar) -> bool:
        return isinstance(element, bool) and element == self.value


@dataclass(frozen=True, kw_only=True)
class NumberBetween(ElementCondition):
    """Require value is an int or float and satisfies all bounds"""

    at_least: float | None = None
    at_most: float | None = None
    above: float | None = None
    below: float | None = None

    def __post_init__(self) -> None:
        attributes = (self.at_least, self.at_most, self.above, self.below)
        if all(bound is None for bound in attributes):
            raise ValueError("NumberBetween needs at least one bound")
        for attribute in attributes:
            if attribute is not None:
                check_attribute_value(attribute)

    def match_one(self, value: Scalar) -> bool:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False

        return _fall_within(
            value,
            at_least=self.at_least,
            at_most=self.at_most,
            above=self.above,
            below=self.below,
        )


@dataclass(frozen=True, kw_only=True)
class DateBetween(ElementCondition):
    """Require value is a date and satisfies all bounds"""

    at_least: date | None = None
    at_most: date | None = None
    above: date | None = None
    below: date | None = None

    def __post_init__(self) -> None:
        attributes = (self.at_least, self.at_most, self.above, self.below)
        if all(bound is None for bound in attributes):
            raise ValueError("DateBetween needs at least one bound")
        for attribute in attributes:
            if attribute is not None:
                check_attribute_value(attribute)

    def match_one(self, value: Scalar) -> bool:
        if not isinstance(value, date):
            return False

        return _fall_within(
            value,
            at_least=self.at_least,
            at_most=self.at_most,
            above=self.above,
            below=self.below,
        )


_ATTRIBUTE_NAME = re.compile(
    r"[a-z][a-z0-9_]*"
)  # a dot would be a nested path in Qdrant


def check_attribute_name(name: str) -> None:
    if not _ATTRIBUTE_NAME.fullmatch(name):
        raise ValueError(
            f"attribute name {name!r} must be lower-case letters, digits and underscores"
        )


def check_attribute_value(value: AttributeValue) -> None:
    match value:
        case datetime():
            raise TypeError(f"attribute values take dates, not datetimes: {value!r}")
        case float() if not math.isfinite(value):
            raise ValueError(f"attribute values must be finite numbers: {value!r}")
        case _:
            pass


def _list_elements(value: AttributeValue | None) -> tuple[Scalar, ...]:
    """A missing attribute has no elements; a single value is a list of one"""
    if value is None:
        return ()
    return value if isinstance(value, tuple) else (value,)


def _fall_within[T: (float, date)](
    value: T,
    *,
    at_least: T | None = None,
    at_most: T | None = None,
    above: T | None = None,
    below: T | None = None,
) -> bool:
    """Whether value meets every bound given."""
    return (
        (at_least is None or value >= at_least)
        and (at_most is None or value <= at_most)
        and (above is None or value > above)
        and (below is None or value < below)
    )
