import unicodedata
from dataclasses import dataclass
from datetime import date, datetime

import pytest

from document_processor.retrieval.attributes import (
    AllOf,
    AnyOf,
    AttributeValue,
    BooleanEquals,
    Condition,
    DateBetween,
    ElementCondition,
    EveryElement,
    IntegerIn,
    NoneOf,
    NumberBetween,
    Scalar,
    TextIn,
    check_attribute_name,
    check_attribute_value,
)

KB = TextIn(values=("kb-1", "kb-2"))
FEBRUARY = DateBetween(at_least=date(2024, 2, 1), below=date(2024, 3, 1))


# TextIn -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("kb-1", True),
        ("kb-3", False),
        (("kb-9", "kb-2"), True),
        (("kb-9",), False),
        ((), False),
        (None, False),
    ],
)
def test_text_in_matches_a_value_or_some_element(
    value: AttributeValue | None, expected: bool
) -> None:
    assert KB.match(value) is expected


def test_text_in_compares_exactly() -> None:
    assert not TextIn(values=("main",)).match("Main")
    assert not TextIn(values=("Điều",)).match(
        unicodedata.normalize("NFD", "Điều")
    )  # same word, decomposed


def test_text_in_never_matches_another_kind() -> None:
    assert not TextIn(values=("1",)).match(1)
    assert not TextIn(values=("True",)).match(True)
    assert not TextIn(values=("2024-01-01",)).match(date(2024, 1, 1))


def test_text_in_needs_a_value() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        TextIn(values=())


# IntegerIn --------------------------------------------------------------------------------


def test_integer_in_matches_a_value_or_some_element() -> None:
    assert IntegerIn(values=(2024,)).match(2024)
    assert IntegerIn(values=(1,)).match((3, 1))
    assert not IntegerIn(values=(1,)).match((3, 2))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_integer_in_never_matches_booleans_floats_or_text(value: Scalar) -> None:
    assert not IntegerIn(values=(1,)).match(value)


def test_integer_in_needs_a_value() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        IntegerIn(values=())


# BooleanEquals ----------------------------------------------------------------------------


def test_boolean_equals_matches_only_that_boolean() -> None:
    assert BooleanEquals(value=True).match(True)
    assert not BooleanEquals(value=True).match(False)
    assert BooleanEquals(value=False).match(False)


@pytest.mark.parametrize("value", [1, 0, "True"])
def test_boolean_equals_never_matches_integers_or_text(value: Scalar) -> None:
    assert not BooleanEquals(value=True).match(value)
    assert not BooleanEquals(value=False).match(value)


def test_boolean_equals_never_matches_a_missing_attribute() -> None:
    assert not BooleanEquals(value=False).match(None)


# NumberBetween ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "matching", "not_matching"),
    [
        (NumberBetween(at_least=2023), 2023, 2022),
        (NumberBetween(above=2023), 2024, 2023),
        (NumberBetween(at_most=2023), 2023, 2024),
        (NumberBetween(below=2023), 2022, 2023),
    ],
)
def test_number_between_bounds_are_inclusive_or_strict_as_named(
    condition: NumberBetween, matching: int, not_matching: int
) -> None:
    assert condition.match(matching)
    assert not condition.match(not_matching)


def test_number_between_compares_integers_and_floats_alike() -> None:
    assert NumberBetween(above=1.5).match(2)
    assert NumberBetween(below=3).match(2.5)


def test_number_between_needs_every_bound_to_hold() -> None:
    condition = NumberBetween(at_least=2023, below=2025)
    assert condition.match(2024)
    assert not condition.match(2025)


def test_number_between_needs_one_element_to_meet_every_bound() -> None:
    assert not NumberBetween(at_least=2023, below=2025).match((2020, 2026))
    assert NumberBetween(at_least=2023, below=2025).match((2020, 2024))


def test_number_between_with_contradictory_bounds_matches_nothing() -> None:
    assert not NumberBetween(above=5, below=1).match(3)


@pytest.mark.parametrize("value", [True, False, "5", date(2024, 1, 1)])
def test_number_between_never_matches_booleans_text_or_dates(value: Scalar) -> None:
    assert not NumberBetween(at_least=0).match(value)


def test_number_between_needs_a_bound() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        NumberBetween()


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")])
def test_number_between_refuses_a_bound_that_is_not_finite(bound: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        NumberBetween(at_most=bound)


# DateBetween ------------------------------------------------------------------------------


def test_date_between_matches_a_month_with_an_exclusive_end() -> None:
    assert FEBRUARY.match(date(2024, 2, 1))
    assert FEBRUARY.match(date(2024, 2, 29))
    assert not FEBRUARY.match(date(2024, 3, 1))
    assert not FEBRUARY.match(date(2024, 1, 31))


@pytest.mark.parametrize("value", [20240210, "2024-02-10", True])
def test_date_between_never_matches_numbers_text_or_booleans(value: Scalar) -> None:
    assert not FEBRUARY.match(value)


def test_date_between_needs_a_bound() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        DateBetween()


def test_date_between_refuses_a_datetime_bound() -> None:
    with pytest.raises(TypeError, match="not datetimes"):
        DateBetween(below=datetime(2024, 3, 1, 9, 30))


# EveryElement -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (("kb-1", "kb-2"), True),
        (("kb-1",), True),
        (("kb-1", "kb-9"), False),
        ((), False),
        (None, False),
        ("kb-2", True),
        ("kb-9", False),
    ],
)
def test_every_element_needs_all_elements_and_at_least_one(
    value: AttributeValue | None, expected: bool
) -> None:
    assert EveryElement(condition=KB).match(value) is expected


def test_every_element_needs_each_element_to_meet_every_bound() -> None:
    since_2023 = EveryElement(condition=NumberBetween(at_least=2023))
    assert since_2023.match((2023, 2024))
    assert not since_2023.match((2020, 2024))


def test_every_element_shows_the_condition_it_wraps() -> None:
    assert EveryElement(condition=KB).subconditions == (KB,)


# Combinators ------------------------------------------------------------------------------


def test_all_of_needs_every_condition() -> None:
    in_both = AllOf(conditions=(TextIn(values=("kb-1",)), TextIn(values=("kb-2",))))
    assert in_both.match(("kb-1", "kb-2", "kb-3"))
    assert not in_both.match(("kb-1",))


def test_any_of_needs_one_condition() -> None:
    early_or_late = AnyOf(
        conditions=(NumberBetween(below=2000), NumberBetween(above=2020))
    )
    assert early_or_late.match(1999)
    assert early_or_late.match(2024)
    assert not early_or_late.match(2010)


def test_none_of_needs_no_condition() -> None:
    not_a_draft = NoneOf(conditions=(TextIn(values=("draft",)),))
    assert not_a_draft.match("final")
    assert not not_a_draft.match("draft")


def test_none_of_matches_a_missing_attribute() -> None:
    assert NoneOf(conditions=(TextIn(values=("draft",)),)).match(None)


def test_combinators_nest() -> None:
    condition = AllOf(conditions=(KB, NoneOf(conditions=(TextIn(values=("kb-2",)),))))
    assert condition.match(("kb-1",))
    assert not condition.match(("kb-1", "kb-2"))


@pytest.mark.parametrize("combinator", [AllOf, AnyOf, NoneOf])
def test_combinators_need_a_condition(combinator: type[AllOf | AnyOf | NoneOf]) -> None:
    with pytest.raises(ValueError, match="at least one condition"):
        combinator(conditions=())


@pytest.mark.parametrize("combinator", [AllOf, AnyOf, NoneOf])
def test_combinators_show_the_conditions_they_combine(
    combinator: type[AllOf | AnyOf | NoneOf],
) -> None:
    assert combinator(conditions=(KB, FEBRUARY)).subconditions == (KB, FEBRUARY)


def test_a_built_in_condition_has_no_subconditions() -> None:
    assert KB.subconditions == ()


# Extending --------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TextStartsWith(ElementCondition):
    prefix: str

    def match_one(self, value: Scalar) -> bool:
        return isinstance(value, str) and value.startswith(self.prefix)


def test_a_new_element_condition_gets_the_list_rules() -> None:
    condition = TextStartsWith(prefix="kb-")
    assert condition.match(("x", "kb-1"))
    assert not condition.match(())
    assert not condition.match(None)


def test_a_new_element_condition_works_with_every_element_and_combinators() -> None:
    every: Condition = EveryElement(condition=TextStartsWith(prefix="kb-"))
    assert every.match(("kb-1", "kb-2"))
    assert not every.match(("kb-1", "x"))
    assert NoneOf(conditions=(TextStartsWith(prefix="x"),)).match("kb-1")


# Checks -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["title", "knowledge_base_ids", "year2024"])
def test_attribute_names_may_be_lower_case_words_and_digits(name: str) -> None:
    check_attribute_name(name)


@pytest.mark.parametrize(
    "name", ["", "Title", "issuer.name", "1st", "ngày", "with space", "with-dash"]
)
def test_attribute_names_refuse_what_stores_cannot_hold(name: str) -> None:
    with pytest.raises(ValueError, match="attribute name"):
        check_attribute_name(name)


@pytest.mark.parametrize(
    "value", ["text", 1, 2.5, True, date(2024, 1, 1), ("a", "b"), (1, 2), ()]
)
def test_attribute_values_accept_every_kind(value: AttributeValue) -> None:
    check_attribute_value(value)


def test_attribute_values_refuse_a_datetime() -> None:
    with pytest.raises(TypeError, match="not datetimes"):
        check_attribute_value(datetime(2024, 1, 18, 8, 0))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_attribute_values_refuse_numbers_that_are_not_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        check_attribute_value(value)
