import dataclasses
import enum
import math
import re
import uuid
from datetime import UTC, date, datetime
from typing import Any

import pytest

from salemdex_retrieval.schema import (
    Field,
    Many,
    ManyValues,
    One,
    OneValue,
    Schema,
    SchemaError,
)


class Level(int):
    pass


class Colour(enum.StrEnum):
    RED = "red"


KINDS = [pytest.param(One, id="One"), pytest.param(Many, id="Many")]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value_type", [str, int, float, bool, uuid.UUID, datetime])
def test_a_field_accepts_each_supported_value_type(
    kind: type[Field[Any]], value_type: type[Any]
) -> None:
    field = kind(value_type)

    assert field.value_type is value_type


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "value_type",
    [
        pytest.param(list, id="list"),
        pytest.param(list[int], id="parameterised list"),
        pytest.param(date, id="date"),
        pytest.param(Level, id="subclass of int"),
        pytest.param(Colour, id="subclass of str"),
        pytest.param(int | None, id="optional"),
    ],
)
def test_a_field_rejects_other_value_types_naming_the_supported_ones(
    kind: type[Field[Any]], value_type: Any
) -> None:
    with pytest.raises(
        TypeError, match="expected str, int, float, bool, UUID, datetime"
    ):
        kind(value_type)


def test_bool_and_int_fields_keep_their_own_value_type() -> None:
    assert One(bool).value_type is bool
    assert One(int).value_type is int


@pytest.mark.parametrize("kind", KINDS)
def test_a_field_is_not_indexed_unless_asked(kind: type[Field[Any]]) -> None:
    assert kind(int).indexed is False
    assert kind(int, indexed=True).indexed is True


def test_fields_take_their_names_from_the_schema_class_body() -> None:
    class Chunks:
        document_id = One(uuid.UUID)
        groups = Many(uuid.UUID)

    assert Chunks.document_id.name == "document_id"
    assert Chunks.groups.name == "groups"


@pytest.mark.parametrize(
    ("field", "example"),
    [
        pytest.param(One(uuid.UUID), "One(UUID)", id="One"),
        pytest.param(Many(str), "Many(str)", id="Many"),
    ],
)
def test_a_field_outside_a_schema_class_body_has_no_name_and_says_how_to_declare_it(
    field: Field[Any], example: str
) -> None:
    with pytest.raises(
        SchemaError, match=rf"field has no name.*my_field = {re.escape(example)}"
    ):
        _ = field.name


def test_aliasing_a_field_in_one_class_body_is_rejected() -> None:
    with pytest.raises(SchemaError, match="already the field 'document_id'.*'doc'"):

        class Chunks:
            document_id = One(uuid.UUID)
            doc = document_id


def test_sharing_a_field_between_two_schemas_is_rejected() -> None:
    shared = One(uuid.UUID)

    class Chunks:
        document_id = shared

    with pytest.raises(
        SchemaError, match="already the field 'document_id'.*in Entries"
    ):

        class Entries:
            document_id = shared

    assert Chunks.document_id.name == "document_id"


def test_the_double_naming_error_names_the_kind_of_field() -> None:
    with pytest.raises(SchemaError, match=r"this Many\(\.\.\.\) is already"):

        class Chunks:
            groups = Many(uuid.UUID)
            teams = groups


def test_a_field_name_cannot_be_reassigned() -> None:
    class Chunks:
        document_id = One(uuid.UUID)

    with pytest.raises(AttributeError):
        Chunks.document_id.name = "groups"  # type: ignore[misc]

    assert Chunks.document_id.name == "document_id"


class Standard(Schema):
    document_id = One(uuid.UUID, indexed=True)
    position = One(int)


def test_a_field_shows_its_kind_type_name_and_indexing() -> None:
    assert repr(Standard.document_id) == "One(UUID, name='document_id', indexed=True)"


def test_a_field_without_a_name_can_still_be_shown() -> None:
    assert repr(Many(str)) == "Many(str, name=None, indexed=False)"


def test_a_schema_lists_its_fields_in_declaration_order() -> None:
    class Chunks(Schema):
        document_id = One(uuid.UUID)
        groups = Many(uuid.UUID)
        version = One(int)

    assert list(Chunks.fields()) == ["document_id", "groups", "version"]
    assert Chunks.fields()["groups"] is Chunks.groups


def test_a_schema_lists_inherited_fields_first() -> None:
    class Chunks(Standard):
        groups = Many(uuid.UUID)

    assert list(Chunks.fields()) == ["document_id", "position", "groups"]
    assert Chunks.fields()["position"] is Standard.position


def test_a_schema_lists_only_fields() -> None:
    class Chunks(Schema):
        """Chunks of documents."""

        document_id = One(uuid.UUID)
        limit = 10

        def describe(self) -> str:
            return "chunks"

    assert list(Chunks.fields()) == ["document_id"]


def test_the_base_schema_and_an_empty_schema_have_no_fields() -> None:
    class Empty(Schema):
        pass

    assert dict(Schema.fields()) == {}
    assert dict(Empty.fields()) == {}


def test_the_fields_of_a_schema_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        Standard.fields()["groups"] = Many(uuid.UUID)  # type: ignore[index]


def test_a_preset_reached_through_two_branches_counts_its_fields_once() -> None:
    class Left(Standard):
        pass

    class Right(Standard):
        pass

    class Both(Left, Right):
        pass

    assert list(Both.fields()) == ["document_id", "position"]


def test_a_schema_ignores_fields_declared_in_a_base_that_is_not_a_schema() -> None:
    class Common:
        collection_id = One(uuid.UUID)

    class Chunks(Common, Schema):
        document_id = One(uuid.UUID)

    assert list(Chunks.fields()) == ["document_id"]


def test_inheriting_two_different_fields_with_one_name_is_rejected() -> None:
    class First(Schema):
        position = One(int)

    class Second(Schema):
        position = One(int)

    with pytest.raises(SchemaError, match="Both inherits two different fields named"):

        class Both(First, Second):
            pass


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(One(str), id="another field"),
        pytest.param(None, id="a plain value"),
    ],
)
def test_an_inherited_field_cannot_be_redefined_or_hidden(replacement: Any) -> None:
    with pytest.raises(SchemaError, match="Chunks.position hides an inherited field"):
        type("Chunks", (Standard,), {"position": replacement})


def test_an_inherited_field_cannot_be_hidden_by_a_method() -> None:
    with pytest.raises(SchemaError, match="Chunks.position hides an inherited field"):

        class Chunks(Standard):
            def position(self) -> int:  # type: ignore[override]
                return 0


@pytest.mark.parametrize("name", ["fields", "_fields"])
def test_a_field_cannot_take_a_name_the_schema_uses(name: str) -> None:
    with pytest.raises(SchemaError, match=f"'{name}' is .*reserved"):
        type("Chunks", (Schema,), {name: Many(str)})


class Record(Schema):
    count = One(int)
    score = One(float)
    title = One(str)
    indexed_at = One(datetime)
    headings = Many(str)
    counts = Many(int)
    groups = Many(uuid.UUID)
    seen_at = Many(datetime)


HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")
LEGAL = uuid.UUID("00000000-0000-0000-0000-00000000000b")
NEW_YEAR = datetime(2026, 1, 1, tzinfo=UTC)
NAIVE_NEW_YEAR = datetime(2026, 1, 1)


def test_one_value_pairs_the_field_with_the_value() -> None:
    assert Record.title.value("Scope") == OneValue(field=Record.title, value="Scope")


@pytest.mark.parametrize(
    ("field", "wrong", "message"),
    [
        pytest.param(Record.count, True, "count is 'bool', expected int", id="bool"),
        pytest.param(
            Record.title, Colour.RED, "title is 'Colour', expected str", id="StrEnum"
        ),
        pytest.param(
            Record.title, None, "title is 'NoneType', expected str", id="None"
        ),
        pytest.param(
            Record.score, True, "score is 'bool', expected float", id="bool for float"
        ),
        pytest.param(
            Record.indexed_at,
            date(2026, 1, 1),
            "indexed_at is 'date', expected datetime",
            id="date",
        ),
    ],
)
def test_one_value_rejects_a_value_not_exactly_of_the_field_type(
    field: One[Any], wrong: Any, message: str
) -> None:
    with pytest.raises(TypeError, match=re.escape(message)):
        field.value(wrong)


def test_one_value_widens_an_int_to_float_for_a_float_field() -> None:
    stored = Record.score.value(3).value

    assert stored == 3.0
    assert type(stored) is float


def test_one_value_accepts_a_datetime_with_a_timezone() -> None:
    assert Record.indexed_at.value(NEW_YEAR).value == NEW_YEAR


def test_one_value_rejects_a_datetime_without_a_timezone() -> None:
    with pytest.raises(TypeError, match="indexed_at is missing a timezone"):
        Record.indexed_at.value(NAIVE_NEW_YEAR)


@pytest.mark.parametrize(
    "wrong", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"]
)
def test_a_float_field_rejects_a_value_that_is_not_finite(wrong: float) -> None:
    message = f"score is {wrong!r}; expected a finite number"
    with pytest.raises(ValueError, match=re.escape(message)):
        Record.score.value(wrong)
    with pytest.raises(ValueError, match=re.escape(message)):
        Record.score.eq(wrong)


@pytest.mark.parametrize("limit", [-(2**63), 2**63 - 1], ids=["smallest", "largest"])
def test_an_int_field_accepts_the_64_bit_limits(limit: int) -> None:
    assert Record.count.value(limit).value == limit


@pytest.mark.parametrize("wrong", [-(2**63) - 1, 2**63], ids=["below", "above"])
def test_an_int_field_rejects_a_value_outside_64_bits(wrong: int) -> None:
    with pytest.raises(ValueError, match="count is outside 64 bits"):
        Record.count.value(wrong)
    with pytest.raises(ValueError, match="count is outside 64 bits"):
        Record.count.eq(wrong)


def test_a_float_field_rejects_an_int_too_large_to_widen() -> None:
    with pytest.raises(ValueError, match="score is an int too large for a float"):
        Record.score.value(10**400)


def test_many_value_names_the_position_of_an_int_outside_64_bits() -> None:
    with pytest.raises(ValueError, match=re.escape("counts[1] is outside 64 bits")):
        Record.counts.value([1, 2**63])


def test_many_value_keeps_the_order_and_repeats_as_a_tuple() -> None:
    field_value = Record.headings.value(["Part 1", "Scope", "Part 1"])

    assert field_value == ManyValues(
        field=Record.headings, values=("Part 1", "Scope", "Part 1")
    )


def test_many_value_accepts_no_values() -> None:
    assert Record.groups.value([]).values == ()


def test_many_value_accepts_any_iterable_such_as_a_generator() -> None:
    assert Record.groups.value(g for g in [HR, LEGAL]).values == (HR, LEGAL)


def test_many_value_names_the_position_of_a_value_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match=re.escape("groups[1] is 'str', expected UUID")):
        Record.groups.value([HR, "legal"])  # type: ignore[list-item]


def test_many_value_names_the_position_of_a_datetime_without_a_timezone() -> None:
    with pytest.raises(TypeError, match=re.escape("seen_at[1] is missing a timezone")):
        Record.seen_at.value([NEW_YEAR, NAIVE_NEW_YEAR])


@pytest.mark.parametrize(
    ("wrong", "type_name"),
    [("intro", "str"), (b"ab", "bytes"), ({"a": 1}, "dict")],
    ids=["str", "bytes", "dict"],
)
def test_many_value_rejects_a_string_bytes_or_mapping_naming_its_type(
    wrong: Any, type_name: str
) -> None:
    message = f"value(...) needs a collection of values, not {type_name!r}"
    with pytest.raises(TypeError, match=re.escape(message)):
        Record.headings.value(wrong)


def test_a_field_value_cannot_be_changed_after_it_is_built() -> None:
    field_value = Record.title.value("Scope")

    with pytest.raises(dataclasses.FrozenInstanceError):
        field_value.value = "Other"  # type: ignore[misc]
