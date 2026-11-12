import enum
import re
import uuid
from datetime import date, datetime
from typing import Any

import pytest

from salemdex_retrieval.schema import Field, Many, One, Schema, SchemaError


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
