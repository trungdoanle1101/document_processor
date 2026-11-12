import enum
import uuid
from datetime import date, datetime
from typing import Any

import pytest

from salemdex_retrieval.schema import One, SchemaError


class Level(int):
    pass


class Colour(enum.StrEnum):
    RED = "red"


@pytest.mark.parametrize("value_type", [str, int, float, bool, uuid.UUID, datetime])
def test_one_accepts_each_supported_value_type(value_type: type[Any]) -> None:
    field = One(value_type)

    assert field.value_type is value_type


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
def test_one_rejects_other_value_types_naming_the_supported_ones(
    value_type: Any,
) -> None:
    with pytest.raises(
        TypeError, match="expected str, int, float, bool, UUID, datetime"
    ):
        One(value_type)


def test_bool_and_int_fields_keep_their_own_value_type() -> None:
    assert One(bool).value_type is bool
    assert One(int).value_type is int


def test_one_is_not_indexed_unless_asked() -> None:
    assert One(int).indexed is False
    assert One(int, indexed=True).indexed is True


def test_one_takes_its_name_from_the_schema_class_body() -> None:
    class Chunks:
        document_id = One(uuid.UUID)
        version = One(int)

    assert Chunks.document_id.name == "document_id"
    assert Chunks.version.name == "version"


def test_one_without_a_schema_class_body_has_no_name() -> None:
    field = One(int)

    with pytest.raises(SchemaError, match="field has no name"):
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


def test_a_field_name_cannot_be_reassigned() -> None:
    class Chunks:
        document_id = One(uuid.UUID)

    with pytest.raises(AttributeError):
        Chunks.document_id.name = "groups"  # type: ignore[misc]

    assert Chunks.document_id.name == "document_id"
