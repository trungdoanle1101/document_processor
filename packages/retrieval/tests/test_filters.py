import dataclasses
import re
import uuid
from typing import Any

import pytest

from salemdex_retrieval.filters import AnyOf, Equals
from salemdex_retrieval.schema import Many, One, Schema


class Chunks(Schema):
    document_id = One(uuid.UUID, indexed=True)
    version = One(int)
    position = One(int)
    groups = Many(uuid.UUID, indexed=True)
    headings = Many(str)


DOCUMENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")
LEGAL = uuid.UUID("00000000-0000-0000-0000-00000000000b")


def test_eq_describes_the_field_and_the_value() -> None:
    where = Chunks.document_id.eq(DOCUMENT_ID)

    assert where == Equals(field=Chunks.document_id, value=DOCUMENT_ID)
    assert where.field.name == "document_id"


def test_eq_on_different_fields_gives_different_filters() -> None:
    assert Chunks.version.eq(1) != Chunks.position.eq(1)


def test_eq_rejects_a_value_not_of_the_field_type() -> None:
    with pytest.raises(TypeError, match="document_id is 'str', expected UUID"):
        Chunks.document_id.eq(str(DOCUMENT_ID))  # type: ignore[arg-type]


def test_a_filter_cannot_be_changed_after_it_is_built() -> None:
    where = Equals(field=Chunks.version, value=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        where.value = 2  # type: ignore[misc]


def test_equals_takes_its_parts_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        Equals(Chunks.version, 1)  # type: ignore[call-arg]


def test_any_of_describes_the_field_and_the_values() -> None:
    where = Chunks.groups.any_of([HR, LEGAL])

    assert where == AnyOf(field=Chunks.groups, values=frozenset({HR, LEGAL}))
    assert where.field.name == "groups"


def test_any_of_ignores_the_order_and_repeats_of_its_values() -> None:
    assert Chunks.groups.any_of([HR, LEGAL]) == Chunks.groups.any_of([LEGAL, HR, HR])


def test_any_of_keeps_its_own_copy_of_the_values() -> None:
    group_ids = [HR]
    where = Chunks.groups.any_of(group_ids)

    group_ids.append(LEGAL)

    assert where == AnyOf(field=Chunks.groups, values=frozenset({HR}))


def test_any_of_accepts_any_iterable_such_as_a_generator() -> None:
    where = Chunks.groups.any_of(group_id for group_id in [HR, LEGAL])

    assert where == AnyOf(field=Chunks.groups, values=frozenset({HR, LEGAL}))


def test_any_of_names_the_position_of_a_value_not_of_the_field_type() -> None:
    with pytest.raises(TypeError, match=re.escape("groups[1] is 'str', expected UUID")):
        Chunks.groups.any_of([HR, str(LEGAL)])  # type: ignore[list-item]


def test_any_of_with_no_values_keeps_no_values() -> None:
    assert Chunks.groups.any_of([]) == AnyOf(field=Chunks.groups, values=frozenset())


@pytest.mark.parametrize(
    ("wrong", "type_name"),
    [("intro", "str"), (b"ab", "bytes"), ({"a": 1}, "dict")],
    ids=["str", "bytes", "dict"],
)
def test_any_of_rejects_a_string_bytes_or_mapping_naming_its_type(
    wrong: Any, type_name: str
) -> None:
    message = f"any_of(...) needs a collection of values, not {type_name!r}"
    with pytest.raises(TypeError, match=re.escape(message)):
        Chunks.headings.any_of(wrong)
