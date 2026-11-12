import dataclasses
import uuid

import pytest

from salemdex_retrieval.filters import Equals
from salemdex_retrieval.schema import One


class Chunks:
    document_id = One(uuid.UUID, indexed=True)
    version = One(int)


DOCUMENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def test_eq_describes_the_field_and_the_value() -> None:
    where = Chunks.document_id.eq(DOCUMENT_ID)

    assert where == Equals(field=Chunks.document_id, value=DOCUMENT_ID)
    assert where.field.name == "document_id"


def test_eq_on_different_fields_gives_different_filters() -> None:
    assert Chunks.version.eq(1) != Equals(field=Chunks.document_id, value=1)


def test_a_filter_cannot_be_changed_after_it_is_built() -> None:
    where = Chunks.version.eq(1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        where.value = 2  # type: ignore[misc]


def test_equals_takes_its_parts_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        Equals(Chunks.version, 1)  # type: ignore[call-arg]
