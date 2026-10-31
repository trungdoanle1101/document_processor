"""Tests for saving and loading PhysicalDocument, focused on list info."""

import json
from typing import Any

import pytest

from document_processor.core.models import BlockType, PhysicalDocument
from document_processor.core.serialization import (
    physical_document_from_dict,
    physical_document_to_dict,
)
from tests.factories import (
    make_block,
    make_declared_numbering,
    make_document,
    make_list_info,
)


def round_trip(doc: PhysicalDocument) -> PhysicalDocument:
    # Through real JSON text, so tuples really become lists on the way.
    return physical_document_from_dict(
        json.loads(json.dumps(physical_document_to_dict(doc)))
    )


def list_document(**list_info_overrides: Any) -> PhysicalDocument:
    block = make_block(
        id="pb1",
        block_type=BlockType.LIST_ITEM,
        text="Phạm vi điều chỉnh",
        list_info=make_list_info(**list_info_overrides),
    )
    return make_document(blocks=(block,))


def list_info_dict(doc: PhysicalDocument) -> dict[str, Any]:
    result: dict[str, Any] = physical_document_to_dict(doc)["document"]["blocks"][0][
        "list_info"
    ]
    return result


def test_round_trip_without_declared_numbering() -> None:
    doc = list_document(declared_numbering=None)
    assert round_trip(doc) == doc


@pytest.mark.parametrize(
    ("num_fmt", "segment_formats"),
    [
        ("decimal", ("decimal",)),
        ("lowerLetter", ("decimal", "lowerLetter")),
        ("bullet", ()),
        ("decimal", ("decimal", "decimal")),
    ],
    ids=["single", "multilevel", "bullet", "repeated"],
)
def test_round_trip_with_declared_numbering(
    num_fmt: str, segment_formats: tuple[str, ...]
) -> None:
    doc = list_document(
        declared_numbering=make_declared_numbering(
            num_fmt=num_fmt, segment_formats=segment_formats
        )
    )
    loaded = round_trip(doc)
    assert loaded == doc
    info = loaded.blocks[0].list_info
    assert info is not None and info.declared_numbering is not None
    assert isinstance(info.declared_numbering.segment_formats, tuple)


def test_declared_numbering_written_as_null_when_absent() -> None:
    assert (
        list_info_dict(list_document(declared_numbering=None))["declared_numbering"]
        is None
    )


def test_declared_numbering_json_shape() -> None:
    doc = list_document(
        declared_numbering=make_declared_numbering(
            num_fmt="lowerLetter", segment_formats=("decimal", "lowerLetter")
        )
    )
    written = json.loads(json.dumps(list_info_dict(doc)))
    assert written["declared_numbering"] == {
        "num_fmt": "lowerLetter",
        "segment_formats": ["decimal", "lowerLetter"],
    }


def test_load_rejects_missing_declared_numbering_key() -> None:
    # Old dumps without the key fail loudly rather than loading as None.
    d = physical_document_to_dict(list_document())
    del d["document"]["blocks"][0]["list_info"]["declared_numbering"]
    with pytest.raises(KeyError, match="declared_numbering"):
        physical_document_from_dict(d)


def test_load_runs_declared_numbering_validation() -> None:
    d = physical_document_to_dict(
        list_document(declared_numbering=make_declared_numbering())
    )
    d["document"]["blocks"][0]["list_info"]["declared_numbering"]["segment_formats"] = [
        " "
    ]
    with pytest.raises(ValueError, match="segment format"):
        physical_document_from_dict(d)
