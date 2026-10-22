"""Tests for saving and loading LogicalDocument (segments + outline)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from document_processor.core.models.logical import BaseSegment, LogicalDocument
from document_processor.core.serialization import (
    LOGICAL_DOCUMENT_KIND,
    LOGICAL_SCHEMA_VERSION,
    PHYSICAL_DOCUMENT_KIND,
    load_logical_document,
    logical_document_from_dict,
    logical_document_to_dict,
    save_logical_document,
)
from tests.factories import (
    heading_for,
    make_image_info,
    make_image_segment,
    make_logical_document,
    make_logical_node,
    make_table_info,
    make_table_segment,
    make_text_segment,
)


def full_document() -> LogicalDocument:
    """Every segment kind, front and end matter, nested nodes, optional fields set and unset.

        0  text     CHÍNH PHỦ                    front matter
        1  heading  Chương I QUY ĐỊNH CHUNG      chuong-I [1, 9)
        2  heading  Điều 1. Phạm vi điều chỉnh     dieu-1 [2, 9)
        3  text     Nghị định này quy định về:
        4  heading  1.                               khoan-1 [4, 7)
        5  text     Kiến trúc tổng thể.
        6  image    (diagram)
        7  heading  (no marker, no title)            khoan-2 [7, 9)
        8  table    (two-row table)
        9  text     Nơi nhận:                    end matter
    """
    nodes = (
        make_logical_node(id="chuong-I", parent_id=None, level="chuong", number="I",
                          marker="Chương I", title="QUY ĐỊNH CHUNG",
                          head_block_ids=("pb2", "pb3"), start=1, end=9),
        make_logical_node(id="chuong-I/dieu-1", parent_id="chuong-I", level="dieu", number="1",
                          marker="Điều 1.", title="Phạm vi điều chỉnh",
                          head_block_ids=("pb4",), start=2, end=9),
        make_logical_node(id="chuong-I/dieu-1/khoan-1", parent_id="chuong-I/dieu-1",
                          level="khoan", number="1", marker="1.", title=None,
                          head_block_ids=("pb6",), start=4, end=7),
        make_logical_node(id="chuong-I/dieu-1/khoan-2", parent_id="chuong-I/dieu-1",
                          level="khoan", number=None, marker=None, title=None,
                          head_block_ids=("pb8",), start=7, end=9),
    )
    segments = (
        make_text_segment(block_id="pb1", text="CHÍNH PHỦ"),
        heading_for(nodes[0]),
        heading_for(nodes[1]),
        make_text_segment(block_id="pb5", text="Nghị định này quy định về:"),
        heading_for(nodes[2]),
        make_text_segment(block_id="pb6", text="Kiến trúc tổng thể."),
        make_image_segment(block_id="pb7", image=make_image_info(displayed_width_pt=None)),
        heading_for(nodes[3]),
        make_table_segment(block_id="pb8", table=make_table_info()),
        make_text_segment(block_id="pb9", text="Nơi nhận:"),
    )
    return make_logical_document(segments=segments, nodes=nodes, body_start=1, body_end=9)


def through_json(d: dict[str, Any]) -> dict[str, Any]:
    """Real JSON text in between, so tuples really become lists."""
    result: dict[str, Any] = json.loads(json.dumps(d, ensure_ascii=False))
    return result


def round_trip(doc: LogicalDocument) -> LogicalDocument:
    return logical_document_from_dict(through_json(logical_document_to_dict(doc)))


# ---------- round trips


def test_full_document_round_trip() -> None:
    doc = full_document()
    assert round_trip(doc) == doc


def test_empty_document_round_trip() -> None:
    doc = make_logical_document()
    assert round_trip(doc) == doc


def test_document_without_hierarchy_round_trip() -> None:
    segments = (make_text_segment(block_id="pb1"), make_text_segment(block_id="pb2"))
    doc = make_logical_document(segments=segments, body_start=0, body_end=2)
    assert round_trip(doc) == doc


def test_loaded_document_rebuilds_its_lookups() -> None:
    loaded = round_trip(full_document())
    assert loaded.root_ids == ("chuong-I",)
    assert loaded.get_children("chuong-I/dieu-1") == (
        "chuong-I/dieu-1/khoan-1",
        "chuong-I/dieu-1/khoan-2",
    )
    assert loaded.get_node("chuong-I").title == "QUY ĐỊNH CHUNG"


def test_loaded_collections_are_tuples() -> None:
    loaded = round_trip(full_document())
    assert isinstance(loaded.segments, tuple)
    assert isinstance(loaded.nodes, tuple)
    assert isinstance(loaded.nodes[0].head_block_ids, tuple)
    table_segment = loaded.segments[8]
    assert isinstance(table_segment.table.cells, tuple)  # type: ignore[union-attr]


def test_segment_kinds_survive() -> None:
    loaded = round_trip(full_document())
    assert [type(s).__name__ for s in loaded.segments] == [
        "TextSegment", "HeadingSegment", "HeadingSegment", "TextSegment", "HeadingSegment",
        "TextSegment", "ImageSegment", "HeadingSegment", "TableSegment", "TextSegment",
    ]


def test_node_order_is_preserved() -> None:
    loaded = round_trip(full_document())
    assert [n.id for n in loaded.nodes] == [n.id for n in full_document().nodes]


# ---------- JSON shape


def test_envelope() -> None:
    d = logical_document_to_dict(full_document())
    assert set(d) == {"kind", "schema_version", "document"}
    assert d["kind"] == LOGICAL_DOCUMENT_KIND
    assert d["schema_version"] == LOGICAL_SCHEMA_VERSION


def test_document_fields() -> None:
    document = logical_document_to_dict(full_document())["document"]
    assert set(document) == {
        "id", "physical_document_id", "source_file_hash", "profile_name",
        "profile_version", "segments", "nodes", "body_start", "body_end",
    }
    assert (document["body_start"], document["body_end"]) == (1, 9)


def test_derived_lookups_are_not_written() -> None:
    text = json.dumps(logical_document_to_dict(full_document()))
    assert "_node_by_id" not in text
    assert "_children_by_id" not in text
    assert "root_ids" not in text


def test_text_segment_shape() -> None:
    segments = through_json(logical_document_to_dict(full_document()))["document"]["segments"]
    assert segments[3] == {"kind": "text", "block_id": "pb5", "text": "Nghị định này quy định về:"}


def test_heading_segment_shape() -> None:
    segments = through_json(logical_document_to_dict(full_document()))["document"]["segments"]
    assert segments[1] == {
        "kind": "heading",
        "block_id": "pb2",
        "node_id": "chuong-I",
        "marker": "Chương I",
        "title": "QUY ĐỊNH CHUNG",
    }


def test_heading_segment_without_marker_or_title_writes_null() -> None:
    segments = through_json(logical_document_to_dict(full_document()))["document"]["segments"]
    assert (segments[7]["marker"], segments[7]["title"]) == (None, None)


def test_image_segment_shape() -> None:
    segments = through_json(logical_document_to_dict(full_document()))["document"]["segments"]
    image = segments[6]
    assert set(image) == {"kind", "block_id", "image"}
    assert image["kind"] == "image"
    assert image["image"]["mime_type"] == "image/png"
    assert image["image"]["displayed_width_pt"] is None


def test_table_segment_shape() -> None:
    segments = through_json(logical_document_to_dict(full_document()))["document"]["segments"]
    table = segments[8]
    assert set(table) == {"kind", "block_id", "table"}
    assert table["kind"] == "table"
    assert table["table"]["n_rows"] == 2
    assert isinstance(table["table"]["cells"], list)


def test_node_shape() -> None:
    nodes = through_json(logical_document_to_dict(full_document()))["document"]["nodes"]
    assert nodes[0] == {
        "id": "chuong-I",
        "parent_id": None,
        "level": "chuong",
        "number": "I",
        "marker": "Chương I",
        "title": "QUY ĐỊNH CHUNG",
        "head_block_ids": ["pb2", "pb3"],
        "start": 1,
        "end": 9,
    }


def test_unset_optional_node_fields_are_written_as_null() -> None:
    nodes = through_json(logical_document_to_dict(full_document()))["document"]["nodes"]
    assert nodes[3]["number"] is None
    assert nodes[3]["marker"] is None
    assert nodes[3]["title"] is None


@dataclass(frozen=True, kw_only=True)
class _UnregisteredSegment(BaseSegment):
    """A new segment kind that serialization doesn't know about yet."""


def test_unknown_segment_type_rejected_on_write() -> None:
    doc = make_logical_document()
    object.__setattr__(doc, "segments", (_UnregisteredSegment(block_id="pb1"),))
    with pytest.raises((TypeError, ValueError), match="_UnregisteredSegment"):
        logical_document_to_dict(doc)


# ---------- loading: envelope checks


def test_wrong_kind_rejected() -> None:
    d = logical_document_to_dict(full_document())
    d["kind"] = PHYSICAL_DOCUMENT_KIND
    with pytest.raises(ValueError, match="kind"):
        logical_document_from_dict(d)


def test_missing_kind_rejected() -> None:
    d = logical_document_to_dict(full_document())
    del d["kind"]
    with pytest.raises(ValueError, match="kind"):
        logical_document_from_dict(d)


@pytest.mark.parametrize("version", [LOGICAL_SCHEMA_VERSION + 1, 0, None, "1"])
def test_wrong_schema_version_rejected(version: Any) -> None:
    d = logical_document_to_dict(full_document())
    d["schema_version"] = version
    with pytest.raises(ValueError, match="schema version"):
        logical_document_from_dict(d)


# ---------- loading: segments


def test_unknown_segment_kind_rejected() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["segments"][3]["kind"] = "paragraph"
    with pytest.raises(ValueError, match=r"'heading'\), got kind='paragraph'"):
        logical_document_from_dict(d)


def test_missing_segment_kind_rejected() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    del d["document"]["segments"][3]["kind"]
    with pytest.raises(ValueError, match="kind=None"):
        logical_document_from_dict(d)


def test_segment_validation_runs_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["segments"][3]["text"] = "   "
    with pytest.raises(ValueError, match="text"):
        logical_document_from_dict(d)


def test_heading_segment_validation_runs_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["segments"][1]["title"] = ""
    with pytest.raises(ValueError, match="'title'"):
        logical_document_from_dict(d)


def test_image_validation_runs_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["segments"][6]["image"]["mime_type"] = "text/plain"
    with pytest.raises(ValueError, match="mime_type"):
        logical_document_from_dict(d)


# ---------- loading: nodes and outline


def test_node_validation_runs_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["nodes"][2]["head_block_ids"] = []
    with pytest.raises(ValueError, match="head_block_ids"):
        logical_document_from_dict(d)


def test_outline_validation_runs_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["nodes"][2]["end"] = 8  # overlaps khoan-2 [7, 9)
    with pytest.raises(ValueError, match="overlaps or precedes"):
        logical_document_from_dict(d)


def test_reordered_nodes_rejected_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    nodes = d["document"]["nodes"]
    nodes[1], nodes[2] = nodes[2], nodes[1]  # a khoan before its Điều
    with pytest.raises(ValueError, match="not an open ancestor"):
        logical_document_from_dict(d)


def test_heading_check_runs_on_load() -> None:
    # A file edited by hand so a node's title no longer matches its heading segment.
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["nodes"][1]["title"] = "Đối tượng áp dụng"
    with pytest.raises(ValueError, match="different title"):
        logical_document_from_dict(d)


def test_heading_segment_for_a_missing_node_rejected_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["segments"][4]["node_id"] = "khoan-9"
    with pytest.raises(ValueError, match="'khoan-9' is not at that node's start"):
        logical_document_from_dict(d)


def test_body_range_validation_runs_on_load() -> None:
    d = through_json(logical_document_to_dict(full_document()))
    d["document"]["body_end"] = 99
    with pytest.raises(ValueError, match="body_end"):
        logical_document_from_dict(d)


# ---------- files


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "doc.logical.json"
    doc = full_document()
    save_logical_document(doc, path)
    assert load_logical_document(path) == doc


def test_save_creates_parent_folders(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "doc.json"
    save_logical_document(full_document(), path)
    assert path.is_file()


def test_saved_file_keeps_vietnamese_readable(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    save_logical_document(full_document(), path)
    text = path.read_text(encoding="utf-8")
    assert "QUY ĐỊNH CHUNG" in text
    assert "\\u" not in text


def test_save_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    save_logical_document(full_document(), str(path))
    assert load_logical_document(str(path)) == full_document()