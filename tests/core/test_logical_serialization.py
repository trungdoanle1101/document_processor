"""Tests for saving and loading LogicalDocument."""

import json
from pathlib import Path
from typing import Any

import pytest

from document_processor.core.models.logical import BlockSpan, LogicalDocument, LogicalNode
from document_processor.core.models.ranges import TextRange
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
    make_block_span,
    make_logical_document,
    make_logical_node,
)


def span(block_id: str, start: int, end: int) -> BlockSpan:
    return make_block_span(block_id=block_id, text_range=TextRange(start, end))


def full_document() -> LogicalDocument:
    """Two roots with front and end matter, and every optional field exercised.

        [front] pb1, pb2
        chuong-1          marker + title on the next block
          dieu-1          marker and title in one block
            khoan-1       auto-numbered: no marker, body only
            khoan-2       body in two blocks
          dieu-2          number None
        chuong-2
        [end]   pb12
    """
    nodes: tuple[LogicalNode, ...] = (
        make_logical_node(
            id="chuong-1",
            level="chuong",
            number="I",
            marker=span("pb3", 0, 8),
            title=(span("pb4", 0, 15),),
            body=(),
            children_ids=("dieu-1", "dieu-2"),
        ),
        make_logical_node(
            id="dieu-1",
            level="dieu",
            number="1",
            marker=span("pb5", 0, 7),
            title=(span("pb5", 8, 26),),
            body=(),
            children_ids=("khoan-1", "khoan-2"),
        ),
        make_logical_node(
            id="khoan-1",
            level="khoan",
            number="1",
            marker=None,
            title=(),
            body=(span("pb6", 0, 40),),
            children_ids=(),
        ),
        make_logical_node(
            id="khoan-2",
            level="khoan",
            number="2",
            marker=span("pb7", 0, 2),
            title=(),
            body=(span("pb7", 3, 30), span("pb8", 0, 12)),
            children_ids=(),
        ),
        make_logical_node(
            id="dieu-2",
            level="dieu",
            number=None,
            marker=span("pb9", 0, 7),
            title=(span("pb9", 8, 20),),
            body=(span("pb10", 0, 5),),
            children_ids=(),
        ),
        make_logical_node(
            id="chuong-2",
            level="chuong",
            number="II",
            marker=span("pb11", 0, 9),
            title=(),
            body=(),
            children_ids=(),
        ),
    )
    return make_logical_document(
        front_matter=(span("pb1", 0, 30), span("pb2", 0, 12)),
        root_ids=("chuong-1", "chuong-2"),
        nodes=nodes,
        end_matter=(span("pb12", 0, 9),),
    )


def saved_dict() -> dict[str, Any]:
    return logical_document_to_dict(full_document())


# --------------------------------------------------------------------------- #
# to_dict
# --------------------------------------------------------------------------- #


class TestToDict:
    def test_envelope(self) -> None:
        d = saved_dict()
        assert d["kind"] == LOGICAL_DOCUMENT_KIND
        assert d["schema_version"] == LOGICAL_SCHEMA_VERSION
        assert set(d) == {"kind", "schema_version", "document"}

    def test_document_has_every_field_and_no_mapping(self) -> None:
        assert set(saved_dict()["document"]) == {
            "id",
            "physical_document_id",
            "source_file_hash",
            "profile_name",
            "profile_version",
            "front_matter",
            "root_ids",
            "nodes",
            "end_matter",
        }

    def test_node_fields(self) -> None:
        node = saved_dict()["document"]["nodes"][0]
        assert set(node) == {
            "id",
            "level",
            "number",
            "marker",
            "title",
            "body",
            "children_ids",
        }

    def test_span_shape(self) -> None:
        marker = saved_dict()["document"]["nodes"][0]["marker"]
        assert marker == {"block_id": "pb3", "text_range": {"start": 0, "end": 8}}

    def test_missing_marker_saved_as_none(self) -> None:
        khoan_1 = saved_dict()["document"]["nodes"][2]
        assert khoan_1["marker"] is None

    def test_json_serializable(self) -> None:
        json.dumps(saved_dict())

    def test_does_not_crash_on_lookup_mapping(self) -> None:
        # asdict() on the whole document would deepcopy the mappingproxy and fail.
        doc = full_document()
        doc.get_node("dieu-1")
        logical_document_to_dict(doc)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_empty_document(self) -> None:
        doc = make_logical_document()
        assert logical_document_from_dict(logical_document_to_dict(doc)) == doc

    def test_full_document(self) -> None:
        doc = full_document()
        assert logical_document_from_dict(logical_document_to_dict(doc)) == doc

    def test_through_json_text(self) -> None:
        doc = full_document()
        text = json.dumps(logical_document_to_dict(doc))
        assert logical_document_from_dict(json.loads(text)) == doc

    def test_sequences_are_tuples_after_load(self) -> None:
        loaded = logical_document_from_dict(saved_dict())
        assert isinstance(loaded.front_matter, tuple)
        assert isinstance(loaded.root_ids, tuple)
        assert isinstance(loaded.nodes, tuple)
        assert isinstance(loaded.end_matter, tuple)
        node = loaded.get_node("khoan-2")
        assert isinstance(node.title, tuple)
        assert isinstance(node.body, tuple)
        assert isinstance(node.children_ids, tuple)

    def test_lookup_works_after_load(self) -> None:
        loaded = logical_document_from_dict(saved_dict())
        assert loaded.get_node("khoan-2").body[1].block_id == "pb8"

    def test_order_preserved(self) -> None:
        loaded = logical_document_from_dict(saved_dict())
        assert [n.id for n in loaded.nodes] == [
            "chuong-1",
            "dieu-1",
            "khoan-1",
            "khoan-2",
            "dieu-2",
            "chuong-2",
        ]

    def test_load_is_idempotent(self) -> None:
        once = logical_document_to_dict(full_document())
        twice = logical_document_to_dict(logical_document_from_dict(once))
        assert once == twice


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


class TestFiles:
    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.logical.json"
        doc = full_document()
        save_logical_document(doc, path)
        assert load_logical_document(path) == doc

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = str(tmp_path / "doc.json")
        doc = full_document()
        save_logical_document(doc, path)
        assert load_logical_document(path) == doc

    def test_creates_parent_folders(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "doc.json"
        save_logical_document(full_document(), path)
        assert path.is_file()

    def test_vietnamese_written_as_is(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.json"
        doc = make_logical_document(
            root_ids=("điều-1",),
            nodes=(make_logical_node(id="điều-1", level="điều"),),
        )
        save_logical_document(doc, path)
        raw = path.read_text(encoding="utf-8")
        assert "điều-1" in raw
        assert "\\u" not in raw
        assert load_logical_document(path) == doc


# --------------------------------------------------------------------------- #
# Rejected input
# --------------------------------------------------------------------------- #


class TestRejectedInput:
    def test_wrong_kind(self) -> None:
        d = saved_dict()
        d["kind"] = PHYSICAL_DOCUMENT_KIND
        with pytest.raises(ValueError, match="Unexpected dictionary kind"):
            logical_document_from_dict(d)

    def test_missing_kind(self) -> None:
        d = saved_dict()
        del d["kind"]
        with pytest.raises(ValueError, match="Unexpected dictionary kind"):
            logical_document_from_dict(d)

    def test_wrong_schema_version(self) -> None:
        d = saved_dict()
        d["schema_version"] = LOGICAL_SCHEMA_VERSION + 1
        with pytest.raises(
            ValueError, match=f"Expected: {LOGICAL_SCHEMA_VERSION}. Actual: {LOGICAL_SCHEMA_VERSION + 1}"
        ):
            logical_document_from_dict(d)

    @pytest.mark.parametrize(
        "field",
        ["id", "physical_document_id", "source_file_hash", "root_ids", "nodes", "end_matter"],
    )
    def test_missing_document_field(self, field: str) -> None:
        d = saved_dict()
        del d["document"][field]
        with pytest.raises(KeyError, match=field):
            logical_document_from_dict(d)

    @pytest.mark.parametrize("field", ["level", "number", "marker", "children_ids"])
    def test_missing_node_field(self, field: str) -> None:
        d = saved_dict()
        del d["document"]["nodes"][0][field]
        with pytest.raises(KeyError, match=field):
            logical_document_from_dict(d)

    def test_physical_document_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.json"
        path.write_text(
            json.dumps({"kind": PHYSICAL_DOCUMENT_KIND, "schema_version": 1, "document": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unexpected dictionary kind"):
            load_logical_document(path)


class TestValidationRunsOnLoad:
    def test_orphan_node(self) -> None:
        d = saved_dict()
        d["document"]["root_ids"] = ["chuong-1"]  # chuong-2 now unreachable
        with pytest.raises(ValueError, match="unreachable"):
            logical_document_from_dict(d)

    def test_nodes_out_of_order(self) -> None:
        d = saved_dict()
        nodes = d["document"]["nodes"]
        nodes[2], nodes[3] = nodes[3], nodes[2]  # swap khoan-1 and khoan-2
        with pytest.raises(ValueError, match="not in pre-order"):
            logical_document_from_dict(d)

    def test_dangling_child(self) -> None:
        d = saved_dict()
        node = d["document"]["nodes"][1]
        node["children_ids"] = [*node["children_ids"], "khoan-9"]
        with pytest.raises(ValueError, match="no node with id khoan-9"):
            logical_document_from_dict(d)

    def test_invalid_text_range(self) -> None:
        d = saved_dict()
        d["document"]["nodes"][0]["marker"]["text_range"] = {"start": 5, "end": 5}
        with pytest.raises(ValueError, match="0 <= start < end"):
            logical_document_from_dict(d)

    def test_empty_block_id(self) -> None:
        d = saved_dict()
        d["document"]["front_matter"][0]["block_id"] = ""
        with pytest.raises(ValueError, match="Empty block_id"):
            logical_document_from_dict(d)

    def test_empty_number(self) -> None:
        d = saved_dict()
        d["document"]["nodes"][1]["number"] = ""
        with pytest.raises(ValueError, match="'number'"):
            logical_document_from_dict(d)