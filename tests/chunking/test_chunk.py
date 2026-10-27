"""Tests for chunking/chunk.py: PathStep and Chunk.

A chunk covers segments [start, end) of a logical document, optionally narrowed
inside one segment (rows of a table, characters of a text). Its path is the
outline from the root down to the node that owns it, and headings are the
labels of that whole path: the breadcrumb for display, citations and payloads.
A step's label is its marker and title; a node without a title (an item) uses
its lead-in instead, the capped opening words the chunker found for it.
Each node starts with its heading segment, so a chunk starting at its owner's
start already has the owner's heading as its first line; contextualize() puts in
front of the text only the headings not already in it (step.start < chunk.start),
so nothing is embedded twice.
"""

import dataclasses
import json
from typing import Any

import pytest

from document_processor.chunking.chunk import Chunk, PathStep
from document_processor.core.models import heading_label

CHUONG = PathStep(node_id="n0", level="chuong", number="I", marker="Chương I", title="QUY ĐỊNH CHUNG", start=1)
DIEU = PathStep(node_id="n4", level="dieu", number="2", marker="Điều 2.", title="Giải thích từ ngữ", start=12)
KHOAN = PathStep(node_id="n14", level="khoan", number="10", marker="10.", title=None, start=40)


def make_chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "id": "c0",
        "logical_document_id": "ld1",
        "part": "body",
        "path": (CHUONG, DIEU, KHOAN),
        "start": 41,
        "end": 42,
        "text": "“Hệ thống Log tập trung”: Là hệ thống cho phép tập trung hóa dữ liệu log",
        "token_count": 13,
        "block_ids": ("pb30",),
    }
    fields.update(overrides)
    return Chunk(**fields)


# ============================================================ PathStep


def test_label_is_marker_and_title() -> None:
    assert DIEU.label == "Điều 2. Giải thích từ ngữ"


def test_label_without_a_title_is_the_marker() -> None:
    assert KHOAN.label == "10."


def test_label_without_a_marker_is_the_title() -> None:
    assert PathStep(node_id="n1", level="x", number=None, marker=None, title="Tiêu đề", start=0).label == "Tiêu đề"


def test_label_is_heading_label() -> None:
    # The breadcrumb and the rendered heading segment spell a heading the same way.
    assert heading_label("Điều 2.", "Giải thích từ ngữ") == DIEU.label
    assert heading_label(None, None) == ""


def test_label_without_a_title_uses_the_lead_in() -> None:
    step = dataclasses.replace(KHOAN, lead_in="Các hệ thống phải đáp ứng các yêu cầu sau:")
    assert step.label == "10. Các hệ thống phải đáp ứng các yêu cầu sau:"


def test_lead_in_defaults_to_none() -> None:
    assert KHOAN.lead_in is None and KHOAN.label == "10."


def test_lead_in_without_a_marker() -> None:
    step = PathStep(node_id="n1", level="x", number=None, marker=None, title=None, start=0, lead_in="Mở đầu")
    assert step.label == "Mở đầu"


def test_title_and_lead_in_together_rejected() -> None:
    with pytest.raises(ValueError, match="'n4' at level 'dieu': title and lead-in cannot both be set"):
        dataclasses.replace(DIEU, lead_in="Trong Quy định này")


@pytest.mark.parametrize("lead_in", ["", "   ", "\n\t", "\u200b"], ids=["empty", "spaces", "whitespace", "zero-width"])
def test_invisible_lead_in_rejected(lead_in: str) -> None:
    with pytest.raises(ValueError, match="'n14' at level 'khoan': lead-in is empty or contains invisible text"):
        dataclasses.replace(KHOAN, lead_in=lead_in)


def test_headings_and_contextualize_use_the_lead_in() -> None:
    khoan = dataclasses.replace(KHOAN, lead_in="Hệ thống gồm:")
    chunk = make_chunk(path=(CHUONG, DIEU, khoan))
    assert chunk.headings[-1] == "10. Hệ thống gồm:"
    assert chunk.contextualize().split("\n")[:3] == [
        "Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "10. Hệ thống gồm:"
    ]


def test_path_step_rejects_a_negative_start() -> None:
    with pytest.raises(ValueError, match="start >= 0"):
        PathStep(node_id="n1", level="x", number=None, marker=None, title=None, start=-1)


def test_path_step_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DIEU.title = "Khác"  # type: ignore[misc]


# ============================================================ Chunk fields


def test_keeps_its_fields() -> None:
    chunk = make_chunk()
    assert (chunk.id, chunk.logical_document_id, chunk.part) == ("c0", "ld1", "body")
    assert chunk.path == (CHUONG, DIEU, KHOAN)
    assert (chunk.start, chunk.end) == (41, 42)
    assert chunk.table_rows is None and chunk.text_span is None
    assert chunk.block_ids == ("pb30",)


def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_chunk().text = "khác"  # type: ignore[misc]


def test_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        Chunk("c0", "ld1", "body", (), 0, 1, None, None, "x", ())  # type: ignore[call-arg, arg-type]


# ---------- token_count


def test_keeps_its_token_count() -> None:
    assert make_chunk(token_count=42).token_count == 42


def test_token_count_may_be_zero() -> None:
    assert make_chunk(token_count=0).token_count == 0


def test_negative_token_count_rejected() -> None:
    with pytest.raises(ValueError, match="token_count >= 0"):
        make_chunk(token_count=-1)


# ---------- headings


def test_headings_are_the_path_labels_root_first() -> None:
    assert make_chunk().headings == ("Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "10.")


def test_headings_include_the_owner_even_when_its_heading_is_in_the_text() -> None:
    # The breadcrumb names every node containing the chunk, wherever the chunk starts.
    chunk = make_chunk(start=40, end=42, text="10. “Hệ thống Log tập trung”: …")
    assert chunk.headings == ("Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "10.")


def test_chunks_of_one_owner_share_their_headings() -> None:
    at_owner_start = make_chunk(start=40, end=41, text="10. Đầu khoản")
    later = make_chunk(start=41, end=42, text="Phần sau")
    assert at_owner_start.headings == later.headings


def test_a_root_chunk_has_its_root_as_heading() -> None:
    chunk = make_chunk(path=(CHUONG,), start=1, end=5, text="Chương I QUY ĐỊNH CHUNG\n…")
    assert chunk.headings == ("Chương I QUY ĐỊNH CHUNG",)


def test_headings_cannot_be_passed_in() -> None:
    with pytest.raises(TypeError):
        make_chunk(headings=("khác",))


def test_no_path_no_headings() -> None:
    assert make_chunk(part="front_matter", path=()).headings == ()


def test_headings_are_in_asdict_and_json() -> None:
    data = dataclasses.asdict(make_chunk())
    assert data["headings"] == ("Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "10.")
    assert data["path"][2]["start"] == 40
    assert json.loads(json.dumps(data, ensure_ascii=False))["headings"][1] == "Điều 2. Giải thích từ ngữ"


def test_chunks_compare_by_their_fields() -> None:
    assert make_chunk() == make_chunk()
    assert make_chunk() != make_chunk(text="khác")


# ---------- node_id


def test_node_id_is_the_owner() -> None:
    assert make_chunk().node_id == "n14"


def test_node_id_outside_every_node() -> None:
    assert make_chunk(part="front_matter", path=()).node_id is None


def test_node_id_of_a_body_chunk_outside_every_node() -> None:
    # Body text between root nodes (a closing remark after a Chương) has no owner.
    assert make_chunk(path=()).node_id is None


# ---------- contextualize


def test_contextualize_puts_the_headings_first() -> None:
    chunk = make_chunk(text="Nội dung")
    assert chunk.contextualize() == "Chương I QUY ĐỊNH CHUNG\nĐiều 2. Giải thích từ ngữ\n10.\nNội dung"


def test_contextualize_leaves_out_the_owner_heading_already_in_the_text() -> None:
    # Starting at the owner's start means the text begins with "10."; no second copy.
    chunk = make_chunk(start=40, end=42, text="10. “Hệ thống Log tập trung”: …")
    assert chunk.contextualize() == "Chương I QUY ĐỊNH CHUNG\nĐiều 2. Giải thích từ ngữ\n10. “Hệ thống Log tập trung”: …"


def test_contextualize_of_a_root_chunk_starting_at_its_owner_is_the_text() -> None:
    chunk = make_chunk(path=(CHUONG,), start=1, end=5, text="Chương I QUY ĐỊNH CHUNG\n…")
    assert chunk.contextualize() == "Chương I QUY ĐỊNH CHUNG\n…"


def test_contextualize_of_chunks_of_one_owner_reads_alike() -> None:
    # Whether the owner's heading is in the text or not, it appears exactly once.
    at_owner_start = make_chunk(start=40, end=41, text="10. Đầu khoản")
    later = make_chunk(start=41, end=42, text="Phần sau")
    assert at_owner_start.contextualize().count("10.") == 1
    assert later.contextualize().splitlines()[:3] == ["Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "10."]


def test_contextualize_without_a_path_is_the_text() -> None:
    assert make_chunk(part="end_matter", path=(), text="Nơi nhận:").contextualize() == "Nơi nhận:"


def test_text_never_contains_the_headings() -> None:
    chunk = make_chunk(text="Nội dung")
    assert chunk.text == "Nội dung"


# ============================================================ checks


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (3, 3), (5, 4)])
def test_range_must_be_non_empty(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="start < end"):
        make_chunk(start=start, end=end)


def test_path_starts_must_increase() -> None:
    late_dieu = PathStep(node_id="n4", level="dieu", number="2", marker="Điều 2.", title=None, start=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        make_chunk(path=(CHUONG, late_dieu, KHOAN))


def test_owner_cannot_start_after_the_chunk() -> None:
    with pytest.raises(ValueError, match=r"owner must not start after the chunk.*owner start=40 > chunk start=30"):
        make_chunk(start=30, end=31)


def test_a_range_of_several_segments() -> None:
    assert (make_chunk(start=50, end=54).start, make_chunk(start=50, end=54).end) == (50, 54)


@pytest.mark.parametrize("part", ["front_matter", "end_matter"])
def test_front_and_end_matter_have_no_path(part: str) -> None:
    with pytest.raises(ValueError, match="path"):
        make_chunk(part=part)
    assert make_chunk(part=part, path=()).part == part


def test_table_rows() -> None:
    chunk = make_chunk(start=47, end=48, table_rows=(1, 12))
    assert chunk.table_rows == (1, 12)


def test_text_span() -> None:
    chunk = make_chunk(start=47, end=48, text_span=(0, 400))
    assert chunk.text_span == (0, 400)


def test_not_both_table_rows_and_text_span() -> None:
    with pytest.raises(ValueError, match="simultaneously"):
        make_chunk(start=47, end=48, table_rows=(0, 2), text_span=(0, 10))


@pytest.mark.parametrize("narrowing", ["table_rows", "text_span"])
def test_narrowing_needs_exactly_one_segment(narrowing: str) -> None:
    with pytest.raises(ValueError, match="end == start \\+ 1"):
        make_chunk(start=47, end=49, **{narrowing: (0, 2)})


@pytest.mark.parametrize("bad", [(2, 2), (3, 1), (-1, 2)])
def test_table_rows_must_be_a_non_empty_range(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="table_rows"):
        make_chunk(start=47, end=48, table_rows=bad)


@pytest.mark.parametrize("bad", [(5, 5), (9, 3), (-2, 4)])
def test_text_span_must_be_a_non_empty_range(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="text_span"):
        make_chunk(start=47, end=48, text_span=bad)


def test_errors_name_the_chunk() -> None:
    with pytest.raises(ValueError, match="'c7'"):
        make_chunk(id="c7", start=2, end=2)