"""Tests for pipeline/review.py: format_outline.

One line per node: indentation by depth, the heading label (marker and title;
"[level]" when both are missing) and the segment range. Front and end matter
get a line each, "none" when empty, in both modes. sections_only keeps only
sections (Chương, Mục, Điều), wherever they sit: Điều may be a root, under a
Chương, or under an optional Mục.
"""

from typing import Any

import pytest

from document_processor.core.models import LogicalDocument, LogicalNode, Segment
from document_processor.pipeline import format_outline
from tests.chunking.test_outline_chunker import regulation
from tests.factories import heading_for, make_logical_document, make_logical_node, make_text_segment

REGULATION_OUTLINE = """front matter  [0, 1)
Chương I QUY ĐỊNH CHUNG  [1, 15)
  Điều 1. Phạm vi điều chỉnh  [2, 7)
    1.  [5, 7)
  Điều 2. Giải thích từ ngữ  [7, 15)
    1.  [8, 10)
    2.  [10, 15)
      a)  [12, 15)
end matter  [15, 17)"""


def outlined(*nodes: LogicalNode, n_segments: int, body: tuple[int, int]) -> LogicalDocument:
    segments: list[Segment] = []
    for index in range(n_segments):
        starting = [n for n in nodes if n.start == index]
        segments.append(heading_for(starting[0]) if starting else make_text_segment(block_id=f"pb{index}", text="x"))
    return make_logical_document(segments=tuple(segments), nodes=nodes, body_start=body[0], body_end=body[1])


def node(node_id: str, parent: str | None, start: int, end: int, **overrides: Any) -> LogicalNode:
    fields: dict[str, Any] = {
        "id": node_id, "parent_id": parent, "level": "dieu", "number": None,
        "marker": f"Điều {node_id}.", "title": None, "head_block_ids": (f"h{node_id}",), "start": start, "end": end,
    }
    fields.update(overrides)
    return make_logical_node(**fields)


# ============================================================ the whole outline


def test_full_outline_of_the_regulation() -> None:
    assert format_outline(regulation()) == REGULATION_OUTLINE


def test_returns_text_without_printing(capsys: pytest.CaptureFixture[str]) -> None:
    text = format_outline(regulation())
    assert capsys.readouterr().out == ""
    assert isinstance(text, str) and not text.endswith("\n")


# ============================================================ depth


def test_sections_only() -> None:
    assert format_outline(regulation(), sections_only=True) == """front matter  [0, 1)
Chương I QUY ĐỊNH CHUNG  [1, 15)
  Điều 1. Phạm vi điều chỉnh  [2, 7)
  Điều 2. Giải thích từ ngữ  [7, 15)
end matter  [15, 17)"""


def test_sections_only_keeps_front_and_end_matter() -> None:
    lines = format_outline(regulation(), sections_only=True).splitlines()
    assert (lines[0], lines[-1]) == ("front matter  [0, 1)", "end matter  [15, 17)")


def test_sections_only_finds_dieu_at_any_depth() -> None:
    # Chương with a Mục, Chương without, and a Điều with no Chương above it: every Điều is shown.
    document = outlined(
        node("I", None, 0, 6, level="chuong", marker="Chương I"),
        node("m1", "I", 1, 4, level="muc", marker="Mục 1"),
        node("1", "m1", 2, 4),
        node("1-1", "1", 3, 4, level="khoan", is_section=False, marker="1."),
        node("2", "I", 4, 6),
        node("2-1", "2", 5, 6, level="khoan", is_section=False, marker="1."),
        node("3", None, 6, 8),
        node("3-1", "3", 7, 8, level="khoan", is_section=False, marker="1."),
        n_segments=8, body=(0, 8),
    )
    assert format_outline(document, sections_only=True).splitlines()[1:-1] == [
        "Chương I  [0, 6)",
        "  Mục 1  [1, 4)",
        "    Điều 1.  [2, 4)",
        "  Điều 2.  [4, 6)",
        "Điều 3.  [6, 8)",
    ]


def test_sections_only_on_a_document_of_items_only() -> None:
    document = outlined(node("1", None, 0, 2, level="gach", is_section=False, marker="-"), n_segments=2, body=(0, 2))
    assert format_outline(document, sections_only=True) == "front matter  none\nend matter  none"


@pytest.mark.parametrize("indent", [1, 4])
def test_indent_width(indent: int) -> None:
    lines = format_outline(regulation(), indent=indent).splitlines()
    assert lines[2] == " " * indent + "Điều 1. Phạm vi điều chỉnh  [2, 7)"
    assert lines[7] == " " * (3 * indent) + "a)  [12, 15)"


def test_siblings_after_a_deep_branch_return_to_their_depth() -> None:
    document = outlined(
        node("1", None, 0, 4), node("1a", "1", 1, 3), node("1a-i", "1a", 2, 3), node("2", None, 4, 5),
        n_segments=5, body=(0, 5),
    )
    assert format_outline(document).splitlines()[1:-1] == [
        "Điều 1.  [0, 4)",
        "  Điều 1a.  [1, 3)",
        "    Điều 1a-i.  [2, 3)",
        "Điều 2.  [4, 5)",
    ]


# ============================================================ labels


def test_title_without_a_marker() -> None:
    document = outlined(node("1", None, 0, 2, marker=None, title="Lời nói đầu"), n_segments=2, body=(0, 2))
    assert format_outline(document).splitlines()[1] == "Lời nói đầu  [0, 2)"


def test_neither_marker_nor_title_shows_the_level() -> None:
    document = outlined(node("1", None, 0, 2, level="gach", marker=None, title=None), n_segments=2, body=(0, 2))
    assert format_outline(document).splitlines()[1] == "[gach]  [0, 2)"


# ============================================================ front and end matter


def test_no_front_matter() -> None:
    document = outlined(node("1", None, 0, 2), n_segments=3, body=(0, 2))
    assert format_outline(document).splitlines()[0] == "front matter  none"


def test_no_end_matter() -> None:
    document = outlined(node("1", None, 1, 3), n_segments=3, body=(1, 3))
    assert format_outline(document).splitlines()[-1] == "end matter  none"


def test_document_without_nodes() -> None:
    document = make_logical_document(
        segments=(make_text_segment(block_id="pb1"), make_text_segment(block_id="pb2")), body_start=0, body_end=2
    )
    assert format_outline(document) == "front matter  none\nend matter  none"


def test_empty_document() -> None:
    assert format_outline(make_logical_document()) == "front matter  none\nend matter  none"