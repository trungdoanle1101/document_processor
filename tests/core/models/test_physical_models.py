"""Validation rules of the physical models.

Each test changes one thing from a valid default (see factories.py) and checks
that exactly that rule fires, or that a legal edge case is accepted.
"""

import pytest

from document_processor.core.models.physical import (
    BlockType,
    CharSpan,
    SpanStyle,
    TextRange,
)
from tests.factories import (
    make_block,
    make_document,
    make_image_info,
    make_list_info,
)


def bold(start: int, end: int) -> CharSpan:
    return CharSpan(style=SpanStyle.BOLD, text_range=TextRange(start=start, end=end))


# ---------- PhysicalBlock: text


@pytest.mark.parametrize(
    "text",
    ["", " ", "\t\n", "\u200b", "\u200b ﻿"],
    ids=["empty", "space", "tab-newline", "zero-width-space", "mixed-invisible"],
)
def test_block_rejects_invisible_text(text: str) -> None:
    with pytest.raises(ValueError, match="invisible"):
        make_block(text=text)


def test_error_message_names_the_block() -> None:
    # Guards against f-string slips such as "Block f{self.id}".
    with pytest.raises(ValueError, match=r"^Block pb7: "):
        make_block(id="pb7", text="")


def test_image_block_may_have_empty_text() -> None:
    block = make_block(
        block_type=BlockType.IMAGE, text="", image_info=make_image_info()
    )
    assert block.text == ""


def test_image_block_may_have_alt_text() -> None:
    block = make_block(
        block_type=BlockType.IMAGE,
        text="Sơ đồ kiến trúc dữ liệu",
        image_info=make_image_info(),
    )
    assert block.text == "Sơ đồ kiến trúc dữ liệu"


# ---------- PhysicalBlock: list_info present exactly on LIST_ITEM


def test_list_item_requires_list_info() -> None:
    with pytest.raises(ValueError, match="list_info"):
        make_block(block_type=BlockType.LIST_ITEM)


def test_list_item_with_list_info_is_accepted() -> None:
    block = make_block(block_type=BlockType.LIST_ITEM, list_info=make_list_info())
    assert block.list_info is not None


def test_paragraph_rejects_list_info() -> None:
    with pytest.raises(ValueError, match="list_info"):
        make_block(block_type=BlockType.PARAGRAPH, list_info=make_list_info())


# ---------- PhysicalBlock: image_info present exactly on IMAGE


def test_image_requires_image_info() -> None:
    with pytest.raises(ValueError, match="image_info"):
        make_block(block_type=BlockType.IMAGE, text="")


def test_paragraph_rejects_image_info() -> None:
    with pytest.raises(ValueError, match="image_info"):
        make_block(block_type=BlockType.PARAGRAPH, image_info=make_image_info())


def test_info_fields_default_to_none() -> None:
    block = make_block()
    assert block.list_info is None
    assert block.image_info is None


# ---------- PhysicalBlock: char spans inside the text


def test_span_ending_exactly_at_text_end_is_accepted() -> None:
    block = make_block(text="abc", char_spans=(bold(0, 3),))
    assert block.char_spans[0].text_range.end == 3


def test_span_past_text_end_is_rejected() -> None:
    with pytest.raises(ValueError, match="char span 1 ends at 4, past text length 3"):
        make_block(text="abc", char_spans=(bold(0, 1), bold(2, 4)))


def test_image_with_empty_text_cannot_have_spans() -> None:
    with pytest.raises(ValueError, match="char span 0"):
        make_block(
            block_type=BlockType.IMAGE,
            text="",
            image_info=make_image_info(),
            char_spans=(bold(0, 1),),
        )


# ---------- ImageInfo


def test_valid_image_info_is_accepted() -> None:
    info = make_image_info()
    assert info.mime_type == "image/png"


def test_image_info_sizes_may_be_unknown() -> None:
    info = make_image_info(displayed_width_pt=None, displayed_height_pt=None)
    assert info.displayed_width_pt is None
    assert info.displayed_height_pt is None


@pytest.mark.parametrize("source_path", ["", "   ", "\u200b"])
def test_image_info_rejects_invisible_source_path(source_path: str) -> None:
    with pytest.raises(ValueError, match="source_path"):
        make_image_info(source_path=source_path)


@pytest.mark.parametrize(
    "mime_type",
    ["image/png", "image/jpeg", "image/svg+xml", "image/x-emf", "image/vnd.ms-photo"],
)
def test_image_info_accepts_mime_type(mime_type: str) -> None:
    assert make_image_info(mime_type=mime_type).mime_type == mime_type


@pytest.mark.parametrize(
    "mime_type",
    [
        "png",
        "image/",
        "/png",
        "Image/PNG",
        "image/png extra",
        "image/png\n",
        "image/.png",
        "application/octet-stream",
    ],
)
def test_image_info_rejects_mime_type(mime_type: str) -> None:
    with pytest.raises(ValueError, match="mime_type"):
        make_image_info(mime_type=mime_type)


def test_image_info_rejects_hash_without_algorithm_prefix() -> None:
    with pytest.raises(ValueError, match="content_hash"):
        make_image_info(content_hash="e3b0c44298fc1c149afbf4c8996fb924")


@pytest.mark.parametrize("field", ["displayed_width_pt", "displayed_height_pt"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_image_info_rejects_non_positive_size(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_image_info(**{field: value})


# ---------- PhysicalDocument


def test_document_with_unique_ids_is_accepted() -> None:
    doc = make_document()
    assert [b.id for b in doc.blocks] == ["pb1", "pb2"]


def test_document_may_have_no_blocks() -> None:
    # A source with only blank lines produces an empty document.
    assert make_document(blocks=()).blocks == ()


def test_document_rejects_duplicate_block_ids() -> None:
    blocks = (make_block(id="pb1"), make_block(id="pb2"), make_block(id="pb1"))
    with pytest.raises(
        ValueError,
        match=r"'nghi-dinh-13\.docx'.*Duplicate block id 'pb1' at position 2",
    ):
        make_document(blocks=blocks)
