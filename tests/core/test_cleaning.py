"""Tests for core/cleaning.py."""

import unicodedata

import pytest

from document_processor.core.cleaning import (
    DEFAULT_STEPS,
    CleaningStep,
    clean_document,
)
from document_processor.core.models.physical import BlockType, CharSpan, SpanStyle
from document_processor.core.models.ranges import TextRange
from tests.factories import (
    make_block,
    make_document,
    make_image_info,
    make_table_block,
    make_table_cell,
    make_table_info,
)

NFC_DIEU = "Điều 1. Phạm vi điều chỉnh"
NFD_DIEU = unicodedata.normalize("NFD", NFC_DIEU)


def clean_one(text: str, **block_fields: object) -> tuple[str, tuple[str, ...]]:
    """Clean a one-block document; return the block's text and changelog."""
    doc = make_document(blocks=(make_block(text=text, **block_fields),))
    block = clean_document(doc).blocks[0]
    return block.text, block.changelog


# --------------------------------------------------------------------------- #
# Default steps
# --------------------------------------------------------------------------- #


class TestDefaultSteps:
    def test_names_and_order(self) -> None:
        assert [s.name for s in DEFAULT_STEPS] == ["nfc", "strip_edges"]

    def test_nfd_becomes_nfc(self) -> None:
        assert NFD_DIEU != NFC_DIEU  # the fixture really is decomposed
        text, changelog = clean_one(NFD_DIEU)
        assert text == NFC_DIEU
        assert changelog == ("clean:nfc",)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("\tĐiều 1. Phạm vi điều chỉnh", "Điều 1. Phạm vi điều chỉnh"),
            ("Kiến trúc dữ liệu. ", "Kiến trúc dữ liệu."),
            ("  “Cloud native”: Là", "“Cloud native”: Là"),
            ("\t​\tĐiều 2.", "Điều 2."),  # mixed whitespace and zero-width
            (" Mục I ", "Mục I"),  # NBSP
            ("﻿Chương I", "Chương I"),  # BOM
            ("Điều 3.\r\n", "Điều 3."),
        ],
        ids=["tab", "trailing-space", "spaces", "mixed", "nbsp", "bom", "crlf"],
    )
    def test_edges_stripped(self, raw: str, expected: str) -> None:
        text, changelog = clean_one(raw)
        assert text == expected
        assert changelog == ("clean:strip_edges",)

    def test_inner_whitespace_kept(self) -> None:
        raw = "Lớp  tích hợp\tngoài​API"
        text, changelog = clean_one(raw)
        assert text == raw
        assert changelog == ()

    def test_both_steps_recorded_in_order(self) -> None:
        text, changelog = clean_one("\t" + NFD_DIEU + " ")
        assert text == NFC_DIEU
        assert changelog == ("clean:nfc", "clean:strip_edges")

    def test_existing_changelog_kept(self) -> None:
        _, changelog = clean_one("\tĐiều 1.", changelog=("adapter:merged",))
        assert changelog == ("adapter:merged", "clean:strip_edges")


# --------------------------------------------------------------------------- #
# Blocks left alone
# --------------------------------------------------------------------------- #


class TestUnchanged:
    def test_clean_block_returned_as_same_object(self) -> None:
        block = make_block(text=NFC_DIEU)
        doc = make_document(blocks=(block,))
        assert clean_document(doc).blocks[0] is block

    def test_table_untouched(self) -> None:
        # A leading tab is an empty first cell, not stray whitespace.
        info = make_table_info(
            n_rows=1,
            n_header_rows=0,
            cells=(
                make_table_cell(row=0, col=0, paragraphs=()),
                make_table_cell(row=0, col=1, paragraphs=(NFD_DIEU,)),
            ),
        )
        table = make_table_block(table_info=info)
        assert table.text.startswith("\t")
        cleaned = clean_document(make_document(blocks=(table,))).blocks[0]
        assert cleaned is table

    def test_image_untouched(self) -> None:
        image = make_block(
            block_type=BlockType.IMAGE,
            text=" ",
            image_info=make_image_info(),
        )
        cleaned = clean_document(make_document(blocks=(image,))).blocks[0]
        assert cleaned is image

    def test_char_spans_allowed_when_nothing_changes(self) -> None:
        span = CharSpan(style=SpanStyle.BOLD, text_range=TextRange(0, 4))
        block = make_block(text=NFC_DIEU, char_spans=(span,))
        assert clean_document(make_document(blocks=(block,))).blocks[0] is block


# --------------------------------------------------------------------------- #
# Document level
# --------------------------------------------------------------------------- #


class TestDocument:
    def test_ids_order_and_identity_kept(self) -> None:
        doc = make_document(
            blocks=(
                make_block(id="pb1", text="\tĐiều 1."),
                make_block(id="pb2", text="Không đổi"),
                make_block(id="pb3", text=NFD_DIEU),
            )
        )
        cleaned = clean_document(doc)
        assert [b.id for b in cleaned.blocks] == ["pb1", "pb2", "pb3"]
        assert cleaned.id == doc.id
        assert cleaned.file_hash == doc.file_hash
        assert cleaned.source_filename == doc.source_filename

    def test_other_block_fields_kept(self) -> None:
        block = make_block(
            text="\tĐiều 1.", source_reference="/w:document/w:body/w:p[3]"
        )
        cleaned = clean_document(make_document(blocks=(block,))).blocks[0]
        assert cleaned.source_reference == block.source_reference
        assert cleaned.paragraph_format == block.paragraph_format
        assert cleaned.block_type == block.block_type

    def test_input_document_not_modified(self) -> None:
        doc = make_document(blocks=(make_block(text="\tĐiều 1."),))
        clean_document(doc)
        assert doc.blocks[0].text == "\tĐiều 1."
        assert doc.blocks[0].changelog == ()

    def test_running_twice_changes_nothing_more(self) -> None:
        doc = make_document(blocks=(make_block(text="\t" + NFD_DIEU),))
        once = clean_document(doc)
        twice = clean_document(once)
        assert twice == once
        assert twice.blocks[0] is once.blocks[0]

    def test_empty_document(self) -> None:
        doc = make_document(blocks=())
        assert clean_document(doc).blocks == ()


# --------------------------------------------------------------------------- #
# Custom steps
# --------------------------------------------------------------------------- #


class TestCustomSteps:
    def test_no_steps_changes_nothing(self) -> None:
        block = make_block(text="\t" + NFD_DIEU)
        doc = make_document(blocks=(block,))
        assert clean_document(doc, steps=()).blocks[0] is block

    def test_only_given_steps_run(self) -> None:
        doc = make_document(blocks=(make_block(text="\t" + NFD_DIEU),))
        block = clean_document(doc, steps=(DEFAULT_STEPS[1],)).blocks[0]
        assert block.text == NFD_DIEU  # stripped but still decomposed
        assert block.changelog == ("clean:strip_edges",)

    def test_custom_step_name_in_changelog(self) -> None:
        upper = CleaningStep("upper", str.upper)
        doc = make_document(blocks=(make_block(text="điều 1."),))
        block = clean_document(doc, steps=(upper,)).blocks[0]
        assert block.text == "ĐIỀU 1."
        assert block.changelog == ("clean:upper",)

    def test_step_order_respected(self) -> None:
        add_a = CleaningStep("add_a", lambda t: t + "a")
        add_b = CleaningStep("add_b", lambda t: t + "b")
        doc = make_document(blocks=(make_block(text="x"),))
        block = clean_document(doc, steps=(add_b, add_a)).blocks[0]
        assert block.text == "xba"
        assert block.changelog == ("clean:add_b", "clean:add_a")

    def test_step_that_undoes_itself_not_recorded(self) -> None:
        # Each step is compared with its own input, so a no-op step is skipped
        # even when an earlier one changed the text.
        identity = CleaningStep("identity", lambda t: t)
        doc = make_document(blocks=(make_block(text="\tx"),))
        block = clean_document(doc, steps=(DEFAULT_STEPS[1], identity)).blocks[0]
        assert block.changelog == ("clean:strip_edges",)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_step_making_text_invisible(self) -> None:
        erase = CleaningStep("erase", lambda t: "")
        doc = make_document(blocks=(make_block(id="pb7", text="Điều 1."),))
        with pytest.raises(ValueError, match=r"pb7.*clean:erase.*invisible"):
            clean_document(doc, steps=(erase,))

    def test_char_spans_with_changed_text(self) -> None:
        span = CharSpan(style=SpanStyle.BOLD, text_range=TextRange(1, 5))
        block = make_block(id="pb9", text="\tĐiều 1.", char_spans=(span,))
        with pytest.raises(ValueError, match=r"pb9.*char_spans"):
            clean_document(make_document(blocks=(block,)))
