import json

import pytest
from pathlib import Path
from document_processor.core.models.physical import (
    BlockType,
    CharSpan,
    DocSourceFormat,
    ListInfo,
    ImageInfo,
    ParagraphAlignment,
    ParagraphFormat,
    PhysicalBlock,
    PhysicalDocument,
    Region,
    SpanStyle,
)
from document_processor.core.models.ranges import TextRange
from document_processor.core.serialization import (
    PHYSICAL_SCHEMA_VERSION,
    physical_document_from_dict,
    physical_document_to_dict,
    save_physical_document,
    load_physical_document
)


def make_block(**overrides) -> PhysicalBlock:
    """A plain body paragraph; tests override only the fields they care about."""
    fields = dict(
        id="pb1",
        block_type=BlockType.PARAGRAPH,
        text="Plain text",
        region=Region.BODY,
        page=None,
        paragraph_format=ParagraphFormat(alignment=None, style_name=None),
        char_spans=(),
        list_info=None,
        image_info=None,
        source_reference="paragraph:0",
        changelog=(),
    )
    fields.update(overrides)
    return PhysicalBlock(**fields)


@pytest.fixture
def rich_document() -> PhysicalDocument:
    blocks = (
        # Centred chapter heading: alignment + style name + two overlapping spans
        make_block(
            id="pb1",
            text="Chương I",
            paragraph_format=ParagraphFormat(
                alignment=ParagraphAlignment.CENTER, style_name="Heading 7"
            ),
            char_spans=(
                CharSpan(style=SpanStyle.BOLD, text_range=TextRange(start=0, end=8)),
                CharSpan(style=SpanStyle.ITALIC, text_range=TextRange(start=7, end=8)),
            ),
            page=1,
            source_reference="paragraph:0",
        ),
        # Numbered list item with a displayed marker
        make_block(
            id="pb2",
            block_type=BlockType.LIST_ITEM,
            text="Cơ quan trung ương.",
            list_info=ListInfo(
                id="L1", is_ordered=True, displayed_marker="1.", nested_level=0
            ),
            source_reference="paragraph:1",
            changelog=("nfc: 1 change",),
        ),
        # Bullet whose marker could not be determined
        make_block(
            id="pb3",
            block_type=BlockType.LIST_ITEM,
            text="Nội dung 1",
            list_info=ListInfo(
                id="L2", is_ordered=False, displayed_marker=None, nested_level=1
            ),
            source_reference="paragraph:2",
        ),
        # Footer text: a non-body region, everything else unknown
        make_block(
            id="pb4", text="Trang 1", region=Region.FOOTER,
            source_reference="footer:0",
        ),
        make_block(
            id="pb5", text="", block_type=BlockType.IMAGE,
            image_info=ImageInfo(
                source_path="word/media/image1.png", mime_type="image/png",
                content_hash="sha256:abvdg", displayed_width_pt=12.3, displayed_height_pt=None,
            ),
            source_reference="image:0"
        )
    )
    return PhysicalDocument(
        id="sha256:abc",
        source_filename="sample.docx",
        source_format=DocSourceFormat.DOCX,
        adapter_name="docx",
        adapter_version="0.1.0",
        file_hash="sha256:abc",
        blocks=blocks,
    )


def test_round_trip_through_dict(rich_document):
    d = physical_document_to_dict(rich_document)
    assert physical_document_from_dict(d) == rich_document


def test_round_trip_through_json_text(rich_document):
    # Real files give lists, not tuples, and strings, not enums
    text = json.dumps(physical_document_to_dict(rich_document), ensure_ascii=False)
    loaded = physical_document_from_dict(json.loads(text))
    assert loaded == rich_document


def test_saved_form_is_plain_json_data(rich_document):
    d = physical_document_to_dict(rich_document)
    assert d["kind"] == "physical_document"
    assert d["schema_version"] == PHYSICAL_SCHEMA_VERSION
    block = d["document"]["blocks"][0]
    assert block["block_type"] == "paragraph"            # enum written as its value
    assert block["paragraph_format"]["alignment"] == "center"
    json.dumps(d)                                         # must not raise


def test_wrong_kind_is_rejected(rich_document):
    d = physical_document_to_dict(rich_document)
    d["kind"] = "logical_document"
    with pytest.raises(ValueError, match="kind"):
        physical_document_from_dict(d)


def test_wrong_schema_version_is_rejected(rich_document):
    d = physical_document_to_dict(rich_document)
    d["schema_version"] = PHYSICAL_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema version"):
        physical_document_from_dict(d)


def test_save_then_load_from_string_path(rich_document, tmp_path):
    save_path = tmp_path / "out" / "doc.json"
    save_physical_document(rich_document, save_path)
    load_path = str(save_path)
    loaded_doc = load_physical_document(load_path)
    assert loaded_doc == rich_document