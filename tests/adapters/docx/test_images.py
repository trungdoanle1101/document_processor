"""Tests for adapters/docx/images.py: one <w:drawing> -> ImageInfo or None.

Drawings are built from XML strings (see drawing_xml.py); related parts are
real python-docx Parts, so no .docx file is needed.
"""

import pytest
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from lxml import etree
from lxml.etree import _Element

from document_processor.adapters.docx.images import (
    EMU_PER_PT,
    read_drawing_image,
)
from document_processor.core.identity import compute_hash

from tests.drawing_xml import EMU_INCH, chart, drawing, pic

PNG_BYTES = b"\x89PNG fake image bytes"


def parse(xml: str) -> _Element:
    return etree.fromstring(xml)


def image_part(
    partname: str = "/word/media/image1.png",
    content_type: str = "image/png",
    blob: bytes = PNG_BYTES,
) -> Part:
    return Part(PackURI(partname), content_type, blob)


def parts(**by_rid: Part) -> dict[str, Part]:
    return dict(by_rid)


DEFAULT_PARTS = parts(rId10=image_part())


# ---------- pictures


def test_inline_picture() -> None:
    info = read_drawing_image(parse(drawing()), DEFAULT_PARTS)
    assert info is not None
    assert info.source_path == "word/media/image1.png"  # no leading "/"
    assert info.mime_type == "image/png"
    assert info.content_hash == compute_hash(PNG_BYTES)


def test_anchored_picture() -> None:
    inline = read_drawing_image(parse(drawing()), DEFAULT_PARTS)
    anchored = read_drawing_image(parse(drawing(anchor=True)), DEFAULT_PARTS)
    assert anchored is not None
    assert anchored == inline


def test_picture_found_by_its_relationship_id() -> None:
    related = parts(
        rId10=image_part("/word/media/image1.png", blob=b"first"),
        rId11=image_part("/word/media/image2.jpeg", "image/jpeg", b"second"),
    )
    info = read_drawing_image(parse(drawing(pic("rId11"))), related)
    assert info is not None
    assert info.source_path == "word/media/image2.jpeg"
    assert info.mime_type == "image/jpeg"
    assert info.content_hash == compute_hash(b"second")


def test_vector_picture_is_accepted() -> None:
    # Diagrams pasted from Visio or PowerPoint often arrive as EMF.
    related = parts(rId10=image_part("/word/media/image1.emf", "image/x-emf"))
    info = read_drawing_image(parse(drawing()), related)
    assert info is not None
    assert info.mime_type == "image/x-emf"


def test_same_bytes_give_the_same_hash() -> None:
    related = parts(
        rId10=image_part("/word/media/image1.png"),
        rId11=image_part("/word/media/image7.png"),
    )
    first = read_drawing_image(parse(drawing(pic("rId10"))), related)
    second = read_drawing_image(parse(drawing(pic("rId11"))), related)
    assert first is not None and second is not None
    assert first.content_hash == second.content_hash
    assert first.source_path != second.source_path


# ---------- displayed size


def test_size_is_converted_from_emu_to_points() -> None:
    info = read_drawing_image(parse(drawing(cx=EMU_INCH, cy=EMU_INCH // 2)), DEFAULT_PARTS)
    assert info is not None
    assert info.displayed_width_pt == 72.0
    assert info.displayed_height_pt == 36.0


def test_emu_per_point() -> None:
    assert EMU_PER_PT == 12700


def test_missing_extent_means_unknown_size() -> None:
    info = read_drawing_image(parse(drawing(extent=False)), DEFAULT_PARTS)
    assert info is not None
    assert info.displayed_width_pt is None
    assert info.displayed_height_pt is None


def test_missing_extent_attribute_means_unknown_for_that_side() -> None:
    info = read_drawing_image(parse(drawing(cx=None)), DEFAULT_PARTS)
    assert info is not None
    assert info.displayed_width_pt is None
    assert info.displayed_height_pt == 36.0


@pytest.mark.parametrize(("cx", "cy"), [(0, EMU_INCH), (EMU_INCH, 0), (0, 0)])
def test_zero_extent_means_unknown_size(cx: int, cy: int) -> None:
    # A collapsed drawing must not fail the whole document.
    info = read_drawing_image(parse(drawing(cx=cx, cy=cy)), DEFAULT_PARTS)
    assert info is not None
    assert (info.displayed_width_pt is None) == (cx == 0)
    assert (info.displayed_height_pt is None) == (cy == 0)


def test_non_numeric_extent_rejected() -> None:
    xml = drawing().replace(f'cx="{EMU_INCH}"', 'cx="wide"')
    with pytest.raises(ValueError, match="wide"):
        read_drawing_image(parse(xml), DEFAULT_PARTS)


# ---------- drawings that are not pictures


def test_chart_is_not_a_picture() -> None:
    assert read_drawing_image(parse(drawing(chart())), DEFAULT_PARTS) is None


def test_linked_picture_has_no_bytes() -> None:
    linked = drawing(pic(embed=None, link="rId30"))
    assert read_drawing_image(parse(linked), DEFAULT_PARTS) is None


def test_drawing_without_container_is_skipped() -> None:
    assert read_drawing_image(parse(drawing(container=False)), DEFAULT_PARTS) is None


# ---------- broken drawings


def test_unknown_relationship_id_rejected() -> None:
    with pytest.raises(ValueError, match="'rId99'"):
        read_drawing_image(parse(drawing(pic("rId99"))), DEFAULT_PARTS)


def test_unknown_relationship_id_raises_value_error_not_key_error() -> None:
    # The reader only wraps ValueError with the file and paragraph.
    with pytest.raises(ValueError) as excinfo:
        read_drawing_image(parse(drawing(pic("rId99"))), {})
    assert not isinstance(excinfo.value, KeyError)


def test_related_part_that_is_not_an_image_rejected() -> None:
    related = parts(rId10=image_part("/word/charts/chart1.xml", "application/xml"))
    with pytest.raises(ValueError, match="mime_type"):
        read_drawing_image(parse(drawing()), related)


def test_several_pictures_in_one_drawing_rejected() -> None:
    # A grouped drawing: not supported until a real file needs it.
    related = parts(rId10=image_part(), rId11=image_part("/word/media/image2.png"))
    with pytest.raises(ValueError, match="0 or 1"):
        read_drawing_image(parse(drawing(pic("rId10"), pic("rId11"))), related)