from lxml.etree import _Element
from typing import Mapping
from docx.opc.part import Part
from ._xml import xpath_element_zero_or_more, xpath_element_or_none, xpath_str_or_none, xpath_int_or_none

from document_processor.core.models.physical import ImageInfo
from document_processor.core.identity import compute_hash

P_DRAWING_XPATH = ".//w:drawing"
CONTAINER_XPATH = "./wp:inline | ./wp:anchor"
EXTENT_PATH = "./wp:extent"
BLIP_EMBED_XPATH = ".//a:blip/@r:embed"

EMU_PER_PT = 12700

def find_drawings(p: _Element) -> list[_Element]:
    return xpath_element_zero_or_more(p, P_DRAWING_XPATH)


def find_container(drawing: _Element) -> _Element | None:
    return xpath_element_or_none(drawing, CONTAINER_XPATH)


def find_extent(container: _Element) -> _Element | None:
    return xpath_element_or_none(container, EXTENT_PATH)

def read_embed(drawing: _Element) -> str | None:
    return xpath_str_or_none(drawing, BLIP_EMBED_XPATH)


def read_extent_cx_cy(extent: _Element) -> tuple[int | None, int | None]:
    """Return pair of (cx, cy) from wp:extent
    """
    cx = xpath_int_or_none(extent, "./@cx")
    cy = xpath_int_or_none(extent, "./@cy")

    
    return (cx, cy)



def read_drawing_image(drawing: _Element, related_parts: Mapping[str, Part]) -> ImageInfo | None:
    container = find_container(drawing)
    if container is None:
        return None
    extent = find_extent(container)
    displayed_width_pt: float | None = None
    displayed_height_pt: float | None = None
    if extent is not None:
        cx, cy = read_extent_cx_cy(extent)
        displayed_width_pt = _emu_to_pt(cx)
        if cy is not None:
            displayed_height_pt = _emu_to_pt(cy)

        
    embed = read_embed(drawing)
    if embed is None:
        return None
    try:
        part = related_parts[embed]
    except KeyError:
        raise ValueError(f"image relationship {embed!r} not found")
    return ImageInfo(
        source_path=str(part.partname).lstrip("/"),
        mime_type=part.content_type,
        content_hash=compute_hash(part.blob),
        displayed_width_pt=displayed_width_pt,
        displayed_height_pt=displayed_height_pt
    )


def _emu_to_pt(emu: int | None) -> float | None:
    if emu is None or emu <= 0:
        return None
    return emu / EMU_PER_PT