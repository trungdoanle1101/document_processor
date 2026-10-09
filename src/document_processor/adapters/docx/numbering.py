from dataclasses import dataclass
from lxml.etree import _Element
from ._xml import xpath_str_or_none, xpath_int_or_none, xpath_element_or_none

NUM_ID_VAL_XPATH = "./w:pPr/w:numPr/w:numId/@w:val"
ILVL_VAL_XPATH = "./w:pPr/w:numPr/w:ilvl/@w:val"
ABSTRACT_NUM_ID_VAL_XPATH_TEMPLATE = (
    './w:num[@w:numId="{num_id}"]/w:abstractNumId/@w:val'
)

LEVEL_XPATH_TEMPLATE = (
    './w:abstractNum[@w:abstractNumId="{abstract_num_id}"]/w:lvl[@w:ilvl="{ilvl}"]'
)
NUM_FMT_VAL_XPATH = "./w:numFmt/@w:val"
LVL_TEXT_VAL_XPATH = "./w:lvlText/@w:val"
START_VAL_XPATH = "./w:start/@w:val"


def read_num_id(element: _Element) -> int | None:
    return xpath_int_or_none(element, NUM_ID_VAL_XPATH)


def read_ilvl(element: _Element) -> int | None:
    return xpath_int_or_none(element, ILVL_VAL_XPATH)


def read_abstract_num_id(numbering_root: _Element, num_id: int) -> int:
    path = ABSTRACT_NUM_ID_VAL_XPATH_TEMPLATE.format(num_id=num_id)
    abstract_num_id = xpath_int_or_none(numbering_root, path)
    if abstract_num_id is None:
        raise ValueError(f"numId {num_id}: no <w:num> with this id in numbering.xml")
    return abstract_num_id


def find_level(numbering_root: _Element, abstract_num_id: int, ilvl: int) -> _Element:
    path = LEVEL_XPATH_TEMPLATE.format(
        abstract_num_id=abstract_num_id, ilvl=ilvl
    )
    level = xpath_element_or_none(numbering_root, path)
    if level is None:
        raise ValueError(f"abstractNumId: {abstract_num_id}, ilvl: {ilvl}. Level does not exist")
    return level


def read_num_fmt(level: _Element) -> str | None:
    return xpath_str_or_none(level, NUM_FMT_VAL_XPATH)


def read_lvl_text(level: _Element) -> str | None:
    return xpath_str_or_none(level, LVL_TEXT_VAL_XPATH)


def read_start(level: _Element) -> int | None:
    return xpath_int_or_none(level, START_VAL_XPATH)


@dataclass(frozen=True)
class LevelDefinition:
    """One list level from numbering.xml, with the standard's defaults applied
    """
    num_fmt: str
    lvl_text: str | None
    start: int


def build_level_definition(level: _Element) -> LevelDefinition:
    num_fmt = read_num_fmt(level)
    lvl_text = read_lvl_text(level)
    start = read_start(level)

    return LevelDefinition(
        num_fmt=num_fmt if num_fmt is not None else "decimal", # ECMA-376: missing numFmt means decimal
        lvl_text=lvl_text,
        start=start if start is not None else 0, # ECMA-376: missing start means 0
    )
