from lxml.etree import _Element
from ._xml import xpath_str_or_none, xpath_int_or_none

NUM_ID_VAL_XPATH = "./w:pPr/w:numPr/w:numId/@w:val"
ILVL_VAL_XPATH = "./w:pPr/w:numPr/w:ilvl/@w:val"
ABSTRACT_NUM_ID_VAL_XPATH_TEMPLATE = (
    './w:num[@w:numId="{num_id}"]/w:abstractNumId/@w:val'
)

NUM_FMT_XPATH_TEMPLATE = './w:abstractNum[@w:abstractNumId="{abstract_num_id}"]/w:lvl[@w:ilvl="{ilvl}"]/w:numFmt/@w:val'
LVL_TEXT_XPATH_TEMPLATE = './w:abstractNum[@w:abstractNumId="{abstract_num_id}"]/w:lvl[@w:ilvl="{ilvl}"]/w:lvlText/@w:val'
START_VAL_XPATH_TEMPLATE = './w:abstractNum[@w:abstractNumId="{abstract_num_id}"]/w:lvl[@w:ilvl="{ilvl}"]/w:start/@w:val'


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


def read_num_fmt(
    numbering_root: _Element, abstract_num_id: int, ilvl: int
) -> str | None:
    return xpath_str_or_none(
        numbering_root,
        NUM_FMT_XPATH_TEMPLATE.format(abstract_num_id=abstract_num_id, ilvl=ilvl),
    )


def read_lvl_text(
    numbering_root: _Element, abstract_num_id: int, ilvl: int
) -> str | None:
    return xpath_str_or_none(
        numbering_root,
        LVL_TEXT_XPATH_TEMPLATE.format(abstract_num_id=abstract_num_id, ilvl=ilvl),
    )


def read_start(numbering_root: _Element, abstract_num_id: int, ilvl: int) -> int | None:
    return xpath_int_or_none(
        numbering_root,
        START_VAL_XPATH_TEMPLATE.format(abstract_num_id=abstract_num_id, ilvl=ilvl),
    )
