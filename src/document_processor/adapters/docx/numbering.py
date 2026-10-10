import re
from dataclasses import dataclass, replace

from lxml.etree import _Element

from document_processor.core.models.physical import ListInfo

from ._xml import xpath_element_or_none, xpath_int_or_none, xpath_str_or_none
from .number_formats import format_number

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


LVL_OVERRIDE_XPATH_TEMPLATE = (
    './w:num[@w:numId="{num_id}"]/w:lvlOverride[@w:ilvl="{ilvl}"]'
)
START_OVERRIDE_XPATH = "./w:startOverride/@w:val"
REPLACEMENT_LEVEL_XPATH = "./w:lvl"

PLACEHOLDER_PATTERN = re.compile(r"%([1-9])")


# --- readers
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
    path = LEVEL_XPATH_TEMPLATE.format(abstract_num_id=abstract_num_id, ilvl=ilvl)
    level = xpath_element_or_none(numbering_root, path)
    if level is None:
        raise ValueError(
            f"abstractNumId: {abstract_num_id}, ilvl: {ilvl}. Level does not exist in numbering.xml"
        )
    return level


def read_num_fmt(level: _Element) -> str | None:
    return xpath_str_or_none(level, NUM_FMT_VAL_XPATH)


def read_lvl_text(level: _Element) -> str | None:
    return xpath_str_or_none(level, LVL_TEXT_VAL_XPATH)


def read_start(level: _Element) -> int | None:
    return xpath_int_or_none(level, START_VAL_XPATH)


def find_level_override(
    numbering_root: _Element, num_id: int, ilvl: int
) -> _Element | None:
    path = LVL_OVERRIDE_XPATH_TEMPLATE.format(num_id=num_id, ilvl=ilvl)
    return xpath_element_or_none(numbering_root, path)


def read_start_override(level_override: _Element) -> int | None:
    return xpath_int_or_none(level_override, START_OVERRIDE_XPATH)


def find_replacement_level(level_override: _Element) -> _Element | None:
    return xpath_element_or_none(level_override, REPLACEMENT_LEVEL_XPATH)


# --- level definitions
@dataclass(frozen=True)
class LevelDefinition:
    """One list level from numbering.xml, with the standard's defaults applied"""

    num_fmt: str
    lvl_text: str | None
    start: int

    @property
    def is_ordered(self) -> bool:
        """ False for bullets, 'none', and levels without text
        """
        return not (
            self.num_fmt == "bullet" or self.num_fmt == "none" or self.lvl_text is None
        )


def build_level_definition(level: _Element) -> LevelDefinition:
    num_fmt = read_num_fmt(level)
    lvl_text = read_lvl_text(level)
    start = read_start(level)

    # ECMA-376: missing numFmt means decimal
    num_fmt = num_fmt if num_fmt is not None else "decimal"

    # ECMA-376: missing start means 0
    start = start if start is not None else 0

    return LevelDefinition(num_fmt=num_fmt, lvl_text=lvl_text, start=start)


def _build_original_level_definition(
    numbering_root: _Element, num_id: int, ilvl: int
) -> LevelDefinition:
    abstract_num_id = read_abstract_num_id(numbering_root, num_id)
    original_level = find_level(numbering_root, abstract_num_id, ilvl)
    return build_level_definition(original_level)


def resolve_level_definition(
    numbering_root: _Element, num_id: int, ilvl: int
) -> LevelDefinition:
    level_override = find_level_override(numbering_root, num_id, ilvl)
    if level_override is None:
        return _build_original_level_definition(numbering_root, num_id, ilvl)

    replacement_level = find_replacement_level(level_override)
    if replacement_level is not None:
        definition = build_level_definition(replacement_level)
    else:
        definition = _build_original_level_definition(numbering_root, num_id, ilvl)

    start_override = read_start_override(level_override)
    if start_override is not None:
        definition = replace(definition, start=start_override)

    return definition


# --- counting
class ListCounter:
    def __init__(self) -> None:
        # Keys: num_id, ilvl
        self._counts: dict[int, dict[int, int]] = {}

    def advance(self, num_id: int, ilvl: int, start: int) -> dict[int, int]:
        if num_id not in self._counts:
            self._counts[num_id] = {}

        counts = self._counts[num_id]
        if ilvl not in counts:
            counts[ilvl] = start - 1
        counts[ilvl] += 1

        # Forget deeper levels. They restart at their own next time.
        all_ilvls = list(counts)
        for i in all_ilvls:
            if i > ilvl:
                del counts[i]

        return dict(counts)


# --- markers
def resolve_marker(
    level_def: LevelDefinition, counts: dict[int, int], formats: dict[int, str]
) -> str | None:
    if level_def.num_fmt == "none" or level_def.lvl_text is None:
        return None

    if level_def.num_fmt == "bullet":
        # TODO: Symbol-font bullets come through as Private Use Area characters.
        # Need to map them via w:lvl/w:rPr/w:rFonts
        return level_def.lvl_text

    def _fill(match: re.Match[str]) -> str:
        ilvl = int(match.group(1)) - 1
        try:
            num = counts[ilvl]
        except KeyError as e:
            raise ValueError(
                f"lvlText {level_def.lvl_text!r}: no count for level {ilvl} (%{ilvl + 1})"
            ) from e

        try:
            num_fmt = formats[ilvl]
        except KeyError as e:
            raise ValueError(
                f"lvlText {level_def.lvl_text!r}: no format for level {ilvl} (%{ilvl + 1})"
            ) from e

        return format_number(num=num, num_fmt=num_fmt)

    replaced = PLACEHOLDER_PATTERN.sub(repl=_fill, string=level_def.lvl_text)
    return replaced


# --- numberer
class Numberer:
    def __init__(self, numbering_root: _Element) -> None:
        self._counter = ListCounter()
        self._root = numbering_root
        self._level_defs: dict[tuple[int, int], LevelDefinition] = {}

    def _get_level_def(self, num_id: int, ilvl: int) -> LevelDefinition:
        key = (num_id, ilvl)

        if key not in self._level_defs:
            ld = resolve_level_definition(self._root, num_id, ilvl)
            self._level_defs[key] = ld

        return self._level_defs[key]

    def next_list_info(self, num_id: int, ilvl: int) -> ListInfo:
        level_def = self._get_level_def(num_id, ilvl)

        formats: dict[int, str] = {}
        if level_def.lvl_text is not None:
            all_levels = [
                int(lvl) - 1 for lvl in PLACEHOLDER_PATTERN.findall(level_def.lvl_text)
            ]
            for level in all_levels:
                ld = self._get_level_def(num_id, level)
                formats[level] = ld.num_fmt

        counts = self._counter.advance(num_id, ilvl, level_def.start)

        displayed_marker = resolve_marker(
            level_def=level_def, counts=counts, formats=formats
        )

        return ListInfo(
            id=str(num_id),
            is_ordered=level_def.is_ordered,
            displayed_marker=displayed_marker,
            nested_level=ilvl,
        )
