from docx.oxml.ns import qn
from lxml.etree import _Element

from document_processor.core.models.physical import ParagraphAlignment, ParagraphFormat

from ._xml import (
    read_on_off,
    xpath_element_or_none,
    xpath_element_zero_or_more,
    xpath_str_or_none,
)

ALIGNMENT_MAPPING = {
    "left": ParagraphAlignment.LEFT,
    "start": ParagraphAlignment.LEFT,
    "right": ParagraphAlignment.RIGHT,
    "end": ParagraphAlignment.RIGHT,
    "center": ParagraphAlignment.CENTER,
    "both": ParagraphAlignment.JUSTIFIED,
    "distribute": ParagraphAlignment.JUSTIFIED,
}

JC_VAL_XPATH = "./w:pPr/w:jc/@w:val"
P_STYLE_VAL_XPATH = "./w:pPr/w:pStyle/@w:val"

STYLE_NAME_VAL_XPATH = "./w:name/@w:val"
STYLE_BASED_ON_VAL_XPATH = "./w:basedOn/@w:val"
STYLE_ID_VAL_XPATH = "./@w:styleId"
STYLE_XPATH = "./w:style"

DEFAULT_PARAGRAPH_STYLE_XPATH = './w:style[@w:type="paragraph" and (@w:default="1" or @w:default="true" or @w:default="on")][last()]'
P_PR_DEFAULTS_XPATH = "./w:docDefaults/w:pPrDefault"

BOLD_XPATH = "./w:rPr/w:b"
R_STYLE_VAL_XPATH = "./w:rPr/w:rStyle/@w:val"
R_PR_DEFAULTS_XPATH = "./w:docDefaults/w:rPrDefault"


# --- readers
def read_jc(element: _Element) -> str | None:
    return xpath_str_or_none(element, JC_VAL_XPATH)


def find_p_pr_defaults(styles_root: _Element) -> _Element | None:
    return xpath_element_or_none(styles_root, P_PR_DEFAULTS_XPATH)


def read_p_style(element: _Element) -> str | None:
    return xpath_str_or_none(element, P_STYLE_VAL_XPATH)


def read_style_name(style: _Element) -> str | None:
    return xpath_str_or_none(style, STYLE_NAME_VAL_XPATH)


def read_style_id(style: _Element) -> str | None:
    return xpath_str_or_none(style, STYLE_ID_VAL_XPATH)


def read_style_based_on(style: _Element) -> str | None:
    return xpath_str_or_none(style, STYLE_BASED_ON_VAL_XPATH)


def find_default_paragraph_style(styles_root: _Element) -> _Element | None:
    return xpath_element_or_none(styles_root, DEFAULT_PARAGRAPH_STYLE_XPATH)


def read_bold(element: _Element) -> bool | None:
    b = xpath_element_or_none(element, BOLD_XPATH)
    return read_on_off(b, "w:b") if b is not None else None


def read_r_style(r: _Element) -> str | None:
    return xpath_str_or_none(r, R_STYLE_VAL_XPATH)


def find_r_pr_defaults(styles_root: _Element) -> _Element | None:
    return xpath_element_or_none(styles_root, R_PR_DEFAULTS_XPATH)


# --- Map to ParagraphAlignment
def map_alignment(raw_alignment_str: str) -> ParagraphAlignment:
    if raw_alignment_str not in ALIGNMENT_MAPPING:
        raise ValueError(
            f"Unknown alignment type {raw_alignment_str!r}. Supporting: {sorted(ALIGNMENT_MAPPING)}"
        )
    return ALIGNMENT_MAPPING[raw_alignment_str]


class Stylist:
    def __init__(self, styles_root: _Element | None) -> None:
        self._styles_by_id: dict[str, _Element] = {}
        self._default_paragraph_style: _Element | None = None
        self._doc_default_alignment: ParagraphAlignment | None = None
        self._doc_default_bold: bool = False

        if styles_root is None:
            return

        # Build styles by style_id dict

        styles = xpath_element_zero_or_more(styles_root, STYLE_XPATH)
        for style in styles:
            style_id = read_style_id(style)
            if style_id is None:
                continue
            if style_id in self._styles_by_id:
                raise ValueError(f"Duplicate style_id found: {style_id!r}")
            self._styles_by_id[style_id] = style

        # Get default style
        self._default_paragraph_style = find_default_paragraph_style(styles_root)
        p_pr_defaults = find_p_pr_defaults(styles_root)
        doc_default_jc = read_jc(p_pr_defaults) if p_pr_defaults is not None else None
        self._doc_default_alignment = (
            map_alignment(doc_default_jc) if doc_default_jc is not None else None
        )

        r_pr_defaults = find_r_pr_defaults(styles_root)
        doc_default_bold = (
            read_bold(r_pr_defaults) if r_pr_defaults is not None else None
        )
        self._doc_default_bold = (
            doc_default_bold if doc_default_bold is not None else False
        )

    def resolve_paragraph_format(self, p_el: _Element) -> ParagraphFormat:

        # style_name
        style = self._paragraph_style(p_el)
        style_name = read_style_name(style) if style is not None else None
        style_id = read_style_id(style) if style is not None else None

        # alignment
        alignment = self._paragraph_alignment(p_el, style_id)

        return ParagraphFormat(alignment=alignment, style_name=style_name)

    def resolve_run_is_bold(self, r_el: _Element) -> bool:
        """Direct w:b, then the character style, then the paragraph style, then the document default

        Simplification: in styles w:b is a toggle (a bold character style on a bold
        paragraph style gives plain text); here the nearest level that sets it wins.
        """
        direct = read_bold(r_el)
        if direct is not None:
            return direct

        from_character_style = self._style_chain_bold(read_r_style(r_el))
        if from_character_style is not None:
            return from_character_style

        p_el = next(r_el.iterancestors(qn("w:p")), None)

        paragraph_style = (
            self._paragraph_style(p_el)
            if p_el is not None
            else self._default_paragraph_style
        )
        if paragraph_style is not None:
            from_paragraph_style = self._style_chain_bold(
                read_style_id(paragraph_style)
            )
            if from_paragraph_style is not None:
                return from_paragraph_style

        return self._doc_default_bold

    def _style_chain_bold(self, style_id: str | None) -> bool | None:
        seen_style_ids: set[str] = set()
        while style_id is not None and style_id not in seen_style_ids:
            seen_style_ids.add(style_id)
            style = self._styles_by_id.get(style_id)
            if style is None:
                break
            bold = read_bold(style)
            if bold is not None:
                return bold
            style_id = read_style_based_on(style)
        return None

    def _paragraph_style(self, p_el: _Element) -> _Element | None:
        p_style = read_p_style(p_el)
        if p_style is None or p_style not in self._styles_by_id:
            return self._default_paragraph_style
        return self._styles_by_id[p_style]

    def _paragraph_alignment(
        self, p_el: _Element, style_id: str | None
    ) -> ParagraphAlignment:

        # Check jc of the paragraph element itself
        el_jc = read_jc(p_el)
        if el_jc is not None:
            return map_alignment(el_jc)

        # Check jc in style
        seen_style_ids: set[str] = set()

        while style_id is not None and style_id not in seen_style_ids:
            seen_style_ids.add(style_id)
            style = self._styles_by_id.get(style_id)

            if style is None:
                break

            style_jc = read_jc(style)
            if style_jc is not None:
                return map_alignment(style_jc)

            style_id = read_style_based_on(style)

        # Check document's default
        if self._doc_default_alignment is not None:
            return self._doc_default_alignment

        return ParagraphAlignment.LEFT
