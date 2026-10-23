import math
import re
from dataclasses import dataclass
from itertools import product

from document_processor.core.registry import Registry
from document_processor.core.text_utils import is_blank_char, normalize_nfc
from document_processor.logical.counter_styles import NAME_PATTERN, CounterStyle
from document_processor.logical.markers import MatchSource


_WHITESPACE_PATTERN = r"[^\S\r\n]+"
_BOUNDARY_PATTERN = r"(?=\s|$)"
_MAX_VARIANTS = 16


@dataclass(frozen=True, kw_only=True)
class Numeral:
    style: str
    text: str
    value: int
    start: int
    end: int


@dataclass(frozen=True, kw_only=True)
class TemplateMatch:
    source: MatchSource
    template: str
    numerals: tuple[Numeral, ...]
    end: int


@dataclass(frozen=True, kw_only=True)
class _Slot:
    styles: tuple[str, ...]


type _TemplatePart = str | _Slot


@dataclass(frozen=True, kw_only=True)
class _Variant:
    template: str
    regex: re.Pattern[str]
    # one per slot, in slot order
    styles: tuple[CounterStyle, ...]

    def match(self, s: str, *, full: bool, source: MatchSource) -> TemplateMatch | None:
        m = self.regex.fullmatch(s) if full else self.regex.match(s)
        if m is None:
            return None
        
        numerals: list[Numeral] = []
        for k, style in enumerate(self.styles):
            group = f"s{k}"
            numeral_text = m.group(group)
            numeral_value = style.to_number(numeral_text)
            if numeral_value is None:
                return None

            start, end = m.span(group)
            numerals.append(
                Numeral(
                    style=style.name,
                    text=numeral_text,
                    value=numeral_value,
                    start=start,
                    end=end
                )
            )
        return TemplateMatch(
            source=source, template=self.template,
            numerals=tuple(numerals), end=m.end()
        )


@dataclass(frozen=True, kw_only=True)
class TemplateMarker:
    variants: tuple[_Variant, ...]

    def match_text(self, text: str) -> list[TemplateMatch]:
        return self._match(text, full=False, source="text")

    def match_list_marker(self, displayed_marker: str) -> list[TemplateMatch]:
        return self._match(displayed_marker, full=True, source="list_marker")
    
    def _match(self, s: str, *, full: bool, source: MatchSource) -> list[TemplateMatch]:
        results: list[TemplateMatch] = []
        
        for v in self.variants:
            marker_match = v.match(s, full=full, source=source)
            if marker_match is None:
                continue
            results.append(marker_match)
           
        return results



def _parse_slot(slot_content: str) -> _Slot:
    styles = slot_content.split("|")
    seen: set[str] = set()
    for style in styles:
        if style in seen:
            raise ValueError(f"style {style!r} appears multiple times in slot {{{slot_content}}}")
        if not NAME_PATTERN.fullmatch(style):
            raise ValueError(f"invalid style string {style!r}. Must fully match '[a-z][a-z0-9_]*'")
        seen.add(style)
    return _Slot(styles=tuple(styles))


def _check_blanks(template: str) -> None:
    if not template:
        raise ValueError("marker template: must not be empty")

    if template.strip() != template:
        raise ValueError(f"marker template {template!r}: must not contain leading or trailing white spaces")

    for i, c in enumerate(template):
        if is_blank_char(c) and c != " ":
            raise ValueError(f"marker template {template!r}: found whitespace character (not plain space) at column {i + 1}. Hint: use plain space only")


def _check_parts(parts: list[_TemplatePart], template: str) -> None:
    for left, right in zip(parts, parts[1:]):
        if isinstance(left, _Slot) and isinstance(right, _Slot):
            raise ValueError(f"marker template {template!r}: two slots side by side not allowed")

    has_literal = any(isinstance(part, str) and part != " " for part in parts)
    if not has_literal:
        raise ValueError(f"marker template {template!r}: no literal text")
    

def _parse_template(template: str) -> tuple[_TemplatePart, ...]:
    _check_blanks(template)

    i = 0
    parts: list[_TemplatePart] = []
    buffer: list[str] = []
    n = len(template)

    def flush() -> None:
        if buffer:
            parts.append("".join(buffer))
            buffer.clear()

    def add_space() -> None:
        flush()
        if parts and parts[-1] != " ":
            parts.append(" ")

    while i < n:
        c = template[i]
        
        if c == " ":
            add_space()
        elif c == "{":
            if i + 1 < n and template[i+1] == "{":
                buffer.append("{")
                i += 1
            else:
                content_start = i + 1
                content_stop = template.find("}", i + 1)
                
                if content_stop == -1:
                    raise ValueError(f"marker template {template!r}, column {i + 1}: brace '{{' is never closed")

                slot_content = template[content_start : content_stop]

                try:
                    slot = _parse_slot(slot_content)
                except ValueError as e:
                    raise ValueError(f"marker template {template!r}, column {content_start}: {e}") from None
                
                flush()
                parts.append(slot)
                i = content_stop
        elif c == "}":
            if i + 1 < n and template[i+1] == "}":
                buffer.append("}")
                i += 1
            else:
                raise ValueError(f"marker template {template!r}, column {i + 1}: lone brace '}}'")
        else:
            buffer.append(c)

        i += 1

    flush()

    _check_parts(parts, template)

    return tuple(parts)


def _literal_pattern(text: str, case_sensitive: bool) -> str:
    literal = re.escape(text)
    return literal if case_sensitive else f"(?i:{literal})"


def _build_regex(
    parts: tuple[_TemplatePart, ...], styles: tuple[CounterStyle, ...], case_sensitive: bool
) -> re.Pattern[str]:
    pieces: list[str] = []
    slots = iter(enumerate(styles))
    for part in parts:
        if isinstance(part, _Slot):
            slot_index, style = next(slots)
            pieces.append(f"(?P<s{slot_index}>{style.numeral_pattern})")
        elif part == " ":
            pieces.append(_WHITESPACE_PATTERN)
        else:
            pieces.append(_literal_pattern(part, case_sensitive))
    pattern = "".join(pieces) + _BOUNDARY_PATTERN
    return re.compile(pattern)


def _compile_variants(
    template: str, styles_registry: Registry[CounterStyle], case_sensitive: bool = False
) -> list[_Variant]:
    
    normalized_template = normalize_nfc(template)
    parts = _parse_template(normalized_template)
    slots = [part for part in parts if isinstance(part, _Slot)]

    try:
        slot_styles = [tuple(styles_registry.get(name) for name in s.styles) for s in slots]
    except KeyError as e:
        raise ValueError(f"marker template {template!r}: {e.args[0]}") from None

    count = math.prod(len(option) for option in slot_styles)
    if count > _MAX_VARIANTS:
        raise ValueError(f"marker template {template!r}: {count} style combinations, at most {_MAX_VARIANTS} allowed")
    
    variants: list[_Variant] = [] 
    for choice in product(*slot_styles):
        regex = _build_regex(parts, choice, case_sensitive)
        variant = _Variant(template=template, regex=regex, styles=choice)
        variants.append(variant)

    return variants


def compile_template_marker(
    templates: list[str], styles_registry: Registry[CounterStyle], *, case_sensitive: bool = False
) -> TemplateMarker:
    if isinstance(templates, str):
        raise ValueError("templates must be a list of string, not one string")

    if not templates:
        raise ValueError("a template marker needs at least one template")

    variants: list[_Variant] = []
    for template in templates:
        variants.extend(_compile_variants(template, styles_registry, case_sensitive))
    return TemplateMarker(variants=tuple(variants))



