from dataclasses import dataclass

from document_processor.core.text_utils import is_blank_char
from .counter_styles import NAME_PATTERN


@dataclass(frozen=True, kw_only=True)
class Numeral:
    style: str
    text: str
    value: int
    start: int
    end: int


@dataclass(frozen=True, kw_only=True)
class MarkerMatch:
    template: str
    numerals: tuple[Numeral, ...]
    end: int


@dataclass(frozen=True)
class Slot:
    styles: tuple[str, ...]


type TemplatePart = str | Slot


def _parse_slot(slot_content: str) -> Slot:
    styles = slot_content.split("|")
    seen: set[str] = set()
    for style in styles:
        if style in seen:
            raise ValueError(f"style {style!r} appears multiple times in slot {{{slot_content}}}")
        if not NAME_PATTERN.fullmatch(style):
            raise ValueError(f"invalid style string {style!r}. Must fully match '[a-z][a-z0-9_]*'")
        seen.add(style)
    return Slot(styles=tuple(styles))



def parse_template(template_str: str) -> tuple[TemplatePart, ...]:
    if not template_str:
        raise ValueError("marker template: must not be empty")

    if template_str.strip() != template_str:
        raise ValueError(f"marker template {template_str!r}: must not contain leading or trailing white spaces")

    for i, c in enumerate(template_str):
        if is_blank_char(c) and c != " ":
            raise ValueError(f"marker template {template_str!r}: found whitespace character (not plain space) at column {i + 1}. Hint: use plain space only")

    i = 0
    parts: list[TemplatePart] = []
    buffer: list[str] = []
    n = len(template_str)

    def flush() -> None:
        if buffer:
            parts.append("".join(buffer))
            buffer.clear()

    def add_space() -> None:
        flush()
        if parts and parts[-1] != " ":
            parts.append(" ")

    while i < n:
        c = template_str[i]
        
        if c == " ":
            add_space()
        elif c == "{":
            if i + 1 < n and template_str[i+1] == "{":
                buffer.append("{")
                i += 1
            else:
                content_start = i + 1
                content_stop = template_str.find("}", i + 1)
                
                if content_stop == -1:
                    raise ValueError(f"marker template {template_str!r}, column {i + 1}: brace '{{' is never closed")

                slot_content = template_str[content_start : content_stop]

                try:
                    slot = _parse_slot(slot_content)
                except ValueError as e:
                    raise ValueError(f"marker template {template_str!r}, column {content_start}: {e}") from None
                
                flush()
                parts.append(slot)
                i = content_stop
        elif c == "}":
            if i + 1 < n and template_str[i+1] == "}":
                buffer.append("}")
                i += 1
            else:
                raise ValueError(f"marker template {template_str!r}, column {i + 1}: lone brace '}}'")
        else:
            buffer.append(c)


        i += 1

    flush()

    for left, right in zip(parts, parts[1:]):
        if isinstance(left, Slot) and isinstance(right, Slot):
            raise ValueError(f"marker template {template_str!r}: two slots side by side not allowed")

    has_literal = any(isinstance(part, str) and part != " " for part in parts)
    if not has_literal:
        raise ValueError(f"marker template {template_str!r}: no literal text")

    return tuple(parts)
