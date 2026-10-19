from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType

from .markers import MatchSource
from .glyph_styles import GlyphStyle
from .registry import Registry

_VARIATION_SELECTORS = frozenset({"\ufe0e", "\ufe0f"})


@dataclass(frozen=True, kw_only=True)
class GlyphMatch:
    source: MatchSource
    style: str

    # 'glyph' is the base character
    glyph: str
    end: int


@dataclass(frozen=True, kw_only=True)
class GlyphMarker:
    by_glyph: Mapping[str, GlyphStyle]

    def _read_glyph(self, s: str) -> tuple[GlyphStyle, str, int] | None:
        if not s:
            return None
        style = self.by_glyph.get(s[0])
        if style is None:
            return None

        end = 2 if len(s) > 1 and s[1] in _VARIATION_SELECTORS else 1
        return style, s[0], end

    def match_text(self, text: str) -> GlyphMatch | None:
        read = self._read_glyph(text)
        if read is None:
            return None
        style, glyph, end = read
        if end >= len(text) or not text[end].isspace():
            return None
        return GlyphMatch(source="text", style=style.name, glyph=glyph, end=end)

    def match_list_marker(self, displayed_marker: str) -> GlyphMatch | None:
        read = self._read_glyph(displayed_marker)
        if read is None:
            return None
        style, glyph, end = read
        if end != len(displayed_marker):
            return None
        return GlyphMatch(source="list_marker", style=style.name, glyph=glyph, end=end)


def compile_glyph_marker(
    names: list[str], styles_registry: Registry[GlyphStyle]
) -> GlyphMarker:
    if isinstance(names, str):
        raise ValueError("names must be a list, not one string")
    if not names:
        raise ValueError("names must not be empty")

    by_glyph: dict[str, GlyphStyle] = {}
    for name in names:
        if name == "word_bullet":
            raise ValueError(f"glyph style {name!r} is reserved for the profile")
        try:
            style = styles_registry.get(name)
        except KeyError as e:
            raise ValueError(f"glyphs {names!r}: {e.args[0]}") from None

        for glyph in style.glyphs:

            by_glyph[glyph] = style
    return GlyphMarker(by_glyph=MappingProxyType(by_glyph))
