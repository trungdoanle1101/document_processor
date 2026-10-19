import unicodedata
from dataclasses import dataclass

from .markers import NAME_PATTERN
from .registry import Registry

@dataclass(frozen=True, kw_only=True)
class GlyphStyle:
    name: str
    glyphs: frozenset[str]

    def __post_init__(self) -> None:
        if not NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid value for attribute 'name': got {self.name!r}. Must fully match '[a-z][a-z0-9_]*'")
        owner = f"GlyphStyle {self.name!r}"
        if not self.glyphs:
            raise ValueError(f"{owner}: glyphs must not be empty")
        
        for g in sorted(self.glyphs):
            if len(g) != 1:
                raise ValueError(f"{owner}: glyph {g!r} must be exactly one character")
            if unicodedata.category(g)[0] not in "PS":
                raise ValueError(f"{owner}: glyph {g!r} (U+{ord(g):04X}) must be a punctuation mark or a symbol")


_BUILTINS = (
    GlyphStyle(name="dash",         glyphs=frozenset({"-", "\u2010", "\u2011", "\u2013", "\u2014", "\u2212"})),
    GlyphStyle(name="plus",         glyphs=frozenset({"+", "\uff0b"})),
    GlyphStyle(name="round_bullet", glyphs=frozenset({"\u2022", "\u25cf"})),
    GlyphStyle(name="white_bullet", glyphs=frozenset({"\u25e6", "\u25cb"})),
    GlyphStyle(name="square",       glyphs=frozenset({"\u25aa", "\u25a0", "\u25a1", "\u2751"})),
    GlyphStyle(name="check",        glyphs=frozenset({"\u2713", "\u2714", "\u2611"})),
    GlyphStyle(name="arrow",        glyphs=frozenset({"\u27a2", "\u27a4", "\u25ba", "\u2192"})),
    GlyphStyle(name="diamond",      glyphs=frozenset({"\u2756", "\u25c6", "\u25c7"})),
)


BUILTIN_GLYPH_STYLES = Registry(
    kind="glyph style",
    items={
        style.name: style for style in _BUILTINS
    }
)
