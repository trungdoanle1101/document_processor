from dataclasses import dataclass
from typing import Literal, get_args

from document_processor.logical.glyph_marker import GlyphMarker, GlyphMatch
from document_processor.logical.markers import NAME_PATTERN
from document_processor.logical.template_marker import TemplateMarker, TemplateMatch

type Marker = TemplateMarker | GlyphMarker

type MarkerMatch = TemplateMatch | GlyphMatch

type TitleRule = Literal["none", "rest_of_paragraph", "next_block"]
TITLE_RULES: tuple[str, ...] = get_args(TitleRule.__value__)


@dataclass(frozen=True, kw_only=True)
class TemplateMarkerSpec:
    templates: tuple[str, ...]
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("templates must not be empty")

        if len(set(self.templates)) != len(self.templates):
            raise ValueError(f"templates must not repeat: {self.templates!r}")


@dataclass(frozen=True, kw_only=True)
class GlyphMarkerSpec:
    glyphs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.glyphs:
            raise ValueError("glyphs must not be empty")

        if len(set(self.glyphs)) != len(self.glyphs):
            raise ValueError(f"glyphs must not repeat: {self.glyphs!r}")


type MarkerSpec = TemplateMarkerSpec | GlyphMarkerSpec


@dataclass(frozen=True, kw_only=True)
class LevelSpec:
    name: str
    marker: MarkerSpec
    is_section: bool
    title: TitleRule = "none"
    closes_on_plain_paragraph: bool = False
    inside: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid level name {self.name!r}: must match {NAME_PATTERN.pattern}")

        if self.title not in TITLE_RULES:
            raise ValueError(f"level {self.name!r}: 'title' must be one of {TITLE_RULES}, got {self.title!r}")
    