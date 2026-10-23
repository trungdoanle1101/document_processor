from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType

from document_processor.logical.glyph_styles import GlyphStyle, BUILTIN_GLYPH_STYLES
from document_processor.logical.counter_styles import CounterStyle, BUILTIN_COUNTER_STYLES
from document_processor.logical.template_marker import TemplateMarker, compile_template_marker
from document_processor.logical.glyph_marker import GlyphMarker, compile_glyph_marker
from document_processor.logical.levels import (
    LevelSpec, TemplateMarkerSpec, Marker, GlyphMarkerSpec, MarkerSpec
)
from document_processor.logical.registry import Registry
from document_processor.logical.markers import NAME_PATTERN, WORD_BULLET
from document_processor.core.text_utils import normalize_nfc


@dataclass(frozen=True, kw_only=True)
class Profile:
    name: str
    version: str
    levels: tuple[LevelSpec, ...] | None
    end_matter: TemplateMarkerSpec | None
    counter_styles: Registry[CounterStyle] = field(default=BUILTIN_COUNTER_STYLES, compare=False)
    glyph_styles: Registry[GlyphStyle] = field(default=BUILTIN_GLYPH_STYLES, compare=False)

    _level_by_name: Mapping[str, LevelSpec] = field(init=False, repr=False, compare=False)
    _rank_by_name: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _marker_by_name: Mapping[str, Marker] = field(init=False, repr=False, compare=False)
    _end_matter_marker: TemplateMarker | None = field(init=False, repr=False, compare=False)
    _word_bullet_level: str | None = field(init=False, repr=False, compare=False)


    def __post_init__(self) -> None:
        if not NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid profile name {self.name!r}: must match {NAME_PATTERN.pattern}")

        if not self.version:
            raise ValueError("profile version must not be empty")

        level_by_name: dict[str, LevelSpec] = {}
        rank_by_name: dict[str, int] = {}
        marker_by_name: dict[str, Marker] = {}
        word_bullet_level: str | None = None

        
        if self.levels is not None:
            if not self.levels:
                raise ValueError(f"profile {self.name!r}: levels must not be empty; use levels=None for no structure")

            all_names = {level.name for level in self.levels}
            template_owner: dict[str, str] = {}
            glyph_owner: dict[str, str] = {}

            first_item: str | None = None

            for rank, level in enumerate(self.levels):
                where = f"profile {self.name!r}, level {level.name!r}"

                if level.is_section and first_item is not None:
                    raise ValueError(f"{where}: a section cannot rank below an item level ({first_item!r})")

                if not level.is_section and first_item is None:
                    first_item = level.name

                if level.name in level_by_name:
                    raise ValueError(f"{where}: duplicate level name")
                
            
                for name in level.inside:
                    if name not in all_names:
                        raise ValueError(f"{where}: inside names unknown level {name!r}")
                    if name not in level_by_name:
                        raise ValueError(f"{where}: inside may only name earlier levels, got {name!r}")

                
                marker_spec = level.marker
                keys = [normalize_nfc(t) for t in marker_spec.templates] if isinstance(marker_spec, TemplateMarkerSpec) else list(marker_spec.glyphs)
                owners = template_owner if isinstance(marker_spec, TemplateMarkerSpec) else glyph_owner
                for key in keys:
                    if key in owners:
                        raise ValueError(f"{where}: {key!r} already used by level {owners[key]!r}")
                    owners[key] = level.name
                
                try:
                    marker_by_name[level.name] = self._compile(marker_spec)
                except ValueError as e:
                    raise ValueError(f"{where}: {e}") from e

                level_by_name[level.name] = level
                rank_by_name[level.name] = rank
                if isinstance(marker_spec, GlyphMarkerSpec) and WORD_BULLET in marker_spec.glyphs:
                    word_bullet_level = level.name

        end_matter_marker: TemplateMarker | None = None
        try: 
            if self.end_matter is not None:
                end_matter_marker = compile_template_marker(
                    templates=list(self.end_matter.templates),
                    styles_registry=self.counter_styles,
                    case_sensitive=self.end_matter.case_sensitive
                )
        except ValueError as e:
            raise ValueError(f"profile {self.name!r}, end_matter: {e}") from e

        object.__setattr__(self, "_level_by_name", MappingProxyType(level_by_name))
        object.__setattr__(self, "_rank_by_name", MappingProxyType(rank_by_name))
        object.__setattr__(self, "_marker_by_name", MappingProxyType(marker_by_name))
        object.__setattr__(self, "_end_matter_marker", end_matter_marker)
        object.__setattr__(self, "_word_bullet_level", word_bullet_level)

    def get_level(self, name: str) -> LevelSpec:
        if name not in self._level_by_name:
            raise KeyError(f"profile {self.name!r}: unknown level {name!r}, known {list(self._level_by_name)}")
        return self._level_by_name[name]

    def get_rank(self, name: str) -> int:
        if name not in self._rank_by_name:
            raise KeyError(f"profile {self.name!r}: unknown level {name!r}, known {list(self._rank_by_name)}")
        return self._rank_by_name[name]

    def get_marker(self, name: str) -> Marker:
        if name not in self._marker_by_name:
            raise KeyError(f"profile {self.name!r}: unknown level {name!r}, known {list(self._marker_by_name)}")
        return self._marker_by_name[name]

    @property
    def end_matter_marker(self) -> TemplateMarker | None:
        return self._end_matter_marker

    @property
    def word_bullet_level(self) -> str | None:
        return self._word_bullet_level


    def _compile(self, marker_spec: MarkerSpec) -> Marker:
        if isinstance(marker_spec, TemplateMarkerSpec):  
            return compile_template_marker(
                templates=list(marker_spec.templates),
                styles_registry=self.counter_styles,
                case_sensitive=marker_spec.case_sensitive
            )
        else:
            glyphs = [g for g in marker_spec.glyphs if g != WORD_BULLET]
            if not glyphs:
                return GlyphMarker(
                    by_glyph=MappingProxyType({})
                )
            else:
                return compile_glyph_marker(
                    names=glyphs, styles_registry=self.glyph_styles
                )