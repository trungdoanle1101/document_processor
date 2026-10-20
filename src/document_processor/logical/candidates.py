from dataclasses import dataclass

from document_processor.core.models import PhysicalBlock, BlockType, Region, ListInfo
from document_processor.logical.levels import MarkerMatch
from document_processor.logical.glyph_marker import GlyphMatch
from document_processor.logical.template_marker import TemplateMatch
from document_processor.logical.profile import Profile
from document_processor.logical.markers import WORD_BULLET


@dataclass(frozen=True, kw_only=True)
class Candidate:
    level: str
    list_match: MarkerMatch | None
    text_match: MarkerMatch | None

    def __post_init__(self) -> None:
        if self.list_match is None and self.text_match is None:
            raise ValueError(f"candidate for level {self.level!r}: at least one of list_match or text_match must be set")
    
        if self.list_match is not None and self.list_match.source != "list_marker":
            raise ValueError(f"candidate for level {self.level!r}: list_match.source must be 'list_marker'")
        
        if self.text_match is not None and self.text_match.source != "text":
            raise ValueError(f"candidate for level {self.level!r}: text_match.source must be 'text'")
        

def find_candidates(profile: Profile, block: PhysicalBlock) -> tuple[Candidate, ...]:
    if block.block_type in (BlockType.TABLE, BlockType.IMAGE):
        return ()
    
    if block.region is not Region.BODY:
        return ()
    
    if profile.levels is None:
        return ()
    
    text = block.text
    displayed_marker = block.list_info.displayed_marker if block.list_info is not None else None

    candidates: list[Candidate] = []
    for level in profile.levels:
        marker = profile.get_marker(level.name)
        list_readings = marker.match_list_marker(displayed_marker) if displayed_marker else []
        text_readings = marker.match_text(text)
    
        unpaired_text: list[MarkerMatch] = list(text_readings)
        for list_match in list_readings:
            partner = next((t for t in unpaired_text if _number_key(t) == _number_key(list_match)), None)
            if partner is not None:
                unpaired_text.remove(partner)
            candidates.append(Candidate(
                level=level.name, list_match=list_match, text_match=partner
            ))
        
        for text_match in unpaired_text:
            candidates.append(Candidate(
                level=level.name, list_match=None, text_match=text_match
            ))

    # Fallback to word bullet
    if (
        profile.word_bullet_level is not None
        and block.list_info is not None
        and _is_word_bullet(block.list_info)
        and all(c.list_match is None for c in candidates)
    ):
        displayed = displayed_marker or ""
        candidates.append(Candidate(
            level=profile.word_bullet_level,
            list_match=GlyphMatch(
                source="list_marker", style=WORD_BULLET, glyph=displayed, end=len(displayed)
            ),
            text_match=None
        ))


    return tuple(sorted(candidates, key=lambda c: profile.get_rank(c.level)))


def _number_key(match: MarkerMatch) -> tuple[object, ...]:
    if isinstance(match, TemplateMatch):
        return tuple((n.style, n.value) for n in match.numerals)
    return (match.style, )

def _is_word_bullet(list_info: ListInfo) -> bool:
    if list_info.declared_numbering is not None:
        return list_info.declared_numbering.num_fmt == "bullet"
    return not list_info.is_ordered