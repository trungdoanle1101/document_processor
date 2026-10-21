import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from document_processor.core.models import BlockType, ListInfo, PhysicalBlock, Region
from document_processor.logical.glyph_marker import GlyphMatch
from document_processor.logical.levels import MarkerMatch
from document_processor.logical.markers import WORD_BULLET
from document_processor.logical.profile import Profile
from document_processor.logical.template_marker import TemplateMatch

DECLARED_FORMAT_STYLES: Mapping[str, str] = MappingProxyType(
    {
        "decimal": "arabic",
        "lowerLetter": "lower_latin",
        "upperLetter": "upper_latin",
        "lowerRoman": "lower_roman",
        "upperRoman": "upper_roman",
    }
)


@dataclass(frozen=True, kw_only=True)
class Candidate:
    level: str
    list_match: MarkerMatch | None
    text_match: MarkerMatch | None

    def __post_init__(self) -> None:
        if self.list_match is None and self.text_match is None:
            raise ValueError(
                f"candidate for level {self.level!r}: at least one of list_match or text_match must be set"
            )

        if self.list_match is not None and self.list_match.source != "list_marker":
            raise ValueError(
                f"candidate for level {self.level!r}: list_match.source must be 'list_marker'"
            )

        if self.text_match is not None and self.text_match.source != "text":
            raise ValueError(
                f"candidate for level {self.level!r}: text_match.source must be 'text'"
            )


@dataclass(frozen=True, kw_only=True)
class BlockCandidates:
    block: PhysicalBlock
    candidates: tuple[Candidate, ...]


type DocumentPart = Literal["front_matter", "body", "end_matter"]


@dataclass(frozen=True, kw_only=True)
class ClassifiedBlock:
    block: PhysicalBlock
    heading: Candidate | None
    part: DocumentPart


def classify_blocks(
    profile: Profile, blocks: Sequence[PhysicalBlock]
) -> tuple[ClassifiedBlock, ...]:
    """Classify every block: pass 1 finds candidates, pass 2 keeps one per level, pass 3 chooses the heading and the document part."""
    items = [
        BlockCandidates(block=b, candidates=find_candidates(profile, b)) for b in blocks
    ]
    narrowed = narrow_candidates(items)
    return choose_headings(profile, narrowed)


def find_candidates(profile: Profile, block: PhysicalBlock) -> tuple[Candidate, ...]:
    if block.block_type in (BlockType.TABLE, BlockType.IMAGE):
        return ()

    if block.region is not Region.BODY:
        return ()

    if profile.levels is None:
        return ()

    text = block.text
    displayed_marker = (
        block.list_info.displayed_marker if block.list_info is not None else None
    )

    candidates: list[Candidate] = []
    for level in profile.levels:
        marker = profile.get_marker(level.name)
        list_readings = (
            marker.match_list_marker(displayed_marker) if displayed_marker else []
        )
        text_readings = marker.match_text(text)

        unpaired_text: list[MarkerMatch] = list(text_readings)
        for list_match in list_readings:
            partner = next(
                (t for t in unpaired_text if _number_key(t) == _number_key(list_match)),
                None,
            )
            if partner is not None:
                unpaired_text.remove(partner)
            candidates.append(
                Candidate(level=level.name, list_match=list_match, text_match=partner)
            )

        for text_match in unpaired_text:
            candidates.append(
                Candidate(level=level.name, list_match=None, text_match=text_match)
            )

    # Fallback to word bullet
    if (
        profile.word_bullet_level is not None
        and block.list_info is not None
        and _is_word_bullet(block.list_info)
        and all(c.list_match is None for c in candidates)
    ):
        displayed = displayed_marker or ""
        candidates.append(
            Candidate(
                level=profile.word_bullet_level,
                list_match=GlyphMatch(
                    source="list_marker",
                    style=WORD_BULLET,
                    glyph=displayed,
                    end=len(displayed),
                ),
                text_match=None,
            )
        )

    return tuple(sorted(candidates, key=lambda c: profile.get_rank(c.level)))


def narrow_candidates(
    block_candidates: Sequence[BlockCandidates],
) -> tuple[BlockCandidates, ...]:
    """For each block, keep at most one candidate per level chosen by the numbering of earlier items at the same level; candidates of different levels are left for the builder."""
    # level -> (style, value)
    last_number_by_level: dict[str, tuple[str, int]] = {}
    narrowed: list[BlockCandidates] = []

    for item in block_candidates:
        chosen_candidates: list[Candidate] = []
        for level, candidates in _group_by_level(item.candidates):
            chosen_candidate = _pick_one_candidate(
                candidates=list(candidates),
                of_block=item.block,
                last_number_at_this_level=last_number_by_level.get(level),
            )
            chosen_candidates.append(chosen_candidate)
            style_and_value = _get_candidate_style_value(chosen_candidate)
            if style_and_value is not None:
                last_number_by_level[level] = style_and_value
        narrowed.append(
            BlockCandidates(block=item.block, candidates=tuple(chosen_candidates))
        )

    return tuple(narrowed)


def choose_headings(
    profile: Profile, items: Sequence[BlockCandidates]
) -> tuple[ClassifiedBlock, ...]:
    """For each block, the heading it becomes (the outermost candidate whose 'inside' is satisfied by the open levels) or None.
    Nothing from the first end-matter block on
    """
    # name of open levels, outermost first
    open_levels: list[str] = []
    in_end_matter = False
    body_started = profile.levels is None
    classified: list[ClassifiedBlock] = []
    for item in items:
        if (
            not in_end_matter
            and body_started
            and _starts_end_matter(profile, item.block)
        ):
            in_end_matter = True

        if in_end_matter:
            classified.append(
                ClassifiedBlock(block=item.block, heading=None, part="end_matter")
            )
            continue

        heading = _choose_heading(profile, item.candidates, open_levels)
        if heading is not None:
            body_started = True
            kept = count_kept_open_by_heading(profile, open_levels, heading.level)
            del open_levels[kept:]
            open_levels.append(heading.level)
        elif is_text_block(item.block):
            kept = count_kept_open_by_plain_paragraph(profile, open_levels)
            del open_levels[kept:]

        classified.append(
            ClassifiedBlock(
                block=item.block,
                heading=heading,
                part="body" if body_started else "front_matter",
            )
        )

    return tuple(classified)


def count_kept_open_by_heading(
    profile: Profile, open_levels: Sequence[str], heading_level: str
) -> int:
    """How many open levels (outermost first) stay open when a heading of 'heading_level' opens:
    the rest, same rank or deeper, close.
    """
    heading_rank = profile.get_rank(heading_level)
    kept = 0
    for open_level in open_levels:
        if profile.get_rank(open_level) >= heading_rank:
            break
        kept += 1
    return kept


def count_kept_open_by_plain_paragraph(
    profile: Profile, open_levels: Sequence[str]
) -> int:
    """How many open levels (outermost first) stay open at a plain paragraph:
    close from the innermost outwards while the level has closes_on_plain_paragraph
    """
    kept = len(open_levels)
    while (
        kept > 0 and profile.get_level(open_levels[kept - 1]).closes_on_plain_paragraph
    ):
        kept -= 1
    return kept


def _starts_end_matter(profile: Profile, block: PhysicalBlock) -> bool:
    marker = profile.end_matter_marker
    if (marker is None) or (block.region is not Region.BODY):
        return False

    matches = marker.match_text(block.text)
    return bool(matches)


def is_text_block(block: PhysicalBlock) -> bool:
    return block.block_type in (BlockType.PARAGRAPH, BlockType.LIST_ITEM)


def _choose_heading(
    profile: Profile, candidates: tuple[Candidate, ...], open_levels: Sequence[str]
) -> Candidate | None:
    for candidate in candidates:
        candidate_level = profile.get_level(candidate.level)
        if not candidate_level.inside or any(
            name in open_levels for name in candidate_level.inside
        ):
            return candidate

    return None


def _pick_one_candidate(
    candidates: list[Candidate],
    of_block: PhysicalBlock,
    last_number_at_this_level: tuple[str, int] | None,
) -> Candidate:
    if len(candidates) == 1:
        return candidates[0]

    # 1. Word declares format: list readings in another style are wrong
    # Filter out candidates with different numbering style from block's declared style
    # Keep candidates with list_match = None (they are matched by text)
    declared_style = _get_declared_num_style(of_block)
    if declared_style is not None:
        style_filtered: list[Candidate] = []
        for candidate in candidates:
            if candidate.list_match is None:
                style_filtered.append(candidate)
            else:
                style_and_value = _get_candidate_style_value(candidate)
                if style_and_value is not None and style_and_value[0] == declared_style:
                    style_filtered.append(candidate)

        candidates = style_filtered or candidates

    # 2. Choose candidate that continues the count of this level
    if last_number_at_this_level is not None:
        last_style, last_value = last_number_at_this_level
        for candidate in candidates:
            style_and_value = _get_candidate_style_value(candidate)
            if style_and_value == (last_style, last_value + 1):
                return candidate

    # 3. Start a new list: value 1
    for candidate in candidates:
        style_and_value = _get_candidate_style_value(candidate)
        if style_and_value is not None and style_and_value[1] == 1:
            return candidate

    # 4. Nothing decides: list or text, then the order in the slot
    if declared_style is None:
        for candidate in candidates:
            if candidate.text_match is not None:
                return candidate

    return candidates[0]


def _group_by_level(
    candidates: tuple[Candidate, ...],
) -> Iterator[tuple[str, Iterator[Candidate]]]:
    return itertools.groupby(candidates, key=lambda c: c.level)


def _get_candidate_style_value(candidate: Candidate) -> tuple[str, int] | None:
    """(style, value) of the candidate"""
    match = candidate.list_match or candidate.text_match
    if not isinstance(match, TemplateMatch) or not match.numerals:
        return None

    last_numeral = match.numerals[-1]
    return (last_numeral.style, last_numeral.value)


def _get_declared_num_style(block: PhysicalBlock) -> str | None:
    if block.list_info is None:
        return None

    declared_numbering = block.list_info.declared_numbering

    if declared_numbering is None:
        return None

    num_fmt = declared_numbering.num_fmt

    if num_fmt not in DECLARED_FORMAT_STYLES:
        return None

    return DECLARED_FORMAT_STYLES[num_fmt]


def _number_key(match: MarkerMatch) -> tuple[object, ...]:
    if isinstance(match, TemplateMatch):
        return tuple((n.style, n.value) for n in match.numerals)
    return (match.style,)


def _is_word_bullet(list_info: ListInfo) -> bool:
    if list_info.declared_numbering is not None:
        return list_info.declared_numbering.num_fmt == "bullet"
    return not list_info.is_ordered
