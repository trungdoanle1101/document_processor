from collections.abc import Sequence
from dataclasses import dataclass

from document_processor.core.identity import compute_hash
from document_processor.core.models import (
    BlockType,
    ImageSegment,
    LogicalDocument,
    LogicalNode,
    PhysicalBlock,
    PhysicalDocument,
    Segment,
    TableSegment,
    TextSegment,
)
from document_processor.logical.candidates import (
    Candidate,
    ClassifiedBlock,
    classify_blocks,
    count_kept_open_by_heading,
    count_kept_open_by_plain_paragraph,
    is_text_block,
)
from document_processor.logical.glyph_marker import GlyphMatch
from document_processor.logical.profile import Profile


@dataclass(kw_only=True)
class _NodeDraft:
    id: str
    parent_id: str | None
    level: str
    number: str | None
    marker: str | None
    title: str | None
    head_block_ids: list[str]
    start: int
    end: int | None = None


@dataclass(frozen=True, kw_only=True)
class _HeadingParts:
    marker: str
    number: str | None
    # The block's text after the marker and its blanks ("" if nothing)
    rest_of_text: str


class _Assembly:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.segments: list[Segment] = []
        self.open_nodes: list[_NodeDraft] = []
        self.all_drafts: list[_NodeDraft] = []
        self.waiting_for_title: _NodeDraft | None = None
        self.body_start: int | None = None
        self.body_end: int | None = None

    def takes_as_title(self, item: ClassifiedBlock) -> bool:
        draft, self.waiting_for_title = self.waiting_for_title, None
        if (
            draft is None
            or item.heading is not None
            or item.part != "body"
            or not is_text_block(item.block)
        ):
            return False
        draft.title = item.block.text
        draft.head_block_ids.append(item.block.id)
        return True

    def add_end_matter_block(self, block: PhysicalBlock) -> None:
        if self.body_end is None:
            self.body_end = len(self.segments)
            self._close_from(0)
        self.segments.append(_block_to_segment(block))

    def add_heading(self, block: PhysicalBlock, heading: Candidate) -> None:
        if self.body_start is None:
            self.body_start = len(self.segments)
        open_levels = [n.level for n in self.open_nodes]
        kept = count_kept_open_by_heading(self.profile, open_levels, heading.level)
        self._close_from(kept)
        draft, body_text = self._new_draft(block, heading)
        self.open_nodes.append(draft)
        self.all_drafts.append(draft)
        if body_text:
            self.segments.append(TextSegment(block_id=block.id, text=body_text))
        draft_level_spec = self.profile.get_level(draft.level)
        if draft.title is None and draft_level_spec.title == "next_block":
            self.waiting_for_title = draft

    def add_plain_text(self, block: PhysicalBlock) -> None:
        open_levels = [n.level for n in self.open_nodes]
        kept = count_kept_open_by_plain_paragraph(self.profile, open_levels)
        self._close_from(kept)
        self.segments.append(_block_to_segment(block))

    def add_table_or_image(self, block: PhysicalBlock) -> None:
        self.segments.append(_block_to_segment(block))

    def finish(self, physical_document: PhysicalDocument) -> LogicalDocument:
        self._close_from(0)

        nodes = tuple(
            LogicalNode(
                id=draft.id,
                parent_id=draft.parent_id,
                level=draft.level,
                number=draft.number,
                marker=draft.marker,
                title=draft.title,
                head_block_ids=tuple(draft.head_block_ids),
                start=draft.start,
                end=draft.end if draft.end is not None else len(self.segments),
            )
            for draft in self.all_drafts
        )

        return LogicalDocument(
            id=compute_hash(f"{physical_document.id}|{self.profile.name}|{self.profile.version}".encode()),
            physical_document_id=physical_document.id,
            source_file_hash=physical_document.file_hash,
            profile_name=self.profile.name,
            profile_version=self.profile.version,
            segments=tuple(self.segments),
            nodes=nodes,
            body_start=self.body_start if self.body_start is not None else 0,
            body_end=self.body_end if self.body_end is not None else len(self.segments)
        )

    def _close_from(self, index: int) -> None:
        """Close open nodes from 'index' inwards: they end before the next segments"""
        for node in self.open_nodes[index:]:
            node.end = len(self.segments)
        del self.open_nodes[index:]

    def _new_draft(
        self, block: PhysicalBlock, heading: Candidate
    ) -> tuple[_NodeDraft, str]:
        heading_parts = _split_heading(block, heading)
        level_spec = self.profile.get_level(heading.level)
        title_rule = level_spec.title

        if title_rule == "rest_of_paragraph" or title_rule == "next_block":
            title = heading_parts.rest_of_text or None
            body_text = ""
        else:
            title = None
            body_text = heading_parts.rest_of_text

        return _NodeDraft(
            id=f"n{len(self.all_drafts)}",
            parent_id=self.open_nodes[-1].id if self.open_nodes else None,
            level=heading.level,
            number=heading_parts.number,
            marker=heading_parts.marker,
            title=title,
            head_block_ids=[block.id],
            start=len(self.segments),
            end=None,
        ), body_text


def build_logical_document(
    profile: Profile, physical_document: PhysicalDocument
) -> LogicalDocument:
    classified = classify_blocks(profile, physical_document.blocks)
    return assemble_document(profile, physical_document, classified)


def assemble_document(
    profile: Profile, physical: PhysicalDocument, classified: Sequence[ClassifiedBlock]
) -> LogicalDocument:
    if not _same_blocks(physical, classified):
        raise ValueError(
            "classified blocks did not come from the same physical document provided"
        )

    assembly = _Assembly(profile)
    for item in classified:
        if assembly.takes_as_title(item):
            continue

        if item.part == "end_matter":
            assembly.add_end_matter_block(item.block)
        elif item.heading is not None:
            assembly.add_heading(item.block, item.heading)
        elif is_text_block(item.block):
            assembly.add_plain_text(item.block)
        else:
            assembly.add_table_or_image(item.block)

    return assembly.finish(physical)


def _block_to_segment(block: PhysicalBlock) -> Segment:
    if block.block_type is BlockType.TABLE and block.table_info is not None:
        return TableSegment(block_id=block.id, table=block.table_info)

    if block.block_type is BlockType.IMAGE and block.image_info is not None:
        return ImageSegment(block_id=block.id, image=block.image_info)

    return TextSegment(block_id=block.id, text=block.text)


def _split_heading(block: PhysicalBlock, candidate: Candidate) -> _HeadingParts:
    match = candidate.text_match or candidate.list_match
    if match is None:
        raise ValueError(
            "internal error: Candidate must have either text_match or list_match not None"
        )
    matched_string = (
        block.text
        if match.source == "text"
        else (
            block.list_info.displayed_marker
            if block.list_info is not None
            and block.list_info.displayed_marker is not None
            else ""
        )
    )

    marker = matched_string[: match.end]
    rest_of_text = matched_string[match.end :].lstrip() if match.source == "text" else block.text
    if isinstance(match, GlyphMatch) or not match.numerals:
        number = None
    else:
        number = matched_string[match.numerals[0].start : match.numerals[-1].end]

    return _HeadingParts(marker=marker, number=number, rest_of_text=rest_of_text)


def _same_blocks(
    physical_document: PhysicalDocument, classified: Sequence[ClassifiedBlock]
) -> bool:
    return [b for b in physical_document.blocks] == [item.block for item in classified]
