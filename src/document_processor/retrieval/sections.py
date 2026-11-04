"""Complete the sections a question is about, so an answer sees every part of them.

Search ranks passages one by one, so a question about a whole section ("what
layers are there?") gets some of its parts and misses others. When two or more
hits sit in the same section, the rest of that section is added, in document
order, ahead of the other hits, as long as it fits a size budget.

Sections are found from each chunk's full address (its longest reference key,
e.g. "chuong:iii/muc:ii/dieu:7/khoan:2/diem:b"), so any profile that numbers its
parts works; chunks without an address are never grouped.
"""

from collections.abc import Sequence
from dataclasses import replace

from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkIndex,
    ChunkKey,
    IndexedChunk,
)
from document_processor.retrieval.search import RetrievedChunk, Search

_NO_FILTER = ChunkFilter()
_MAX_SECTION_CHUNKS = 500  # a section with more is treated as too big


def complete_sections(
    search: Search,
    *,
    index: ChunkIndex,
    max_chars: int,
    chunk_filter: ChunkFilter = _NO_FILTER,
    min_hits: int = 2,
) -> Search:
    """search, with its hits completed by the rest of any section min_hits of them share.

    chunk_filter: added chunks must meet it too; pass the filter the search uses.
    max_chars: most text added in all (the hits themselves don't count); 0 adds nothing.
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must not be negative, got {max_chars}")
    if min_hits < 2:
        raise ValueError(f"min_hits must be at least 2, got {min_hits}")

    def completed(question: str, *, limit: int) -> list[RetrievedChunk]:
        hits = search(question, limit=limit)
        if max_chars == 0:
            return hits
        return _complete(
            hits,
            index=index,
            chunk_filter=chunk_filter,
            max_chars=max_chars,
            min_hits=min_hits,
        )

    return completed


def _complete(
    hits: Sequence[RetrievedChunk],
    *,
    index: ChunkIndex,
    chunk_filter: ChunkFilter,
    max_chars: int,
    min_hits: int,
) -> list[RetrievedChunk]:
    found = {hit.chunk.key: hit for hit in hits}
    passages: list[RetrievedChunk] = []
    taken: set[ChunkKey] = set()
    used = 0

    for (document_id, section), best in _shared_sections(hits, min_hits):
        members = _section_chunks(index, chunk_filter, document_id, section)
        if members is None:
            continue  # out of the filter's reach, or too many chunks
        new = [chunk for chunk in members if chunk.key not in taken]
        size = sum(len(chunk.text) for chunk in new if chunk.key not in found)
        if used + size > max_chars:
            continue  # too big: its hits stay among the others
        for chunk in new:
            # a hit keeps its own scores; an added chunk borrows those of the best hit
            hit = found.get(chunk.key) or replace(
                best, chunk=chunk, found_through=(), completed_section=section
            )
            passages.append(hit)
            taken.add(chunk.key)
        used += size

    passages += [hit for hit in hits if hit.chunk.key not in taken]
    return passages


def _shared_sections(
    hits: Sequence[RetrievedChunk], min_hits: int
) -> list[tuple[tuple[str, str], RetrievedChunk]]:
    """(document id, section address) shared by min_hits or more hits, with its best hit,
    best first. Hits come best first, so each group's first member is its best."""
    groups: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for hit in hits:
        parent = _parent_address(hit.chunk)
        if parent is not None:
            groups.setdefault((hit.chunk.document_id, parent), []).append(hit)
    return [
        (key, members[0]) for key, members in groups.items() if len(members) >= min_hits
    ]


def _section_chunks(
    index: ChunkIndex, chunk_filter: ChunkFilter, document_id: str, section: str
) -> list[IndexedChunk] | None:
    """The section's chunks the filter allows, in document order; None if it can't be had."""
    if (
        chunk_filter.document_ids is not None
        and document_id not in chunk_filter.document_ids
    ):
        return None
    # the index narrows to chunks citable by the section's address; the
    # filter's own chunk and reference restrictions are checked below
    narrowed = replace(
        chunk_filter,
        document_ids=(document_id,),
        chunk_keys=None,
        reference_keys=(section,),
    )
    chunks = index.list_chunks(narrowed, limit=_MAX_SECTION_CHUNKS)
    if len(chunks) == _MAX_SECTION_CHUNKS:
        return None
    return [
        chunk
        for chunk in chunks
        if _is_within(chunk, section) and _meets_keys(chunk, chunk_filter)
    ]


def _meets_keys(chunk: IndexedChunk, chunk_filter: ChunkFilter) -> bool:
    """Whether the chunk meets the filter's chunk and reference restrictions."""
    if chunk_filter.chunk_keys is not None and chunk.key not in chunk_filter.chunk_keys:
        return False
    return chunk_filter.reference_keys is None or bool(
        set(chunk.reference_keys) & set(chunk_filter.reference_keys)
    )


def _full_address(chunk: IndexedChunk) -> str | None:
    """The address from the document's root: the reference key with the most steps."""
    if not chunk.reference_keys:
        return None
    return max(chunk.reference_keys, key=lambda key: key.count("/"))


def _parent_address(chunk: IndexedChunk) -> str | None:
    """The section a chunk sits in; None at the top, where it would be the whole document."""
    address = _full_address(chunk)
    if address is None or "/" not in address:
        return None
    return address.rsplit("/", 1)[0]


def _is_within(chunk: IndexedChunk, section: str) -> bool:
    """The section itself or anything below it; a short key that happens to match is not."""
    address = _full_address(chunk)
    return address is not None and (
        address == section or address.startswith(section + "/")
    )