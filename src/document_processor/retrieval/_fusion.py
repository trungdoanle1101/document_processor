from collections.abc import Sequence
from dataclasses import dataclass

from document_processor.retrieval.index import (
    ChunkHit,
    ChunkKey,
    GeneratedEntry,
    GeneratedEntryKey,
    IndexedChunk,
)

_RRF_OFFSET = 60


@dataclass(frozen=True, kw_only=True)
class BestMatch:
    chunk: IndexedChunk
    score: float
    entry: GeneratedEntry | None


def fuse(rankings: Sequence[Sequence[BestMatch]], limit: int) -> list[ChunkHit]:
    if len(rankings) == 1:
        hits: list[ChunkHit] = [
            _build_hit(match.chunk, match.score, [(1, match)]) for match in rankings[0]
        ]
    else:
        scores: dict[ChunkKey, float] = {}
        matches: dict[ChunkKey, list[tuple[int, BestMatch]]] = {}

        for ranking in rankings:
            for rank, match in enumerate(ranking, start=1):
                key = match.chunk.key
                scores[key] = scores.get(key, 0.0) + 1 / (_RRF_OFFSET + rank)
                matches.setdefault(key, []).append((rank, match))
        hits = []
        for key, score in scores.items():
            chunk = matches[key][0][1].chunk
            hits.append(_build_hit(chunk, score, matches[key]))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:limit]


def _build_hit(
    chunk: IndexedChunk, score: float, ranked_matches: Sequence[tuple[int, BestMatch]]
) -> ChunkHit:
    through: dict[GeneratedEntryKey, GeneratedEntry] = {}
    for _, match in sorted(ranked_matches, key=lambda ranked: ranked[0]):
        if match.entry is not None:
            through.setdefault(match.entry.key, match.entry)
    return ChunkHit(chunk=chunk, score=score, found_through=tuple(through.values()))
