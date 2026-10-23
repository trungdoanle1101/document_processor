import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from document_processor.chunking.chunk import Chunk, PathStep

CHUNK_KIND = "chunk"
CHUNK_SCHEMA_VERSION = 1


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "kind": CHUNK_KIND,
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk": asdict(chunk),
    }


def chunk_from_dict(d: dict[str, Any]) -> Chunk:
    kind = d.get("kind")
    if kind != CHUNK_KIND:
        raise ValueError(
            f"Unexpected dictionary kind: Expected: {CHUNK_KIND!r}. Actual: {kind!r}"
        )
    schema_version = d.get("schema_version")
    if schema_version != CHUNK_SCHEMA_VERSION:
        raise ValueError(
            f"Mismatched schema version: Expected: {CHUNK_SCHEMA_VERSION}. Actual: {schema_version}"
        )

    chunk_dict = d["chunk"]

    chunk = Chunk(
        id=chunk_dict["id"],
        logical_document_id=chunk_dict["logical_document_id"],
        part=chunk_dict["part"],
        path=tuple(PathStep(**step) for step in chunk_dict["path"]),
        start=chunk_dict["start"],
        end=chunk_dict["end"],
        table_rows=_pair_or_none(chunk_dict["table_rows"]),
        text_span=_pair_or_none(chunk_dict["text_span"]),
        text=chunk_dict["text"],
        block_ids=tuple(chunk_dict["block_ids"]),
    )

    headings = tuple(chunk_dict["headings"])
    if chunk.headings != headings:
        raise ValueError(
            f"chunk {chunk.id!r}: mismatched headings in dictionary and in Chunk instance. In dict: {headings}, in Chunk instance: {chunk.headings}"
        )

    return chunk


def save_chunks(chunks: Iterable[Chunk], file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_dict = chunk_to_dict(chunk)
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")


def load_chunks(file_path: str | Path) -> tuple[Chunk, ...]:
    results: list[Chunk] = []
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                chunk_dict = json.loads(line)
                results.append(chunk_from_dict(chunk_dict))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{file_path}, line {line_number}: {error}") from error
    return tuple(results)


def _pair_or_none(value: list[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    first, second = value
    return first, second
