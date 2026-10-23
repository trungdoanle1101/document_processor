"""Tests for chunking/serialization.py: chunks as JSON Lines.

Each line is one self-describing record, {"kind": "chunk", "schema_version": 1,
"chunk": {...}}, so files can be streamed, split or concatenated (a whole batch
in one file). The record includes the derived `headings`; loading checks it
against the path and drops it. Tuples come back as tuples, PathSteps as
PathSteps, unset narrowing as None.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from document_processor.chunking.chunk import Chunk, PathStep
from document_processor.chunking.serialization import (
    CHUNK_KIND,
    CHUNK_SCHEMA_VERSION,
    chunk_from_dict,
    chunk_to_dict,
    load_chunks,
    save_chunks,
)
from tests.chunking.test_outline_chunker import chunker, regulation

CHUONG = PathStep(node_id="n0", level="chuong", number="I", marker="Chương I", title="QUY ĐỊNH CHUNG", start=1)
DIEU = PathStep(node_id="n3", level="dieu", number="2", marker="Điều 2.", title="Giải thích từ ngữ", start=7)


def make_chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "id": "c3",
        "logical_document_id": "ld1",
        "part": "body",
        "path": (CHUONG, DIEU),
        "start": 8,
        "end": 10,
        "text": "1.\nHệ thống là tập hợp các thành phần",
        "block_ids": ("pb9",),
    }
    fields.update(overrides)
    return Chunk(**fields)


def through_json(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(d, ensure_ascii=False))
    return result


def round_trip(chunk: Chunk) -> Chunk:
    return chunk_from_dict(through_json(chunk_to_dict(chunk)))


# ============================================================ round trips


def test_round_trip() -> None:
    chunk = make_chunk()
    assert round_trip(chunk) == chunk


def test_path_steps_come_back_as_path_steps() -> None:
    loaded = round_trip(make_chunk())
    assert all(isinstance(step, PathStep) for step in loaded.path)
    assert loaded.path == (CHUONG, DIEU)
    assert loaded.headings == ("Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ")


def test_unset_narrowing_stays_none() -> None:
    loaded = round_trip(make_chunk())
    assert (loaded.table_rows, loaded.text_span) == (None, None)


def test_table_rows_come_back_as_a_tuple() -> None:
    loaded = round_trip(make_chunk(start=14, end=15, table_rows=(1, 12)))
    assert loaded.table_rows == (1, 12)
    assert isinstance(loaded.table_rows, tuple)


def test_text_span_comes_back_as_a_tuple() -> None:
    loaded = round_trip(make_chunk(start=9, end=10, text_span=(0, 400)))
    assert loaded.text_span == (0, 400)


def test_collections_are_tuples() -> None:
    loaded = round_trip(make_chunk())
    assert isinstance(loaded.path, tuple) and isinstance(loaded.block_ids, tuple)


def test_front_matter_chunk_without_a_path() -> None:
    chunk = make_chunk(part="front_matter", path=(), start=0, end=1, text="BAN HÀNH QUY ĐỊNH")
    assert round_trip(chunk) == chunk


def test_every_chunk_of_a_real_run_round_trips() -> None:
    for max_tokens in (3, 12, 60):
        chunks = chunker(max_tokens).chunk(regulation())
        assert tuple(round_trip(c) for c in chunks) == chunks


# ============================================================ record shape


def test_record_shape() -> None:
    d = through_json(chunk_to_dict(make_chunk()))
    assert set(d) == {"kind", "schema_version", "chunk"}
    assert (d["kind"], d["schema_version"]) == (CHUNK_KIND, CHUNK_SCHEMA_VERSION) == ("chunk", 1)
    assert set(d["chunk"]) == {
        "id", "logical_document_id", "part", "path", "headings", "start", "end",
        "table_rows", "text_span", "text", "block_ids",
    }


def test_headings_are_written_for_consumers() -> None:
    d = through_json(chunk_to_dict(make_chunk()))
    assert d["chunk"]["headings"] == ["Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ"]


def test_path_step_shape() -> None:
    d = through_json(chunk_to_dict(make_chunk()))
    assert d["chunk"]["path"][1] == {
        "node_id": "n3", "level": "dieu", "number": "2", "marker": "Điều 2.",
        "title": "Giải thích từ ngữ", "start": 7,
    }


def test_unset_narrowing_is_written_as_null() -> None:
    d = through_json(chunk_to_dict(make_chunk()))
    assert (d["chunk"]["table_rows"], d["chunk"]["text_span"]) == (None, None)


# ============================================================ loading refusals


def test_wrong_kind_rejected() -> None:
    d = chunk_to_dict(make_chunk())
    d["kind"] = "segment"
    with pytest.raises(ValueError, match="'segment'"):
        chunk_from_dict(d)


def test_missing_kind_rejected() -> None:
    d = chunk_to_dict(make_chunk())
    del d["kind"]
    with pytest.raises(ValueError, match="None"):
        chunk_from_dict(d)


@pytest.mark.parametrize("version", [CHUNK_SCHEMA_VERSION + 1, 0, None, "1"])
def test_wrong_schema_version_rejected(version: Any) -> None:
    d = chunk_to_dict(make_chunk())
    d["schema_version"] = version
    with pytest.raises(ValueError, match="schema version"):
        chunk_from_dict(d)


def test_headings_that_disagree_with_the_path_are_rejected() -> None:
    # A file edited by hand: the breadcrumb no longer matches the path it is derived from.
    d = through_json(chunk_to_dict(make_chunk()))
    d["chunk"]["headings"] = ["Chương II", "Điều 2. Giải thích từ ngữ"]
    with pytest.raises(ValueError, match="headings"):
        chunk_from_dict(d)


def test_chunk_checks_run_on_load() -> None:
    d = through_json(chunk_to_dict(make_chunk()))
    d["chunk"]["end"] = d["chunk"]["start"]
    with pytest.raises(ValueError, match="start < end"):
        chunk_from_dict(d)


# ============================================================ files


def test_save_and_load(tmp_path: Path) -> None:
    chunks = chunker(12).chunk(regulation())
    path = tmp_path / "quy_dinh.chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks


def test_one_record_per_line(tmp_path: Path) -> None:
    chunks = chunker(12).chunk(regulation())
    path = tmp_path / "c.jsonl"
    save_chunks(chunks, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    assert [json.loads(line)["chunk"]["id"] for line in lines] == [c.id for c in chunks]


def test_saved_file_keeps_vietnamese_readable(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    save_chunks([make_chunk()], path)
    text = path.read_text(encoding="utf-8")
    assert "Giải thích từ ngữ" in text
    assert "\\u" not in text


def test_save_creates_parent_folders(tmp_path: Path) -> None:
    path = tmp_path / "out" / "nested" / "c.jsonl"
    save_chunks([make_chunk()], path)
    assert load_chunks(path) == (make_chunk(),)


def test_save_accepts_any_iterable_and_a_string_path(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    save_chunks((c for c in [make_chunk()]), str(path))
    assert load_chunks(str(path)) == (make_chunk(),)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    save_chunks([], path)
    assert path.read_text(encoding="utf-8") == ""
    assert load_chunks(path) == ()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record = json.dumps(chunk_to_dict(make_chunk()), ensure_ascii=False)
    path.write_text(f"\n{record}\n\n{record}\n\n", encoding="utf-8")
    assert load_chunks(path) == (make_chunk(), make_chunk())


def test_files_can_be_concatenated(tmp_path: Path) -> None:
    # Two documents' chunks in one file: every line still says what it is.
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_chunks([make_chunk(logical_document_id="ld1")], first)
    save_chunks([make_chunk(logical_document_id="ld2")], second)
    both = tmp_path / "both.jsonl"
    both.write_text(first.read_text(encoding="utf-8") + second.read_text(encoding="utf-8"), encoding="utf-8")
    assert [c.logical_document_id for c in load_chunks(both)] == ["ld1", "ld2"]


def test_a_bad_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    good = json.dumps(chunk_to_dict(make_chunk()), ensure_ascii=False)
    bad = good.replace('"kind": "chunk"', '"kind": "segment"')
    path.write_text(f"{good}\n{good}\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"c\.jsonl, line 3: .*'segment'"):
        load_chunks(path)


def test_a_line_that_is_not_json_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(chunk_to_dict(make_chunk()), ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(path)


def test_a_missing_field_is_reported_with_its_line_number(tmp_path: Path) -> None:
    d = through_json(chunk_to_dict(make_chunk()))
    del d["chunk"]["block_ids"]
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(d, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: .*block_ids"):
        load_chunks(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "missing.jsonl")