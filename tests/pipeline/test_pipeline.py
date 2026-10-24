"""Tests for pipeline/pipeline.py: Pipeline, PipelineResult, PipelineFailure.

A Pipeline holds configuration only: a profile (a registry name or a Profile),
a chunker, and cleaning steps (DEFAULT_STEPS unless told otherwise; () skips
cleaning). run(path) does extract → clean → build → chunk and returns every
stage; errors propagate. run_batch(paths) runs each file and turns input
problems (ValueError, KeyError, OSError) into PipelineFailure values, in input
order; anything else is a bug and still raises. result.save(out_dir) writes
<stem>.physical.json, <stem>.logical.json and <stem>.chunks.jsonl, replacing
files already there, and returns their paths. result.summarize() counts what
the run produced and warns about chunks over the chunker's budget (each chunk
records its own token_count) and about a document where the profile found no
headings.
"""

from pathlib import Path

import pytest

from document_processor.core.models import LogicalDocument
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from document_processor.pipeline import (
    PROFILES,
    Chunk,
    OutlineChunker,
    Pipeline,
    PipelineFailure,
    PipelineResult,
    ResultSummary,
    ReviewWarning,
    SavedFiles,
    WordTokenizer,
    build_logical_document,
    clean_document,
    extract_physical_document,
    load_chunks,
    load_logical_document,
    load_physical_document,
)
from tests.adapters.docx.test_adapter import build_docx, p

REGULATION = """BAN HÀNH QUY ĐỊNH
Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy định này quy định về quản lý kiến trúc tổng thể.
2. Quy định này áp dụng cho toàn hệ thống.
Điều 2. Giải thích từ ngữ
Trong Quy định này, các từ ngữ dưới đây được hiểu như sau:
Nơi nhận:
- Lưu VT
"""


def chunker(max_tokens: int = 200) -> OutlineChunker:
    return OutlineChunker(tokenizer=WordTokenizer(), max_tokens=max_tokens)


def pipeline(**overrides: object) -> Pipeline:
    fields: dict[str, object] = {"profile": "vn_regulation", "chunker": chunker()}
    fields.update(overrides)
    return Pipeline(**fields)  # type: ignore[arg-type]


def write_txt(folder: Path, name: str, text: str = REGULATION) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def write_docx(folder: Path, name: str, *paragraphs: str) -> Path:
    path = folder / name
    path.write_bytes(build_docx("".join(p(text) for text in paragraphs)))
    return path


# ============================================================ construction


def test_profile_by_name() -> None:
    assert pipeline(profile="vn_regulation")._resolved_profile is PROFILES.get("vn_regulation")


def test_profile_object() -> None:
    assert pipeline(profile=VN_REGULATION)._resolved_profile is VN_REGULATION


def test_unknown_profile_name_fails_when_the_pipeline_is_built() -> None:
    # Not at the first file of a batch.
    with pytest.raises(KeyError, match="unknown profile 'vn_regulaton'"):
        pipeline(profile="vn_regulaton")


def test_keeps_what_the_user_passed() -> None:
    built = pipeline(profile="vn_regulation")
    assert built.profile == "vn_regulation"
    assert "_resolved_profile" not in repr(built)


def test_is_frozen() -> None:
    with pytest.raises(AttributeError):
        pipeline().profile = "other"  # type: ignore[misc]


def test_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        Pipeline("vn_regulation", chunker())  # type: ignore[call-arg]


def test_cleaning_is_on_by_default() -> None:
    from document_processor.core.cleaning import DEFAULT_STEPS

    assert pipeline().cleaning_steps == DEFAULT_STEPS


# ============================================================ run


def test_run_returns_every_stage(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "quy_dinh.txt")
    result = pipeline().run(path)
    assert isinstance(result, PipelineResult)
    assert result.source == path
    assert result.physical.source_filename == "quy_dinh.txt"
    assert isinstance(result.logical, LogicalDocument)
    assert result.chunks and all(isinstance(c, Chunk) for c in result.chunks)


def test_stages_belong_together(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    assert result.logical.physical_document_id == result.physical.id
    assert {c.logical_document_id for c in result.chunks} == {result.logical.id}
    block_ids = {b.id for b in result.physical.blocks}
    assert all(set(c.block_ids) <= block_ids for c in result.chunks)


def test_same_as_running_the_steps_by_hand(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "quy_dinh.txt")
    physical = clean_document(extract_physical_document(path))
    logical = build_logical_document(VN_REGULATION, physical)
    result = pipeline().run(path)
    assert (result.physical, result.logical, result.chunks) == (physical, logical, chunker().chunk(logical))


def test_the_regulation_is_read_by_the_profile(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    assert [(n.level, n.marker) for n in result.logical.nodes] == [
        ("chuong", "Chương I"),
        ("dieu", "Điều 1."),
        ("khoan", "1."),
        ("khoan", "2."),
        ("dieu", "Điều 2."),
    ]
    assert [c.part for c in result.chunks] == ["front_matter", "body", "body", "end_matter"]


def test_accepts_a_string_path(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "quy_dinh.txt")
    assert pipeline().run(str(path)).source == path


def test_runs_a_docx_too(tmp_path: Path) -> None:
    path = write_docx(tmp_path, "quy_dinh.docx", "Điều 1. Phạm vi điều chỉnh", "1. Nội dung khoản")
    result = pipeline().run(path)
    assert [n.level for n in result.logical.nodes] == ["dieu", "khoan"]


def test_the_chunker_decides_the_chunks(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "quy_dinh.txt")
    assert len(pipeline(chunker=chunker(5)).run(path).chunks) > len(pipeline(chunker=chunker(200)).run(path).chunks)


# ============================================================ cleaning


def test_cleaned_document_is_kept(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "a.txt", "  Nội dung có khoảng trắng  \n")
    (block,) = pipeline().run(path).physical.blocks
    assert block.text == "Nội dung có khoảng trắng"
    assert "clean:strip_edges" in block.changelog


def test_no_cleaning_steps_skips_cleaning(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "a.txt", "  Nội dung có khoảng trắng  \n")
    (block,) = pipeline(cleaning_steps=()).run(path).physical.blocks
    assert block.text == "  Nội dung có khoảng trắng  "
    assert block.changelog == ()


# ============================================================ errors in run


def test_run_raises_for_an_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"x")
    with pytest.raises(KeyError, match=".pdf"):
        pipeline().run(path)


def test_run_raises_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline().run(tmp_path / "missing.txt")


# ============================================================ run_batch


def test_batch_keeps_the_input_order(tmp_path: Path) -> None:
    paths = [write_txt(tmp_path, f"{name}.txt") for name in ("c", "a", "b")]
    outcomes = pipeline().run_batch(paths)
    assert [o.source for o in outcomes] == paths
    assert all(isinstance(o, PipelineResult) for o in outcomes)


def test_batch_turns_input_problems_into_failures(tmp_path: Path) -> None:
    good = write_txt(tmp_path, "good.txt")
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not a zip file")
    unknown = tmp_path / "scan.pdf"
    unknown.write_bytes(b"x")
    missing = tmp_path / "missing.txt"
    not_utf8 = tmp_path / "old.txt"
    not_utf8.write_bytes("Đơn".encode("cp1258"))
    last = write_txt(tmp_path, "last.txt")

    outcomes = pipeline().run_batch([good, broken, unknown, missing, not_utf8, last])

    assert [type(o).__name__ for o in outcomes] == [
        "PipelineResult",
        "PipelineFailure",
        "PipelineFailure",
        "PipelineFailure",
        "PipelineFailure",
        "PipelineResult",
    ]
    failures = [o for o in outcomes if isinstance(o, PipelineFailure)]
    assert [f.source for f in failures] == [broken, unknown, missing, not_utf8]
    assert failures[0].error.startswith("ValueError: ") and "broken.docx" in failures[0].error
    assert failures[1].error.startswith("KeyError: ") and ".pdf" in failures[1].error
    assert failures[2].error.startswith("FileNotFoundError: ")
    assert failures[3].error.startswith("ValueError: ") and "line 1" in failures[3].error


def test_batch_accepts_any_iterable_and_string_paths(tmp_path: Path) -> None:
    paths = (str(write_txt(tmp_path, f"{i}.txt")) for i in range(2))
    outcomes = pipeline().run_batch(paths)
    assert len(outcomes) == 2 and all(isinstance(o.source, Path) for o in outcomes)


def test_empty_batch() -> None:
    assert pipeline().run_batch([]) == ()


class _BrokenChunker:
    """A chunker with a bug: errors that are not input problems must stop the batch."""

    max_tokens = 100

    def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]:
        raise TypeError("bug in the chunker")


def test_a_bug_still_stops_the_batch(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "a.txt")
    with pytest.raises(TypeError, match="bug in the chunker"):
        pipeline(chunker=_BrokenChunker()).run_batch([path])


def test_failures_are_plain_data(tmp_path: Path) -> None:
    (failure,) = pipeline().run_batch([tmp_path / "missing.txt"])
    assert isinstance(failure, PipelineFailure)
    assert isinstance(failure.error, str)



# ============================================================ save


def test_save_writes_three_files_named_after_the_source(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    files = result.save(tmp_path / "out")
    assert files == SavedFiles(
        physical=tmp_path / "out" / "quy_dinh.physical.json",
        logical=tmp_path / "out" / "quy_dinh.logical.json",
        chunks=tmp_path / "out" / "quy_dinh.chunks.jsonl",
    )
    assert files.physical.is_file() and files.logical.is_file() and files.chunks.is_file()


def test_saved_stages_load_back_equal(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    files = result.save(tmp_path / "out")
    assert load_physical_document(files.physical) == result.physical
    assert load_logical_document(files.logical) == result.logical
    assert load_chunks(files.chunks) == result.chunks


def test_save_creates_the_folder(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "a.txt"))
    files = result.save(tmp_path / "out" / "nested")
    assert files.chunks.parent == tmp_path / "out" / "nested"
    assert files.chunks.is_file()


def test_save_accepts_a_string_folder(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "a.txt"))
    assert result.save(str(tmp_path / "out")).logical == tmp_path / "out" / "a.logical.json"


def test_the_source_extension_is_not_part_of_the_names(tmp_path: Path) -> None:
    result = pipeline().run(write_docx(tmp_path, "quy_dinh.docx", "Điều 1. A", "Nội dung"))
    assert result.save(tmp_path / "out").physical.name == "quy_dinh.physical.json"


def test_saving_again_replaces_the_files(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "quy_dinh.txt")
    out = tmp_path / "out"
    pipeline(chunker=chunker(5)).run(path).save(out)
    result = pipeline(chunker=chunker(200)).run(path)
    files = result.save(out)
    assert load_chunks(files.chunks) == result.chunks


def test_same_file_name_in_two_folders_replaces_each_others_outputs(tmp_path: Path) -> None:
    # Documented behaviour: outputs are named after the file name only.
    first = write_txt(tmp_path / "2024", "quy_dinh.txt", "Điều 1. Cũ\n")
    second = write_txt(tmp_path / "2025", "quy_dinh.txt", "Điều 1. Mới\n")
    out = tmp_path / "out"
    pipeline().run(first).save(out)
    files = pipeline().run(second).save(out)
    assert [n.title for n in load_logical_document(files.logical).nodes] == ["Mới"]


def test_saving_every_result_of_a_batch(tmp_path: Path) -> None:
    paths = [write_txt(tmp_path, f"{name}.txt") for name in ("a", "b")]
    outcomes = pipeline().run_batch([*paths, tmp_path / "missing.txt"])
    saved = [o.save(tmp_path / "out") for o in outcomes if isinstance(o, PipelineResult)]
    assert [f.chunks.name for f in saved] == ["a.chunks.jsonl", "b.chunks.jsonl"]



# ============================================================ summarize


def test_result_records_the_chunkers_budget(tmp_path: Path) -> None:
    assert pipeline(chunker=chunker(37)).run(write_txt(tmp_path, "a.txt")).max_tokens == 37


def test_summary_counts(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    summary = result.summarize()
    assert isinstance(summary, ResultSummary)
    assert summary.source == result.source
    assert summary.block_count == len(result.physical.blocks) == 10
    assert summary.segment_count == len(result.logical.segments)
    assert summary.chunk_count == len(result.chunks)
    assert summary.chunk_tokens == tuple(c.token_count for c in result.chunks)


def test_summary_counts_nodes_per_level_in_outline_order(tmp_path: Path) -> None:
    summary = pipeline().run(write_txt(tmp_path, "quy_dinh.txt")).summarize()
    assert summary.nodes_by_levels == (("chuong", 1), ("dieu", 2), ("khoan", 2))


def test_summary_front_and_end_matter(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "quy_dinh.txt"))
    summary = result.summarize()
    assert summary.front_matter == (0, result.logical.body_start) == (0, 1)
    assert summary.end_matter == (result.logical.body_end, len(result.logical.segments))
    assert summary.end_matter[0] < summary.end_matter[1]  # "Nơi nhận:" was found


def test_a_clean_document_has_no_warnings(tmp_path: Path) -> None:
    assert pipeline().run(write_txt(tmp_path, "quy_dinh.txt")).summarize().warnings == ()


def test_chunk_over_budget_is_warned(tmp_path: Path) -> None:
    long_line = " ".join(["từ"] * 12)
    path = write_txt(tmp_path, "a.txt", f"Điều 1. Phạm vi\n{long_line}\nNgắn\n")
    result = pipeline(chunker=chunker(5)).run(path)
    over = [c for c in result.chunks if c.token_count > 5]
    assert len(over) == 1
    (warning,) = result.summarize().warnings
    assert isinstance(warning, ReviewWarning)
    assert warning.code == "chunk_over_budget"
    assert warning.chunk_id == over[0].id
    assert str(over[0].token_count) in warning.message and "5" in warning.message


def test_every_chunk_over_budget_gets_its_own_warning(tmp_path: Path) -> None:
    long_line = " ".join(["từ"] * 12)
    path = write_txt(tmp_path, "a.txt", f"Điều 1. Phạm vi\n{long_line}\n{long_line}\n")
    warnings = pipeline(chunker=chunker(5)).run(path).summarize().warnings
    assert [w.code for w in warnings] == ["chunk_over_budget", "chunk_over_budget"]
    assert len({w.chunk_id for w in warnings}) == 2


def test_a_chunk_exactly_at_the_budget_is_not_warned(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "a.txt", "Điều 1. Phạm vi\nmột hai ba bốn năm\n")
    result = pipeline(chunker=chunker(5)).run(path)
    assert max(c.token_count for c in result.chunks) == 5
    assert result.summarize().warnings == ()


def test_no_headings_is_warned(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "a.txt", "Một đoạn văn\nMột đoạn khác\n")
    result = pipeline().run(path)
    assert result.logical.nodes == ()
    warnings = result.summarize().warnings
    assert [w.code for w in warnings] == ["no_headings"]
    assert "vn_regulation" in warnings[0].message
    assert warnings[0].chunk_id is None


def test_summary_of_a_result_without_chunks(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n\n", encoding="utf-8")
    summary = pipeline().run(path).summarize()
    assert (summary.block_count, summary.chunk_count, summary.chunk_tokens) == (0, 0, ())
    assert [w.code for w in summary.warnings] == ["no_headings"]


def test_summary_records_the_budget(tmp_path: Path) -> None:
    assert pipeline(chunker=chunker(37)).run(write_txt(tmp_path, "a.txt")).summarize().max_tokens == 37


def _summary(chunk_tokens: tuple[int, ...]) -> ResultSummary:
    return ResultSummary(
        source=Path("a.txt"), block_count=0, segment_count=0, nodes_by_levels=(), front_matter=(0, 0),
        end_matter=(0, 0), chunk_count=len(chunk_tokens), chunk_tokens=chunk_tokens, warnings=(), max_tokens=300,
    )


def test_chunk_size_statistics() -> None:
    summary = _summary((40, 12, 847, 180))
    assert (summary.smallest_chunk_tokens, summary.median_chunk_tokens, summary.largest_chunk_tokens) == (12, 110, 847)


def test_median_of_an_even_count_falls_between_two_sizes() -> None:
    assert _summary((180, 181)).median_chunk_tokens == 180.5


def test_median_of_an_odd_count_is_the_middle_size() -> None:
    assert _summary((5, 300, 7)).median_chunk_tokens == 7


def test_statistics_of_no_chunks_are_none() -> None:
    summary = _summary(())
    assert (summary.smallest_chunk_tokens, summary.median_chunk_tokens, summary.largest_chunk_tokens) == (None, None, None)


def test_statistics_come_from_the_runs_chunks(tmp_path: Path) -> None:
    result = pipeline(chunker=chunker(5)).run(write_txt(tmp_path, "quy_dinh.txt"))
    summary = result.summarize()
    sizes = [c.token_count for c in result.chunks]
    assert summary.smallest_chunk_tokens == min(sizes)
    assert summary.largest_chunk_tokens == max(sizes)


# ============================================================ ResultSummary.format


def test_format_a_clean_document(tmp_path: Path) -> None:
    summary = pipeline().run(write_txt(tmp_path, "quy_dinh.txt")).summarize()
    assert summary.format() == "\n".join([
        "quy_dinh.txt",
        "  10 blocks → 11 segments, 5 nodes, 4 chunks",
        "  front matter [0, 1) · end matter [9, 11)",
        "  nodes: chuong 1 · dieu 2 · khoan 2",
        "  chunk tokens: min 4 · median 12 · max 29 (budget 200)",
        "  no warnings",
    ])


def test_format_lists_each_warning(tmp_path: Path) -> None:
    long_line = " ".join(["từ"] * 12)
    path = write_txt(tmp_path, "a.txt", f"Điều 1. Phạm vi\n{long_line}\nNgắn\n")
    lines = pipeline(chunker=chunker(5)).run(path).summarize().format().splitlines()
    assert lines[4] == "  chunk tokens: min 1 · median 6 · max 12 (budget 5)"
    assert lines[5:] == ["  ⚠ chunk_over_budget  chunk c0 has 12 tokens (budget 5)"]


def test_format_without_nodes_or_chunks(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n", encoding="utf-8")
    lines = pipeline().run(path).summarize().format().splitlines()
    assert lines[:5] == [
        "empty.txt",
        "  0 blocks → 0 segments, 0 nodes, 0 chunks",
        "  front matter [0, 0) · end matter [0, 0)",
        "  nodes: none",
        "  chunk tokens: none",
    ]
    assert lines[5].startswith("  ⚠ no_headings  profile 'vn_regulation'")


def test_format_shows_the_file_name_not_the_whole_path(tmp_path: Path) -> None:
    path = write_txt(tmp_path / "deep" / "folder", "quy_dinh.txt")
    assert pipeline().run(path).summarize().format().splitlines()[0] == "quy_dinh.txt"


def test_format_does_not_end_with_a_newline(tmp_path: Path) -> None:
    assert not pipeline().run(write_txt(tmp_path, "a.txt")).summarize().format().endswith("\n")


# ============================================================ level order


def test_result_records_the_profiles_level_order(tmp_path: Path) -> None:
    result = pipeline().run(write_txt(tmp_path, "a.txt"))
    assert result.level_order == tuple(level.name for level in VN_REGULATION.levels or ())
    assert result.level_order[:3] == ("chuong", "muc", "dieu")


def test_levels_are_summarized_in_profile_order_not_first_appearance(tmp_path: Path) -> None:
    # Chương I has a Điều and a dash item before Chương II brings the first Mục.
    text = """Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi
1. Nội dung khoản
- gạch đầu dòng
Chương II
TỔ CHỨC
Mục 1
THÀNH PHẦN
Điều 2. Thành phần
a) Điểm a
"""
    result = pipeline().run(write_txt(tmp_path, "a.txt", text))
    first_seen = list(dict.fromkeys(n.level for n in result.logical.nodes))
    assert first_seen.index("gach") < first_seen.index("muc")  # the case this test is about
    summary = result.summarize()
    assert [level for level, _ in summary.nodes_by_levels] == ["chuong", "muc", "dieu", "khoan", "diem", "gach"]
    assert dict(summary.nodes_by_levels) == {"chuong": 2, "muc": 1, "dieu": 2, "khoan": 1, "diem": 1, "gach": 1}