"""The plain profile: no recognisable structure, so the whole text is body, cut by size."""

from pathlib import Path

import pytest

from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.logical.profiles.plain import PLAIN
from document_processor.preparation import PreparedDocument, Preparer

REGULATION = """QUY ĐỊNH QUẢN LÝ DỮ LIỆU
Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy định này áp dụng cho dữ liệu khách hàng.
2. Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày.
Điều 2. Thời hạn lưu trữ
Dữ liệu lưu trữ dài hạn được lưu trên Tape.
Nơi nhận:
- Như trên
"""


def prepare(tmp_path: Path, text: str, *, max_tokens: int = 200) -> PreparedDocument:
    path = tmp_path / "document.txt"
    path.write_text(text, encoding="utf-8")
    chunker = OutlineChunker(tokenizer=WordTokenizer(), max_tokens=max_tokens)
    return Preparer(profile=PLAIN, chunker=chunker).prepare(path)


def test_plain_describes_no_structure() -> None:
    assert (PLAIN.name, PLAIN.version) == ("plain", "1")
    assert PLAIN.levels is None
    assert PLAIN.end_matter is None


def test_plain_finds_no_headings_even_in_a_regulation(tmp_path: Path) -> None:
    prepared = prepare(tmp_path, REGULATION)

    assert prepared.logical.nodes == ()
    assert (prepared.logical.profile_name, prepared.logical.profile_version) == (
        "plain",
        "1",
    )


def test_the_whole_text_is_body(tmp_path: Path) -> None:
    # no end matter: "Nơi nhận:" is just text to plain
    prepared = prepare(tmp_path, REGULATION)

    assert prepared.logical.body_start == 0
    assert prepared.logical.body_end == len(prepared.logical.segments)
    assert {chunk.part for chunk in prepared.chunks} == {"body"}
    assert "Nơi nhận" in "".join(chunk.text for chunk in prepared.chunks)


PARAGRAPH_WORDS = 12  # "Đoạn 0: nội dung mô tả hoạt động trong kỳ báo cáo."
REPORT = "\n".join(
    f"Đoạn {i}: nội dung mô tả hoạt động trong kỳ báo cáo." for i in range(40)
)


@pytest.mark.parametrize("max_tokens", [PARAGRAPH_WORDS, 25, 60])
def test_the_text_is_cut_by_size(tmp_path: Path, max_tokens: int) -> None:
    text = REPORT

    prepared = prepare(tmp_path, text, max_tokens=max_tokens)

    assert len(prepared.chunks) > 1
    assert all(chunk.token_count <= max_tokens for chunk in prepared.chunks)


def test_a_paragraph_longer_than_the_budget_stays_whole_and_is_reported(
    tmp_path: Path,
) -> None:
    prepared = prepare(tmp_path, REPORT, max_tokens=PARAGRAPH_WORDS - 1)

    assert all(chunk.token_count == PARAGRAPH_WORDS for chunk in prepared.chunks)
    warnings = [warning.code for warning in prepared.summarize().warnings]
    assert "chunk_over_budget" in warnings


def test_every_word_lands_in_a_chunk(tmp_path: Path) -> None:
    prepared = prepare(tmp_path, REGULATION, max_tokens=10)

    chunked = " ".join(chunk.text for chunk in prepared.chunks).split()
    assert chunked == REGULATION.split()


def test_the_summary_warns_that_there_are_no_headings(tmp_path: Path) -> None:
    warnings = prepare(tmp_path, REGULATION).summarize().warnings

    assert "no_headings" in [warning.code for warning in warnings]
