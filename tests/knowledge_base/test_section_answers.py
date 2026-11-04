"""Asking with section completion on: the model sees every part of a section the hits share."""

from pathlib import Path

from tests.knowledge_base.helpers import chunks_of, make_fakes, write


def layer(number: int, name: str) -> str:
    # long enough that each khoản becomes its own chunk (the fake chunker allows 200 words)
    return f"{number}. Lớp {name}: " + " ".join(
        f"thành phần {name} {i}" for i in range(80)
    )


LAYERS = ("ngoài", "trong", "ứng dụng", "dữ liệu", "bảo mật")
REGULATION = "\n".join(
    [
        "QUY ĐỊNH KIẾN TRÚC TÍCH HỢP",
        "Điều 1. Các lớp tích hợp",
        *(layer(i, name) for i, name in enumerate(LAYERS, start=1)),
        "Điều 2. Hiệu lực",
        "1. Quy định có hiệu lực từ ngày ký.",
    ]
)
QUESTION = "Kiến trúc tích hợp gồm những lớp nào?"


def test_every_layer_reaches_the_model_when_hits_share_the_article(
    tmp_path: Path,
) -> None:
    fakes = make_fakes(section_chars=50_000)
    fakes.indexer.add_document(write(tmp_path, "kt.txt", REGULATION), access_group="it")
    layer_chunks = [
        chunk.chunk_id
        for chunk in chunks_of(fakes, "kt")
        if any(key.startswith("dieu:1/khoan:") for key in chunk.reference_keys)
    ]
    assert len(layer_chunks) == 5  # one chunk per khoản, as the test needs

    answer = fakes.knowledge_base.ask(QUESTION, groups=["it"])

    shown = [passage.chunk.chunk_id for passage in answer.passages]
    assert set(layer_chunks) <= set(shown)
    # the article's chunks come first, in document order
    assert shown[: len(layer_chunks)] == layer_chunks
    added = [p for p in answer.passages if p.completed_section is not None]
    assert added and {p.completed_section for p in added} == {"dieu:1"}


def test_with_completion_off_only_the_hits_are_shown(tmp_path: Path) -> None:
    fakes = make_fakes(section_chars=0)
    fakes.indexer.add_document(write(tmp_path, "kt.txt", REGULATION), access_group="it")
    answer = fakes.knowledge_base.ask(QUESTION, groups=["it"])
    assert len(answer.passages) <= 3  # the helper's passages setting
    assert all(p.completed_section is None for p in answer.passages)


def test_a_reader_without_access_gets_nothing_added(tmp_path: Path) -> None:
    fakes = make_fakes(section_chars=50_000)
    fakes.indexer.add_document(write(tmp_path, "kt.txt", REGULATION), access_group="it")
    answer = fakes.knowledge_base.ask(QUESTION, groups=["hr"])
    assert answer.passages == ()
    assert not answer.found