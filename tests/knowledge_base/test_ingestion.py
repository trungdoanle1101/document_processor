"""Indexer: adding, refreshing and removing documents keeps the index and the registry in step."""

from datetime import date
from pathlib import Path

import pytest

from document_processor.chunking import WordTokenizer
from tests.knowledge_base.helpers import (
    DATA_REGULATION,
    LEAVE_REGULATION,
    chunks_of,
    make_fakes,
    write,
)
from tests.retrieval.stub_embedder import StubEmbedder


def test_adding_indexes_the_chunks_with_their_group_and_part(tmp_path: Path) -> None:
    fakes = make_fakes()
    result = fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION), access_group="it"
    )

    assert result.action == "indexed"
    chunks = chunks_of(fakes, "du-lieu")
    assert result.record.chunk_count == len(chunks) > 0
    assert {c.attributes["access_group"] for c in chunks} == {"it"}
    assert {"body"} <= {c.attributes["part"] for c in chunks}
    assert (result.record.version, result.record.title, result.record.source) == (
        1,
        "du-lieu",
        str(tmp_path / "du-lieu.txt"),
    )


def test_the_provenance_names_what_produced_the_chunks(tmp_path: Path) -> None:
    fakes = make_fakes()
    record = fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="it"
    ).record
    assert record.provenance.profile.startswith("vn_regulation@")
    assert (
        record.provenance.chunker,
        record.provenance.embedder,
        record.provenance.collection,
    ) == (
        "outline/200",
        fakes.embedder.identity,
        "test",
    )
    assert record.provenance.source_hash.startswith("sha256:")


def test_adding_an_unchanged_document_does_nothing(tmp_path: Path) -> None:
    fakes = make_fakes()
    path = write(tmp_path, "a.txt", DATA_REGULATION)
    fakes.indexer.add_document(path, access_group="it")
    batches = len(fakes.embedder.passage_batches)

    result = fakes.indexer.add_document(path, access_group="it")
    assert (result.action, result.record.version) == ("unchanged", 1)
    assert len(fakes.embedder.passage_batches) == batches


def test_a_new_group_or_attributes_rewrite_the_chunks_without_embedding_again(
    tmp_path: Path,
) -> None:
    fakes = make_fakes()
    path = write(tmp_path, "a.txt", DATA_REGULATION)
    fakes.indexer.add_document(
        path, access_group="it", attributes={"kind": "quy_dinh", "year": 2025}
    )
    batches = len(fakes.embedder.passage_batches)

    result = fakes.indexer.add_document(
        path, access_group="all", attributes={"kind": "quy_dinh"}
    )
    assert (result.action, result.record.version) == ("attributes updated", 2)
    assert len(fakes.embedder.passage_batches) == batches
    for chunk in chunks_of(fakes, "a"):
        assert chunk.attributes["access_group"] == "all"
        assert chunk.attributes["kind"] == "quy_dinh"
        assert "year" not in chunk.attributes


def test_a_changed_file_is_indexed_again_as_a_new_version(tmp_path: Path) -> None:
    fakes = make_fakes()
    path = write(tmp_path, "a.txt", DATA_REGULATION)
    fakes.indexer.add_document(path, access_group="it")
    write(
        tmp_path,
        "a.txt",
        DATA_REGULATION + "Điều 3. Hiệu lực\n1. Quy định có hiệu lực từ ngày ký.\n",
    )

    result = fakes.indexer.add_document(path, access_group="it")
    assert (result.action, result.record.version) == ("indexed", 2)
    assert any("hiệu lực" in c.text for c in chunks_of(fakes, "a"))
    assert [r.version for r in fakes.registry.history("a")] == [1, 2]


def test_force_indexes_even_when_nothing_changed(tmp_path: Path) -> None:
    fakes = make_fakes()
    path = write(tmp_path, "a.txt", DATA_REGULATION)
    fakes.indexer.add_document(path, access_group="it")
    assert (
        fakes.indexer.add_document(path, access_group="it", force=True).action
        == "indexed"
    )


def test_the_id_and_title_can_be_chosen(tmp_path: Path) -> None:
    fakes = make_fakes()
    record = fakes.indexer.add_document(
        write(tmp_path, "file name.txt", DATA_REGULATION),
        access_group="it",
        document_id="qd-du-lieu",
        title="Quy định quản lý dữ liệu",
        attributes={"issued_on": date(2025, 3, 1)},
    ).record
    assert (record.document_id, record.title) == (
        "qd-du-lieu",
        "Quy định quản lý dữ liệu",
    )
    assert chunks_of(fakes, "qd-du-lieu")[0].attributes["issued_on"] == date(2025, 3, 1)


@pytest.mark.parametrize("name", ["access_group", "part"])
def test_attributes_the_app_sets_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="set by the app"):
        make_fakes().indexer.add_document(
            write(tmp_path, "a.txt", DATA_REGULATION),
            access_group="it",
            attributes={name: "x"},
        )


def test_removing_takes_the_document_out_but_keeps_its_history(tmp_path: Path) -> None:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="it"
    )
    fakes.indexer.add_document(
        write(tmp_path, "b.txt", LEAVE_REGULATION), access_group="hr"
    )

    fakes.indexer.remove_document("a")
    assert chunks_of(fakes, "a") == []
    assert chunks_of(fakes, "b") != []
    assert fakes.registry.get("a") is None
    assert len(fakes.registry.history("a")) == 1


def test_removing_an_unknown_document_is_refused() -> None:
    with pytest.raises(KeyError, match="no document 'x'"):
        make_fakes().indexer.remove_document("x")


def test_a_readded_document_continues_its_versions(tmp_path: Path) -> None:
    fakes = make_fakes()
    path = write(tmp_path, "a.txt", DATA_REGULATION)
    fakes.indexer.add_document(path, access_group="it")
    fakes.indexer.remove_document("a")
    assert fakes.indexer.add_document(path, access_group="it").record.version == 2


# Stale documents ----------------------------------------------------------------------------


def test_nothing_is_stale_right_after_indexing(tmp_path: Path) -> None:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="it"
    )
    assert fakes.indexer.find_stale() == []


def test_a_changed_or_missing_file_makes_a_document_stale(tmp_path: Path) -> None:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="it"
    )
    fakes.indexer.add_document(
        write(tmp_path, "b.txt", LEAVE_REGULATION), access_group="hr"
    )
    write(tmp_path, "a.txt", DATA_REGULATION + "Điều 3. Mới\n")
    (tmp_path / "b.txt").unlink()

    stale = {s.record.document_id: s.reasons for s in fakes.indexer.find_stale()}
    assert stale == {"a": ("source_hash",), "b": ("source missing",)}


def test_another_embedder_makes_every_document_stale(tmp_path: Path) -> None:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="it"
    )
    # the same registry, seen by an app now configured with another embedder
    other = make_fakes(
        embedder=StubEmbedder(dimension=16, tokenizer=WordTokenizer()),
        registry=fakes.registry,
    )
    assert [s.reasons for s in other.indexer.find_stale()] == [("embedder",)]


def test_refreshing_indexes_changed_documents_and_leaves_missing_ones(
    tmp_path: Path,
) -> None:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="it"
    )
    fakes.indexer.add_document(
        write(tmp_path, "b.txt", LEAVE_REGULATION), access_group="hr"
    )
    write(tmp_path, "a.txt", DATA_REGULATION + "Điều 3. Mới\n")
    (tmp_path / "b.txt").unlink()

    results = fakes.indexer.refresh_stale()
    assert [(r.record.document_id, r.action, r.record.version) for r in results] == [
        ("a", "indexed", 2)
    ]
    assert fakes.registry.get("b") is not None
