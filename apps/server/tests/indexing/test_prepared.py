"""Prepared output: the settings that describe it, and saving, loading and deleting it."""

import io
import json
import uuid
from pathlib import Path
from typing import BinaryIO

import pytest

from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.preparation import Preparer
from kb_server.indexing.prepared import (
    PreparationSettings,
    PreparedParts,
    delete_prepared,
    load_prepared,
    save_prepared,
)
from kb_server.storage.files import (
    FileStore,
    LocalFileStore,
    prepared_folder,
    prepared_version_folder,
)

REGULATION = """QUY ĐỊNH QUẢN LÝ DỮ LIỆU
Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy định này áp dụng cho dữ liệu khách hàng tại ngân hàng.
2. Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày.
Điều 2. Thời hạn lưu trữ
Dữ liệu lưu trữ dài hạn được lưu trên Tape.
Nơi nhận:
- Như trên
"""

SETTINGS = PreparationSettings(
    profiles=("vn_regulation@4",),
    fallback_profile="plain@1",
    chunker={"kind": "OutlineChunker", "max_tokens": 512, "table_layout": "grid"},
    embedder="huggingface:BAAI/bge-m3",
)

DOCUMENT_ID = uuid.UUID("0b5e4c1a-6f1d-4a43-9c55-3d2b8e0f7a19")
OTHER_DOCUMENT_ID = uuid.UUID("7d2f9e30-1c4b-4e8a-b6a1-52c0d9e4f3b8")


class StopsBeforeChunks:
    """A store whose writes fail at chunks.jsonl, like a worker stopping mid-save."""

    def __init__(self, store: FileStore) -> None:
        self._store = store

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        if key.endswith("/chunks.jsonl"):
            raise OSError("worker stopped")
        self._store.put(key, data, size=size, media_type=media_type)

    def open(self, key: str) -> BinaryIO:
        return self._store.open(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def delete_folder(self, folder: str) -> int:
        return self._store.delete_folder(folder)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture
def store(root: Path) -> FileStore:
    return LocalFileStore(root)


@pytest.fixture
def parts(tmp_path: Path) -> PreparedParts:
    """A real regulation, prepared by the framework (whitespace tokens: no models)."""
    path = tmp_path / "quy_dinh.txt"
    path.write_text(REGULATION, encoding="utf-8")
    preparer = Preparer(
        profile="vn_regulation",
        chunker=OutlineChunker(tokenizer=WordTokenizer(), max_tokens=50),
    )
    prepared = preparer.prepare(path)
    assert prepared.logical.nodes  # the profile found the structure
    return PreparedParts(logical=prepared.logical, chunks=prepared.chunks)


def version_path(root: Path, version: int = 1) -> Path:
    return root / prepared_version_folder(DOCUMENT_ID, version)


# the settings


def test_describe_gives_the_settings_in_full() -> None:
    assert SETTINGS.describe() == {
        "profiles": ["vn_regulation@4"],
        "fallback_profile": "plain@1",
        "chunker": {
            "kind": "OutlineChunker",
            "max_tokens": 512,
            "table_layout": "grid",
        },
        "embedder": "huggingface:BAAI/bge-m3",
    }
    json.dumps(SETTINGS.describe())  # plain JSON, ready for provenance


def test_describe_is_a_copy() -> None:
    described = SETTINGS.describe()
    described["chunker"]["max_tokens"] = 1
    described["profiles"].append("numbered@1")

    assert SETTINGS.chunker["max_tokens"] == 512
    assert SETTINGS.profiles == ("vn_regulation@4",)


def test_settings_can_be_compared() -> None:
    # how a later --stale check tells outdated output apart
    same = PreparationSettings(
        profiles=("vn_regulation@4",),
        fallback_profile="plain@1",
        chunker={"table_layout": "grid", "max_tokens": 512, "kind": "OutlineChunker"},
        embedder="huggingface:BAAI/bge-m3",
    )
    newer = PreparationSettings(
        profiles=("vn_regulation@5",),
        fallback_profile="plain@1",
        chunker=dict(SETTINGS.chunker),
        embedder="huggingface:BAAI/bge-m3",
    )

    assert same.describe() == SETTINGS.describe()
    assert newer.describe() != SETTINGS.describe()


# saving and loading


def test_saved_parts_load_back_the_same(store: FileStore, parts: PreparedParts) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)

    loaded = load_prepared(store, DOCUMENT_ID, 1)

    assert loaded == parts
    assert loaded is not None
    assert len(loaded.chunks) == len(parts.chunks) > 1


def test_vietnamese_text_survives_unchanged(
    store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)

    loaded = load_prepared(store, DOCUMENT_ID, 1)

    assert loaded is not None
    text = "".join(chunk.text for chunk in loaded.chunks)
    assert "Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày" in text


def test_the_files_are_readable_json_in_the_versions_folder(
    root: Path, store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    folder = version_path(root)

    assert sorted(path.name for path in folder.iterdir()) == [
        "chunks.jsonl",
        "logical.json",
    ]
    logical_text = (folder / "logical.json").read_text(encoding="utf-8")
    json.loads(logical_text)
    assert "Điều 1" in logical_text  # not \u escapes
    lines = (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(parts.chunks)
    assert all(json.loads(line) for line in lines)


def test_nothing_saved_loads_as_none(store: FileStore) -> None:
    assert load_prepared(store, DOCUMENT_ID, 1) is None


def test_an_outline_without_its_chunks_loads_as_none(
    root: Path, store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    (version_path(root) / "chunks.jsonl").unlink()

    assert load_prepared(store, DOCUMENT_ID, 1) is None


def test_a_document_with_no_chunks_round_trips(
    store: FileStore, parts: PreparedParts
) -> None:
    empty = PreparedParts(logical=parts.logical, chunks=())

    save_prepared(store, DOCUMENT_ID, 1, empty)

    assert load_prepared(store, DOCUMENT_ID, 1) == empty


def test_a_damaged_file_raises(
    root: Path, store: FileStore, parts: PreparedParts
) -> None:
    # a bug or a damaged store: it must surface, not pass for "nothing saved"
    save_prepared(store, DOCUMENT_ID, 1, parts)
    (version_path(root) / "logical.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_prepared(store, DOCUMENT_ID, 1)


@pytest.mark.parametrize("version", [0, -1])
def test_a_version_below_1_is_refused(
    store: FileStore, parts: PreparedParts, version: int
) -> None:
    with pytest.raises(ValueError):
        save_prepared(store, DOCUMENT_ID, version, parts)
    with pytest.raises(ValueError):
        load_prepared(store, DOCUMENT_ID, version)


# saving replaces


def test_saving_again_replaces_the_parts(
    store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    fewer = PreparedParts(logical=parts.logical, chunks=parts.chunks[:1])

    save_prepared(store, DOCUMENT_ID, 1, fewer)

    assert load_prepared(store, DOCUMENT_ID, 1) == fewer


def test_saving_twice_is_harmless(store: FileStore, parts: PreparedParts) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    save_prepared(store, DOCUMENT_ID, 1, parts)

    assert load_prepared(store, DOCUMENT_ID, 1) == parts


def test_saving_clears_other_files_in_the_versions_folder(
    root: Path, store: FileStore, parts: PreparedParts
) -> None:
    # e.g. the old review.html: the prepare handler writes a new one after saving
    save_prepared(store, DOCUMENT_ID, 1, parts)
    review = f"{prepared_version_folder(DOCUMENT_ID, 1)}/review.html"
    store.put(review, io.BytesIO(b"<html>"), size=6, media_type="text/html")

    save_prepared(store, DOCUMENT_ID, 1, parts)

    assert not store.exists(review)


def test_a_save_that_stops_halfway_never_mixes_old_and_new(
    store: FileStore, parts: PreparedParts
) -> None:
    # the old chunks must not be loaded next to a new outline
    save_prepared(store, DOCUMENT_ID, 1, parts)
    fewer = PreparedParts(logical=parts.logical, chunks=parts.chunks[:1])

    with pytest.raises(OSError, match="worker stopped"):
        save_prepared(StopsBeforeChunks(store), DOCUMENT_ID, 1, fewer)

    assert load_prepared(store, DOCUMENT_ID, 1) is None


def test_saving_again_repairs_a_save_that_stopped_halfway(
    store: FileStore, parts: PreparedParts
) -> None:
    with pytest.raises(OSError):
        save_prepared(StopsBeforeChunks(store), DOCUMENT_ID, 1, parts)

    save_prepared(store, DOCUMENT_ID, 1, parts)

    assert load_prepared(store, DOCUMENT_ID, 1) == parts


# versions and documents are kept apart


def test_saving_a_version_leaves_the_other_versions_alone(
    store: FileStore, parts: PreparedParts
) -> None:
    fewer = PreparedParts(logical=parts.logical, chunks=parts.chunks[:1])
    save_prepared(store, DOCUMENT_ID, 1, parts)

    save_prepared(store, DOCUMENT_ID, 2, fewer)

    assert load_prepared(store, DOCUMENT_ID, 1) == parts
    assert load_prepared(store, DOCUMENT_ID, 2) == fewer


def test_version_1_doesnt_touch_version_10(
    store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 10, parts)

    save_prepared(store, DOCUMENT_ID, 1, parts)

    assert load_prepared(store, DOCUMENT_ID, 10) == parts


def test_another_documents_parts_are_separate(
    store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)

    assert load_prepared(store, OTHER_DOCUMENT_ID, 1) is None


# deleting


def test_deleting_removes_every_version_of_the_document(
    root: Path, store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    save_prepared(store, DOCUMENT_ID, 2, parts)
    review = f"{prepared_version_folder(DOCUMENT_ID, 2)}/review.html"
    store.put(review, io.BytesIO(b"<html>"), size=6, media_type="text/html")

    assert delete_prepared(store, DOCUMENT_ID) == 5

    assert load_prepared(store, DOCUMENT_ID, 1) is None
    assert load_prepared(store, DOCUMENT_ID, 2) is None
    assert not (root / prepared_folder(DOCUMENT_ID)).exists()


def test_deleting_leaves_other_documents_alone(
    store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    save_prepared(store, OTHER_DOCUMENT_ID, 1, parts)

    delete_prepared(store, DOCUMENT_ID)

    assert load_prepared(store, OTHER_DOCUMENT_ID, 1) == parts


def test_deleting_a_document_with_nothing_prepared_deletes_nothing(
    store: FileStore,
) -> None:
    assert delete_prepared(store, DOCUMENT_ID) == 0


def test_a_deleted_document_can_be_prepared_again(
    store: FileStore, parts: PreparedParts
) -> None:
    save_prepared(store, DOCUMENT_ID, 1, parts)
    delete_prepared(store, DOCUMENT_ID)

    save_prepared(store, DOCUMENT_ID, 1, parts)

    assert load_prepared(store, DOCUMENT_ID, 1) == parts
