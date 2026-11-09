"""The cache of prepared documents: settings fingerprints, and saving and loading parts."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.preparation import Preparer
from kb_server.indexing.derived import (
    PreparationSettings,
    PreparedParts,
    load_prepared,
    save_prepared,
)
from kb_server.storage.files import FileStore, LocalFileStore, derived_prefix

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
    profile="vn_regulation@4",
    chunker={"kind": "outline", "max_tokens": 512, "table_layout": "grid"},
    embedder="huggingface:BAAI/bge-m3",
)


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    root = tmp_path / "store"
    root.mkdir()
    return LocalFileStore(root)


@pytest.fixture
def prefix() -> str:
    return derived_prefix(
        hashlib.sha256(REGULATION.encode()).hexdigest(), SETTINGS.fingerprint()
    )


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


def with_changes(**changes: Any) -> PreparationSettings:
    values: dict[str, Any] = {
        "profile": SETTINGS.profile,
        "chunker": dict(SETTINGS.chunker),
        "embedder": SETTINGS.embedder,
    }
    values.update(changes)
    return PreparationSettings(**values)


# the settings fingerprint


def test_a_fingerprint_is_12_lowercase_hex_characters() -> None:
    fingerprint = SETTINGS.fingerprint()

    assert len(fingerprint) == 12
    assert set(fingerprint) <= set("0123456789abcdef")


def test_the_same_settings_give_the_same_fingerprint() -> None:
    assert with_changes().fingerprint() == SETTINGS.fingerprint()


def test_the_order_of_the_chunker_options_doesnt_matter() -> None:
    reordered = with_changes(
        chunker={"table_layout": "grid", "max_tokens": 512, "kind": "outline"}
    )

    assert reordered.fingerprint() == SETTINGS.fingerprint()


@pytest.mark.parametrize(
    "changes",
    [
        {"profile": "vn_regulation@5"},
        {"profile": "plain@1"},
        {"chunker": {"kind": "outline", "max_tokens": 256, "table_layout": "grid"}},
        {
            "chunker": {
                "kind": "outline",
                "max_tokens": 512,
                "table_layout": "key_value",
            }
        },
        {"chunker": {"kind": "outline", "max_tokens": 512}},
        {"embedder": "huggingface:BAAI/bge-m3@another-revision"},
    ],
    ids=[
        "profile version",
        "profile",
        "max tokens",
        "an option",
        "an option removed",
        "embedder",
    ],
)
def test_any_change_gives_a_new_fingerprint(changes: dict[str, Any]) -> None:
    assert with_changes(**changes).fingerprint() != SETTINGS.fingerprint()


def test_describe_gives_the_settings_in_full() -> None:
    assert SETTINGS.describe() == {
        "profile": "vn_regulation@4",
        "chunker": {"kind": "outline", "max_tokens": 512, "table_layout": "grid"},
        "embedder": "huggingface:BAAI/bge-m3",
    }
    json.dumps(SETTINGS.describe())  # plain JSON, ready for provenance


def test_describe_is_a_copy() -> None:
    described = SETTINGS.describe()
    described["chunker"]["max_tokens"] = 1

    assert SETTINGS.chunker["max_tokens"] == 512


def test_the_fingerprint_fits_derived_prefix() -> None:
    digest = hashlib.sha256(b"x").hexdigest()

    assert derived_prefix(digest, SETTINGS.fingerprint()).endswith(
        SETTINGS.fingerprint()
    )


# saving and loading


def test_saved_parts_load_back_the_same(
    store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    save_prepared(store, prefix, parts)

    loaded = load_prepared(store, prefix)

    assert loaded == parts
    assert loaded is not None
    assert len(loaded.chunks) == len(parts.chunks) > 1


def test_vietnamese_text_survives_unchanged(
    store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    save_prepared(store, prefix, parts)

    loaded = load_prepared(store, prefix)

    assert loaded is not None
    text = "".join(chunk.text for chunk in loaded.chunks)
    assert "Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày" in text


def test_the_files_are_readable_json(
    tmp_path: Path, store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    save_prepared(store, prefix, parts)
    folder = tmp_path / "store" / prefix

    assert sorted(path.name for path in folder.iterdir()) == [
        "chunks.jsonl",
        "logical.json",
    ]
    json.loads((folder / "logical.json").read_text(encoding="utf-8"))
    lines = (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(parts.chunks)
    assert all(json.loads(line) for line in lines)
    assert "Điều 1" in (folder / "logical.json").read_text(
        encoding="utf-8"
    )  # not \u escapes


def test_nothing_saved_loads_as_none(store: FileStore, prefix: str) -> None:
    assert load_prepared(store, prefix) is None


def test_an_outline_without_its_chunks_loads_as_none(
    tmp_path: Path, store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    # a worker that stopped between the two saves
    save_prepared(store, prefix, parts)
    (tmp_path / "store" / prefix / "chunks.jsonl").unlink()

    assert load_prepared(store, prefix) is None


def test_saving_again_completes_a_half_saved_pair(
    tmp_path: Path, store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    save_prepared(store, prefix, parts)
    (tmp_path / "store" / prefix / "chunks.jsonl").unlink()

    save_prepared(store, prefix, parts)

    assert load_prepared(store, prefix) == parts


def test_saving_twice_is_harmless(
    store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    save_prepared(store, prefix, parts)
    save_prepared(store, prefix, parts)

    assert load_prepared(store, prefix) == parts


def test_a_document_with_no_chunks_round_trips(
    store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    empty = PreparedParts(logical=parts.logical, chunks=())

    save_prepared(store, prefix, empty)

    assert load_prepared(store, prefix) == empty


def test_a_damaged_cache_file_raises(
    tmp_path: Path, store: FileStore, prefix: str, parts: PreparedParts
) -> None:
    # a bug or a damaged store: it must surface, not be quietly prepared again forever
    save_prepared(store, prefix, parts)
    (tmp_path / "store" / prefix / "logical.json").write_text(
        "{not json", encoding="utf-8"
    )

    with pytest.raises(json.JSONDecodeError):
        load_prepared(store, prefix)


def test_different_settings_keep_separate_caches(
    store: FileStore, parts: PreparedParts
) -> None:
    digest = hashlib.sha256(REGULATION.encode()).hexdigest()
    first = derived_prefix(digest, SETTINGS.fingerprint())
    second = derived_prefix(
        digest, with_changes(profile="vn_regulation@5").fingerprint()
    )

    save_prepared(store, first, parts)

    assert load_prepared(store, second) is None
