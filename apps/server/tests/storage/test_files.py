"""The file store: where files go (originals and prepared output), and LocalFileStore.

The store tests use the `store` fixture, typed as the FileStore protocol, so they can run
against S3FileStore too once it exists (by parametrising the fixture).
"""

import hashlib
import io
import uuid
from pathlib import Path
from typing import Any

import pytest

from kb_server.storage.files import (
    FileStore,
    LocalFileStore,
    original_key,
    prepared_folder,
    prepared_version_folder,
)

CONTENT = "Điều 1. Phạm vi điều chỉnh\n".encode()


def key_for(data: bytes) -> str:
    return original_key(hashlib.sha256(data).hexdigest())


def put_bytes(store: FileStore, key: str, data: bytes) -> None:
    store.put(key, io.BytesIO(data), size=len(data), media_type="text/plain")


def read(store: FileStore, key: str) -> bytes:
    with store.open(key) as stored:
        return stored.read()


class BrokenStream(io.RawIOBase):
    """A stream that fails partway, like a client disconnecting mid-upload."""

    def __init__(self) -> None:
        self._sent = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        if not self._sent:
            self._sent = True
            buffer[:5] = b"start"
            return 5
        raise ConnectionError("client went away")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture
def store(root: Path) -> FileStore:
    return LocalFileStore(root)


# original_key


def test_original_key_spreads_files_by_the_first_two_byte_pairs() -> None:
    digest = hashlib.sha256(b"x").hexdigest()

    assert (
        original_key(digest) == f"originals/sha256/{digest[:2]}/{digest[2:4]}/{digest}"
    )


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "abc",
        "A" * 64,  # uppercase would give a second key for the same content
        "g" * 64,
        "a" * 63,
        "a" * 65,
        "a" * 64 + "\n",
        "../" + "a" * 61,
    ],
    ids=[
        "empty",
        "short",
        "uppercase",
        "not hex",
        "63 chars",
        "65 chars",
        "trailing newline",
        "path",
    ],
)
def test_original_key_refuses_anything_but_a_sha256_hex_digest(digest: str) -> None:
    with pytest.raises(ValueError):
        original_key(digest)


# storing and reading


def test_a_stored_file_reads_back_the_same(store: FileStore) -> None:
    key = key_for(CONTENT)

    put_bytes(store, key, CONTENT)

    assert read(store, key) == CONTENT


def test_a_large_file_is_stored_whole(store: FileStore) -> None:
    data = bytes(range(256)) * 40_000  # about 10 MB, more than one copy chunk
    key = key_for(data)

    put_bytes(store, key, data)

    assert read(store, key) == data


def test_an_empty_file_can_be_stored(store: FileStore) -> None:
    key = key_for(b"")

    put_bytes(store, key, b"")

    assert store.exists(key)
    assert read(store, key) == b""


def test_putting_the_same_key_again_keeps_the_first_content(store: FileStore) -> None:
    key = key_for(CONTENT)
    put_bytes(store, key, CONTENT)

    put_bytes(store, key, b"something else")

    assert read(store, key) == CONTENT


def test_put_copies_from_where_the_stream_is(store: FileStore) -> None:
    # the upload code hashes the stream first: it must rewind before putting
    key = key_for(CONTENT)
    data = io.BytesIO(CONTENT)
    hashlib.sha256(data.read())

    data.seek(0)
    store.put(key, data, size=len(CONTENT), media_type="text/plain")

    assert read(store, key) == CONTENT


def test_files_land_under_the_root_at_their_key(root: Path, store: FileStore) -> None:
    key = key_for(CONTENT)

    put_bytes(store, key, CONTENT)

    assert (root / key).read_bytes() == CONTENT


# exists and open


def test_exists_before_and_after_put(store: FileStore) -> None:
    key = key_for(CONTENT)
    assert not store.exists(key)

    put_bytes(store, key, CONTENT)

    assert store.exists(key)


def test_a_folder_is_not_a_stored_file(store: FileStore) -> None:
    key = key_for(CONTENT)
    put_bytes(store, key, CONTENT)
    folder = key.rsplit("/", 1)[0]

    assert not store.exists(folder)
    with pytest.raises(FileNotFoundError):
        store.open(folder)


def test_opening_a_missing_key_raises_file_not_found(store: FileStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.open(key_for(b"never stored"))


# keys that must not reach the disk


@pytest.mark.parametrize(
    "key",
    [
        "",
        ".",
        "..",
        "../outside",
        "a/../../outside",
        "/etc/passwd",
        "originals/../../x",
    ],
)
def test_keys_outside_the_root_are_refused(
    root: Path, store: FileStore, key: str
) -> None:
    with pytest.raises(ValueError):
        put_bytes(store, key, b"x")
    with pytest.raises(ValueError):
        store.exists(key)
    with pytest.raises(ValueError):
        store.open(key)
    with pytest.raises(ValueError):
        store.delete_folder(key)

    # nothing was written next to the store, and the store itself is still there
    assert sorted(path.name for path in root.parent.iterdir()) == ["store"]


def test_a_key_that_stays_inside_after_dots_is_allowed(store: FileStore) -> None:
    put_bytes(store, "a/../b", b"x")

    assert read(store, "b") == b"x"


# failures


def test_a_failed_put_leaves_nothing_behind(root: Path, store: FileStore) -> None:
    key = key_for(CONTENT)

    with pytest.raises(ConnectionError):
        store.put(
            key, io.BufferedReader(BrokenStream()), size=100, media_type="text/plain"
        )

    assert not store.exists(key)
    assert [path for path in root.rglob("*") if path.is_file()] == []


def test_a_successful_put_leaves_no_temporary_files(
    root: Path, store: FileStore
) -> None:
    key = key_for(CONTENT)

    put_bytes(store, key, CONTENT)

    assert [path for path in root.rglob("*") if path.is_file()] == [root / key]


# the root folder


def test_the_root_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalFileStore(tmp_path / "missing")

    assert not (tmp_path / "missing").exists()  # and isn't created


def test_the_root_must_be_a_folder(tmp_path: Path) -> None:
    a_file = tmp_path / "a-file"
    a_file.write_bytes(b"")

    with pytest.raises(ValueError):
        LocalFileStore(a_file)


def test_a_relative_root_is_resolved(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(root.parent)
    store = LocalFileStore(Path("store"))
    key = key_for(CONTENT)

    put_bytes(store, key, CONTENT)
    monkeypatch.chdir("/")  # the store keeps working from anywhere

    assert read(store, key) == CONTENT


# prepared folders

DOCUMENT_ID = uuid.UUID("0b5e4c1a-6f1d-4a43-9c55-3d2b8e0f7a19")


def test_a_documents_prepared_folder_is_named_by_its_id() -> None:
    assert prepared_folder(DOCUMENT_ID) == f"prepared/{DOCUMENT_ID}"


def test_a_versions_folder_sits_inside_its_documents_folder() -> None:
    assert (
        prepared_version_folder(DOCUMENT_ID, 3) == f"{prepared_folder(DOCUMENT_ID)}/v3"
    )


@pytest.mark.parametrize("version", [0, -1])
def test_a_version_below_1_is_refused(version: int) -> None:
    with pytest.raises(ValueError):
        prepared_version_folder(DOCUMENT_ID, version)


def test_prepared_folders_are_valid_store_folders(store: FileStore) -> None:
    key = f"{prepared_version_folder(DOCUMENT_ID, 1)}/logical.json"

    put_bytes(store, key, b"{}")

    assert read(store, key) == b"{}"


# delete_folder


def test_delete_folder_deletes_its_files_including_subfolders(
    root: Path, store: FileStore
) -> None:
    for key in ["a/one", "a/b/two", "a/b/c/three"]:
        put_bytes(store, key, b"x")

    assert store.delete_folder("a") == 3

    for key in ["a/one", "a/b/two", "a/b/c/three"]:
        assert not store.exists(key)
    assert not (root / "a").exists()


def test_delete_folder_leaves_other_folders_alone(store: FileStore) -> None:
    put_bytes(store, "a/one", b"x")
    put_bytes(store, "b/one", b"y")
    put_bytes(store, "parent/a/one", b"z")

    store.delete_folder("a")

    assert read(store, "b/one") == b"y"
    assert read(store, "parent/a/one") == b"z"


def test_a_folder_doesnt_match_a_longer_name(store: FileStore) -> None:
    # "v1" must not take "v10" with it: a folder is matched as folder + "/"
    put_bytes(store, "doc/v1/logical.json", b"1")
    put_bytes(store, "doc/v10/logical.json", b"10")

    assert store.delete_folder("doc/v1") == 1

    assert read(store, "doc/v10/logical.json") == b"10"


def test_deleting_a_folder_inside_a_folder_keeps_the_rest(store: FileStore) -> None:
    put_bytes(store, "doc/v1/logical.json", b"1")
    put_bytes(store, "doc/v2/logical.json", b"2")

    store.delete_folder("doc/v1")

    assert not store.exists("doc/v1/logical.json")
    assert read(store, "doc/v2/logical.json") == b"2"


def test_deleting_a_missing_folder_deletes_nothing(store: FileStore) -> None:
    assert store.delete_folder("never/made") == 0


def test_deleting_twice_is_harmless(store: FileStore) -> None:
    put_bytes(store, "a/one", b"x")

    assert store.delete_folder("a") == 1
    assert store.delete_folder("a") == 0


def test_an_empty_folder_counts_as_nothing(root: Path, store: FileStore) -> None:
    put_bytes(store, "doc/v1/logical.json", b"1")
    store.delete_folder("doc/v1")  # leaves an empty doc/ behind on disk

    assert store.delete_folder("doc") == 0
    assert not (root / "doc").exists()


def test_a_file_is_not_a_folder(store: FileStore) -> None:
    put_bytes(store, "a/one", b"x")

    with pytest.raises(ValueError):
        store.delete_folder("a/one")

    assert read(store, "a/one") == b"x"  # and it wasn't deleted


def test_files_can_be_stored_again_after_their_folder_is_deleted(
    store: FileStore,
) -> None:
    put_bytes(store, "a/one", b"old")
    store.delete_folder("a")

    put_bytes(store, "a/one", b"new")

    assert read(store, "a/one") == b"new"


def test_deleting_a_folder_spares_originals(store: FileStore) -> None:
    original = key_for(CONTENT)
    put_bytes(store, original, CONTENT)
    put_bytes(store, f"{prepared_folder(DOCUMENT_ID)}/v1/logical.json", b"{}")

    assert store.delete_folder(prepared_folder(DOCUMENT_ID)) == 1

    assert read(store, original) == CONTENT
