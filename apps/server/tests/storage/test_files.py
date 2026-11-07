"""The file store: content-addressed keys, and LocalFileStore.

The store tests use the `store` fixture, typed as the FileStore protocol, so they can run
against S3FileStore too once it exists (by parametrising the fixture).
"""

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from kb_server.storage.files import FileStore, LocalFileStore, original_key

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

    # nothing was written next to the store
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
