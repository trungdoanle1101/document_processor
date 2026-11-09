import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT = re.compile(r"[0-9a-f]{12}")


class FileStore(Protocol):
    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None: ...
    def open(self, key: str) -> BinaryIO: ...
    def exists(self, key: str) -> bool: ...


class LocalFileStore:
    def __init__(self, root: Path) -> None:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(
                f"file store folder {root} doesn't exist or isn't a folder"
            )
        self._root = root

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        path = self._path(key)
        if path.is_file():
            return
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as temp_file:
                shutil.copyfileobj(data, temp_file)
                temp_file.flush()
                os.fsync(fd)
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def open(self, key: str) -> BinaryIO:
        """Open a stored file for reading; a missing key raises FileNotFoundError."""
        try:
            return self._path(key).open("rb")
        except IsADirectoryError:
            raise FileNotFoundError(f"no stored file at {key!r}") from None

    def exists(self, key: str) -> bool:
        """Check if key exists and is a file."""
        return self._path(key).is_file()

    def _path(self, key: str) -> Path:
        """The path for a key; refuses keys that aren't a file inside the root."""
        key_path = Path(key)

        if key_path.is_absolute():
            raise ValueError("absolute paths aren't allowed")

        resolved_path = (self._root / key_path).resolve()
        if resolved_path == self._root or not resolved_path.is_relative_to(self._root):
            raise ValueError(f"key {key!r} isn't a file path inside the store")

        return resolved_path


def original_key(sha256: str) -> str:
    """Decides where a file is stored, based on its content hash."""
    if not _SHA256.fullmatch(sha256):
        raise ValueError(f"not a sha256 hex digest: {sha256!r}")
    return f"originals/{_by_sha256(sha256)}"


def derived_prefix(file_sha256: str, fingerprint: str) -> str:
    """Decides where a file's prepared parts are kept."""
    if not _SHA256.fullmatch(file_sha256):
        raise ValueError(f"file_sha256 is not a sha256 hex digest: {file_sha256!r}")
    if not _FINGERPRINT.fullmatch(fingerprint):
        raise ValueError(f"fingerprint is not a 12-char hex digest: {fingerprint!r}")
    return f"derived/{_by_sha256(file_sha256)}/{fingerprint}"


def _by_sha256(sha256: str) -> str:
    return f"sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"
