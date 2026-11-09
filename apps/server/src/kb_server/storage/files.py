import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol

_SHA256 = re.compile(r"[0-9a-f]{64}")


class FileStore(Protocol):
    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        """Store data at key. If a file is already there, keep it and ignore data."""
        ...

    def open(self, key: str) -> BinaryIO: ...
    def exists(self, key: str) -> bool: ...
    def delete_folder(self, folder: str) -> int:
        """A folder is a key without the last /name part.
        This method deletes every file whose key starts with folder + '/';
        a missing folder deletes nothing.
        """
        ...


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

    def delete_folder(self, folder: str) -> int:
        """Delete the files in the folder (including subfolders) and return number of files deleted."""
        folder_path = self._path(folder)
        if folder_path.is_file():
            raise ValueError(f"{folder!r} names a file, not a folder")
        if not folder_path.exists():
            return 0
        file_count = sum(1 for item in folder_path.rglob("*") if item.is_file())
        shutil.rmtree(folder_path)
        return file_count

    def _path(self, key: str) -> Path:
        """The path for a key; refuses keys that aren't a path inside the root."""
        key_path = Path(key)

        if key_path.is_absolute():
            raise ValueError("absolute paths aren't allowed")

        resolved_path = (self._root / key_path).resolve()
        if resolved_path == self._root or not resolved_path.is_relative_to(self._root):
            raise ValueError(f"key {key!r} isn't a path inside the store")

        return resolved_path


def original_key(sha256: str) -> str:
    """Decides where a file is stored, based on its content hash."""
    if not _SHA256.fullmatch(sha256):
        raise ValueError(f"not a sha256 hex digest: {sha256!r}")
    return f"originals/{_by_sha256(sha256)}"


def prepared_folder(document_id: uuid.UUID) -> str:
    """Everything prepared for a document: deleted with the document."""
    return f"prepared/{document_id}"


def prepared_version_folder(document_id: uuid.UUID, version: int) -> str:
    """One version's prepared output. Refuses version < 1."""
    if version < 1:
        raise ValueError(f"version must be >= 1, got {version}")
    return f"{prepared_folder(document_id)}/v{version}"


def _by_sha256(sha256: str) -> str:
    return f"sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"
