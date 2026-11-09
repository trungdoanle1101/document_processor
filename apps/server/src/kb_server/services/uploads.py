import codecs
import hashlib
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile, is_zipfile

from kb_server.domain.errors import FileTooLarge, InvalidRequest, UnsupportedFileType
from kb_server.domain.files import DOCX, TEXT

_MB = 1024 * 1024
_CHUNK_SIZE = 1 * _MB


@dataclass(frozen=True, kw_only=True)
class StagedFile:
    path: Path
    sha256: str
    size: int
    media_type: str
    extension: str


@contextmanager
def stage_upload(
    source: BinaryIO, *, max_bytes: int
) -> Generator[StagedFile, None, None]:
    """Copy an upload to a temp file, hashing it, refusing it if too big or an unknown type."""
    fd, name = tempfile.mkstemp()
    path = Path(name)

    try:
        sha256, size = _copy(source, fd, max_bytes)
        media_type, extension = _detect(path)
        yield StagedFile(
            path=path,
            sha256=sha256,
            size=size,
            media_type=media_type,
            extension=extension,
        )
    finally:
        path.unlink(missing_ok=True)


def _copy(source: BinaryIO, fd: int, max_bytes: int) -> tuple[str, int]:
    sha256 = hashlib.sha256()
    size = 0
    with os.fdopen(fd, mode="wb") as temp:
        while chunk := source.read(_CHUNK_SIZE):
            size += len(chunk)
            if size > max_bytes:
                raise FileTooLarge(
                    f"the file is larger than the allowed limit of {max_bytes // _MB} MB"
                )
            sha256.update(chunk)
            temp.write(chunk)
        if size == 0:
            raise InvalidRequest("the file is empty")
    return sha256.hexdigest(), size


def _detect(path: Path) -> tuple[str, str]:
    if _is_docx(path):
        return DOCX, ".docx"
    if _is_text(path):
        return TEXT, ".txt"
    raise UnsupportedFileType("only .docx and .txt files are supported")


def _is_docx(path: Path) -> bool:
    if not is_zipfile(path):
        return False
    try:
        with ZipFile(path) as archive:
            return "word/document.xml" in archive.namelist()
    except BadZipFile:
        return False


def _is_text(path: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open(mode="rb") as f:
            while chunk := f.read(_CHUNK_SIZE):
                if b"\0" in chunk:
                    return False
                decoder.decode(chunk)
            # a file can't end halfway through a character
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True
