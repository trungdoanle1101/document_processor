"""Tests for pipeline/extraction.py and the ADAPTERS registry: picking the adapter by file extension.

`extract_physical_document(path)` looks the lowercased suffix up in ADAPTERS
and lets that adapter read the file. Keys are lowercase suffixes with the dot,
so ".DOCX" works and a mistyped key fails a test instead of never matching.
"""

from pathlib import Path

import pytest

from document_processor.adapters import ADAPTERS, Adapter, DocxAdapter, PlainTextAdapter
from document_processor.pipeline import extract_physical_document
from document_processor.core.models.physical import DocSourceFormat
from tests.adapters.docx.test_adapter import build_docx, p


def write_docx(folder: Path, name: str, *paragraphs: str) -> Path:
    path = folder / name
    path.write_bytes(build_docx("".join(p(text) for text in paragraphs)))
    return path


def write_txt(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


# ============================================================ the registry


def test_registry_knows_docx_and_txt() -> None:
    assert ADAPTERS.names() == (".docx", ".txt")
    assert isinstance(ADAPTERS.get(".docx"), DocxAdapter)
    assert isinstance(ADAPTERS.get(".txt"), PlainTextAdapter)


def test_keys_are_lowercase_suffixes_with_the_dot() -> None:
    for key in ADAPTERS.names():
        assert key.startswith(".") and key == key.lower() and " " not in key, key


def test_every_adapter_names_itself() -> None:
    for key in ADAPTERS.names():
        adapter: Adapter = ADAPTERS.get(key)
        assert adapter.name and adapter.version, key


# ============================================================ extraction


def test_docx_is_read_by_the_docx_adapter(tmp_path: Path) -> None:
    path = write_docx(tmp_path, "quy_dinh.docx", "Điều 1. Phạm vi điều chỉnh", "Nội dung")
    document = extract_physical_document(path)
    assert document.source_format is DocSourceFormat.DOCX
    assert (document.adapter_name, document.source_filename) == ("docx", "quy_dinh.docx")
    assert [b.text for b in document.blocks] == ["Điều 1. Phạm vi điều chỉnh", "Nội dung"]


def test_txt_is_read_by_the_plain_text_adapter(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "quy_dinh.txt", "Điều 1. Phạm vi điều chỉnh\n\nNội dung\n")
    document = extract_physical_document(path)
    assert document.source_format is DocSourceFormat.TXT
    assert document.adapter_name == "plain_text"
    assert [b.text for b in document.blocks] == ["Điều 1. Phạm vi điều chỉnh", "Nội dung"]


def test_same_result_as_calling_the_adapter_directly(tmp_path: Path) -> None:
    path = write_docx(tmp_path, "a.docx", "Nội dung")
    assert extract_physical_document(path) == DocxAdapter().parse_file(path)


def test_accepts_a_string_path(tmp_path: Path) -> None:
    path = write_txt(tmp_path, "a.txt", "Nội dung")
    assert extract_physical_document(str(path)).blocks[0].text == "Nội dung"


@pytest.mark.parametrize("name", ["QUY_DINH.DOCX", "quy_dinh.Docx"])
def test_extension_case_does_not_matter(tmp_path: Path, name: str) -> None:
    document = extract_physical_document(write_docx(tmp_path, name, "Nội dung"))
    assert document.source_format is DocSourceFormat.DOCX


def test_uppercase_txt(tmp_path: Path) -> None:
    document = extract_physical_document(write_txt(tmp_path, "GHI_CHU.TXT", "Nội dung"))
    assert document.source_format is DocSourceFormat.TXT


# ============================================================ refusals


@pytest.mark.parametrize("name", ["quy_dinh.pdf", "quy_dinh.doc", "quy_dinh"])
def test_unknown_extension_names_what_is_supported(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")
    with pytest.raises(KeyError, match=r"unknown adapter .*; known: \.docx, \.txt"):
        extract_physical_document(path)


def test_unknown_extension_is_refused_before_reading(tmp_path: Path) -> None:
    # The file does not even exist: the extension alone decides.
    with pytest.raises(KeyError, match="'.pdf'"):
        extract_physical_document(tmp_path / "missing.pdf")


@pytest.mark.parametrize("name", ["missing.docx", "missing.txt"])
def test_missing_file(tmp_path: Path, name: str) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        extract_physical_document(tmp_path / name)


def test_a_folder_named_like_a_document_is_not_a_file(tmp_path: Path) -> None:
    folder = tmp_path / "bundle.docx"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        extract_physical_document(folder)


def test_a_broken_docx_is_reported_by_its_adapter(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip file")
    with pytest.raises(ValueError, match="broken.docx"):
        extract_physical_document(path)


def test_text_that_is_not_utf8_is_reported_by_its_adapter(tmp_path: Path) -> None:
    path = tmp_path / "cp1258.txt"
    path.write_bytes("Đơn".encode("cp1258"))  # an older Vietnamese code page, not UTF-8
    with pytest.raises(ValueError, match="cp1258.txt.*line 1"):
        extract_physical_document(path)