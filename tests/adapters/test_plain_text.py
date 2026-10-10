import pytest

from document_processor.adapters import PlainTextAdapter


def parse(data: bytes):
    adapter = PlainTextAdapter()
    return adapter.parse_bytes(data, "doc.txt")


def test_skips_blank_lines_and_keeps_line_numbers():
    doc = parse(b"Line one\n\n \t\nLine four")

    actual_texts = [b.text for b in doc.blocks]
    expected_texts = ["Line one", "Line four"]

    actual_refs = [b.source_reference for b in doc.blocks]
    expected_refs = ["line:1", "line:4"]

    assert actual_texts == expected_texts
    assert actual_refs == expected_refs


@pytest.mark.parametrize("data", [b"A\nB", b"A\r\nB", b"A\rB"])
def test_line_endings_give_same_blocks(data):
    doc = parse(data)
    actual = [b.text for b in doc.blocks]
    expected = ["A", "B"]
    assert actual == expected


def test_invalid_utf_names_file_and_line():
    with pytest.raises(ValueError, match=r"doc\.txt.*line 2"):
        parse(b"ok\n\xff\xfe")


def test_parse_file_matches_parse_bytes(tmp_path):
    data = "Chương I\nĐiều 1. Phạm vi".encode()
    filename = "rule.txt"
    path = tmp_path / filename
    path.write_bytes(data)
    adapter = PlainTextAdapter()
    assert adapter.parse_file(path) == adapter.parse_bytes(data, filename)


def test_missing_file():
    filename = "file_not_exist.txt"
    adapter = PlainTextAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.parse_file(filename)


def test_wrong_file_extension(tmp_path):
    data = "Chương I\nĐiều 1. Phạm vi".encode()
    filename = "rule.md"
    path = tmp_path / filename
    path.write_bytes(data)
    adapter = PlainTextAdapter()
    with pytest.raises(ValueError, match=r"Unexpected file type"):
        adapter.parse_file(path)
