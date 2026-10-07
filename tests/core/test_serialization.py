from document_processor.core.serialization import physical_document_to_dict, physical_document_from_dict
from document_processor.adapters import PlainTextAdapter


def test_physical_document_to_dict():
    adapter = PlainTextAdapter()
    doc = adapter.parse_bytes(b"Line 1\n\n\tLine 3   ", filename="doc.txt")
    d = physical_document_to_dict(doc)
    print(d)


def test_round_trip_to_dict_then_from_dict():
    adapter = PlainTextAdapter()
    doc = adapter.parse_bytes(b"Line 1\n\n\tLine 3   ", filename="doc.txt")
    d = physical_document_to_dict(doc)
    new_doc = physical_document_from_dict(d)
    assert doc == new_doc