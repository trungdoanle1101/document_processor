from document_processor.core.identity import compute_hash

def test_hash_empty_data():
    hash_empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert compute_hash(b"") == f"sha256:{hash_empty}"