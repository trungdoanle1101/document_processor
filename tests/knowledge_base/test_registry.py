"""SqliteDocumentRegistry: current versions, kept history, and records that survive a round trip."""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from knowledge_base.registry import (
    DocumentRecord,
    Provenance,
    SqliteDocumentRegistry,
    provenance_differences,
    record_from_json,
    record_to_json,
)

PROVENANCE = Provenance(
    source_hash="sha256:abc",
    profile="vn_regulation@1",
    chunker="outline/512",
    embedder="stub",
    collection="kb",
)


def record(
    document_id: str = "a", *, version: int = 1, access_group: str = "it"
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        title=f"Title {document_id}",
        source=f"/docs/{document_id}.docx",
        access_group=access_group,
        attributes={
            "kind": "quy_dinh",
            "year": 2025,
            "issued_on": date(2025, 3, 1),
            "tags": ("a", "b"),
        },
        version=version,
        provenance=PROVENANCE,
        chunk_count=12,
        indexed_at=datetime(2026, 11, 3, 9, 0, tzinfo=UTC),
    )


def test_a_record_survives_json() -> None:
    original = record()
    assert record_from_json(record_to_json(original)) == original


def test_a_newer_schema_is_refused() -> None:
    text = record_to_json(record()).replace(
        '"schema_version": 1', '"schema_version": 99'
    )
    with pytest.raises(ValueError, match="schema"):
        record_from_json(text)


def test_saving_versions_keeps_the_latest_current_and_all_in_history() -> None:
    registry = SqliteDocumentRegistry(":memory:")
    registry.save(record(version=1))
    registry.save(replace(record(version=2), chunk_count=15))

    current = registry.get("a")
    assert current is not None and (current.version, current.chunk_count) == (2, 15)
    assert [r.version for r in registry.history("a")] == [1, 2]


@pytest.mark.parametrize("version", [1, 3])
def test_a_version_out_of_step_is_refused(version: int) -> None:
    registry = SqliteDocumentRegistry(":memory:")
    registry.save(record(version=1))
    with pytest.raises(ValueError, match="version"):
        registry.save(record(version=version))


def test_listing_filters_by_access_group() -> None:
    registry = SqliteDocumentRegistry(":memory:")
    registry.save(record("b", access_group="hr"))
    registry.save(record("a", access_group="it"))
    registry.save(record("c", access_group="public"))

    assert [r.document_id for r in registry.list_documents()] == ["a", "b", "c"]
    assert [
        r.document_id for r in registry.list_documents(access_groups=["it", "public"])
    ] == ["a", "c"]
    assert registry.list_documents(access_groups=[]) == []


def test_deleting_forgets_the_document_but_not_its_history() -> None:
    registry = SqliteDocumentRegistry(":memory:")
    registry.save(record())
    registry.delete("a")
    assert registry.get("a") is None
    assert registry.list_documents() == []
    assert len(registry.history("a")) == 1


def test_records_persist_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "kb.sqlite3"
    first = SqliteDocumentRegistry(path)
    first.save(record())
    first.close()
    assert SqliteDocumentRegistry(path).get("a") == record()


def test_provenance_differences_name_the_fields() -> None:
    newer = replace(PROVENANCE, embedder="bge-m3", chunker="outline/256")
    assert provenance_differences(PROVENANCE, newer) == ["chunker", "embedder"]
    assert provenance_differences(PROVENANCE, PROVENANCE) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"document_id": " "}, "document_id"),
        ({"access_group": ""}, "access_group"),
        ({"version": 0}, "version"),
    ],
)
def test_bad_records_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(record(), **changes)  # type: ignore[arg-type]
