"""Tests for integrations/huggingface/reference.py: ModelReference and model_reference.json.

A ModelReference names one exact model: a hub id and a full commit hash.
There is no "latest": a vector space is only meaningful for exact weights, so
every spec pins a commit and upgrading is a deliberate edit. The same
reference is written into each downloaded folder as model_reference.json,
and loading compares the two.
"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from document_processor.integrations.huggingface.reference import (
    MODEL_REFERENCE_FILE,
    ModelReference,
    read_model_reference,
    write_model_reference,
)

COMMIT = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3 = ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=COMMIT)


def write_json(folder: Path, contents: object) -> None:
    (folder / MODEL_REFERENCE_FILE).write_text(json.dumps(contents), encoding="utf-8")


# ============================================================ ModelReference


def test_reference_keeps_its_fields() -> None:
    assert (BGE_M3.hub_model_id, BGE_M3.commit_hash) == ("BAAI/bge-m3", COMMIT)


def test_both_fields_are_required() -> None:
    with pytest.raises(TypeError, match="commit_hash"):
        ModelReference(hub_model_id="BAAI/bge-m3")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="hub_model_id"):
        ModelReference(commit_hash=COMMIT)  # type: ignore[call-arg]


def test_empty_hub_model_id_is_refused() -> None:
    with pytest.raises(ValueError, match="^hub_model_id must not be empty$"):
        ModelReference(hub_model_id="", commit_hash=COMMIT)


@pytest.mark.parametrize(
    "commit_hash",
    [
        "",
        "main",
        "v1.0",
        COMMIT[:7],  # a short hash: unambiguous on the hub today, not forever
        COMMIT + "0",
        COMMIT[:-1],
        COMMIT.upper(),  # the hub reports hashes in lower case; compared exactly
        "g" + COMMIT[1:],
        f" {COMMIT[1:]}",
        f"{COMMIT}\n",
    ],
)
def test_commit_hash_must_be_40_lower_case_hex(commit_hash: str) -> None:
    with pytest.raises(
        ValueError, match="^commit_hash must be 40-character lower-case hex, got "
    ) as raised:
        ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=commit_hash)
    assert repr(commit_hash) in str(raised.value)


def test_reference_is_frozen_and_takes_keywords_only() -> None:
    with pytest.raises(FrozenInstanceError):
        BGE_M3.commit_hash = "0" * 40  # type: ignore[misc]
    with pytest.raises(TypeError):
        ModelReference("BAAI/bge-m3", COMMIT)  # type: ignore[call-arg]


def test_references_compare_by_value() -> None:
    # Loading checks the folder with ==, so value equality is the contract.
    assert ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=COMMIT) == BGE_M3
    assert ModelReference(hub_model_id="BAAI/bge-m3", commit_hash="0" * 40) != BGE_M3
    assert ModelReference(hub_model_id="baai/bge-m3", commit_hash=COMMIT) != BGE_M3


def test_references_can_be_dictionary_keys() -> None:
    assert {BGE_M3: "folder"}[
        ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=COMMIT)
    ] == "folder"


# ============================================================ model_reference.json


def test_file_name() -> None:
    assert MODEL_REFERENCE_FILE == "model_reference.json"


def test_written_reference_reads_back(tmp_path: Path) -> None:
    write_model_reference(tmp_path, BGE_M3)
    assert read_model_reference(tmp_path) == BGE_M3
    assert read_model_reference(str(tmp_path)) == BGE_M3


def test_file_is_indented_json_ending_in_a_newline(tmp_path: Path) -> None:
    write_model_reference(tmp_path, BGE_M3)
    assert (tmp_path / MODEL_REFERENCE_FILE).read_text(encoding="utf-8") == (
        '{\n  "hub_model_id": "BAAI/bge-m3",\n  "commit_hash": "' + COMMIT + '"\n}\n'
    )


def test_writing_replaces_an_existing_file(tmp_path: Path) -> None:
    write_model_reference(
        tmp_path, ModelReference(hub_model_id="org/old", commit_hash="0" * 40)
    )
    write_model_reference(tmp_path, BGE_M3)
    assert read_model_reference(tmp_path) == BGE_M3


def test_extra_keys_are_ignored(tmp_path: Path) -> None:
    # Room for later fields (who downloaded it, when) without breaking older readers.
    write_json(
        tmp_path,
        {
            "hub_model_id": "BAAI/bge-m3",
            "commit_hash": COMMIT,
            "downloaded_at": "2026-10-28",
        },
    )
    assert read_model_reference(tmp_path) == BGE_M3


def test_missing_file_says_how_to_get_one(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="^cannot find model_reference.json in .*download_model.*copy the whole folder",
    ) as raised:
        read_model_reference(tmp_path)
    assert str(tmp_path) in str(raised.value)
    assert raised.value.__suppress_context__  # raised "from None": no second traceback


def test_missing_folder_is_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cannot find model_reference.json"):
        read_model_reference(tmp_path / "nowhere")


def test_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    (tmp_path / MODEL_REFERENCE_FILE).write_text(
        "{hub_model_id: BAAI/bge-m3", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="is not valid JSON") as raised:
        read_model_reference(tmp_path)
    assert str(tmp_path / MODEL_REFERENCE_FILE) in str(raised.value)
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "contents",
    [
        {"hub_model_id": "BAAI/bge-m3"},
        {"commit_hash": COMMIT},
        {},
        {"hub_model_id": "BAAI/bge-m3", "revision": COMMIT},  # the field's old name
        ["BAAI/bge-m3", COMMIT],
        "BAAI/bge-m3",
        None,
    ],
)
def test_file_without_both_fields_is_malformed(
    tmp_path: Path, contents: object
) -> None:
    write_json(tmp_path, contents)
    with pytest.raises(
        ValueError, match="is malformed: needs hub_model_id and commit_hash$"
    ) as raised:
        read_model_reference(tmp_path)
    assert str(tmp_path / MODEL_REFERENCE_FILE) in str(raised.value)


def test_file_with_a_bad_commit_hash_names_the_file(tmp_path: Path) -> None:
    write_json(tmp_path, {"hub_model_id": "BAAI/bge-m3", "commit_hash": "main"})
    with pytest.raises(
        ValueError, match="commit_hash must be 40-character lower-case hex, got 'main'$"
    ) as raised:
        read_model_reference(tmp_path)
    assert str(raised.value).startswith(f"{tmp_path / MODEL_REFERENCE_FILE}: ")


def test_file_with_an_empty_hub_model_id_names_the_file(tmp_path: Path) -> None:
    write_json(tmp_path, {"hub_model_id": "", "commit_hash": COMMIT})
    with pytest.raises(ValueError, match="hub_model_id must not be empty$") as raised:
        read_model_reference(tmp_path)
    assert str(raised.value).startswith(f"{tmp_path / MODEL_REFERENCE_FILE}: ")


def test_a_commit_hash_that_is_not_a_string_is_malformed(tmp_path: Path) -> None:
    # The pattern match raises TypeError on a number, which reads as malformed.
    write_json(tmp_path, {"hub_model_id": "BAAI/bge-m3", "commit_hash": 5617})
    with pytest.raises(
        ValueError, match="is malformed: needs hub_model_id and commit_hash$"
    ):
        read_model_reference(tmp_path)


def test_bad_json_is_not_reported_twice(tmp_path: Path) -> None:
    # JSONDecodeError is a ValueError too; it must get its own message, not "<path>: <path> ...".
    (tmp_path / MODEL_REFERENCE_FILE).write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        read_model_reference(tmp_path)
    assert str(raised.value).count(str(tmp_path / MODEL_REFERENCE_FILE)) == 1
