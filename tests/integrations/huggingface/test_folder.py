"""Tests for integrations/huggingface/_folder.py: the checks a model folder must pass before loading.

Three small checks, each raising one clear error:

- check_is_folder: the path exists and is a folder;
- check_reference_matches: its model_reference.json names exactly the model
  the spec wants (same hub id, same commit);
- check_has_modules_json: sentence-transformers would otherwise load the
  folder as plain mean pooling without normalization, with no visible
  warning, which is wrong for most models and fails only as worse search.

The embedder runs all three, in that order, so the most basic problem is
reported first. None of them imports transformers or torch.
"""

from pathlib import Path

import pytest

from document_processor.integrations.huggingface._folder import (
    MODULES_JSON,
    check_has_modules_json,
    check_is_folder,
    check_reference_matches,
)
from document_processor.integrations.huggingface.reference import ModelReference, write_model_reference

COMMIT = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3 = ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=COMMIT)


def model_folder(tmp_path: Path, reference: ModelReference | None = BGE_M3, *, modules_json: bool = True) -> Path:
    folder = tmp_path / "bge-m3"
    folder.mkdir()
    if reference is not None:
        write_model_reference(folder, reference)
    if modules_json:
        (folder / MODULES_JSON).write_text("[]", encoding="utf-8")
    return folder


def check_all(folder: Path, expected: ModelReference) -> None:
    """The embedder's order."""
    check_is_folder(folder)
    check_reference_matches(folder, expected)
    check_has_modules_json(folder)


# ============================================================ check_is_folder


def test_a_folder_passes(tmp_path: Path) -> None:
    check_is_folder(tmp_path)


def test_an_empty_folder_passes_this_check(tmp_path: Path) -> None:
    # Whether the folder holds a model is the next check's job.
    folder = tmp_path / "empty"
    folder.mkdir()
    check_is_folder(folder)


def test_a_missing_folder_says_to_download(tmp_path: Path) -> None:
    folder = tmp_path / "bge-m3"
    with pytest.raises(FileNotFoundError, match="^no model folder at .*; download it with download_model\\(\\) first$") as raised:
        check_is_folder(folder)
    assert str(folder) in str(raised.value)


def test_a_file_is_not_a_folder(tmp_path: Path) -> None:
    path = tmp_path / "bge-m3"
    path.write_text("not a folder", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="is a file; pass the folder that holds the model files$"):
        check_is_folder(path)


def test_a_model_file_inside_the_folder_is_not_a_folder(tmp_path: Path) -> None:
    folder = model_folder(tmp_path)
    with pytest.raises(NotADirectoryError):
        check_is_folder(folder / MODULES_JSON)


# ============================================================ check_reference_matches


def test_the_expected_model_passes(tmp_path: Path) -> None:
    check_reference_matches(model_folder(tmp_path), BGE_M3)


def test_another_commit_is_refused_naming_both(tmp_path: Path) -> None:
    folder = model_folder(tmp_path, ModelReference(hub_model_id="BAAI/bge-m3", commit_hash="0" * 40))
    with pytest.raises(ValueError) as raised:
        check_reference_matches(folder, BGE_M3)
    message = str(raised.value)
    assert message == (
        f"{folder} holds BAAI/bge-m3@{'0' * 40}, but the spec wants BAAI/bge-m3@{COMMIT}; "
        "download that commit into another folder"
    )


def test_another_model_is_refused(tmp_path: Path) -> None:
    other = ModelReference(hub_model_id="intfloat/multilingual-e5-large", commit_hash=COMMIT)
    folder = model_folder(tmp_path, other)
    with pytest.raises(ValueError, match="holds intfloat/multilingual-e5-large@.*but the spec wants BAAI/bge-m3@"):
        check_reference_matches(folder, BGE_M3)


def test_hub_ids_are_compared_exactly(tmp_path: Path) -> None:
    # The hub treats ids case-insensitively, but a folder records one spelling;
    # matching exactly keeps the check simple and the identity stable.
    folder = model_folder(tmp_path, ModelReference(hub_model_id="baai/bge-m3", commit_hash=COMMIT))
    with pytest.raises(ValueError, match="holds baai/bge-m3@"):
        check_reference_matches(folder, BGE_M3)


def test_a_folder_without_a_reference_says_how_to_get_one(tmp_path: Path) -> None:
    # The message is read_model_reference's own, not repeated here.
    folder = model_folder(tmp_path, reference=None)
    with pytest.raises(FileNotFoundError, match="cannot find model_reference.json in .*download_model"):
        check_reference_matches(folder, BGE_M3)


def test_a_broken_reference_file_is_reported_by_the_reader(tmp_path: Path) -> None:
    folder = model_folder(tmp_path, reference=None)
    (folder / "model_reference.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        check_reference_matches(folder, BGE_M3)


# ============================================================ check_has_modules_json


def test_a_folder_with_modules_json_passes(tmp_path: Path) -> None:
    check_has_modules_json(model_folder(tmp_path))


def test_a_folder_without_modules_json_is_refused(tmp_path: Path) -> None:
    folder = model_folder(tmp_path, modules_json=False)
    with pytest.raises(FileNotFoundError) as raised:
        check_has_modules_json(folder)
    assert str(raised.value) == (
        f"{folder} has no modules.json, so sentence-transformers would load the model as mean pooling; "
        "this model can't be used as it is"
    )


def test_modules_json_must_be_a_file(tmp_path: Path) -> None:
    folder = model_folder(tmp_path, modules_json=False)
    (folder / MODULES_JSON).mkdir()
    with pytest.raises(FileNotFoundError, match="has no modules.json"):
        check_has_modules_json(folder)


def test_modules_json_in_a_subfolder_does_not_count(tmp_path: Path) -> None:
    folder = model_folder(tmp_path, modules_json=False)
    (folder / "onnx").mkdir()
    (folder / "onnx" / MODULES_JSON).write_text("[]", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="has no modules.json"):
        check_has_modules_json(folder)


def test_only_the_file_is_looked_at_not_its_contents(tmp_path: Path) -> None:
    # sentence-transformers reads the contents; a broken file fails there, loudly.
    folder = model_folder(tmp_path, modules_json=False)
    (folder / MODULES_JSON).write_text("not json", encoding="utf-8")
    check_has_modules_json(folder)


# ============================================================ the order


def test_a_complete_folder_passes_all_three(tmp_path: Path) -> None:
    check_all(model_folder(tmp_path), BGE_M3)


def test_a_missing_folder_is_reported_before_anything_inside_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="^no model folder at"):
        check_all(tmp_path / "nowhere", BGE_M3)


def test_a_wrong_model_is_reported_before_a_missing_modules_json(tmp_path: Path) -> None:
    other = ModelReference(hub_model_id="BAAI/bge-m3", commit_hash="0" * 40)
    folder = model_folder(tmp_path, other, modules_json=False)
    with pytest.raises(ValueError, match="but the spec wants"):
        check_all(folder, BGE_M3)


def test_an_empty_folder_is_reported_as_missing_its_reference(tmp_path: Path) -> None:
    folder = tmp_path / "bge-m3"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="cannot find model_reference.json"):
        check_all(folder, BGE_M3)