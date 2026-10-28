from pathlib import Path

from document_processor.integrations.huggingface.reference import (
    ModelReference,
    read_model_reference,
)

MODULES_JSON = "modules.json"


def check_is_folder(folder: Path) -> None:
    """The model folder exists and is a folder"""
    if not folder.exists():
        raise FileNotFoundError(
            f"no model folder at {folder}; download it with download_model() first"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"{folder} is a file; pass the folder that holds the model files"
        )


def check_reference_matches(folder: Path, expected: ModelReference) -> None:
    """The folder holds exactly the expected model, per its model_reference"""
    found = read_model_reference(folder)
    if found != expected:
        raise ValueError(
            f"{folder} holds {found.hub_model_id}@{found.commit_hash}, "
            f"but the spec wants {expected.hub_model_id}@{expected.commit_hash}; "
            "download that commit into another folder"
        )


def check_has_modules_json(folder: Path) -> None:
    """sentence-transformers would otherwise load the folder as mean pooling, silently"""
    file_path = folder / MODULES_JSON
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{folder} has no {MODULES_JSON}, so sentence-transformers would load the model as mean pooling; "
            f"this model can't be used as it is"
        )
