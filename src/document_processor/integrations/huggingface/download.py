import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from document_processor.integrations.huggingface.reference import (
    ModelReference,
    write_model_reference,
)

IGNORED_FILES = (
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.onnx_data",
    "*.gguf",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
    "*.md",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "*.gif",
    ".gitattributes",
)


def download_model(reference: ModelReference, folder: Path | str) -> None:
    """Download reference's model into folder, skipping IGNORED_FILES. The only function that uses the network"""
    folder = Path(folder)
    if folder.exists():
        if not folder.is_dir():
            raise NotADirectoryError(
                f"{folder} is a file; pass the folder that holds (or will hold) the model files"
            )
        if any(folder.iterdir()):
            raise OSError(f"{folder} is not empty; delete it or choose another folder")

    partial = folder.with_name(f".{folder.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)

    snapshot_download(
        repo_id=reference.hub_model_id,
        revision=reference.commit_hash,
        local_dir=partial,
        ignore_patterns=list(IGNORED_FILES),
    )
    write_model_reference(partial, reference)

    if folder.exists():
        folder.rmdir()
    partial.rename(folder)
