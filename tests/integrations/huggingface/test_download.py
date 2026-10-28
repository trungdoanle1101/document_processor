"""Tests for integrations/huggingface/download.py: download_model.

download_model is the only function that uses the network, so here
snapshot_download is replaced by a FakeHub that writes the repo's files,
filtered by huggingface_hub's own pattern matcher, so IGNORED_FILES is tested
as the hub applies it. It can also stop halfway, like a dropped connection.

The reference already names an exact commit, so nothing is resolved: that
commit is downloaded and the reference is written into the folder as
model_reference.json. The download goes to a hidden sibling folder
(.<name>.partial) and is renamed into place only once complete, so the
target folder either holds a complete model or does not exist. A leftover
.partial folder from an interrupted run is removed by the next run.
"""

from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

# What snapshot_download itself uses to apply ignore_patterns.
from huggingface_hub.utils._paths import filter_repo_objects  # noqa: E402

from document_processor.integrations.huggingface import download  # noqa: E402
from document_processor.integrations.huggingface.download import IGNORED_FILES, download_model  # noqa: E402
from document_processor.integrations.huggingface.reference import (  # noqa: E402
    MODEL_REFERENCE_FILE,
    ModelReference,
    read_model_reference,
)

COMMIT = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3 = ModelReference(hub_model_id="BAAI/bge-m3", commit_hash=COMMIT)

# The files of BAAI/bge-m3, as listed on the hub.
BGE_M3_FILES = (
    ".gitattributes",
    "1_Pooling/config.json",
    "README.md",
    "colbert_linear.pt",
    "config.json",
    "config_sentence_transformers.json",
    "imgs/bm25.jpg",
    "imgs/long.jpg",
    "imgs/miracl.jpg",
    "imgs/mkqa.jpg",
    "imgs/nqa.jpg",
    "imgs/others.webp",
    "long.jpg",
    "modules.json",
    "onnx/Constant_7_attr__value",
    "onnx/config.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "onnx/sentencepiece.bpe.model",
    "onnx/special_tokens_map.json",
    "onnx/tokenizer.json",
    "onnx/tokenizer_config.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "sparse_linear.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class FakeHub:
    """Stands in for snapshot_download; records every call."""

    def __init__(
        self,
        *,
        files: tuple[str, ...] = ("config.json", "pytorch_model.bin", "tokenizer.json"),
        fail_after: int | None = None,
    ) -> None:
        self.files = files
        self.fail_after = fail_after
        self.calls: list[dict[str, object]] = []

    def snapshot_download(
        self, repo_id: str, *, revision: str, local_dir: Path, ignore_patterns: list[str]
    ) -> str:
        self.calls.append(
            {"repo_id": repo_id, "revision": revision, "local_dir": Path(local_dir), "ignore_patterns": ignore_patterns}
        )
        kept = list(filter_repo_objects(self.files, ignore_patterns=ignore_patterns))
        for written, name in enumerate(kept):
            if self.fail_after is not None and written == self.fail_after:
                raise ConnectionError("connection reset")
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"contents of {name}", encoding="utf-8")
        return str(local_dir)


def use_hub(monkeypatch: pytest.MonkeyPatch, fake: FakeHub) -> FakeHub:
    monkeypatch.setattr(download, "snapshot_download", fake.snapshot_download)
    return fake


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    return use_hub(monkeypatch, FakeHub())


def files_in(folder: Path) -> set[str]:
    return {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}


def partial_of(folder: Path) -> Path:
    return folder.with_name(f".{folder.name}.partial")


# ============================================================ downloading


def test_download_fills_the_folder_and_records_the_reference(tmp_path: Path, hub: FakeHub) -> None:
    folder = tmp_path / "models" / "bge-m3"
    download_model(BGE_M3, folder)
    assert files_in(folder) == {"config.json", "pytorch_model.bin", "tokenizer.json", MODEL_REFERENCE_FILE}
    assert read_model_reference(folder) == BGE_M3


def test_the_referenced_commit_is_downloaded(tmp_path: Path, hub: FakeHub) -> None:
    download_model(BGE_M3, tmp_path / "bge-m3")
    assert hub.calls[0]["repo_id"] == "BAAI/bge-m3"
    assert hub.calls[0]["revision"] == COMMIT


def test_one_download_per_call(tmp_path: Path, hub: FakeHub) -> None:
    download_model(BGE_M3, tmp_path / "bge-m3")
    assert len(hub.calls) == 1


def test_accepts_a_string_folder(tmp_path: Path, hub: FakeHub) -> None:
    download_model(BGE_M3, str(tmp_path / "bge-m3"))
    assert read_model_reference(tmp_path / "bge-m3") == BGE_M3


def test_an_existing_empty_folder_is_used(tmp_path: Path, hub: FakeHub) -> None:
    folder = tmp_path / "bge-m3"
    folder.mkdir()
    download_model(BGE_M3, folder)
    assert folder.is_dir() and MODEL_REFERENCE_FILE in files_in(folder)


def test_downloads_into_a_hidden_sibling_first(tmp_path: Path, hub: FakeHub) -> None:
    download_model(BGE_M3, tmp_path / "bge-m3")
    assert hub.calls[0]["local_dir"] == tmp_path / ".bge-m3.partial"


def test_no_partial_folder_is_left_behind(tmp_path: Path, hub: FakeHub) -> None:
    download_model(BGE_M3, tmp_path / "bge-m3")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bge-m3"]


def test_two_models_side_by_side(tmp_path: Path, hub: FakeHub) -> None:
    other = ModelReference(hub_model_id="intfloat/multilingual-e5-large", commit_hash="0" * 40)
    download_model(BGE_M3, tmp_path / "bge-m3")
    download_model(other, tmp_path / "e5")
    assert read_model_reference(tmp_path / "bge-m3") == BGE_M3
    assert read_model_reference(tmp_path / "e5") == other


# ============================================================ which files


def test_the_ignored_files_are_passed_to_the_hub(tmp_path: Path, hub: FakeHub) -> None:
    download_model(BGE_M3, tmp_path / "bge-m3")
    assert hub.calls[0]["ignore_patterns"] == list(IGNORED_FILES)


def test_bge_m3_downloads_what_the_embedder_loads_and_skips_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_hub(monkeypatch, FakeHub(files=BGE_M3_FILES))
    folder = tmp_path / "bge-m3"
    download_model(BGE_M3, folder)
    downloaded = files_in(folder)
    for needed in (
        "config.json",
        "pytorch_model.bin",
        "sparse_linear.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        # The model's own description of itself, read as features.
        "1_Pooling/config.json",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
    ):
        assert needed in downloaded, needed
    assert not any(name.startswith(("onnx/", "imgs/")) for name in downloaded)
    assert not {"README.md", "long.jpg", ".gitattributes"} & downloaded


@pytest.mark.parametrize(
    "name",
    [
        "onnx/model.onnx",
        "onnx/nested/deeper/file.bin",
        "openvino/openvino_model.xml",
        "model.onnx",
        "model.onnx_data",
        "model-q4_k_m.gguf",
        "tf_model.h5",
        "flax_model.msgpack",
        "rust_model.ot",
        "README.md",
        "docs/usage.md",
        "imgs/a.jpg",
        "b.jpeg",
        "c.png",
        "d.webp",
        "e.gif",
        ".gitattributes",
    ],
)
def test_each_kind_of_ignored_file_is_skipped(name: str) -> None:
    assert list(filter_repo_objects([name], ignore_patterns=list(IGNORED_FILES))) == []


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "sparse_linear.pt",
        "tokenizer.json",
        "1_Pooling/config.json",
    ],
)
def test_model_files_are_not_skipped(name: str) -> None:
    assert list(filter_repo_objects([name], ignore_patterns=list(IGNORED_FILES))) == [name]


# ============================================================ refusals, before any network use


def test_a_file_is_not_a_folder(tmp_path: Path, hub: FakeHub) -> None:
    path = tmp_path / "bge-m3"
    path.write_text("not a folder", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="is a file; pass the folder"):
        download_model(BGE_M3, path)
    assert hub.calls == []


def test_a_folder_with_files_is_refused(tmp_path: Path, hub: FakeHub) -> None:
    folder = tmp_path / "bge-m3"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(OSError, match="is not empty; delete it or choose another folder$"):
        download_model(BGE_M3, folder)
    assert hub.calls == []
    assert files_in(folder) == {"config.json"}


def test_an_already_downloaded_model_is_not_downloaded_again(tmp_path: Path, hub: FakeHub) -> None:
    folder = tmp_path / "bge-m3"
    download_model(BGE_M3, folder)
    with pytest.raises(OSError, match="is not empty"):
        download_model(BGE_M3, folder)
    assert len(hub.calls) == 1


# ============================================================ interrupted downloads


def test_an_interrupted_download_leaves_no_model_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_hub(monkeypatch, FakeHub(fail_after=2))
    folder = tmp_path / "bge-m3"
    with pytest.raises(ConnectionError):
        download_model(BGE_M3, folder)
    assert not folder.exists()
    assert len(files_in(partial_of(folder))) == 2
    assert MODEL_REFERENCE_FILE not in files_in(partial_of(folder))


def test_an_interrupted_download_keeps_an_empty_folder_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_hub(monkeypatch, FakeHub(fail_after=1))
    folder = tmp_path / "bge-m3"
    folder.mkdir()
    with pytest.raises(ConnectionError):
        download_model(BGE_M3, folder)
    assert folder.is_dir() and not any(folder.iterdir())


def test_the_next_run_starts_over_and_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bge-m3"
    use_hub(monkeypatch, FakeHub(files=("config.json", "stale.bin", "pytorch_model.bin"), fail_after=2))
    with pytest.raises(ConnectionError):
        download_model(BGE_M3, folder)
    assert "stale.bin" in files_in(partial_of(folder))

    use_hub(monkeypatch, FakeHub(files=("config.json", "pytorch_model.bin")))
    download_model(BGE_M3, folder)
    # Nothing from the interrupted run survives into the model folder.
    assert files_in(folder) == {"config.json", "pytorch_model.bin", MODEL_REFERENCE_FILE}
    assert not partial_of(folder).exists()


def test_reading_an_interrupted_download_finds_no_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_hub(monkeypatch, FakeHub(fail_after=1))
    folder = tmp_path / "bge-m3"
    with pytest.raises(ConnectionError):
        download_model(BGE_M3, folder)
    with pytest.raises(FileNotFoundError, match="cannot find model_reference.json"):
        read_model_reference(folder)