"""Tests for chunking/huggingface_tokenizer.py: counting with a Hugging Face model's tokenizer.

The tokenizer is always read from a local folder (local_files_only). Only when
the folder holds no tokenizer_config.json and hub_model_id is given is it
downloaded, once, with the given revision, and saved there. The limit comes from
the tokenizer's model_max_length (None for the huge placeholder of configs
without one) unless model_max_tokens overrides it. Paths that name a tokenizer
file, or an existing file, are refused before anything is read.

No test uses the network: tokenizers are built here from a tiny vocabulary and
saved with save_pretrained, and the Hub download is replaced by a stub. The
module is skipped when the huggingface extra (transformers) is not installed.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers")

from tokenizers import Tokenizer as BackendTokenizer  # noqa: E402
from tokenizers import models, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from document_processor.chunking import OutlineChunker  # noqa: E402
from document_processor.chunking import huggingface_tokenizer as module  # noqa: E402
from document_processor.chunking.huggingface_tokenizer import HuggingFaceTokenizer  # noqa: E402
from document_processor.chunking.tokenizer import Tokenizer  # noqa: E402
from tests.chunking.test_outline_chunker import regulation  # noqa: E402

VOCAB = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "điều": 3, "1.": 4, "phạm": 5, "vi": 6, "hệ": 7, "thống": 8}


def tiny_tokenizer(model_max_length: int | None = 512) -> PreTrainedTokenizerFast:
    """A word-level tokenizer that, like most models, wraps encodings in [CLS] … [SEP]."""
    backend = BackendTokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")  # type: ignore[no-untyped-call]
    if model_max_length is not None:
        fast.model_max_length = model_max_length
    return fast


def saved_folder(folder: Path, model_max_length: int | None = 512) -> Path:
    tiny_tokenizer(model_max_length).save_pretrained(folder)
    return folder


class HubStub:
    """Stands in for AutoTokenizer: Hub ids get the tiny tokenizer, folders load for real."""

    def __init__(self) -> None:
        self.hub_calls: list[tuple[str, str | None]] = []
        self.folder_calls: list[dict[str, Any]] = []

    def from_pretrained(self, name_or_path: Any, **kwargs: Any) -> Any:
        if isinstance(name_or_path, str) and "/" in name_or_path and not Path(name_or_path).exists():
            self.hub_calls.append((name_or_path, kwargs.get("revision")))
            return tiny_tokenizer()
        self.folder_calls.append(kwargs)
        return PreTrainedTokenizerFast.from_pretrained(name_or_path, **kwargs)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> HubStub:
    stub = HubStub()
    monkeypatch.setattr(module, "AutoTokenizer", stub)
    return stub


# ======================================================================
# Counting
# ======================================================================


def test_counts_tokens_of_the_folders_tokenizer(tmp_path: Path) -> None:
    tokenizer = HuggingFaceTokenizer(saved_folder(tmp_path / "tok"))
    # The Whitespace pre-tokenizer splits "1." into "1" and ".": Điều | 1 | . | Phạm | vi
    assert tokenizer.count_tokens("Điều 1. Phạm vi") == 5


def test_special_tokens_are_not_counted(tmp_path: Path) -> None:
    # The model adds [CLS] and [SEP] once per embedded text, not once per piece: pieces must add up.
    tokenizer = HuggingFaceTokenizer(saved_folder(tmp_path / "tok"))
    assert tokenizer.count_tokens("hệ thống") == 2
    assert tokenizer.count_tokens("hệ thống hệ thống") == 2 * tokenizer.count_tokens("hệ thống")


def test_unknown_words_still_count(tmp_path: Path) -> None:
    assert HuggingFaceTokenizer(saved_folder(tmp_path / "tok")).count_tokens("xyz abc") == 2


def test_empty_text_is_zero(tmp_path: Path) -> None:
    assert HuggingFaceTokenizer(saved_folder(tmp_path / "tok")).count_tokens("") == 0


def test_is_a_tokenizer_the_chunker_accepts(tmp_path: Path) -> None:
    tokenizer: Tokenizer = HuggingFaceTokenizer(saved_folder(tmp_path / "tok"))
    chunks = OutlineChunker(tokenizer=tokenizer, max_tokens=60).chunk(regulation())
    assert chunks and all(chunk.token_count == tokenizer.count_tokens(chunk.text) for chunk in chunks)


# ======================================================================
# The model's limit
# ======================================================================


def test_limit_comes_from_the_config(tmp_path: Path) -> None:
    assert HuggingFaceTokenizer(saved_folder(tmp_path / "tok", 512)).model_max_tokens == 512


def test_placeholder_limit_means_unknown(tmp_path: Path) -> None:
    # A config without a limit reports int(1e30).
    assert HuggingFaceTokenizer(saved_folder(tmp_path / "tok", None)).model_max_tokens is None


def test_explicit_limit_overrides_the_config(tmp_path: Path) -> None:
    folder = saved_folder(tmp_path / "tok", 512)
    assert HuggingFaceTokenizer(folder, model_max_tokens=8192).model_max_tokens == 8192
    assert HuggingFaceTokenizer(saved_folder(tmp_path / "none", None), model_max_tokens=256).model_max_tokens == 256


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_rejected_before_anything_is_read(tmp_path: Path, hub: HubStub, limit: int) -> None:
    with pytest.raises(ValueError, match=f"model_max_tokens must be > 0, got {limit}"):
        HuggingFaceTokenizer(tmp_path / "missing", hub_model_id="BAAI/bge-m3", model_max_tokens=limit)
    assert hub.hub_calls == [] and not (tmp_path / "missing").exists()


def test_chunker_budget_above_the_limit_is_rejected(tmp_path: Path) -> None:
    tokenizer = HuggingFaceTokenizer(saved_folder(tmp_path / "tok", 512))
    with pytest.raises(ValueError, match="max_tokens=513 is above the tokenizer's model limit of 512"):
        OutlineChunker(tokenizer=tokenizer, max_tokens=513)


# ======================================================================
# Folder, offline
# ======================================================================


def test_reads_the_folder_with_local_files_only(tmp_path: Path, hub: HubStub) -> None:
    HuggingFaceTokenizer(saved_folder(tmp_path / "tok"))
    assert hub.folder_calls == [{"local_files_only": True}]
    assert hub.hub_calls == []


def test_accepts_a_string_path(tmp_path: Path) -> None:
    assert HuggingFaceTokenizer(str(saved_folder(tmp_path / "tok"))).count_tokens("hệ") == 1


def test_missing_tokenizer_without_a_hub_id_says_how_to_fetch_it(tmp_path: Path, hub: HubStub) -> None:
    with pytest.raises(FileNotFoundError, match=r"no tokenizer in folder .*missing; pass hub_model_id"):
        HuggingFaceTokenizer(tmp_path / "missing")
    assert hub.hub_calls == [] and hub.folder_calls == []


def test_empty_existing_folder_without_a_hub_id(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no tokenizer in folder"):
        HuggingFaceTokenizer(tmp_path / "empty")


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
def test_a_tokenizer_file_name_is_refused(tmp_path: Path, hub: HubStub, name: str) -> None:
    with pytest.raises(ValueError, match=f"names a tokenizer file; pass the folder that holds it: {tmp_path / 'tok'}"):
        HuggingFaceTokenizer(tmp_path / "tok" / name, hub_model_id="BAAI/bge-m3")
    assert hub.hub_calls == []


def test_an_existing_tokenizer_file_is_refused_too(tmp_path: Path) -> None:
    folder = saved_folder(tmp_path / "tok")
    with pytest.raises(ValueError, match="names a tokenizer file"):
        HuggingFaceTokenizer(folder / "tokenizer.json")


def test_another_existing_file_is_refused(tmp_path: Path, hub: HubStub) -> None:
    file = tmp_path / "notes.txt"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="notes.txt is a file; pass the folder"):
        HuggingFaceTokenizer(file, hub_model_id="BAAI/bge-m3")
    assert hub.hub_calls == []


# ======================================================================
# Downloading, only when the folder has no tokenizer
# ======================================================================


def test_downloads_once_into_a_new_nested_folder(tmp_path: Path, hub: HubStub) -> None:
    folder = tmp_path / "models" / "tokenizers" / "bge-m3"
    tokenizer = HuggingFaceTokenizer(folder, hub_model_id="BAAI/bge-m3")
    assert hub.hub_calls == [("BAAI/bge-m3", None)]
    assert (folder / "tokenizer_config.json").is_file() and (folder / "tokenizer.json").is_file()
    assert tokenizer.count_tokens("hệ thống") == 2
    assert hub.folder_calls == [{"local_files_only": True}]  # what was saved is what is read


def test_revision_is_passed_to_the_download(tmp_path: Path, hub: HubStub) -> None:
    HuggingFaceTokenizer(tmp_path / "tok", hub_model_id="BAAI/bge-m3", revision="5617a9f")
    assert hub.hub_calls == [("BAAI/bge-m3", "5617a9f")]


def test_an_existing_tokenizer_is_never_downloaded_again(tmp_path: Path, hub: HubStub) -> None:
    folder = tmp_path / "tok"
    HuggingFaceTokenizer(folder, hub_model_id="BAAI/bge-m3")
    HuggingFaceTokenizer(folder, hub_model_id="BAAI/bge-m3")
    HuggingFaceTokenizer(folder, hub_model_id="BAAI/bge-m3", revision="other")  # the folder wins
    assert hub.hub_calls == [("BAAI/bge-m3", None)]


def test_a_folder_saved_elsewhere_loads_without_a_hub_id(tmp_path: Path, hub: HubStub) -> None:
    # Fetched once on a machine with internet, copied to one without.
    folder = tmp_path / "tok"
    HuggingFaceTokenizer(folder, hub_model_id="BAAI/bge-m3")
    assert HuggingFaceTokenizer(folder).count_tokens("phạm vi") == 2
    assert hub.hub_calls == [("BAAI/bge-m3", None)]