from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

# configs without a limit report a huge placeholder (int(1e30))
_UNKNOWN_LIMIT = 1_000_000

_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}


class HuggingFaceTokenizer:
    """Counts tokens with a Hugging Face model's own tokenizer, loaded from a local folder.

    The folder is always where it is read from. Only when it holds no tokenizer and
    hub_model_id is given is the tokenizer downloaded, once, and saved there; pass
    revision (a commit hash) to pin which version that is. To upgrade, use a new folder.
    """

    def __init__(
        self,
        folder: Path | str,
        *,
        hub_model_id: str | None = None,
        revision: str | None = None,
        model_max_tokens: int | None = None,
    ) -> None:
        if model_max_tokens is not None and model_max_tokens <= 0:
            raise ValueError(f"model_max_tokens must be > 0, got {model_max_tokens}")

        folder = Path(folder)
        if folder.name in _TOKENIZER_FILES:
            raise ValueError(
                f"{folder} names a tokenizer file; pass the folder that holds it: {folder.parent}"
            )

        if folder.exists() and not folder.is_dir():
            raise NotADirectoryError(
                f"{folder} is a file; pass the folder that holds (or will hold) the tokenizer"
            )

        if not (folder / "tokenizer_config.json").is_file():
            if hub_model_id is None:
                raise FileNotFoundError(
                    f"no tokenizer in folder {folder}; pass hub_model_id (e.g. 'BAAI/bge-m3') to download it once"
                )
            downloaded = AutoTokenizer.from_pretrained(hub_model_id, revision=revision)
            downloaded.save_pretrained(folder)
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            folder, local_files_only=True
        )
        self._model_max_tokens = (
            model_max_tokens
            if model_max_tokens is not None
            else _known_limit(self._tokenizer)
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def model_max_tokens(self) -> int | None:
        return self._model_max_tokens


def _known_limit(tokenizer: PreTrainedTokenizerBase) -> int | None:
    limit = tokenizer.model_max_length
    return limit if limit < _UNKNOWN_LIMIT else None
