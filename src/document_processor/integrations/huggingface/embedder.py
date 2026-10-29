import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, cast

import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from document_processor.chunking import Tokenizer
from document_processor.integrations.huggingface._folder import (
    check_has_modules_json,
    check_is_folder,
    check_reference_matches,
)
from document_processor.integrations.huggingface._prompts import (
    declared_prompts,
    resolve_prompt_names,
)
from document_processor.integrations.huggingface.model_specs import HuggingFaceModelSpec
from document_processor.integrations.huggingface.reference import ModelReference
from document_processor.retrieval.embedding import Embedding

# bump whenever a change to this module alters the vector
EMBEDDER_VERSION = 1

type Device = Literal["auto", "cuda", "mps", "cpu"]


class _ModelTokenizer:
    """Counts what the model receives: the document prompt, the text and the special tokens."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_text: str,
        max_length: int | None,
        lock: Lock,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_text = prompt_text
        self._max_length = max_length
        self._lock = lock

    def count_tokens(self, text: str) -> int:
        with self._lock:
            encoded = encode_text(self._tokenizer, self._prompt_text + text)
        return len(encoded)

    @property
    def model_max_tokens(self) -> int | None:
        return self._max_length
    

@dataclass(frozen=True, kw_only=True)
class EmbeddingSettings:
    """What decides the vectors, apart from the weights. Exposed so the app can show it."""
    # The text actually prepended
    query_prompt: str
    document_prompt: str

    # max_seq_length
    max_length: int | None

    sparse_head: str | None


class HuggingFaceEmbedder:

    def __init__(
        self,
        *,
        spec: HuggingFaceModelSpec,
        folder: Path | str,
        device: Device = "auto",
        batch_size: int = 32,
    ) -> None:
        # Folder
        folder = Path(folder)
        check_is_folder(folder)
        check_reference_matches(folder, spec.reference)
        check_has_modules_json(folder)

        # Batch
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        # Device
        resolved_device = _resolve_device(device)

        # Model
        model = SentenceTransformer(
            str(folder),
            device=resolved_device,
            local_files_only=True,
            model_kwargs={"dtype": torch.float32}
        )

        # Prompt
        query_prompt_name, document_prompt_name = resolve_prompt_names(
            model.prompts,
            query=spec.query_prompt_name,
            document=spec.document_prompt_name,
        )
        prompts = declared_prompts(model.prompts)
        query_prompt = prompts[query_prompt_name] if query_prompt_name is not None else ""
        document_prompt = prompts[document_prompt_name] if document_prompt_name is not None else ""

        # Dense dimension
        reference = spec.reference
        embedding_dim = model.get_embedding_dimension()
        if embedding_dim is None:
            raise ValueError(
                f"model {reference.hub_model_id}@{reference.commit_hash}: cannot find embedding dimension"
            )

        self._dense_dimension = embedding_dim
        self._device = resolved_device
        self._query_prompt_name = query_prompt_name
        self._document_prompt_name = document_prompt_name
        self._model = model
        self._lock = Lock()
        self._tokenizer = _ModelTokenizer(
            tokenizer=self._model.tokenizer,
            prompt_text=document_prompt,
            max_length=self._model.max_seq_length,
            lock=self._lock
        )
        self._batch_size = batch_size
        self._embedding_settings = EmbeddingSettings(
            query_prompt=query_prompt,
            document_prompt=document_prompt,
            max_length=self._model.max_seq_length,
            sparse_head=None
        )
        self._model_identity = model_identity(spec.reference, self._embedding_settings)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer
    
    @property
    def embedding_settings(self) -> EmbeddingSettings:
        return self._embedding_settings
    
    @property
    def device(self) -> str:
        return self._device

    @property
    def dense_dimension(self) -> int:
        return self._dense_dimension
    
    @property
    def has_sparse(self) -> bool:
        return False

    @property
    def model_identity(self) -> str:
        return self._model_identity

    def encode_query(self, text: str) -> Embedding:
        return self._encode([text], self._query_prompt_name)[0]
    
    def encode_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return self._encode(texts, self._document_prompt_name)

    def _encode(self, texts: Sequence[str], prompt_name: str | None) -> list[Embedding]:
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results: dict[int, Embedding] = {}
        for start in range(0, len(order), self._batch_size):
            batch = order[start : start + self._batch_size]
            with self._lock, torch.inference_mode():
                outputs = cast(
                    list[dict[str, torch.Tensor]],
                    self._model.encode(
                        [texts[index] for index in batch],
                        prompt_name=prompt_name,
                        output_value=None,
                        batch_size=len(batch),
                        show_progress_bar=False,
                    ),
                )
                for index, output in zip(batch, outputs, strict=True):
                    results[index] = self._to_embedding(output, texts[index])

        return [results[index] for index in range(len(texts))]
    
    def _to_embedding(self, output: Mapping[str, torch.Tensor], text: str) -> Embedding:
        dense = output["sentence_embedding"].float()
        if not torch.isfinite(dense).all():
            raise ValueError(
                f"the model produced NaN or infinite values for {text[:60]!r}"
            )
        return Embedding(
            dense=tuple(dense.tolist()),
            sparse=None
        )


def encode_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """The token ids a model receives for text: special tokens included, never truncated.

    verbose=False: the chunker counts oversized sections exactly in order to split them,
    so transformers' "longer than the maximum" warning would only mislead.
    """
    encoding = tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)
    ids: list[int] = encoding["input_ids"]
    return ids


def model_identity(reference: ModelReference, settings: EmbeddingSettings) -> str:
    settings_json = json.dumps(asdict(settings), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    settings_hash = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()[:12]
    return f"{reference.hub_model_id}@{reference.commit_hash}/{settings_hash}/v{EMBEDDER_VERSION}"


def _resolve_device(device: Device) -> Literal["cuda", "mps", "cpu"]:
    cuda_available = torch.cuda.is_available()
    if device == "cuda":
        if not cuda_available:
            raise ValueError(
                "device 'cuda' was asked for, but PyTorch found no CUDA-compatible GPU; use 'auto' or 'cpu'"
            )
        return "cuda"

    mps_available = torch.backends.mps.is_available()
    if device == "mps":
        if not mps_available:
            raise ValueError(
                "device 'mps' was asked for, but PyTorch found no MPS-compatible GPU; use 'auto' or 'cpu'"
            )
        return "mps"

    if device == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
    return "cpu"
