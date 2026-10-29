import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
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
from document_processor.retrieval.embedding import (
    DenseVector,
    DenseVectorSpec,
    Embedding,
    VectorSpec,
)

# bump whenever a change to this module alters the vectors
EMBEDDER_VERSION = 1

DENSE_VECTOR_NAME = "dense"

type Device = Literal["auto", "cuda", "mps", "cpu"]
type ResolvedDevice = Literal["cuda", "mps", "cpu"]


class _ModelTokenizer:
    """Counts what the model receives: the passage prompt, the text and the special tokens."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_text: str,
        model_max_tokens: int | None,
        lock: Lock,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_text = prompt_text
        self._model_max_tokens = model_max_tokens
        self._lock = lock

    def count_tokens(self, text: str) -> int:
        with self._lock:
            ids = tokenize(self._tokenizer, self._prompt_text + text)
        return len(ids)

    @property
    def model_max_tokens(self) -> int | None:
        return self._model_max_tokens


@dataclass(frozen=True, kw_only=True)
class EmbedderSettings:
    """What decides the vectors, apart from the weights. Exposed so the app can show it."""

    # The text actually prepended; "" for none
    query_prompt_text: str
    passage_prompt_text: str

    # sentence-transformers' max_seq_length: where texts are cut
    model_max_tokens: int | None

    sparse_head: str | None


class HuggingFaceEmbedder:
    """A sentence-transformers model from a folder made by download_model; dense vectors only, for now."""

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

        # Model, in float32 whatever precision the checkpoint was saved in
        model = SentenceTransformer(
            str(folder),
            device=resolved_device,
            local_files_only=True,
            model_kwargs={"dtype": torch.float32},
        )

        # Prompts
        query_prompt_name, passage_prompt_name = resolve_prompt_names(
            model.prompts,
            query=spec.query_prompt_name,
            passage=spec.passage_prompt_name,
        )
        prompts = declared_prompts(model.prompts)
        query_prompt_text = (
            prompts[query_prompt_name] if query_prompt_name is not None else ""
        )
        passage_prompt_text = (
            prompts[passage_prompt_name] if passage_prompt_name is not None else ""
        )

        # Dense dimension
        reference = spec.reference
        dimension = model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(
                f"model {reference.hub_model_id}@{reference.commit_hash}: cannot find embedding dimension"
            )

        self._model = model
        self._device = resolved_device
        self._batch_size = batch_size
        self._query_prompt_name = query_prompt_name
        self._passage_prompt_name = passage_prompt_name
        self._vector_specs: Mapping[str, VectorSpec] = MappingProxyType(
            {DENSE_VECTOR_NAME: DenseVectorSpec(dimension=dimension)}
        )
        self._lock = Lock()
        self._tokenizer = _ModelTokenizer(
            tokenizer=model.tokenizer,
            prompt_text=passage_prompt_text,
            model_max_tokens=model.max_seq_length,
            lock=self._lock,
        )
        self._settings = EmbedderSettings(
            query_prompt_text=query_prompt_text,
            passage_prompt_text=passage_prompt_text,
            model_max_tokens=model.max_seq_length,
            sparse_head=None,
        )
        self._identity = embedder_identity(reference, self._settings)

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def vector_specs(self) -> Mapping[str, VectorSpec]:
        return self._vector_specs

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def settings(self) -> EmbedderSettings:
        return self._settings

    @property
    def device(self) -> str:
        return self._device

    def encode_query(self, text: str) -> Embedding:
        return self._encode([text], self._query_prompt_name)[0]

    def encode_passages(self, texts: Sequence[str]) -> list[Embedding]:
        return self._encode(texts, self._passage_prompt_name)

    def _encode(self, texts: Sequence[str], prompt_name: str | None) -> list[Embedding]:
        # Sort so each batch holds similar lengths (less padding); put results back in input order.
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results: dict[int, Embedding] = {}
        for start in range(0, len(order), self._batch_size):
            batch = order[start : start + self._batch_size]
            with (
                self._lock,
                torch.inference_mode(),
            ):  # a query waits for at most one batch
                outputs = cast(
                    list[
                        dict[str, torch.Tensor]
                    ],  # output_value=None always gives this shape
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
            vectors={DENSE_VECTOR_NAME: DenseVector(values=tuple(dense.tolist()))}
        )


def tokenize(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """The token ids a model receives for text: special tokens included, never truncated.

    verbose=False: the chunker counts oversized sections exactly in order to split them,
    so transformers' "longer than the maximum" warning would only mislead.
    """
    encoding = tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)
    ids: list[int] = encoding["input_ids"]
    return ids


def embedder_identity(reference: ModelReference, settings: EmbedderSettings) -> str:
    settings_json = json.dumps(
        asdict(settings), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    settings_hash = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()[:12]
    return f"{reference.hub_model_id}@{reference.commit_hash}/{settings_hash}/v{EMBEDDER_VERSION}"


def _resolve_device(device: Device) -> ResolvedDevice:
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
