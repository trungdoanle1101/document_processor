from collections.abc import Sequence
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from document_processor.integrations.huggingface._device import Device, resolve_device
from document_processor.integrations.huggingface._folder import (
    check_is_folder,
    check_reference_matches,
)
from document_processor.integrations.huggingface.model_specs import (
    HuggingFaceRerankerSpec,
)
from document_processor.integrations.huggingface.reference import ModelReference

# bump whenever a change to this module alters the scores
RERANKER_VERSION = 1


class HuggingFaceReranker:
    """A cross-encoder from a folder made by download_model; scores (question, passage) pairs, 0 to 1."""

    def __init__(
        self,
        *,
        spec: HuggingFaceRerankerSpec,
        folder: Path | str,
        device: Device = "auto",
        batch_size: int = 16,
    ) -> None:
        # Folder
        folder = Path(folder)
        check_is_folder(folder)
        check_reference_matches(folder, spec.reference)

        # Batch
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Device
        resolved_device = resolve_device(device)

        reference = spec.reference

        # Model, in float32 whatever precision the checkpoint was saved in
        model = CrossEncoder(
            str(folder),
            device=resolved_device,
            local_files_only=True,
            max_length=spec.max_length,
            model_kwargs={"dtype": torch.float32},
        )
        if model.num_labels != 1:
            raise ValueError(
                f"model {reference.hub_model_id}@{reference.commit_hash} has {model.num_labels} outputs; a reranker needs 1"
            )

        self._model = model
        self._device = resolved_device
        self._batch_size = batch_size
        self._lock = Lock()
        self._identity = reranker_identity(reference, spec.max_length)

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def device(self) -> str:
        return self._device

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        "One score per passage, in order; 0 to 1, higher answers the question better."
        if not passages:
            return []

        pairs = [(question, p) for p in passages]
        with self._lock, torch.inference_mode():
            scores = np.asarray(
                self._model.predict(
                    pairs, batch_size=self._batch_size, show_progress_bar=False
                ),
                dtype=np.float32,
            )
        if not np.isfinite(scores).all():
            raise ValueError(
                f"the reranker produced NaN or infinite values for {question[:60]!r}"
            )
        return [float(score) for score in scores]


def reranker_identity(reference: ModelReference, max_length: int) -> str:
    return f"{reference.hub_model_id}@{reference.commit_hash}/max{max_length}/v{RERANKER_VERSION}"
