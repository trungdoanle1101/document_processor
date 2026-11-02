"""Tests for integrations/huggingface/reranker.py: HuggingFaceReranker.

The reranker loads a cross-encoder folder made by download_model (a plain
transformers model with one output, like bge-reranker-v2-m3; no modules.json),
checks it is the spec's exact model, and scores each (question, passage) pair
through sentence-transformers' CrossEncoder, which applies a sigmoid: 0 to 1.

Scores are checked against the same model run directly with transformers, one
pair at a time, so batching and order are checked against code that does neither.

No test uses the network or a GPU.
"""

import threading
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from document_processor.integrations.huggingface import (
    HuggingFaceRerankerSpec,
    ModelReference,
)
from document_processor.integrations.huggingface.reranker import (
    RERANKER_VERSION,
    HuggingFaceReranker,
)
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import IndexedChunk
from document_processor.retrieval.reranker import Reranker
from document_processor.retrieval.search import search_chunks
from tests.integrations.huggingface.tiny_model import (
    TINY_REFERENCE,
    TINY_TOKENIZER_LIMIT,
    build_tiny_cross_encoder,
)
from tests.retrieval.stub_embedder import StubEmbedder

MAX_LENGTH = 32
SPEC = HuggingFaceRerankerSpec(reference=TINY_REFERENCE, max_length=MAX_LENGTH)
QUESTION = "điều 14 quy định"
PASSAGES = ["hệ thống dữ liệu", "điều 14 khoản 1", "phạm vi điều_chỉnh", "quy định"]


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_cross_encoder(tmp_path_factory.mktemp("reranker") / "tiny")


@pytest.fixture(scope="module")
def reranker(folder: Path) -> HuggingFaceReranker:
    return HuggingFaceReranker(spec=SPEC, folder=folder, device="cpu")


def direct_score(folder: Path, question: str, passage: str) -> float:
    """The model's own output for one pair, without sentence-transformers: sigmoid of the logit."""
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(
        folder, dtype=torch.float32
    )
    inputs = tokenizer(
        question, passage, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        logit = model(**inputs).logits[0, 0]
    return float(torch.sigmoid(logit))


# The protocol ----------------------------------------------------------------------------


def test_is_a_reranker(reranker: HuggingFaceReranker) -> None:
    as_protocol: Reranker = reranker
    assert as_protocol.identity == reranker.identity


def test_one_score_per_passage_in_order_matching_the_model_run_directly(
    folder: Path, reranker: HuggingFaceReranker
) -> None:
    scores = reranker.score(QUESTION, PASSAGES)
    expected = [direct_score(folder, QUESTION, passage) for passage in PASSAGES]
    assert scores == pytest.approx(expected, abs=1e-6)


def test_scores_are_plain_floats_between_0_and_1(reranker: HuggingFaceReranker) -> None:
    scores = reranker.score(QUESTION, PASSAGES)
    assert all(type(score) is float for score in scores)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_a_pair_scores_the_same_alone_and_in_any_batch(folder: Path) -> None:
    one_at_a_time = HuggingFaceReranker(
        spec=SPEC, folder=folder, device="cpu", batch_size=1
    )
    all_at_once = HuggingFaceReranker(
        spec=SPEC, folder=folder, device="cpu", batch_size=16
    )
    batched = all_at_once.score(QUESTION, PASSAGES)
    assert one_at_a_time.score(QUESTION, PASSAGES) == pytest.approx(batched, abs=1e-6)
    assert all_at_once.score(QUESTION, PASSAGES[2:3]) == pytest.approx(
        batched[2:3], abs=1e-6
    )


def test_no_passages_give_no_scores_without_running_the_model(
    reranker: HuggingFaceReranker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model ran")

    monkeypatch.setattr(reranker._model, "predict", fail)
    assert reranker.score(QUESTION, []) == []


def test_a_passage_longer_than_max_length_is_cut_not_refused(
    reranker: HuggingFaceReranker,
) -> None:
    """Past MAX_LENGTH tokens, more text changes nothing: the pair was cut there."""
    long = " ".join(["dữ liệu"] * MAX_LENGTH)
    longer = long + " " + " ".join(["hệ thống"] * MAX_LENGTH)
    assert reranker.score(QUESTION, [long]) == pytest.approx(
        reranker.score(QUESTION, [longer]), abs=1e-6
    )


def test_concurrent_calls_give_the_same_scores(reranker: HuggingFaceReranker) -> None:
    expected = reranker.score(QUESTION, PASSAGES)
    results: list[list[float]] = []

    def run() -> None:
        results.append(reranker.score(QUESTION, PASSAGES))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 4
    assert all(r == pytest.approx(expected, abs=1e-6) for r in results)


# Identity and device -----------------------------------------------------------------------


def test_the_identity_names_the_model_commit_length_and_version(
    reranker: HuggingFaceReranker,
) -> None:
    assert reranker.identity == (
        f"{TINY_REFERENCE.hub_model_id}@{TINY_REFERENCE.commit_hash}/max{MAX_LENGTH}/v{RERANKER_VERSION}"
    )


def test_a_different_max_length_is_a_different_identity(
    folder: Path, reranker: HuggingFaceReranker
) -> None:
    other = HuggingFaceRerankerSpec(
        reference=TINY_REFERENCE, max_length=MAX_LENGTH // 2
    )
    assert (
        HuggingFaceReranker(spec=other, folder=folder, device="cpu").identity
        != reranker.identity
    )


def test_it_reports_the_device_it_runs_on(reranker: HuggingFaceReranker) -> None:
    assert reranker.device == "cpu"


# What it refuses ---------------------------------------------------------------------------


def test_a_missing_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_model"):
        HuggingFaceReranker(spec=SPEC, folder=tmp_path / "nothing", device="cpu")


def test_a_folder_holding_another_model_is_refused(folder: Path) -> None:
    other = HuggingFaceRerankerSpec(
        reference=ModelReference(
            hub_model_id="BAAI/bge-reranker-v2-m3", commit_hash="f" * 40
        ),
        max_length=MAX_LENGTH,
    )
    with pytest.raises(ValueError, match="but the spec wants BAAI/bge-reranker-v2-m3"):
        HuggingFaceReranker(spec=other, folder=folder, device="cpu")


def test_a_model_with_more_than_one_output_is_refused(tmp_path: Path) -> None:
    two_outputs = build_tiny_cross_encoder(tmp_path / "two", num_labels=2)
    with pytest.raises(ValueError, match="has 2 outputs; a reranker needs 1"):
        HuggingFaceReranker(spec=SPEC, folder=two_outputs, device="cpu")


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(folder: Path, batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        HuggingFaceReranker(
            spec=SPEC, folder=folder, device="cpu", batch_size=batch_size
        )


def test_nan_scores_are_refused(tmp_path: Path) -> None:
    broken = HuggingFaceReranker(
        spec=SPEC,
        folder=build_tiny_cross_encoder(tmp_path / "nan", nan_weights=True),
        device="cpu",
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        broken.score(QUESTION, PASSAGES)


def test_the_tiny_tokenizer_limit_is_above_max_length() -> None:
    """Otherwise the tokenizer's own limit, not the spec's, would decide where pairs are cut."""
    assert TINY_TOKENIZER_LIMIT > MAX_LENGTH


# Through search ----------------------------------------------------------------------------


def test_search_chunks_reorders_by_its_scores(reranker: HuggingFaceReranker) -> None:
    embedder = StubEmbedder()
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    chunks = [
        IndexedChunk(
            document_id="law",
            chunk_id=f"c{i}",
            position=i,
            text=text,
            embedded_text=text,
        )
        for i, text in enumerate(PASSAGES)
    ]
    index.replace_document("law", chunks, embedder.encode_passages(PASSAGES))

    results = search_chunks(
        QUESTION, embedder=embedder, index=index, reranker=reranker, limit=4
    )

    expected = dict(zip(PASSAGES, reranker.score(QUESTION, PASSAGES), strict=True))
    assert [r.rerank_score for r in results] == pytest.approx(
        [expected[r.chunk.text] for r in results], abs=1e-6
    )
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
