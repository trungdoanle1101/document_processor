"""Tests for integrations/huggingface/embedder.py: HuggingFaceEmbedder, dense vectors only.

The embedder loads a model folder made by download_model, checks it (a
folder, the spec's exact model, modules.json), and wraps sentence-transformers,
which applies the model's own pooling, normalization and length. Prompts are
the model's own, resolved by _prompts.py, and passed by name. Weights are
loaded in float32 whatever precision the checkpoint was saved in. Every
embedding holds one vector, named "dense", as its vector_specs say.

The tiny model here pools the CLS token and normalizes, like bge-m3, so each
vector is checked against that computed directly with transformers, one text
at a time: sorting, batching, padding and restoring the input order are then
checked against code that does none of them.

embedder.tokenizer counts what the model receives (the passage prompt, the
text and the special tokens) and reports max_seq_length as its limit, so a
chunk that fits the chunker's budget fits the model.

The identity is the reference, a hash of the settings actually used, and
EMBEDDER_VERSION; equal identities mean the same vectors.

No test uses the network or a GPU.
"""

import math
import threading
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from document_processor.chunking import OutlineChunker  # noqa: E402
from document_processor.chunking.tokenizer import Tokenizer  # noqa: E402
from document_processor.integrations.huggingface.embedder import (  # noqa: E402
    DENSE_VECTOR_NAME,
    EMBEDDER_VERSION,
    EmbedderSettings,
    HuggingFaceEmbedder,
    embedder_identity,
)
from document_processor.integrations.huggingface.model_specs import HuggingFaceModelSpec  # noqa: E402
from document_processor.integrations.huggingface.reference import ModelReference  # noqa: E402
from document_processor.retrieval.embedding import (
    DenseVectorSpec,
    Embedder,
    Embedding,
    check_embedding,
)  # noqa: E402
from tests.chunking.test_outline_chunker import regulation  # noqa: E402
from tests.integrations.huggingface.tiny_model import (  # noqa: E402
    TINY_HIDDEN_SIZE,
    TINY_REFERENCE,
    TINY_TOKENIZER_LIMIT,
    build_tiny_model,
    build_tiny_sentence_transformer,
)

SPEC = HuggingFaceModelSpec(reference=TINY_REFERENCE)
E5_PROMPTS = {"query": "query: ", "passage": "passage: "}
TEXTS = [
    "điều 14 khoản 1",
    "phạm vi",
    "hệ thống dữ liệu quy định điều 1 khoản 14",
    "điều",
]


@pytest.fixture(scope="module")
def plain_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """No prompts, like bge-m3."""
    return build_tiny_sentence_transformer(tmp_path_factory.mktemp("models") / "plain")


@pytest.fixture(scope="module")
def e5_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Declares "query" and "passage", like e5."""
    return build_tiny_sentence_transformer(
        tmp_path_factory.mktemp("models") / "e5", prompts=E5_PROMPTS
    )


@pytest.fixture(scope="module")
def plain(plain_folder: Path) -> HuggingFaceEmbedder:
    return HuggingFaceEmbedder(spec=SPEC, folder=plain_folder, device="cpu")


@pytest.fixture(scope="module")
def e5(e5_folder: Path) -> HuggingFaceEmbedder:
    return HuggingFaceEmbedder(spec=SPEC, folder=e5_folder, device="cpu")


def cls_vector(folder: Path, text: str) -> list[float]:
    """CLS pooling and normalization, straight from transformers, for one text."""
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModel.from_pretrained(
        folder, local_files_only=True, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        hidden = model(**tokenizer(text, return_tensors="pt")).last_hidden_state
    vector: list[float] = torch.nn.functional.normalize(hidden[0, 0], dim=-1).tolist()
    return vector


def assert_close(
    left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]
) -> None:
    assert len(left) == len(right)
    assert all(
        math.isclose(a, b, abs_tol=1e-5) for a, b in zip(left, right, strict=True)
    )


def make_settings(**overrides: object) -> EmbedderSettings:
    values: dict[str, object] = {
        "query_prompt_text": "",
        "passage_prompt_text": "",
        "model_max_tokens": 512,
        "sparse_head": None,
    }
    values.update(overrides)
    return EmbedderSettings(**values)  # type: ignore[arg-type]


def dense(embedding: Embedding) -> tuple[float, ...]:
    return embedding.dense(DENSE_VECTOR_NAME).values


def plain_specs() -> dict[str, DenseVectorSpec]:
    return {"dense": DenseVectorSpec(dimension=TINY_HIDDEN_SIZE)}


# ============================================================ the protocol


def test_is_an_embedder(plain: HuggingFaceEmbedder) -> None:
    embedder: Embedder = plain  # mypy checks the protocol here
    assert embedder.vector_specs == {
        "dense": DenseVectorSpec(dimension=TINY_HIDDEN_SIZE)
    }
    assert embedder.tokenizer is not None


def test_every_embedding_matches_the_vector_specs(plain: HuggingFaceEmbedder) -> None:
    for embedding in [*plain.encode_passages(TEXTS), plain.encode_query("điều 14")]:
        check_embedding(embedding, plain.identity, plain.vector_specs)
        assert list(embedding.vectors) == ["dense"]
        assert all(type(value) is float for value in dense(embedding))


def test_vector_specs_cannot_be_changed(plain: HuggingFaceEmbedder) -> None:
    with pytest.raises(TypeError):
        plain.vector_specs["extra"] = DenseVectorSpec(dimension=1)  # type: ignore[index]


def test_vectors_are_normalized(plain: HuggingFaceEmbedder) -> None:
    for embedding in plain.encode_passages(TEXTS):
        assert math.isclose(
            sum(value * value for value in dense(embedding)), 1.0, abs_tol=1e-5
        )


def test_no_texts_give_no_embeddings(plain: HuggingFaceEmbedder) -> None:
    assert plain.encode_passages([]) == []


# ============================================================ the vectors


def test_vectors_match_cls_pooling_computed_directly(
    plain: HuggingFaceEmbedder, plain_folder: Path
) -> None:
    for text, embedding in zip(TEXTS, plain.encode_passages(TEXTS), strict=True):
        assert_close(dense(embedding), cls_vector(plain_folder, text))


def test_input_order_survives_sorting_by_length(plain: HuggingFaceEmbedder) -> None:
    # TEXTS is not in length order, so the batches are not in input order.
    alone = [plain.encode_passages([text])[0] for text in TEXTS]
    for together, single in zip(plain.encode_passages(TEXTS), alone, strict=True):
        assert_close(dense(together), dense(single))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 32])
def test_batch_size_does_not_change_the_vectors(
    plain_folder: Path, plain: HuggingFaceEmbedder, batch_size: int
) -> None:
    other = HuggingFaceEmbedder(
        spec=SPEC, folder=plain_folder, device="cpu", batch_size=batch_size
    )
    for left, right in zip(
        other.encode_passages(TEXTS), plain.encode_passages(TEXTS), strict=True
    ):
        assert_close(dense(left), dense(right))


def test_repeated_texts_give_the_same_vector(plain: HuggingFaceEmbedder) -> None:
    first, second, third = plain.encode_passages(["phạm vi", "điều 14", "phạm vi"])
    assert_close(dense(first), dense(third))
    assert dense(first) != dense(second)


def test_a_query_without_prompts_equals_the_passage(plain: HuggingFaceEmbedder) -> None:
    assert_close(
        dense(plain.encode_query("điều 14")),
        dense(plain.encode_passages(["điều 14"])[0]),
    )


def test_weights_load_as_float32_from_a_bfloat16_checkpoint(tmp_path: Path) -> None:
    folder = build_tiny_sentence_transformer(tmp_path / "bf16", save_as_bfloat16=True)
    embedder = HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu")
    # Read from the private model: the weights' rounding to bfloat16 is already
    # in the file, so the vectors alone cannot show the precision they ran at.
    assert {parameter.dtype for parameter in embedder._model.parameters()} == {
        torch.float32
    }


def test_non_finite_values_are_refused(tmp_path: Path) -> None:
    folder = build_tiny_sentence_transformer(tmp_path / "nan", nan_weights=True)
    embedder = HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu")
    with pytest.raises(
        ValueError, match="^the model produced NaN or infinite values for 'điều 14'$"
    ):
        embedder.encode_query("điều 14")


def test_the_refused_text_is_cut_to_60_characters(tmp_path: Path) -> None:
    folder = build_tiny_sentence_transformer(tmp_path / "nan", nan_weights=True)
    embedder = HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu")
    long_text = "điều " * 40
    with pytest.raises(ValueError) as raised:
        embedder.encode_passages([long_text])
    assert repr(long_text[:60]) in str(raised.value)


# ============================================================ prompts


def test_e5_queries_and_passages_get_their_prompts(
    e5: HuggingFaceEmbedder, e5_folder: Path
) -> None:
    assert_close(
        dense(e5.encode_query("điều 14")), cls_vector(e5_folder, "query: điều 14")
    )
    assert_close(
        dense(e5.encode_passages(["điều 14"])[0]),
        cls_vector(e5_folder, "passage: điều 14"),
    )


def test_queries_and_passages_differ_when_prompts_do(e5: HuggingFaceEmbedder) -> None:
    assert dense(e5.encode_query("điều 14")) != dense(
        e5.encode_passages(["điều 14"])[0]
    )


def test_the_settings_show_the_prompt_texts(
    e5: HuggingFaceEmbedder, plain: HuggingFaceEmbedder
) -> None:
    assert (e5.settings.query_prompt_text, e5.settings.passage_prompt_text) == (
        "query: ",
        "passage: ",
    )
    assert (plain.settings.query_prompt_text, plain.settings.passage_prompt_text) == (
        "",
        "",
    )


def test_a_named_prompt_is_used(e5_folder: Path) -> None:
    spec = HuggingFaceModelSpec(reference=TINY_REFERENCE, passage_prompt_name="query")
    embedder = HuggingFaceEmbedder(spec=spec, folder=e5_folder, device="cpu")
    assert_close(
        dense(embedder.encode_passages(["điều 14"])[0]),
        cls_vector(e5_folder, "query: điều 14"),
    )


def test_a_prompt_the_model_lacks_is_refused_at_load(e5_folder: Path) -> None:
    spec = HuggingFaceModelSpec(reference=TINY_REFERENCE, query_prompt_name="question")
    with pytest.raises(
        ValueError, match="query prompt 'question' is not one the model has"
    ):
        HuggingFaceEmbedder(spec=spec, folder=e5_folder, device="cpu")


def test_unconventional_prompt_names_must_be_named(tmp_path: Path) -> None:
    folder = build_tiny_sentence_transformer(
        tmp_path / "nomic",
        prompts={"search_query": "query: ", "search_document": "passage: "},
    )
    with pytest.raises(
        ValueError,
        match="name the ones to use in the spec: query_prompt_name and passage_prompt_name",
    ):
        HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu")
    named = HuggingFaceModelSpec(
        reference=TINY_REFERENCE,
        query_prompt_name="search_query",
        passage_prompt_name="search_document",
    )
    embedder = HuggingFaceEmbedder(spec=named, folder=folder, device="cpu")
    assert_close(
        dense(embedder.encode_query("điều 14")), cls_vector(folder, "query: điều 14")
    )


# ============================================================ the tokenizer


def test_counts_the_text_and_the_special_tokens(plain: HuggingFaceEmbedder) -> None:
    # <s> điều 14 </s>
    assert plain.tokenizer.count_tokens("điều 14") == 4


def test_counts_the_passage_prompt_too(e5: HuggingFaceEmbedder) -> None:
    # <s> passage: điều 14 </s>: what the model receives for a passage
    assert e5.tokenizer.count_tokens("điều 14") == 5


def test_long_text_is_counted_in_full(plain: HuggingFaceEmbedder) -> None:
    assert plain.tokenizer.count_tokens(" ".join(["điều"] * 100)) == 102


def test_long_text_logs_no_warning(
    plain: HuggingFaceEmbedder, capfd: pytest.CaptureFixture[str]
) -> None:
    plain.tokenizer.count_tokens(" ".join(["điều"] * 100))
    captured = capfd.readouterr()
    assert "longer than the specified maximum" not in captured.err + captured.out


def test_the_limit_is_what_the_model_runs_at(tmp_path: Path) -> None:
    # The tokenizer's own limit is 48; sentence-transformers runs this model at 32.
    folder = build_tiny_sentence_transformer(tmp_path / "short", max_seq_length=32)
    embedder = HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu")
    assert embedder.tokenizer.model_max_tokens == 32
    assert embedder.settings.model_max_tokens == 32


def test_the_model_cuts_texts_at_the_same_limit(tmp_path: Path) -> None:
    folder = build_tiny_sentence_transformer(tmp_path / "short", max_seq_length=8)
    embedder = HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu")
    words = (
        "điều khoản phạm vi hệ thống dữ liệu"  # 8 words, 10 tokens: past the limit of 8
    )
    assert embedder.tokenizer.count_tokens(words) == 10
    cut = embedder.encode_passages([words])[0]
    assert_close(dense(cut), dense(embedder.encode_passages([words + " quy định"])[0]))


def test_the_chunker_accepts_it(plain: HuggingFaceEmbedder) -> None:
    tokenizer: Tokenizer = plain.tokenizer  # mypy checks the protocol here
    chunks = OutlineChunker(tokenizer=tokenizer, max_tokens=TINY_TOKENIZER_LIMIT).chunk(
        regulation()
    )
    assert chunks
    assert all(
        chunk.token_count == tokenizer.count_tokens(chunk.text) for chunk in chunks
    )
    assert all(chunk.token_count <= TINY_TOKENIZER_LIMIT for chunk in chunks)


def test_the_chunker_refuses_a_budget_above_the_model_limit(
    plain: HuggingFaceEmbedder,
) -> None:
    with pytest.raises(
        ValueError, match=f"above the tokenizer's model limit of {TINY_TOKENIZER_LIMIT}"
    ):
        OutlineChunker(tokenizer=plain.tokenizer, max_tokens=TINY_TOKENIZER_LIMIT + 1)


def test_counting_and_encoding_from_many_threads(plain: HuggingFaceEmbedder) -> None:
    # One embedder serves the job thread and request handlers; the lock keeps
    # the shared tokenizer from being used by two threads at once.
    expected_count = plain.tokenizer.count_tokens("điều 14 khoản 1")
    expected_vector = dense(plain.encode_query("điều 14"))
    failures: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(20):
                assert plain.tokenizer.count_tokens("điều 14 khoản 1") == expected_count
                assert_close(dense(plain.encode_query("điều 14")), expected_vector)
        except BaseException as error:  # noqa: BLE001 (collected and re-raised below)
            failures.append(error)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


# ============================================================ the identity


def test_identity_names_the_model_and_the_version(plain: HuggingFaceEmbedder) -> None:
    prefix = f"{TINY_REFERENCE.hub_model_id}@{TINY_REFERENCE.commit_hash}/"
    assert plain.identity.startswith(prefix)
    assert plain.identity.endswith(f"/v{EMBEDDER_VERSION}")
    assert len(plain.identity.removeprefix(prefix).split("/")[0]) == 12


def test_identity_is_embedder_identity_of_the_settings(
    plain: HuggingFaceEmbedder,
) -> None:
    assert plain.identity == embedder_identity(TINY_REFERENCE, plain.settings)


def test_the_same_settings_give_the_same_identity() -> None:
    assert embedder_identity(TINY_REFERENCE, make_settings()) == embedder_identity(
        TINY_REFERENCE, make_settings()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"query_prompt_text": "query: "},
        {"passage_prompt_text": "passage: "},
        {"model_max_tokens": 256},
        {"model_max_tokens": None},
        {"sparse_head": "bge-m3-lexical"},
    ],
)
def test_each_setting_changes_the_identity(change: dict[str, object]) -> None:
    assert embedder_identity(
        TINY_REFERENCE, make_settings(**change)
    ) != embedder_identity(TINY_REFERENCE, make_settings())


def test_another_commit_changes_the_identity() -> None:
    other = ModelReference(
        hub_model_id=TINY_REFERENCE.hub_model_id, commit_hash="f" * 40
    )
    assert embedder_identity(other, make_settings()) != embedder_identity(
        TINY_REFERENCE, make_settings()
    )


def test_prompt_texts_with_newlines_and_vietnamese_are_hashed_stably() -> None:
    settings = make_settings(
        query_prompt_text="Instruct: Tìm điều khoản liên quan.\nQuery: "
    )
    assert embedder_identity(TINY_REFERENCE, settings) == embedder_identity(
        TINY_REFERENCE, settings
    )
    assert "\n" not in embedder_identity(TINY_REFERENCE, settings)


def test_naming_a_prompt_or_finding_it_by_convention_is_the_same_identity(
    e5_folder: Path, e5: HuggingFaceEmbedder
) -> None:
    named = HuggingFaceModelSpec(
        reference=TINY_REFERENCE,
        query_prompt_name="query",
        passage_prompt_name="passage",
    )
    assert (
        HuggingFaceEmbedder(spec=named, folder=e5_folder, device="cpu").identity
        == e5.identity
    )


def test_models_with_different_prompts_have_different_identities(
    plain: HuggingFaceEmbedder, e5: HuggingFaceEmbedder
) -> None:
    assert plain.identity != e5.identity


def test_batch_size_is_not_part_of_the_identity(
    plain_folder: Path, plain: HuggingFaceEmbedder
) -> None:
    other = HuggingFaceEmbedder(
        spec=SPEC, folder=plain_folder, device="cpu", batch_size=1
    )
    assert other.identity == plain.identity


def test_precision_of_the_checkpoint_is_not_part_of_the_identity(
    tmp_path: Path, plain: HuggingFaceEmbedder
) -> None:
    folder = build_tiny_sentence_transformer(tmp_path / "bf16", save_as_bfloat16=True)
    assert (
        HuggingFaceEmbedder(spec=SPEC, folder=folder, device="cpu").identity
        == plain.identity
    )


# ============================================================ the folder


def test_a_missing_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="^no model folder at"):
        HuggingFaceEmbedder(spec=SPEC, folder=tmp_path / "nowhere")


def test_another_model_is_refused(plain_folder: Path) -> None:
    other = HuggingFaceModelSpec(
        reference=ModelReference(hub_model_id="BAAI/bge-m3", commit_hash="0" * 40)
    )
    with pytest.raises(ValueError, match="but the spec wants BAAI/bge-m3@"):
        HuggingFaceEmbedder(spec=other, folder=plain_folder)


def test_a_folder_without_modules_json_is_refused(tmp_path: Path) -> None:
    # sentence-transformers would load it as mean pooling, silently.
    folder = build_tiny_model(tmp_path / "plain", sparse_head=False)
    with pytest.raises(FileNotFoundError, match="has no modules.json"):
        HuggingFaceEmbedder(spec=SPEC, folder=folder)


def test_a_folder_without_a_reference_is_refused(tmp_path: Path) -> None:
    folder = build_tiny_sentence_transformer(tmp_path / "plain", reference=None)
    with pytest.raises(FileNotFoundError, match="cannot find model_reference.json"):
        HuggingFaceEmbedder(spec=SPEC, folder=folder)


def test_accepts_a_string_folder(plain_folder: Path) -> None:
    assert (
        HuggingFaceEmbedder(
            spec=SPEC, folder=str(plain_folder), device="cpu"
        ).vector_specs
        == plain_specs()
    )


# ============================================================ device and batch size


def test_cpu_is_cpu(plain: HuggingFaceEmbedder) -> None:
    assert plain.device == "cpu"


@pytest.mark.skipif(
    torch.cuda.is_available() or torch.backends.mps.is_available(),
    reason="a GPU is present",
)
def test_auto_without_a_gpu_is_cpu(plain_folder: Path) -> None:
    assert HuggingFaceEmbedder(spec=SPEC, folder=plain_folder).device == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="a CUDA GPU is present")
def test_cuda_without_a_gpu_is_refused(plain_folder: Path) -> None:
    with pytest.raises(
        ValueError,
        match="^device 'cuda' was asked for, but PyTorch found no CUDA-compatible GPU",
    ):
        HuggingFaceEmbedder(spec=SPEC, folder=plain_folder, device="cuda")


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="an MPS GPU is present")
def test_mps_without_a_gpu_is_refused(plain_folder: Path) -> None:
    with pytest.raises(
        ValueError,
        match="^device 'mps' was asked for, but PyTorch found no MPS-compatible GPU",
    ):
        HuggingFaceEmbedder(spec=SPEC, folder=plain_folder, device="mps")


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_refused(
    plain_folder: Path, batch_size: int
) -> None:
    with pytest.raises(
        ValueError, match=f"^batch_size must be positive, got {batch_size}$"
    ):
        HuggingFaceEmbedder(spec=SPEC, folder=plain_folder, batch_size=batch_size)
