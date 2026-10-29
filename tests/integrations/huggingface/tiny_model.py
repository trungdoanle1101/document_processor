"""Tiny model folders, laid out like a downloaded bge-m3, for tests.

A two-layer XLM-RoBERTa with hidden size 16 and a word-level vocabulary of a
few Vietnamese words. Its vectors mean nothing, but it runs through the same
loading code as the real model, in milliseconds. Weights come from a fixed
seed, so two folders built alike are identical.

- build_tiny_tokenizer_folder: only the tokenizer files and model_reference.json.
- build_tiny_model: a plain transformers folder (no modules.json), optionally
  with sparse_linear.pt.
- build_tiny_sentence_transformer: the folder in sentence-transformers'
  format, as the embedder loads it: CLS pooling, normalization, optional
  prompts.
"""

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from document_processor.integrations.huggingface.reference import ModelReference, write_model_reference

TINY_HUB_MODEL_ID = "test/tiny-xlmr"
TINY_COMMIT = "0123456789abcdef0123456789abcdef01234567"
TINY_REFERENCE = ModelReference(hub_model_id=TINY_HUB_MODEL_ID, commit_hash=TINY_COMMIT)
TINY_HIDDEN_SIZE = 16
TINY_TOKENIZER_LIMIT = 48

# XLM-RoBERTa's special tokens and ids: <s>=0, <pad>=1, </s>=2, <unk>=3.
VOCAB = {
    "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
    "điều": 4, "khoản": 5, "phạm": 6, "vi": 7, "điều_chỉnh": 8, "hệ": 9, "thống": 10,
    "dữ": 11, "liệu": 12, "quy": 13, "định": 14, "query:": 15, "passage:": 16, "1": 17, "14": 18,
}
SPECIAL_IDS = frozenset({0, 1, 2, 3})


def tiny_tokenizer(model_max_length: int | None) -> PreTrainedTokenizerFast:
    """Word-level, splitting on spaces, wrapping every text in <s> … </s>."""
    backend = BackendTokenizer(models.WordLevel(vocab=VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        bos_token="<s>", eos_token="</s>", cls_token="<s>", sep_token="</s>",
        pad_token="<pad>", unk_token="<unk>",
    )
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length
    return tokenizer


def build_tiny_tokenizer_folder(
    folder: Path,
    *,
    tokenizer_max_length: int | None = TINY_TOKENIZER_LIMIT,
    reference: ModelReference | None = TINY_REFERENCE,
) -> Path:
    """The tokenizer files and model_reference.json; None leaves the limit or the reference out."""
    folder.mkdir(parents=True, exist_ok=True)
    tiny_tokenizer(tokenizer_max_length).save_pretrained(folder)
    if reference is not None:
        write_model_reference(folder, reference)
    return folder


def build_tiny_model(
    folder: Path,
    *,
    sparse_head: bool = True,
    sparse_head_size: int = TINY_HIDDEN_SIZE,
    tokenizer_max_length: int | None = TINY_TOKENIZER_LIMIT,
    reference: ModelReference | None = TINY_REFERENCE,
    seed: int = 0,
) -> Path:
    """A whole tiny model folder; each argument breaks or changes one part of it. Needs torch."""
    import torch
    from transformers import XLMRobertaConfig, XLMRobertaModel

    build_tiny_tokenizer_folder(folder, tokenizer_max_length=tokenizer_max_length, reference=reference)
    torch.manual_seed(seed)
    config = XLMRobertaConfig(
        vocab_size=len(VOCAB),
        hidden_size=TINY_HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=1, bos_token_id=0, eos_token_id=2,
    )
    XLMRobertaModel(config).save_pretrained(folder)  # type: ignore[no-untyped-call]
    if sparse_head:
        head = torch.nn.Linear(sparse_head_size, 1)
        torch.save(head.state_dict(), folder / "sparse_linear.pt")
    return folder


def build_tiny_sentence_transformer(
    folder: Path,
    *,
    prompts: Mapping[str, str] | None = None,
    max_seq_length: int = TINY_TOKENIZER_LIMIT,
    reference: ModelReference | None = TINY_REFERENCE,
    save_as_bfloat16: bool = False,
    nan_weights: bool = False,
    seed: int = 0,
) -> Path:
    """A tiny model saved in sentence-transformers' format: CLS pooling, then normalization.

    Laid out like bge-m3's folder (modules.json, 1_Pooling/, 2_Normalize/, the
    transformer at the root) plus model_reference.json. Needs torch and
    sentence-transformers. prompts=None declares none.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer

    with tempfile.TemporaryDirectory() as scratch:
        plain = build_tiny_model(Path(scratch) / "plain", sparse_head=False, reference=None, seed=seed)
        transformer = Transformer(str(plain), max_seq_length=max_seq_length)
        pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="cls")
        # device="cpu": left out, sentence-transformers would pick a GPU (CUDA or Apple's MPS) if present.
        model = SentenceTransformer(
            modules=[transformer, pooling, Normalize()], prompts=dict(prompts or {}), device="cpu"
        )
        if nan_weights:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(float("nan"))
        if save_as_bfloat16:
            model.to(torch.bfloat16)
        shutil.rmtree(folder, ignore_errors=True)
        # No model card: generating one imports sentence-transformers' training and
        # hub machinery (model lookups, CodeCarbon), which tests never need.
        model.save(str(folder), create_model_card=False)
    if reference is not None:
        write_model_reference(folder, reference)
    return folder