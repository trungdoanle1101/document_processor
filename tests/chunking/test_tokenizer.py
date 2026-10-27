"""Tests for chunking/tokenizer.py: the Tokenizer protocol and WordTokenizer.

A tokenizer counts tokens and states its model's hard input limit
(model_max_tokens), None when there is no fixed limit. WordTokenizer counts
whitespace-separated words and has no limit.
"""

from dataclasses import dataclass

import pytest

from document_processor.chunking.tokenizer import Tokenizer, WordTokenizer


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("", 0),
        ("   ", 0),
        ("Điều", 1),
        ("Điều 9. Phạm vi điều chỉnh", 6),  # Vietnamese words are space-separated syllables
        ("  a\tb\nc  ", 3),  # any whitespace separates, edges ignored
        ("STT | Nội dung", 4),  # punctuation standing alone counts as a word
    ],
)
def test_word_tokenizer_counts_words(text: str, count: int) -> None:
    assert WordTokenizer().count_tokens(text) == count


def test_any_object_with_count_tokens_and_a_limit_is_a_tokenizer() -> None:
    # The protocol is structural: an embedding model's tokenizer only needs these two.
    class CharCount:
        def count_tokens(self, text: str) -> int:
            return len(text)

        @property
        def model_max_tokens(self) -> int | None:
            return 512

    tokenizer: Tokenizer = CharCount()
    assert (tokenizer.count_tokens("abc"), tokenizer.model_max_tokens) == (3, 512)


def test_a_frozen_dataclass_field_satisfies_the_limit() -> None:
    @dataclass(frozen=True)
    class Fixed:
        model_max_tokens: int | None

        def count_tokens(self, text: str) -> int:
            return 1

    tokenizer: Tokenizer = Fixed(model_max_tokens=8192)
    assert tokenizer.model_max_tokens == 8192


def test_word_tokenizer_has_no_limit() -> None:
    assert WordTokenizer().model_max_tokens is None


def test_word_tokenizer_is_a_tokenizer() -> None:
    tokenizer: Tokenizer = WordTokenizer()
    assert tokenizer.count_tokens("một hai") == 2