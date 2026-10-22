"""Tests for chunking/tokenizer.py: the Tokenizer protocol and WordTokenizer."""

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


def test_any_object_with_count_tokens_is_a_tokenizer() -> None:
    # The protocol is structural: an embedding model's tokenizer only needs count_tokens.
    class CharCount:
        def count_tokens(self, text: str) -> int:
            return len(text)

    tokenizer: Tokenizer = CharCount()
    assert tokenizer.count_tokens("abc") == 3


def test_word_tokenizer_is_a_tokenizer() -> None:
    tokenizer: Tokenizer = WordTokenizer()
    assert tokenizer.count_tokens("một hai") == 2