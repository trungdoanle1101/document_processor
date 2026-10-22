from typing import Protocol


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


class WordTokenizer:
    """Counts whitespace-separated words.
    Used mostly for tests and quick runs, not for production budgets.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())
