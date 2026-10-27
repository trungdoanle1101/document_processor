from typing import Protocol


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...

    @property
    def model_max_tokens(self) -> int | None:
        """The model's hard input limit in tokens; None when there is no fixed limit"""
        ...


class WordTokenizer:
    """Counts whitespace-separated words.
    Used mostly for tests and quick runs, not for production budgets.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())
    
    @property
    def model_max_tokens(self) -> int | None:
        return None
    
