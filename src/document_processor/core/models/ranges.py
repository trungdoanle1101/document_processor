from dataclasses import dataclass


@dataclass(frozen=True)
class TextRange:
    """Half open [start, end) offsets in code points to block's text as stored"""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (self.start < 0) or (self.start >= self.end):
            raise ValueError(
                f"Invalid range definition. Required: 0 <= start < end. Got start: {self.start}, end: {self.end}."
            )

    def slice(self, text: str) -> str:
        if self.end > len(text):
            raise ValueError(
                f"Invalid TextRange applied on text: end {self.end} > len {len(text)}"
            )
        return text[self.start : self.end]
