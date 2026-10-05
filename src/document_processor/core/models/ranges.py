from dataclasses import dataclass


@dataclass(frozen=True)
class TextRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if (self.start < 0) or (self.start >= self.end):
            raise ValueError(
                f"Invalid range definition. Required: 0 <= start < end. Got start: {self.start}, end: {self.end}."
            )
