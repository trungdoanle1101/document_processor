from collections.abc import Iterable

from document_processor.core.text_utils import is_invisible


def require_visible(owner: str, name: str, value: str) -> None:
    if is_invisible(value):
        raise ValueError(f"{owner}: {name!r} cannot be empty or invisible")
    

def require_visible_or_none(owner: str, name: str, value: str | None) -> None:
    if value is not None:
        require_visible(owner, name, value)


def require_unique(owner: str, name: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"{owner}: duplicate value {v!r} in {name!r}")
        seen.add(v)


def require_range(
        owner: str, start: int, end: int, lower: int = 0, upper: int | None=None,
        start_label: str = "start", end_label: str = "end"
) -> None:
    if not ((lower <= start <= end) and (upper is None or end <= upper)):
        bound = "" if upper is None else f" <= {upper}"
        raise ValueError(f"{owner}: required {lower} <= {start_label} <= {end_label}{bound}, got {start_label}={start}, {end_label}={end}") 
