from collections.abc import Sequence

from document_processor.chunking import PathStep


def address_keys(path: Sequence[PathStep]) -> tuple[str, ...]:
    """Every way the nodes on this path can be cited, e.g. "dieu:14/khoan:2": each ancestor's
    full address and all its shorter endings. Stops at the first step without a number."""
    parts: list[str] = []
    keys: dict[str, None] = {}
    for step in path:
        if step.number is None:
            break
        parts.append(f"{step.level}:{step.number.casefold()}")
        for i in range(len(parts)):
            keys["/".join(parts[i:])] = None

    return tuple(keys)
