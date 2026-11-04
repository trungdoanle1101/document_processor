"""Who may read what: every chunk carries its document's access group, and every search filters on it."""

from collections.abc import Sequence

from document_processor.retrieval.attributes import Condition, TextIn
from document_processor.retrieval.index import ChunkFilter

ACCESS_GROUP = "access_group"  # the chunk attribute naming who may read it
PART = "part"  # set by ingest: front_matter, body or end_matter


def readable_by(groups: Sequence[str], *, body_only: bool = True) -> ChunkFilter:
    """The chunks a reader in these groups may see; body_only leaves out front and end matter."""
    if not groups:
        raise ValueError("a reader needs at least one access group")
    conditions: dict[str, Condition] = {ACCESS_GROUP: TextIn(values=tuple(groups))}
    if body_only:
        conditions[PART] = TextIn(values=("body",))
    return ChunkFilter(attributes=conditions)
