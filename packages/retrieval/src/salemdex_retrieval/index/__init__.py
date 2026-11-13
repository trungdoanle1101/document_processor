"""Indexes: where chunks are stored and found.

Backends needing an optional extra (Qdrant) aren't imported here; import them from
their own module.
"""

from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.index.protocol import Index

__all__ = ["InMemoryIndex", "Index"]
