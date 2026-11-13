"""The Qdrant backend. Needs the `qdrant` extra: `salemdex-retrieval[qdrant]`."""

from salemdex_retrieval.index.qdrant.index import QdrantIndex

__all__ = ["QdrantIndex"]
