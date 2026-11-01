from collections.abc import Mapping
from document_processor.retrieval.index import ChunkIndex
from document_processor.retrieval.embedding import VectorSpec
from qdrant_client import QdrantClient


class QdrantChunkIndex(ChunkIndex):
    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str,
        embedder_identity: str,
        vector_specs: Mapping[str, VectorSpec],
        candidates_per_vector: int = 50,
    ):
        pass
