from collections.abc import Iterable
from typing import Final

from qdrant_client import QdrantClient, models

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.index.protocol import check_batch
from salemdex_retrieval.index.qdrant.payload import encode_payload
from salemdex_retrieval.schema import Schema


class QdrantIndex:
    def __init__(
        self, *, schema: type[Schema], client: QdrantClient, collection: str
    ) -> None:
        self.schema: Final = schema
        self._client = client
        self._collection = collection

    def put(self, chunks: Iterable[Chunk]) -> None:
        chunks = check_batch(chunks, schema=self.schema)
        points = [
            models.PointStruct(id=chunk.id, payload=encode_payload(chunk), vector={})
            for chunk in chunks
        ]

        self._client.upload_points(
            collection_name=self._collection, points=points, wait=True
        )
