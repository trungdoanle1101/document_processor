from collections.abc import Sequence
from types import MappingProxyType

from document_processor.chunking import Chunk
from document_processor.retrieval.attributes import Attributes
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkIndex, IndexedChunk
from document_processor.retrieval.reference import address_keys


def to_indexed_chunks(
    chunks: Sequence[Chunk],
    *,
    document_id: str,
    attributes: Attributes = MappingProxyType({}),
) -> list[IndexedChunk]:
    """The chunker's chunks as the index stores them, in document order"""
    if "part" in attributes:
        raise ValueError(
            'attribute name "part" is reserved: ingest sets it from each chunk; use a different name if needed'
        )

    return [
        IndexedChunk(
            document_id=document_id,
            chunk_id=chunk.id,
            position=position,
            text=chunk.text,
            embedded_text=chunk.contextualize(),
            headings=chunk.headings,
            reference_keys=address_keys(chunk.path),
            attributes={**attributes, "part": chunk.part},
        )
        for position, chunk in enumerate(chunks)
    ]


def index_document(
    chunks: Sequence[Chunk],
    *,
    document_id: str,
    embedder: Embedder,
    index: ChunkIndex,
    attributes: Attributes = MappingProxyType({}),
) -> None:
    """Embed the chunks and make them the document's only chunks in the index"""
    indexed_chunks = to_indexed_chunks(
        chunks, document_id=document_id, attributes=attributes
    )
    embeddings = embedder.encode_passages(
        [chunk.embedded_text for chunk in indexed_chunks]
    )
    index.replace_document(document_id, indexed_chunks, embeddings)
