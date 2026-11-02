import argparse
import time
from pathlib import Path

from qdrant_client import QdrantClient

from document_processor.chunking.outline_chunker import OutlineChunker
from document_processor.integrations.huggingface.download import download_model
from document_processor.integrations.huggingface.embedder import HuggingFaceEmbedder
from document_processor.integrations.huggingface.model_specs import BGE_M3
from document_processor.integrations.qdrant.index import QdrantChunkIndex
from document_processor.pipeline.pipeline import Pipeline, PipelineResult
from document_processor.retrieval.attributes import TextIn
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkFilter, ChunkIndex
from document_processor.retrieval.ingest import index_document
from document_processor.retrieval.search import RetrievedChunk, search_chunks

MODEL_FOLDER = "huggingface/BAAI/bge_m3"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "architecture-bge-m3"
MAX_TOKENS = 512
LIMIT = 5


def open_embedder() -> HuggingFaceEmbedder:
    if not Path(MODEL_FOLDER).exists():
        print("Download start...")
        download_model(BGE_M3.reference, MODEL_FOLDER)
        print("Download completed")

    return HuggingFaceEmbedder(spec=BGE_M3, folder=MODEL_FOLDER)


def open_index(embedder: Embedder, *, recreate: bool) -> QdrantChunkIndex:
    client = QdrantClient(url=QDRANT_URL)
    if recreate:
        client.delete_collection(collection_name=COLLECTION)
    return QdrantChunkIndex(
        client=client,
        collection=COLLECTION,
        embedder_identity=embedder.identity,
        vector_specs=embedder.vector_specs,
        indexed_attributes={"part": "text"},
    )


def index_folder(folder: Path, embedder: HuggingFaceEmbedder, index: ChunkIndex) -> None:
    chunker = OutlineChunker(
        tokenizer=embedder.tokenizer,
        max_tokens=MAX_TOKENS,
        include_breadcrumb_in_size=True,
        keep_sections_apart=True,
    )
    pipeline = Pipeline(profile="vn_regulation", chunker=chunker)
    for outcome in pipeline.run_batch(sorted(folder.glob("*.docx"))):
        if isinstance(outcome, PipelineResult):
            print(outcome.summarize().format())
            start = time.perf_counter()
            index_document(outcome.chunks, document_id=outcome.source.stem, embedder=embedder, index=index)
            stop = time.perf_counter()
            execution_time = stop - start
            print(f"Indexing {outcome.source.stem!r} took {execution_time:.1f} seconds")
        else:
            print(f"{outcome.source}: {outcome.error}")

def ask_questions(embedder: Embedder, index: ChunkIndex) -> None:
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            return
        hits = search_chunks(question, embedder=embedder, index=index, limit=LIMIT,
                             chunk_filter=ChunkFilter(attributes={"part": TextIn(values=("body",))}))
        print_hits(hits)
    

def print_hits(hits: list[RetrievedChunk]) -> None:
    for pos, hit in enumerate(hits, start=1):
        text = hit.chunk.text
        print(f"{pos} {hit.index_score:.3f} {hit.chunk.document_id}")
        print(f"  {" > ".join(hit.chunk.headings)}")
        print(f"  Text: {" ".join(text.split())[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index files and ask questions about them.")
    commands = parser.add_subparsers(dest="command", required=True)
    index_command = commands.add_parser("index", help="parse, chunk, embed, and store every .docx in a folder")
    index_command.add_argument("folder", type=Path)
    index_command.add_argument("--recreate", action="store_true", help="delete the collection first")
    commands.add_parser("ask", help="ask questions; a blank line quits")
    args = parser.parse_args()

    embedder = open_embedder()
    print("Embedder loaded")
    index = open_index(embedder, recreate=args.command == "index" and args.recreate)
    if args.command == "index":
        index_folder(args.folder, embedder, index)
    else:
        ask_questions(embedder, index)


if __name__ == "__main__":
    main()