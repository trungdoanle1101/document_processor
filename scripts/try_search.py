import argparse
import statistics
import time
from functools import partial
from pathlib import Path

from qdrant_client import QdrantClient

from document_processor.chunking.outline_chunker import OutlineChunker
from document_processor.integrations.huggingface.download import (
    download_model,
    model_folder,
)
from document_processor.integrations.huggingface.embedder import HuggingFaceEmbedder
from document_processor.integrations.huggingface.model_specs import (
    BGE_M3,
    BGE_RERANKER_V2_M3,
)
from document_processor.integrations.huggingface.reranker import HuggingFaceReranker
from document_processor.integrations.qdrant.index import QdrantChunkIndex
from document_processor.preparation.preparer import PreparedDocument, Preparer
from document_processor.retrieval.attributes import TextIn
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.evaluation import (
    MATCH_LEVELS,
    EvaluationReport,
    MatchLevel,
    evaluate,
    format_summary,
    load_questions,
)
from document_processor.retrieval.index import ChunkFilter, ChunkIndex
from document_processor.retrieval.ingest import index_document
from document_processor.retrieval.reranker import Reranker
from document_processor.retrieval.search import RetrievedChunk, search_chunks

MODELS_FOLDER = "huggingface"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "architecture-bge-m3"
MAX_TOKENS = 512
LIMIT = 5
RERANK_DEPTH = 20  # hits the reranker sees before the top LIMIT are kept
BODY_ONLY = ChunkFilter(attributes={"part": TextIn(values=("body",))})


def open_embedder() -> HuggingFaceEmbedder:
    resolved_folder = model_folder(BGE_M3.reference, models_folder=MODELS_FOLDER)
    if not Path(resolved_folder).exists():
        print("Download start...")
        download_model(BGE_M3.reference, resolved_folder)
        print("Download completed")

    return HuggingFaceEmbedder(spec=BGE_M3, folder=resolved_folder, device="cpu")


def open_reranker() -> HuggingFaceReranker:
    resolved_folder = model_folder(
        BGE_RERANKER_V2_M3.reference, models_folder=MODELS_FOLDER
    )
    if not Path(resolved_folder).exists():
        print("Reranker download start...")
        download_model(BGE_RERANKER_V2_M3.reference, resolved_folder)
        print("Reranker download completed")

    return HuggingFaceReranker(
        spec=BGE_RERANKER_V2_M3, folder=resolved_folder, device="auto"
    )


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


def index_folder(
    folder: Path, embedder: HuggingFaceEmbedder, index: ChunkIndex
) -> None:
    chunker = OutlineChunker(
        tokenizer=embedder.tokenizer,
        max_tokens=MAX_TOKENS,
        include_breadcrumb_in_size=True,
        keep_sections_apart=True,
    )
    preparer = Preparer(profile="vn_regulation", chunker=chunker)
    for outcome in preparer.prepare_many(sorted(folder.glob("*.docx"))):
        if isinstance(outcome, PreparedDocument):
            print(outcome.summarize().format())
            start = time.perf_counter()
            index_document(
                outcome.chunks,
                document_id=outcome.source.stem,
                embedder=embedder,
                index=index,
            )
            stop = time.perf_counter()
            execution_time = stop - start
            print(f"Indexing {outcome.source.stem!r} took {execution_time:.1f} seconds")
        else:
            print(f"{outcome.source}: {outcome.error}")


def ask_questions(
    embedder: Embedder,
    index: ChunkIndex,
    *,
    reranker: Reranker | None,
    rerank_depth: int,
) -> None:
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            return
        hits = search_chunks(
            question,
            embedder=embedder,
            index=index,
            limit=LIMIT,
            chunk_filter=BODY_ONLY,
            reranker=reranker,
            rerank_depth=rerank_depth,
        )
        print_hits(hits)


def print_hits(hits: list[RetrievedChunk]) -> None:
    for pos, hit in enumerate(hits, start=1):
        lines = hit.chunk.text.splitlines()
        rerank = "" if hit.rerank_score is None else f" rerank {hit.rerank_score:.3f}"
        print(f"{pos} index {hit.index_score:.3f}{rerank} {hit.chunk.document_id}")
        print(f"  {' > '.join(hit.chunk.headings)}")
        for line in lines[:6]:
            print(f"    {line[:160]}")
        if len(lines) > 6:
            print("    …")
        print()
    print("---------------------------------------------------")


def run_evaluation(
    questions_path: Path,
    embedder: Embedder,
    index: ChunkIndex,
    *,
    depth: int,
    misses_at: MatchLevel,
    reranker: Reranker | None,
    rerank_depth: int,
) -> None:
    questions = load_questions(questions_path)
    search = partial(
        search_chunks,
        embedder=embedder,
        index=index,
        chunk_filter=BODY_ONLY,
        reranker=reranker,
        rerank_depth=rerank_depth,
    )
    if reranker is None:
        print("no reranker")
    else:
        print(f"reranker {reranker.identity}, over the top {max(depth, rerank_depth)}")
    start = time.perf_counter()
    report = evaluate(questions, search, depth=depth)
    print(f"{len(questions)} questions in {time.perf_counter() - start:.1f} seconds\n")
    print(format_summary(report))
    print_misses(report, misses_at)
    print_top_scores(report)


def print_misses(report: EvaluationReport, level: MatchLevel) -> None:
    misses = report.misses(level=level)
    print(
        f"\n{len(misses)} missed at the {level} level (nothing matched in the top {report.depth}):"
    )
    for miss in misses:
        question = miss.question
        print(f"- {question.id} [{', '.join(question.tags)}] {question.question}")
        print(
            f"  expected: {question.location or ', '.join(a.reference_key or a.document_id for a in question.answers)}"
        )
        for rank, hit in enumerate(miss.hits[:3], start=1):
            print(
                f"  got {rank}: {' > '.join(hit.chunk.headings) or hit.chunk.document_id}"
            )


def print_top_scores(report: EvaluationReport) -> None:
    """If unanswerable questions score clearly lower, a threshold for "I don't know" is possible."""
    print("\nbest score per question:")
    for label, answerable in (("answerable", True), ("unanswerable", False)):
        scores = report.top_scores(answerable=answerable)
        if scores:
            print(
                f"  {label:12}  min {min(scores):.3f}  median {statistics.median(scores):.3f}"
                f"  max {max(scores):.3f}  ({len(scores)} questions)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index files and ask questions about them."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index_command = commands.add_parser(
        "index", help="parse, chunk, embed, and store every .docx in a folder"
    )
    index_command.add_argument("folder", type=Path)
    index_command.add_argument(
        "--recreate", action="store_true", help="delete the collection first"
    )
    ask_command = commands.add_parser("ask", help="ask questions; a blank line quits")
    evaluate_command = commands.add_parser(
        "evaluate", help="score search against a questions file"
    )
    evaluate_command.add_argument(
        "questions", type=Path, help="a .jsonl questions file"
    )
    evaluate_command.add_argument(
        "--depth", type=int, default=10, help="hits looked at per question"
    )
    evaluate_command.add_argument(
        "--misses-at",
        choices=MATCH_LEVELS,
        default="section",
        help="which level's misses to list",
    )
    for command in (ask_command, evaluate_command):
        command.add_argument(
            "--rerank",
            action="store_true",
            help="rerank the hits with the cross-encoder",
        )
        command.add_argument(
            "--rerank-depth",
            type=int,
            default=RERANK_DEPTH,
            help="how many hits the reranker sees",
        )
    args = parser.parse_args()

    embedder = open_embedder()
    print("Embedder loaded")
    index = open_index(embedder, recreate=args.command == "index" and args.recreate)
    reranker = open_reranker() if args.command != "index" and args.rerank else None
    if reranker is not None:
        print("Reranker loaded")
    if args.command == "index":
        index_folder(args.folder, embedder, index)
    elif args.command == "evaluate":
        run_evaluation(
            args.questions,
            embedder,
            index,
            depth=args.depth,
            misses_at=args.misses_at,
            reranker=reranker,
            rerank_depth=args.rerank_depth,
        )
    else:
        ask_questions(
            embedder, index, reranker=reranker, rerank_depth=args.rerank_depth
        )


if __name__ == "__main__":
    main()
