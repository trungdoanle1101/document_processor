"""scripts/smoke_test.py, run against the app and a worker with a fake embedder.

The index is the real QdrantChunkIndex on Qdrant's in-memory mode, so the script's Qdrant
queries meet the payload the worker really writes. Instead of waiting, the script's polls run
the worker until it's idle.
"""

import importlib.util
import sys
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from document_processor.integrations.qdrant.index import QdrantChunkIndex
from kb_server.domain.users import User
from kb_server.indexing.handlers import IndexTools, JobTools
from kb_server.indexing.worker import Worker
from kb_server.storage.files import FileStore
from kb_server.storage.models import IndexingJobRow
from kb_server.wiring import QDRANT_INDEXED_ATTRIBUTES, Services
from tests.indexing.test_handlers import (
    INDEX_LABEL,
    REGULATION,
    CountingEmbedder,
    make_chooser,
)
from tests.indexing.test_worker import UnreachableEmbedder
from tests.support import PASSWORD

SCRIPT = Path(__file__).parents[1] / "scripts" / "smoke_test.py"
QDRANT_COLLECTION = "kb_chunks"
ADMIN_EMAIL = "admin@example.com"
# the waits run the worker, so a check that never passes fails within seconds
TIMEOUT_SECONDS = 5.0

# in-memory Qdrant warns that payload indexes do nothing there; the server ones are real
pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

MakeWorker = Callable[..., Worker]
RunSmokeTest = Callable[..., None]


@pytest.fixture
def smoke(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script, imported from its file: scripts/ isn't a package."""
    spec = importlib.util.spec_from_file_location("smoke_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "smoke_test", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def make_worker(
    services: Services, file_store: FileStore, qdrant: QdrantClient
) -> MakeWorker:
    """make_worker(embedder=…, max_attempts=…): a worker over the test's rows and Qdrant."""

    def make(
        *, embedder: CountingEmbedder | None = None, max_attempts: int = 3
    ) -> Worker:
        embedder = embedder or CountingEmbedder()
        index = QdrantChunkIndex(
            qdrant,
            collection=QDRANT_COLLECTION,
            embedder_identity=embedder.identity,
            vector_specs=embedder.vector_specs,
            indexed_attributes=QDRANT_INDEXED_ATTRIBUTES,
        )
        return Worker(
            session_factory=services.session_factory,
            tools=JobTools(
                profile_chooser=make_chooser(),
                index_tools=IndexTools(
                    embedder=embedder, index=index, index_label=INDEX_LABEL
                ),
                file_store=file_store,
            ),
            max_attempts=max_attempts,
            poll_interval=timedelta(seconds=1),
            job_timeout=timedelta(hours=1),
            now=services.clock,
        )

    return make


@pytest.fixture
def http(
    client: TestClient, make_user: Callable[..., User], smoke: ModuleType
) -> TestClient:
    """The test client at the API's base, logged in as an admin, as the script does."""
    make_user(ADMIN_EMAIL, role="admin")
    client.base_url = client.base_url.join(smoke.API)
    smoke._log_in(client, email=ADMIN_EMAIL, password=PASSWORD)
    return client


@pytest.fixture
def run_smoke_test(
    smoke: ModuleType,
    http: TestClient,
    qdrant: QdrantClient,
    file_store_root: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> RunSmokeTest:
    """run_smoke_test(file, worker=…, keep=…): the script's run, its waits running the worker."""

    def run(file: Path, *, worker: Worker, keep: bool = False) -> None:
        monkeypatch.setattr(
            smoke.time, "sleep", _running_until_idle(worker, session_factory)
        )
        smoke.run_smoke_test(
            file,
            collection_slug="smoke-test",
            timeout_seconds=TIMEOUT_SECONDS,
            keep=keep,
            http=http,
            probes=smoke.Probes(
                qdrant=qdrant,
                qdrant_collection=QDRANT_COLLECTION,
                store_root=file_store_root,
            ),
        )

    return run


@pytest.fixture
def regulation(tmp_path: Path) -> Path:
    file = tmp_path / "quy-dinh.txt"
    file.write_text(REGULATION, encoding="utf-8")
    return file


def test_the_smoke_test_passes_against_the_app_and_a_worker(
    run_smoke_test: RunSmokeTest,
    make_worker: MakeWorker,
    regulation: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_smoke_test(regulation, worker=make_worker())

    output = capsys.readouterr().out
    for line in (
        "prepared and indexed",
        "points in Qdrant",
        "prepared output saved",
        "the document reads as gone at once",
        "chunks and prepared output removed",
        "original removed",
    ):
        assert line in output
    assert "Điều" not in output  # ids and counts only, never document text


def test_a_failed_job_is_reported_by_its_type_and_the_document_cleaned_up(
    run_smoke_test: RunSmokeTest,
    make_worker: MakeWorker,
    smoke: ModuleType,
    http: TestClient,
    regulation: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker = make_worker(
        embedder=UnreachableEmbedder("Điều 1 couldn't be embedded"), max_attempts=1
    )

    with pytest.raises(smoke.SmokeTestFailed) as failed:
        run_smoke_test(regulation, worker=worker)

    message = str(failed.value)
    assert "the index job" in message
    assert "failed with ConnectionError" in message
    assert "Điều" not in message  # the type only: the message may quote the document
    document_id = _uploaded_id(capsys.readouterr().out)
    assert http.get(f"/documents/{document_id}").status_code == 404


def test_a_file_already_in_the_collection_is_refused_naming_its_document(
    run_smoke_test: RunSmokeTest,
    make_worker: MakeWorker,
    smoke: ModuleType,
    regulation: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker = make_worker()
    run_smoke_test(regulation, worker=worker, keep=True)
    document_id = _uploaded_id(capsys.readouterr().out)

    with pytest.raises(smoke.SmokeTestFailed, match=f"already document {document_id}"):
        run_smoke_test(regulation, worker=worker, keep=True)


def _running_until_idle(
    worker: Worker, session_factory: sessionmaker[Session]
) -> Callable[[float], None]:
    """A stand-in for time.sleep that runs the worker until no job is left.

    The jobs the worker queued are then dated after the ones before them. Why: a test is one
    transaction, so now() gives every job the same created_at, and the script reads the
    document's newest job to tell when it's indexed.
    """
    rounds = 0

    def run_until_idle(seconds: float) -> None:
        nonlocal rounds
        rounds += 1
        with session_factory.begin() as session:
            before = set(session.scalars(select(IndexingJobRow.id)))
        while worker.run_once():
            pass
        with session_factory.begin() as session:
            session.execute(
                update(IndexingJobRow)
                .where(IndexingJobRow.id.not_in(before))
                .values(
                    created_at=IndexingJobRow.created_at + timedelta(minutes=rounds)
                )
            )

    return run_until_idle


def _uploaded_id(output: str) -> str:
    """The document id from the script's "uploaded document <id>" line."""
    after: Any = output.split("uploaded document ")[1]
    return str(after.split()[0])
