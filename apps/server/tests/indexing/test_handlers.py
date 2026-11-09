"""Choosing how to prepare a document, and the prepare job's handler.

No database: a ClaimedJob is built by hand, the store is a LocalFileStore in a temporary
folder, and chunks are sized in words, so no model is loaded.
"""

import hashlib
import io
import json
import logging
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import docx
import pytest

from document_processor.chunking import Chunk, OutlineChunker, WordTokenizer
from document_processor.core.models import LogicalDocument
from document_processor.logical.profiles import PROFILES
from kb_server.domain.documents import Document, DocumentStatus, DocumentVersion
from kb_server.domain.files import DOCX, TEXT, StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.handlers import (
    ClaimedJob,
    PermanentJobError,
    ProfileChooser,
    run_prepare_job,
)
from kb_server.indexing.prepared import load_prepared
from kb_server.storage.files import (
    FileStore,
    LocalFileStore,
    original_key,
    prepared_version_folder,
)

NOW = datetime(2026, 11, 9, 9, 0, tzinfo=UTC)

REGULATION = """QUY ĐỊNH QUẢN LÝ DỮ LIỆU
Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy định này áp dụng cho dữ liệu khách hàng tại ngân hàng.
2. Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày.
Điều 2. Thời hạn lưu trữ
Dữ liệu lưu trữ dài hạn được lưu trên Tape.
Nơi nhận:
- Như trên
"""

PLAIN_TEXT = """Ghi chú cuộc họp về kế hoạch sao lưu.
Nhóm vận hành sẽ kiểm tra lịch sao lưu hằng tuần.
Mọi thay đổi cần được báo trước cho bộ phận an ninh thông tin.
"""


def label(name: str) -> str:
    profile = PROFILES.get(name)
    return f"{profile.name}@{profile.version}"


VN_REGULATION = label("vn_regulation")
PLAIN = label("plain")


def make_chooser(
    profiles: tuple[str, ...] = ("vn_regulation",), fallback: str = "plain"
) -> ProfileChooser:
    return ProfileChooser.from_profiles(
        profiles=profiles,
        fallback_profile=fallback,
        chunker=OutlineChunker(tokenizer=WordTokenizer(), max_tokens=50),
        embedder_identity="words",
    )


def docx_of(text: str) -> bytes:
    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def damaged_docx() -> bytes:
    # passes the upload check (a zip with word/document.xml) but isn't a Word file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<x/>")
        archive.writestr("word/document.xml", "<x/>")
    return buffer.getvalue()


def text_bytes(text: str) -> bytes:
    return text.encode("utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture
def store(root: Path) -> FileStore:
    return LocalFileStore(root)


@pytest.fixture
def chooser() -> ProfileChooser:
    return make_chooser()


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where the handler's temporary folders go, so tests can check they're cleaned up."""
    folder = tmp_path / "temp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


def stage(
    store: FileStore,
    content: bytes,
    *,
    media_type: str = TEXT,
    status: DocumentStatus = "active",
    document_id: uuid.UUID | None = None,
    version: int = 1,
    stored: bool = True,
) -> ClaimedJob:
    """A claimed prepare job, with its original in the store (unless stored=False)."""
    sha256 = hashlib.sha256(content).hexdigest()
    key = original_key(sha256)
    if stored:
        store.put(key, io.BytesIO(content), size=len(content), media_type=media_type)
    document_id = document_id or uuid.uuid4()
    uploader = uuid.uuid4()
    file = StoredFile(
        id=uuid.uuid4(),
        sha256=sha256,
        size=len(content),
        media_type=media_type,
        storage_key=key,
        created_at=NOW,
    )
    return ClaimedJob(
        job=IndexingJob(
            id=uuid.uuid4(),
            kind="prepare",
            document_id=document_id,
            version=version,
            status="running",
            attempts=1,
            error=None,
            created_by=uploader,
            created_at=NOW,
            started_at=NOW,
            finished_at=None,
        ),
        document=Document(
            id=document_id,
            collection_id=uuid.uuid4(),
            title="Quy định",
            status=status,
            current_version=version,
            created_by=uploader,
            created_at=NOW,
            updated_at=NOW,
        ),
        version=DocumentVersion(
            document_id=document_id,
            version=version,
            file_id=file.id,
            original_filename="quy-dinh.txt",
            provenance=None,
            uploaded_by=uploader,
            created_at=NOW,
        ),
        file=file,
    )


def saved_files(root: Path, job: ClaimedJob) -> list[str]:
    folder = root / prepared_version_folder(job.document.id, job.version.version)
    if not folder.exists():
        return []
    return sorted(path.name for path in folder.iterdir())


def load(
    store: FileStore, job: ClaimedJob
) -> tuple[LogicalDocument, tuple[Chunk, ...]]:
    parts = load_prepared(store, job.document.id, job.version.version)
    assert parts is not None
    return parts.logical, parts.chunks


def write(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


class FailingPuts:
    """A store that fails every write, like S3 being unreachable."""

    def __init__(self, store: FileStore) -> None:
        self._store = store

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        raise ConnectionError("store unreachable")

    def open(self, key: str) -> BinaryIO:
        return self._store.open(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def delete_folder(self, folder: str) -> int:
        return self._store.delete_folder(folder)


# building a ProfileChooser


def test_the_settings_name_the_candidates_and_the_fallback_with_versions(
    chooser: ProfileChooser,
) -> None:
    described = chooser.settings.describe()

    assert described["profiles"] == [VN_REGULATION]
    assert described["fallback_profile"] == PLAIN
    assert described["embedder"] == "words"


def test_the_chunker_is_described_by_its_type_and_options(
    chooser: ProfileChooser,
) -> None:
    chunker = chooser.settings.describe()["chunker"]

    assert chunker["kind"] == "OutlineChunker"
    assert chunker["max_tokens"] == 50
    assert "tokenizer" not in chunker  # the embedder's identity stands for it
    json.dumps(chooser.settings.describe())  # plain JSON, for provenance


def test_there_is_one_preparer_per_candidate() -> None:
    chooser = make_chooser(profiles=("plain", "vn_regulation"))

    assert len(chooser.preparers) == 2
    assert chooser.settings.profiles == (PLAIN, VN_REGULATION)


@pytest.mark.parametrize(
    ("profiles", "fallback"),
    [(("no_such_profile",), "plain"), (("vn_regulation",), "no_such_profile")],
    ids=["candidate", "fallback"],
)
def test_an_unknown_profile_is_refused_when_building(
    profiles: tuple[str, ...], fallback: str
) -> None:
    # caught when the worker starts, not on the first job
    with pytest.raises(KeyError):
        make_chooser(profiles=profiles, fallback=fallback)


def test_a_chunker_that_cant_be_described_is_refused() -> None:
    class NotADataclass:
        max_tokens = 50

        def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]:
            return ()

    with pytest.raises(TypeError, match="NotADataclass"):
        ProfileChooser.from_profiles(
            profiles=("vn_regulation",),
            fallback_profile="plain",
            chunker=NotADataclass(),
            embedder_identity="words",
        )


# choosing a profile


def test_a_regulation_is_prepared_with_the_first_candidate(
    tmp_path: Path, chooser: ProfileChooser
) -> None:
    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == VN_REGULATION
    assert chosen.profiles_without_structure == ()
    assert chosen.prepared.logical.profile_name == "vn_regulation"


def test_text_without_structure_falls_back(
    tmp_path: Path, chooser: ProfileChooser
) -> None:
    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(PLAIN_TEXT)))

    assert chosen.profile == PLAIN
    assert chosen.profiles_without_structure == (VN_REGULATION,)
    assert chosen.prepared.chunks  # still chunked, by size


def test_candidates_that_find_nothing_are_skipped_in_order(tmp_path: Path) -> None:
    chooser = make_chooser(profiles=("plain", "vn_regulation"))

    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == VN_REGULATION
    assert chosen.profiles_without_structure == (PLAIN,)


def test_the_first_candidate_that_finds_structure_wins(tmp_path: Path) -> None:
    # nothing after it is tried: the order is the preference
    chooser = make_chooser(profiles=("vn_regulation", "plain"))

    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == VN_REGULATION
    assert chosen.profiles_without_structure == ()


def test_with_no_candidates_the_fallback_is_used(tmp_path: Path) -> None:
    chooser = make_chooser(profiles=())

    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == PLAIN
    assert chosen.profiles_without_structure == ()


def test_a_word_file_is_prepared_too(tmp_path: Path, chooser: ProfileChooser) -> None:
    chosen = chooser.prepare(write(tmp_path, "a.docx", docx_of(REGULATION)))

    assert chosen.profile == VN_REGULATION


# the prepare handler


def test_preparing_saves_the_parts_and_the_review_page(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_prepare_job(job, chooser=chooser, store=store)

    assert saved_files(root, job) == ["chunks.jsonl", "logical.json", "review.html"]
    logical, chunks = load(store, job)
    assert logical.profile_name == "vn_regulation"
    assert "Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày" in "".join(
        chunk.text for chunk in chunks
    )


def test_the_review_page_is_html(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_prepare_job(job, chooser=chooser, store=store)

    with store.open(
        f"{prepared_version_folder(job.document.id, 1)}/review.html"
    ) as page:
        html = page.read().decode("utf-8")
    assert "<html" in html.lower()
    assert "Điều 1" in html


def test_the_result_says_how_it_was_prepared(
    store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(PLAIN_TEXT))

    result = run_prepare_job(job, chooser=chooser, store=store)

    assert result.profile == PLAIN
    assert result.profiles_without_structure == (VN_REGULATION,)
    assert result.settings == chooser.settings.describe()


def test_the_provenance_is_the_choice_and_the_settings(
    store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(PLAIN_TEXT))

    provenance = run_prepare_job(job, chooser=chooser, store=store).provenance()

    assert provenance == {
        "profile": PLAIN,
        "profiles_without_structure": [VN_REGULATION],
        "settings": chooser.settings.describe(),
    }
    assert json.loads(json.dumps(provenance)) == provenance  # plain JSON


@pytest.mark.parametrize(
    ("status", "should_index"), [("active", True), ("draft", False)]
)
def test_only_an_active_document_is_indexed_next(
    store: FileStore,
    chooser: ProfileChooser,
    status: DocumentStatus,
    should_index: bool,
) -> None:
    job = stage(store, text_bytes(REGULATION), status=status)

    result = run_prepare_job(job, chooser=chooser, store=store)

    assert result.should_index is should_index


def test_a_word_file_is_handled(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, docx_of(REGULATION), media_type=DOCX)

    result = run_prepare_job(job, chooser=chooser, store=store)

    assert result.profile == VN_REGULATION
    assert saved_files(root, job) == ["chunks.jsonl", "logical.json", "review.html"]


def test_preparing_again_replaces_the_old_output(
    store: FileStore, chooser: ProfileChooser
) -> None:
    first = stage(store, text_bytes(PLAIN_TEXT))
    run_prepare_job(first, chooser=chooser, store=store)
    again = stage(store, text_bytes(REGULATION), document_id=first.document.id)

    run_prepare_job(again, chooser=chooser, store=store)

    logical, chunks = load(store, first)
    assert logical.profile_name == "vn_regulation"
    assert "kế hoạch sao lưu" not in "".join(chunk.text for chunk in chunks)


def test_other_versions_are_left_alone(
    store: FileStore, chooser: ProfileChooser
) -> None:
    first = stage(store, text_bytes(PLAIN_TEXT))
    run_prepare_job(first, chooser=chooser, store=store)
    second = stage(
        store, text_bytes(REGULATION), document_id=first.document.id, version=2
    )

    run_prepare_job(second, chooser=chooser, store=store)

    assert load(store, first)[0].profile_name == "plain"
    assert load(store, second)[0].profile_name == "vn_regulation"


def test_the_original_is_left_as_it_was(
    store: FileStore, chooser: ProfileChooser
) -> None:
    content = text_bytes(REGULATION)
    job = stage(store, content)

    run_prepare_job(job, chooser=chooser, store=store)

    with store.open(job.file.storage_key) as original:
        assert original.read() == content


def test_no_temporary_files_are_left(
    temp_root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_prepare_job(job, chooser=chooser, store=store)

    assert list(temp_root.iterdir()) == []


# failures that retrying can't fix


def test_a_missing_original_fails_for_good(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION), stored=False)

    with pytest.raises(PermanentJobError, match="missing") as raised:
        run_prepare_job(job, chooser=chooser, store=store)

    assert job.file.sha256 in str(raised.value)
    assert raised.value.__suppress_context__  # one clear error in the traceback
    assert saved_files(root, job) == []


def test_an_unsupported_file_type_fails_for_good(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, b"%PDF-1.7\n", media_type="application/pdf")

    with pytest.raises(PermanentJobError, match="application/pdf"):
        run_prepare_job(job, chooser=chooser, store=store)

    assert saved_files(root, job) == []


def test_a_damaged_word_file_fails_for_good_and_says_why(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, damaged_docx(), media_type=DOCX)

    with pytest.raises(PermanentJobError) as raised:
        run_prepare_job(job, chooser=chooser, store=store)

    cause = raised.value.__cause__
    assert cause is not None  # the full error stays available to the worker's log
    assert type(cause).__name__ in str(raised.value)
    assert len(str(raised.value)) <= 500
    assert saved_files(root, job) == []


def test_failing_for_good_leaves_no_temporary_files(
    temp_root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, damaged_docx(), media_type=DOCX)

    with pytest.raises(PermanentJobError):
        run_prepare_job(job, chooser=chooser, store=store)

    assert list(temp_root.iterdir()) == []


# failures worth retrying pass through unchanged


def test_an_os_error_while_preparing_is_not_permanent(
    store: FileStore, chooser: ProfileChooser, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(self: ProfileChooser, path: Path) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ProfileChooser, "prepare", disk_full)
    job = stage(store, text_bytes(REGULATION))

    with pytest.raises(OSError, match="No space left") as raised:
        run_prepare_job(job, chooser=chooser, store=store)

    assert not isinstance(raised.value, PermanentJobError)


def test_a_store_that_cant_be_written_is_not_permanent(
    store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    with pytest.raises(ConnectionError):
        run_prepare_job(job, chooser=chooser, store=FailingPuts(store))


# logging


def test_warnings_are_logged_as_counts_without_document_text(
    store: FileStore, chooser: ProfileChooser, caplog: pytest.LogCaptureFixture
) -> None:
    job = stage(store, text_bytes(PLAIN_TEXT))

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.handlers"):
        run_prepare_job(job, chooser=chooser, store=store)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert str(job.document.id) in messages[0]
    assert PLAIN in messages[0]
    assert "no_headings=1" in messages[0]
    assert "sao lưu" not in messages[0]  # never the document's text


def test_a_clean_preparation_logs_nothing(
    store: FileStore, chooser: ProfileChooser, caplog: pytest.LogCaptureFixture
) -> None:
    job = stage(store, text_bytes(REGULATION))

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.handlers"):
        run_prepare_job(job, chooser=chooser, store=store)

    assert caplog.records == []
