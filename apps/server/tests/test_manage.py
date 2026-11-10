"""kb-server's commands refuse to start when the file store can't be written.

Each command is run through main, as the console script runs it, with settings from the
environment. No database is needed: serve's check runs before uvicorn starts, and the
worker's runs before any model loads or database query.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from kb_server import manage

# nothing listens on port 1; the checks under test never connect
UNUSED_DATABASE_URL = "postgresql+psycopg://kb:kb@127.0.0.1:1/kb"

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root writes to read-only folders anyway",
)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The storage folder the commands read from KB_STORAGE_PATH."""
    folder = tmp_path / "storage"
    folder.mkdir()
    monkeypatch.setenv("KB_DATABASE_URL", UNUSED_DATABASE_URL)
    monkeypatch.setenv("KB_STORAGE_PATH", str(folder))
    return folder


@pytest.fixture
def read_only_storage(storage: Path) -> Iterator[Path]:
    storage.chmod(0o555)
    yield storage
    storage.chmod(0o755)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, bool]]:
    """Each time uvicorn would have started, instead of starting it."""
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        manage,
        "serve",
        lambda host, port, reload=False: calls.append((host, port, reload)),
    )
    return calls


@needs_permissions
def test_serve_refuses_to_start_when_the_store_cant_be_written(
    read_only_storage: Path, served: list[tuple[str, int, bool]]
) -> None:
    with pytest.raises(SystemExit) as raised:
        manage.main(["serve"])

    assert str(raised.value.code).startswith(
        "Can't start: can't write to the file store: "
    )
    assert "Permission denied" in str(raised.value.code)
    assert served == []


def test_serve_starts_when_the_store_can_be_written(
    storage: Path, served: list[tuple[str, int, bool]]
) -> None:
    manage.main(["serve", "--port", "8123"])

    assert served == [("127.0.0.1", 8123, False)]
    assert [path for path in storage.rglob("*") if path.is_file()] == []


@needs_permissions
def test_the_worker_refuses_to_start_before_loading_any_model(
    read_only_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load_models(*args: object) -> None:
        raise AssertionError("the worker loaded its models before checking the store")

    monkeypatch.setattr(manage, "open_job_tools", load_models)
    # run_worker configures logging for the whole process; keep it out of the test run
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(SystemExit) as raised:
        manage.main(["worker"])

    assert str(raised.value.code).startswith(
        "Can't start: can't write to the file store: "
    )
