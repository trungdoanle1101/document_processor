"""The wiring: what's built for the worker, and what the other processes don't load."""

import subprocess
import sys
from pathlib import Path

from kb_server.indexing.handlers import _index_attributes
from kb_server.storage.files import LocalFileStore
from kb_server.wiring import QDRANT_INDEXED_ATTRIBUTES
from tests.indexing.test_handlers import stage


def test_qdrant_indexes_exactly_the_attributes_the_index_job_writes(
    tmp_path: Path,
) -> None:
    # a renamed attribute would otherwise leave search filtering on a field with no index
    job = stage(LocalFileStore(tmp_path), b"text")

    assert set(QDRANT_INDEXED_ATTRIBUTES) == set(_index_attributes(job))


WORKER_ONLY_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "qdrant_client",
)


def test_the_api_and_the_commands_load_none_of_the_workers_dependencies(
    tmp_path: Path,
) -> None:
    # W5. A fresh interpreter, because this test process has loaded them already; run
    # from an empty folder so no .env is read.
    script = (
        "import sys\n"
        "import kb_server.app, kb_server.manage\n"
        "kb_server.app.create_app()\n"
        f"print(','.join(m for m in {WORKER_ONLY_MODULES!r} if m in sys.modules))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""
