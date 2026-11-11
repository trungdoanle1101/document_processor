"""Smoke test: one real document through upload, prepare, index and delete.

Run from apps/server, with the API (kb-server serve) and the worker (kb-server worker) running:

    uv run python scripts/smoke_test.py ../../data/<file>.docx --email <admin email>

The password comes from KB_SMOKE_PASSWORD, or is asked for. The Qdrant and file-store
settings come from apps/server/.env, as for the server. The document is uploaded as private,
into its own collection (smoke-test unless --collection says otherwise), so no reader sees it.

Prints ids, counts and statuses only, never document text, so the output is safe to share.
Exits 0 when every check passed, 1 when one failed.
"""

import argparse
import getpass
import os
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from document_processor.integrations.qdrant._payload import (
    ATTRIBUTES,
    DOCUMENT_ID,
    decode_attributes,
)
from kb_server.settings import Settings
from kb_server.storage.files import (
    original_key,
    prepared_folder,
    prepared_version_folder,
)

API = "/api/v1"
POLL_SECONDS = 2.0
# step 4 runs right after the step 3 commit, so the original goes within seconds
ORIGINAL_GRACE_SECONDS = 30.0
PREPARED_FILES = ("chunks.jsonl", "logical.json", "review.html")


class SmokeTestFailed(Exception):
    """A step didn't do what it should; the message says which step and what was seen."""


@dataclass(frozen=True, kw_only=True)
class Probes:
    """What the script looks at besides the API: the index, and the store when it's local."""

    qdrant: QdrantClient
    qdrant_collection: str
    store_root: Path | None  # None: an S3 store, not checked


def main() -> int:
    args = _parse_args()
    settings = Settings()
    password = os.environ.get("KB_SMOKE_PASSWORD") or getpass.getpass("password: ")
    api_key = settings.qdrant_api_key
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key.get_secret_value() if api_key is not None else None,
    )

    with closing(qdrant), httpx.Client(base_url=args.url + API, timeout=120) as http:
        probes = Probes(
            qdrant=qdrant,
            qdrant_collection=settings.qdrant_collection,
            store_root=settings.storage_path
            if settings.storage_kind == "local"
            else None,
        )
        try:
            _log_in(http, email=args.email, password=password)
            run_smoke_test(
                args.file,
                collection_slug=args.collection,
                timeout_seconds=args.timeout_minutes * 60,
                keep=args.keep,
                http=http,
                probes=probes,
            )
        except SmokeTestFailed as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        finally:
            _log_out(http)

    print("passed")
    return 0


def run_smoke_test(
    file: Path,
    *,
    collection_slug: str,
    timeout_seconds: float,
    keep: bool,
    http: httpx.Client,
    probes: Probes,
) -> None:
    """Upload the file, wait for it to be indexed, check the results, then delete it.

    A failed check deletes the document anyway, so a rerun isn't refused as a duplicate,
    unless keep is set.
    """
    collection_id = find_or_create_collection(http, collection_slug)
    document = upload_private(http, file, collection_id=collection_id)
    document_id = uuid.UUID(document["id"])
    _report(f"uploaded document {document_id}")

    try:
        seconds = wait_until_indexed(http, document, timeout_seconds=timeout_seconds)
        _report(f"prepared and indexed in {seconds:.0f}s")
        check_index(probes, document_id, collection_id=collection_id)
        check_prepared_output(probes, document_id, sha256=document["file"]["sha256"])
    except SmokeTestFailed:
        if not keep:
            _request_deletion(http, document_id)
            print(f"document {document_id} deletion requested, to clean up")
        raise

    if keep:
        print(f"kept document {document_id} (--keep)")
        return
    delete_and_check(
        http,
        probes,
        document_id,
        sha256=document["file"]["sha256"],
        timeout_seconds=timeout_seconds,
    )


def find_or_create_collection(http: httpx.Client, slug: str) -> uuid.UUID:
    """The id of the collection with this slug, creating it (with no groups) if it's new."""
    response = http.get("/collections")
    _expect(response, 200, "listing collections")
    for collection in response.json():
        if collection["slug"] == slug:
            _report(f"using collection {slug} ({collection['id']})")
            return uuid.UUID(collection["id"])

    response = http.post(
        "/collections",
        json={"slug": slug, "title": "Smoke test", "group_ids": []},
    )
    _expect(response, 201, f"creating collection {slug}")
    collection_id = uuid.UUID(response.json()["id"])
    _report(f"created collection {slug} ({collection_id})")
    return collection_id


def upload_private(
    http: httpx.Client, file: Path, *, collection_id: uuid.UUID
) -> dict[str, Any]:
    """Upload the file as a private document; return the document as the API returns it."""
    with file.open("rb") as content:
        response = http.post(
            "/documents",
            data={"collection_id": str(collection_id), "private": "true"},
            files={"file": (file.name, content, "application/octet-stream")},
        )
    if response.status_code == 409:
        existing = response.json()["error"]["details"]["document_id"]
        raise SmokeTestFailed(
            f"this file is already document {existing} in the collection; "
            "delete it first, or use another --collection"
        )
    _expect(response, 201, "uploading")
    document: dict[str, Any] = response.json()
    return document


def wait_until_indexed(
    http: httpx.Client, document: dict[str, Any], *, timeout_seconds: float
) -> float:
    """Wait until the prepare job and the index job it queues are done; return the seconds.

    SmokeTestFailed as soon as either job fails, naming the error's type only: its message
    could quote the document.
    """
    prepare_job_id = document["job"]["job_id"]

    def is_indexed() -> bool:
        job = _get_document(http, document["id"])["job"]
        if job["status"] == "failed":
            kind = "prepare" if job["job_id"] == prepare_job_id else "index"
            raise SmokeTestFailed(
                f"the {kind} job {job['job_id']} failed with {_error_type(job['error'])}; "
                f"the full error is in GET {API}/documents/{document['id']}"
            )
        # the prepare job queues the index job, which then becomes the latest job
        return bool(job["job_id"] != prepare_job_id and job["status"] == "done")

    return _wait_for("the document to be indexed", is_indexed, timeout=timeout_seconds)


def check_index(
    probes: Probes, document_id: uuid.UUID, *, collection_id: uuid.UUID
) -> None:
    """The document has points, and its chunks carry its collection, version and no groups."""
    count = _count_points(probes, document_id)
    if count == 0:
        raise SmokeTestFailed(
            f"the index job finished, but Qdrant has no points for {document_id}"
        )

    attributes = _attributes_of_one_point(probes, document_id)
    expected = {
        "access_group_ids": (),
        "collection_id": str(collection_id),
        "document_version": 1,
    }
    wrong = {
        name: attributes.get(name)
        for name, value in expected.items()
        if attributes.get(name) != value
    }
    if wrong:
        raise SmokeTestFailed(f"index attributes differ from {expected}: {wrong}")
    _report(f"{count} points in Qdrant, attributes {sorted(attributes)}")


def check_prepared_output(
    probes: Probes, document_id: uuid.UUID, *, sha256: str
) -> None:
    """The original and version 1's prepared files are in a local store."""
    if probes.store_root is None:
        print("  (an S3 store: prepared output not checked)")
        return

    if not (probes.store_root / original_key(sha256)).is_file():
        raise SmokeTestFailed(f"the original of {document_id} isn't in the store")
    folder = probes.store_root / prepared_version_folder(document_id, 1)
    missing = [name for name in PREPARED_FILES if not (folder / name).is_file()]
    if missing:
        raise SmokeTestFailed(f"prepared output of {document_id} lacks {missing}")
    _report(f"prepared output saved; review page: {(folder / 'review.html').resolve()}")


def delete_and_check(
    http: httpx.Client,
    probes: Probes,
    document_id: uuid.UUID,
    *,
    sha256: str,
    timeout_seconds: float,
) -> None:
    """Delete the document and check it's gone from the API at once, then from everywhere."""
    response = _request_deletion(http, document_id)
    _expect(response, 202, "deleting")
    if response.content:
        raise SmokeTestFailed(f"DELETE answered 202 with a body: {response.content!r}")
    _expect(http.get(f"/documents/{document_id}"), 404, "reading the deleted document")
    _report("delete accepted; the document reads as gone at once")

    def is_removed() -> bool:
        return _count_points(probes, document_id) == 0 and not _prepared_exists(
            probes, document_id
        )

    seconds = _wait_for(
        "the worker to remove the chunks and prepared output",
        is_removed,
        timeout=timeout_seconds,
    )
    _report(f"chunks and prepared output removed in {seconds:.0f}s")
    _check_original_removed(probes, document_id, sha256=sha256)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One real document through upload, prepare, index and delete."
    )
    parser.add_argument("file", type=Path, help="a .docx or .txt to upload")
    parser.add_argument("--email", required=True, help="an admin's email")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="the API's address"
    )
    parser.add_argument(
        "--collection", default="smoke-test", help="the collection's slug"
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=30.0,
        help="how long to wait for each stage (embedding on a CPU is slow)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the document in place, to look at it; skips the delete checks",
    )
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"{args.file} isn't a file")
    return args


def _log_in(http: httpx.Client, *, email: str, password: str) -> None:
    response = http.post("/auth/login", json={"email": email, "password": password})
    _expect(response, 200, "logging in")
    http.headers["Authorization"] = f"Bearer {response.json()['token']}"
    _report("logged in")


def _log_out(http: httpx.Client) -> None:
    """End the session, so the token doesn't outlive the run. Ignores a failed login."""
    if "Authorization" in http.headers:
        http.post("/auth/logout")


def _get_document(http: httpx.Client, document_id: str) -> dict[str, Any]:
    response = http.get(f"/documents/{document_id}")
    _expect(response, 200, "reading the document")
    document: dict[str, Any] = response.json()
    return document


def _request_deletion(http: httpx.Client, document_id: uuid.UUID) -> httpx.Response:
    return http.delete(f"/documents/{document_id}")


def _document_filter(document_id: uuid.UUID) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=DOCUMENT_ID, match=models.MatchValue(value=str(document_id))
            )
        ]
    )


def _count_points(probes: Probes, document_id: uuid.UUID) -> int:
    result = probes.qdrant.count(
        probes.qdrant_collection,
        count_filter=_document_filter(document_id),
        exact=True,
    )
    return result.count


def _attributes_of_one_point(probes: Probes, document_id: uuid.UUID) -> dict[str, Any]:
    """One point's attributes. Only that part of the payload is fetched: the rest is text."""
    points, _ = probes.qdrant.scroll(
        probes.qdrant_collection,
        scroll_filter=_document_filter(document_id),
        limit=1,
        with_payload=[ATTRIBUTES],
        with_vectors=False,
    )
    if not points or points[0].payload is None:
        raise SmokeTestFailed(f"no point of {document_id} carries attributes")
    return dict(decode_attributes(points[0].payload[ATTRIBUTES]))


def _prepared_exists(probes: Probes, document_id: uuid.UUID) -> bool:
    if probes.store_root is None:
        return False
    return (probes.store_root / prepared_folder(document_id)).exists()


def _check_original_removed(
    probes: Probes, document_id: uuid.UUID, *, sha256: str
) -> None:
    """Wait briefly for the original to go. Still there: another document may use it."""
    if probes.store_root is None:
        print("  (an S3 store: original not checked)")
        return

    original = probes.store_root / original_key(sha256)
    try:
        _wait_for(
            "the original",
            lambda: not original.exists(),
            timeout=ORIGINAL_GRACE_SECONDS,
        )
    except SmokeTestFailed:
        print(
            f"  note: the original of {document_id} is still stored; that's right if "
            "another document uses the same file, else check the worker's log"
        )
        return
    _report("original removed")


def _wait_for(what: str, check: Callable[[], bool], *, timeout: float) -> float:
    """Call check every POLL_SECONDS until it's true; return the seconds it took."""
    started = time.monotonic()
    while not check():
        if time.monotonic() - started > timeout:
            raise SmokeTestFailed(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(POLL_SECONDS)
    return time.monotonic() - started


def _expect(response: httpx.Response, status: int, doing: str) -> None:
    """SmokeTestFailed unless the response has this status, with the API's error code."""
    if response.status_code == status:
        return
    raise SmokeTestFailed(
        f"{doing}: expected {status}, got {response.status_code} {_error_code(response)}"
    )


def _error_code(response: httpx.Response) -> str:
    """The API's error code, never its message (it may quote the request)."""
    try:
        return str(response.json()["error"]["code"])
    except (ValueError, KeyError, TypeError):
        return ""


def _error_type(error: str | None) -> str:
    """The exception type from a job's recorded error, "Type: message"."""
    return (error or "an unknown error").split(":", 1)[0]


def _report(line: str) -> None:
    print(f"ok  {line}")


if __name__ == "__main__":
    sys.exit(main())
