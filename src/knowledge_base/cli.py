"""Command line: python -m knowledge_base <command>.

add PATH...          index files (or refresh them when they changed)
remove ID            take a document out of the index and the registry
list                 the documents held
stale                documents whose chunks would now come out differently
refresh              index the stale documents again
ask QUESTION         answer a question, as a given reader
serve                run the HTTP API

add and remove take --server URL to have a running server do the work with the
models it already loaded, instead of loading them again here.
"""

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from document_processor.retrieval.attributes import AttributeValue
from document_processor.retrieval.chat_model import ChatModelError
from knowledge_base.assistant import format_reply
from knowledge_base.ingestion import parse_attribute
from knowledge_base.settings import AppSettings, load_settings
from knowledge_base.wiring import Services, build_services

DEFAULT_CONFIG = Path("knowledge_base.toml")
TOKEN_VARIABLE = "KNOWLEDGE_BASE_ADMIN_TOKEN"


class ServerResponse(Protocol):
    @property
    def status_code(self) -> int: ...
    @property
    def is_success(self) -> bool: ...
    @property
    def text(self) -> str: ...
    def json(self) -> Any: ...


class ServerClient(Protocol):
    """The little of an HTTP client that --server needs; httpx.Client is one."""

    def post(
        self, url: str, *, json: Any, headers: Mapping[str, str]
    ) -> ServerResponse: ...
    def delete(self, url: str, *, headers: Mapping[str, str]) -> ServerResponse: ...
    def close(self) -> None: ...


def connect(url: str) -> ServerClient:
    import httpx

    # indexing a long document can take minutes
    return httpx.Client(base_url=url, timeout=httpx.Timeout(600.0, connect=10.0))


def main(
    argv: Sequence[str] | None = None,
    *,
    build: Callable[[AppSettings], Services] = build_services,
    open_client: Callable[[str], ServerClient] = connect,
) -> int:
    args = _parser().parse_args(argv)
    settings = _settings(args.config)
    if getattr(args, "server", None):
        client = open_client(args.server)
        try:
            return _send(args, client, _token(args, settings))
        finally:
            client.close()

    # each part is built on first use, so a command loads only what it touches
    services = build(settings)
    match args.command:
        case "add":
            attributes = _attributes(args.attribute)
            for path in args.paths:
                result = services.indexer.add_document(
                    path,
                    access_group=args.group,
                    document_id=args.id if len(args.paths) == 1 else None,
                    title=args.title if len(args.paths) == 1 else None,
                    attributes=attributes,
                    force=args.force,
                )
                record = result.record
                _print_added(
                    record.document_id,
                    result.action,
                    record.version,
                    record.chunk_count,
                )
        case "remove":
            services.indexer.remove_document(args.document_id)
            print(f"{args.document_id}: removed")
        case "list":
            groups = [args.group] if args.group else _all_groups(settings)
            for record in services.registry.list_documents(access_groups=groups):
                print(
                    f"{record.document_id}  v{record.version}  [{record.access_group}]  "
                    f"{record.chunk_count} chunks  {record.title}"
                )
        case "stale":
            stale = services.indexer.find_stale()
            for item in stale:
                print(f"{item.record.document_id}: {', '.join(item.reasons)}")
            if not stale:
                print("nothing is stale")
        case "refresh":
            for result in services.indexer.refresh_stale():
                print(
                    f"{result.record.document_id}: {result.action} (v{result.record.version})"
                )
        case "ask":
            knowledge_base = services.knowledge_base
            try:
                answer = knowledge_base.ask(
                    args.question, groups=settings.groups_of(args.reader)
                )
            except ChatModelError as error:
                print(f"the chat model could not answer: {error}", file=sys.stderr)
                return 1
            print(format_reply(answer, knowledge_base))
        case "serve":
            import uvicorn

            from knowledge_base.api import create_api

            uvicorn.run(create_api(services, settings), host=args.host, port=args.port)
    return 0


def _send(args: argparse.Namespace, client: ServerClient, token: str | None) -> int:
    """add or remove done by a running server."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if args.command == "remove":
        response = client.delete(
            f"/v1/documents/{quote(args.document_id, safe='')}", headers=headers
        )
        if not response.is_success:
            return _fail(args.document_id, response)
        print(f"{args.document_id}: removed")
        return 0

    _attributes(args.attribute)  # refuse a malformed one before sending anything
    for path in args.paths:
        body = {
            "path": str(Path(path).resolve()),  # the server reads the file itself
            "group": args.group,
            "document_id": args.id if len(args.paths) == 1 else None,
            "title": args.title if len(args.paths) == 1 else None,
            "attributes": {
                name: value
                for name, _, value in (pair.partition("=") for pair in args.attribute)
            },
            "force": args.force,
        }
        response = client.post("/v1/documents", json=body, headers=headers)
        if not response.is_success:
            return _fail(str(path), response)
        added = response.json()
        _print_added(
            added["document_id"],
            added["action"],
            added["version"],
            added["chunk_count"],
        )
    return 0


def _fail(what: str, response: ServerResponse) -> int:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    print(f"{what}: {detail} (HTTP {response.status_code})", file=sys.stderr)
    return 1


def _print_added(document_id: str, action: str, version: int, chunk_count: int) -> None:
    print(f"{document_id}: {action} (v{version}, {chunk_count} chunks)")


def _token(args: argparse.Namespace, settings: AppSettings) -> str | None:
    """--token, else the environment, else the settings file the server also reads."""
    token: str | None = args.token or os.environ.get(TOKEN_VARIABLE)
    return token or settings.admin_token


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge_base", description="A knowledge base of documents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"settings file (default: {DEFAULT_CONFIG} if present)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    server = argparse.ArgumentParser(add_help=False)
    server.add_argument(
        "--server",
        metavar="URL",
        help="have this running server do it, e.g. http://127.0.0.1:8000",
    )
    server.add_argument(
        "--token",
        help=f"admin token for --server (default: ${TOKEN_VARIABLE}, then admin_token)",
    )

    add = commands.add_parser(
        "add", parents=[server], help="index files, or refresh them when they changed"
    )
    add.add_argument("paths", nargs="+", type=Path)
    add.add_argument("--group", required=True, help="access group allowed to read them")
    add.add_argument("--id", help="document id (one file only; default: the file name)")
    add.add_argument("--title", help="title (one file only; default: the file name)")
    add.add_argument(
        "--attribute",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="filterable attribute; a YYYY-MM-DD value is a date",
    )
    add.add_argument(
        "--force", action="store_true", help="index again even if nothing changed"
    )

    remove = commands.add_parser("remove", parents=[server], help="take a document out")
    remove.add_argument("document_id")

    listing = commands.add_parser("list", help="the documents held")
    listing.add_argument("--group", help="only this access group")

    commands.add_parser(
        "stale", help="documents whose chunks would now come out differently"
    )
    commands.add_parser("refresh", help="index the stale documents again")

    ask = commands.add_parser("ask", help="answer a question")
    ask.add_argument("question")
    ask.add_argument("--reader", help="who is asking (decides the access groups)")

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _settings(config: Path | None) -> AppSettings:
    if config is not None:
        return load_settings(config)
    return load_settings(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else AppSettings()


def _attributes(pairs: Sequence[str]) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"--attribute {pair!r}: expected NAME=VALUE")
        attributes[name] = parse_attribute(value)
    return attributes


def _all_groups(settings: AppSettings) -> list[str]:
    groups = {group for groups in settings.users.values() for group in groups}
    return sorted(groups | set(settings.default_groups))