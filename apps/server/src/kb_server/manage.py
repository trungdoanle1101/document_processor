import argparse
import getpass
import logging
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from types import FrameType

import uvicorn
from pydantic import ValidationError
from pydantic_settings import SettingsError
from sqlalchemy.exc import OperationalError

from kb_server.domain.errors import AdminExists, EmailTaken, StoreNotWritable
from kb_server.domain.users import MIN_PASSWORD_LEN
from kb_server.settings import Settings
from kb_server.storage.files import check_store_writable
from kb_server.wiring import (
    Services,
    WorkerSetupError,
    build_services,
    build_user_service,
    build_worker,
    close_services,
    open_job_tools,
)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="kb-server admin panel")
    commands = parser.add_subparsers(dest="command", required=True)

    # create-admin
    create_admin_parser = commands.add_parser(
        "create-admin", help="create a new admin if there's none yet"
    )
    create_admin_parser.add_argument("--email", required=True)
    create_admin_parser.add_argument("--name", required=True, help="display name")

    # serve
    serve_parser = commands.add_parser("serve", help="run the server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--reload", action="store_true", help="restart server when files change"
    )

    # worker
    commands.add_parser("worker", help="run the indexing jobs until stopped")

    args = parser.parse_args(argv)

    try:
        match args.command:
            case "create-admin":
                services = load_services()
                try:
                    create_admin(services, args.email, args.name)
                finally:
                    close_services(services)
            case "serve":
                services = load_services()
                try:
                    exit_unless_store_writable(services)
                finally:
                    close_services(services)
                serve(args.host, args.port, args.reload)
            case "worker":
                services = load_services()
                run_worker(services)

    except KeyboardInterrupt:
        sys.exit(130)


def create_admin(
    services: Services,
    email: str,
    display_name: str,
    *,
    read_password: Callable[[str], str] = getpass.getpass,
) -> None:
    """Ask for a password, then create the first admin."""
    password = read_password("Password: ")
    if len(password) < MIN_PASSWORD_LEN:
        sys.exit(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    password_verify = read_password("Repeat password: ")
    if password != password_verify:
        sys.exit("The passwords don't match.")

    try:
        with services.session_factory.begin() as session:
            user_service_ = build_user_service(session, services)
            user = user_service_.create_first_admin(
                email=email, display_name=display_name, password=password
            )

    except AdminExists:
        sys.exit("An admin already exists. Ask them to create your account.")
    except EmailTaken:
        sys.exit(f"A user with email {email} already exists.")
    except OperationalError:
        url = services.engine.url.render_as_string(hide_password=True)
        sys.exit(f"Can't reach the database at {url}. Is it running?")
    print(f"Created admin {user.email}.")


def serve(host: str, port: int, reload: bool = False) -> None:
    """Run the api with uvicorn."""
    uvicorn.run("kb_server.app:app", host=host, port=port, reload=reload)


def run_worker(services: Services) -> None:
    """Run jobs until SIGTERM or SIGINT. Requeues stuck jobs first."""
    logging.basicConfig(
        level=services.settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        exit_unless_store_writable(services)
        with open_job_tools(services) as job_tools:
            worker = build_worker(services, job_tools=job_tools)
            worker.recover_stuck_jobs()
            worker.run_forever(stop=stop_on_signals())
    except WorkerSetupError as error:
        sys.exit(f"The worker can't start: {error}")
    except OperationalError:
        url = services.engine.url.render_as_string(hide_password=True)
        sys.exit(f"Can't reach the database at {url}. Is it running?")
    finally:
        close_services(services)


def load_services() -> Services:
    """Services from the environment and .env; exits with a message if a setting is wrong."""
    try:
        settings = Settings()
    # SettingsError: a value that can't be parsed, e.g. KB_PROFILES that isn't JSON
    except (ValidationError, SettingsError) as error:
        sys.exit(f"Invalid settings (environment or apps/server/.env):\n{error}")
    try:
        return build_services(settings)
    # e.g. the file store's folder doesn't exist
    except ValueError as error:
        sys.exit(f"Can't start: {error}")


def stop_on_signals() -> threading.Event:
    """Stop the worker upon a SIGTERM or a SIGINT."""
    stop = threading.Event()

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        logger.info(
            "got %s: stopping after the current job", signal.Signals(signum).name
        )
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return stop


def exit_unless_store_writable(services: Services) -> None:
    """Exit with one line if the file store can't be written: uploads and prepare jobs would fail."""
    try:
        check_store_writable(services.files)
    except StoreNotWritable as error:
        sys.exit(f"Can't start: {error}")


if __name__ == "__main__":
    main()
