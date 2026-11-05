import argparse
import getpass
import sys
from collections.abc import Callable, Sequence

import uvicorn
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from kb_server.domain.errors import AdminExists, EmailTaken
from kb_server.domain.users import MIN_PASSWORD_LEN
from kb_server.settings import Settings
from kb_server.wiring import (
    Services,
    build_services,
    build_user_service,
    close_services,
)


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

    args = parser.parse_args(argv)

    try:
        match args.command:
            case "create-admin":
                try:
                    services = build_services(Settings())
                except ValidationError as error:
                    sys.exit(
                        f"Invalid settings (environment or apps/server/.env):\n{error}"
                    )

                try:
                    create_admin(services, args.email, args.name)
                finally:
                    close_services(services)
            case "serve":
                serve(args.host, args.port, args.reload)
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


if __name__ == "__main__":
    main()
