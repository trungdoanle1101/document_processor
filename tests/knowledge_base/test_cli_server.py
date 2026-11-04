"""add and remove --server: the running server does the work, and nothing is loaded here."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledge_base.api import create_api
from knowledge_base.cli import TOKEN_VARIABLE, ServerClient, main
from knowledge_base.settings import AppSettings, parse_settings
from knowledge_base.wiring import Services
from tests.knowledge_base.helpers import (
    DATA_REGULATION,
    LEAVE_REGULATION,
    Fakes,
    chunks_of,
    make_fakes,
    write,
)

TOKEN = "a-long-enough-admin-token"
SERVER = "http://127.0.0.1:8000"


@pytest.fixture
def fakes() -> Fakes:
    return make_fakes()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "kb.toml"
    path.write_text(f'admin_token = "{TOKEN}"\n', encoding="utf-8")
    return path


def never_build(settings: AppSettings) -> Services:
    raise AssertionError("--server must not load models")


def run(fakes: Fakes, config: Path, *args: str, server_token: str = TOKEN) -> int:
    api = create_api(fakes.services, parse_settings({"admin_token": server_token}))

    def open_client(url: str) -> ServerClient:
        assert url == SERVER
        return TestClient(api)

    return main(
        ["--config", str(config), *args], build=never_build, open_client=open_client
    )


def test_add_and_remove_through_the_server(
    fakes: Fakes, config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = write(tmp_path, "du-lieu.txt", DATA_REGULATION)
    second = write(tmp_path, "nghi-phep.txt", LEAVE_REGULATION)

    assert (
        run(
            fakes,
            config,
            "add",
            str(first),
            str(second),
            "--group",
            "it",
            "--server",
            SERVER,
        )
        == 0
    )
    out = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in out] == ["du-lieu", "nghi-phep"]
    assert all("indexed (v1, " in line for line in out)

    assert run(fakes, config, "remove", "du-lieu", "--server", SERVER) == 0
    assert capsys.readouterr().out == "du-lieu: removed\n"
    assert chunks_of(fakes, "du-lieu") == []


def test_the_id_title_and_attributes_are_sent(
    fakes: Fakes, config: Path, tmp_path: Path
) -> None:
    path = write(tmp_path, "file.txt", DATA_REGULATION)
    options = [
        "--id",
        "qd-1",
        "--title",
        "Quy định",
        "--attribute",
        "issued_on=2025-03-01",
    ]
    run(fakes, config, "add", str(path), "--group", "it", "--server", SERVER, *options)

    record = fakes.registry.get("qd-1")
    assert record is not None and record.title == "Quy định"
    assert str(chunks_of(fakes, "qd-1")[0].attributes["issued_on"]) == "2025-03-01"


def test_a_relative_path_is_sent_as_absolute(
    fakes: Fakes, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path, "a.txt", DATA_REGULATION)
    monkeypatch.chdir(tmp_path)
    run(fakes, config, "add", "a.txt", "--group", "it", "--server", SERVER)
    record = fakes.registry.get("a")
    assert record is not None and record.source == str(tmp_path / "a.txt")


def test_the_token_can_come_from_the_option_or_the_environment(
    fakes: Fakes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    path = str(write(tmp_path, "a.txt", DATA_REGULATION))

    assert (
        run(
            fakes,
            empty,
            "add",
            path,
            "--group",
            "it",
            "--server",
            SERVER,
            "--token",
            TOKEN,
        )
        == 0
    )
    monkeypatch.setenv(TOKEN_VARIABLE, TOKEN)
    assert run(fakes, empty, "remove", "a", "--server", SERVER) == 0


def test_a_refusal_is_reported_and_fails(
    fakes: Fakes, config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = str(write(tmp_path, "a.txt", DATA_REGULATION))
    assert (
        run(
            fakes,
            config,
            "add",
            path,
            "--group",
            "it",
            "--server",
            SERVER,
            server_token="another-admin-token-x",
        )
        == 1
    )
    assert "a valid admin token is required (HTTP 401)" in capsys.readouterr().err

    assert run(fakes, config, "remove", "nope", "--server", SERVER) == 1
    assert "no document 'nope' (HTTP 404)" in capsys.readouterr().err


def test_a_malformed_attribute_is_refused_before_sending(
    fakes: Fakes, config: Path, tmp_path: Path
) -> None:
    path = str(write(tmp_path, "a.txt", DATA_REGULATION))
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        run(
            fakes,
            config,
            "add",
            path,
            "--group",
            "it",
            "--server",
            SERVER,
            "--attribute",
            "x",
        )
    assert fakes.registry.get("a") is None
