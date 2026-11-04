"""The command line, with the services built from fakes."""

from datetime import date
from pathlib import Path

import pytest

from knowledge_base.cli import main
from knowledge_base.settings import AppSettings
from knowledge_base.wiring import Services
from tests.knowledge_base.helpers import (
    DATA_REGULATION,
    LEAVE_REGULATION,
    Fakes,
    chunks_of,
    make_fakes,
    write,
)


@pytest.fixture
def fakes() -> Fakes:
    return make_fakes()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "kb.toml"
    path.write_text(
        'default_groups = ["public"]\n[users]\n"an@bank.vn" = ["it"]\n',
        encoding="utf-8",
    )
    return path


def run(fakes: Fakes, config: Path, *args: str) -> int:
    def build(settings: AppSettings) -> Services:
        assert settings.groups_of("an@bank.vn") == ("it",)  # the config was read
        return fakes.services

    return main(["--config", str(config), *args], build=build)


def test_add_list_and_remove(
    fakes: Fakes, config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = write(tmp_path, "du-lieu.txt", DATA_REGULATION)
    second = write(tmp_path, "nghi-phep.txt", LEAVE_REGULATION)

    assert run(fakes, config, "add", str(first), str(second), "--group", "it") == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in out] == ["du-lieu", "nghi-phep"]
    assert all("indexed (v1, " in line for line in out)

    run(fakes, config, "list")
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == [
        "du-lieu",
        "nghi-phep",
    ]

    run(fakes, config, "remove", "du-lieu")
    assert capsys.readouterr().out == "du-lieu: removed\n"
    run(fakes, config, "list", "--group", "it")
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == [
        "nghi-phep"
    ]


def test_add_takes_an_id_title_and_attributes(
    fakes: Fakes, config: Path, tmp_path: Path
) -> None:
    path = write(tmp_path, "file.txt", DATA_REGULATION)
    options = ["--id", "qd-1", "--title", "Quy định"]
    attributes = ["--attribute", "kind=quy_dinh", "--attribute", "issued_on=2025-03-01"]
    run(fakes, config, "add", str(path), "--group", "it", *options, *attributes)

    record = fakes.registry.get("qd-1")
    assert record is not None and record.title == "Quy định"
    chunk = chunks_of(fakes, "qd-1")[0]
    assert (chunk.attributes["kind"], chunk.attributes["issued_on"]) == (
        "quy_dinh",
        date(2025, 3, 1),
    )


def test_a_malformed_attribute_is_refused(
    fakes: Fakes, config: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        run(
            fakes,
            config,
            "add",
            str(write(tmp_path, "a.txt", DATA_REGULATION)),
            "--group",
            "it",
            "--attribute",
            "x",
        )


def test_stale_and_refresh(
    fakes: Fakes, config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "a.txt", DATA_REGULATION)
    run(fakes, config, "add", str(path), "--group", "it")
    capsys.readouterr()

    run(fakes, config, "stale")
    assert capsys.readouterr().out == "nothing is stale\n"

    write(tmp_path, "a.txt", DATA_REGULATION + "Điều 3. Mới\n")
    run(fakes, config, "stale")
    assert capsys.readouterr().out == "a: source_hash\n"
    run(fakes, config, "refresh")
    assert capsys.readouterr().out == "a: indexed (v2)\n"


def test_ask_answers_as_the_reader(
    fakes: Fakes, config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(
        fakes,
        config,
        "add",
        str(write(tmp_path, "a.txt", DATA_REGULATION)),
        "--group",
        "it",
    )
    capsys.readouterr()

    run(fakes, config, "ask", "Dữ liệu Staging lưu bao lâu?", "--reader", "an@bank.vn")
    assert capsys.readouterr().out.startswith(
        "Dữ liệu được lưu tối đa 07 ngày [1].\n\nNguồn:\n[1] a › "
    )

    run(fakes, config, "ask", "Dữ liệu Staging lưu bao lâu?")
    assert capsys.readouterr().out.startswith("Không tìm thấy")
