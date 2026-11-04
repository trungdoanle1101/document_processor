"""Settings: TOML in, frozen settings out; anything left out keeps its default."""

from pathlib import Path

import pytest

from knowledge_base.settings import AppSettings, load_settings, parse_settings

EXAMPLE = Path(__file__).parents[2] / "knowledge_base.example.toml"


def test_an_empty_file_gives_the_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("", encoding="utf-8")
    assert load_settings(path) == AppSettings()


def test_the_example_file_is_valid() -> None:
    settings = load_settings(EXAMPLE)
    assert settings.profile == "vn_regulation"
    assert settings.groups_of("an@example.com") == ("public", "it")


def test_sections_override_only_what_they_name() -> None:
    settings = parse_settings(
        {
            "collection": "kb",
            "registry": "kb.sqlite3",
            "embedder": {"folder": "models/bge", "device": "cpu"},
            "chat": {"model": "qwen3:1.7b", "think": True},
            "answer": {"min_score": 0.5},
        }
    )
    assert (settings.collection, settings.registry) == ("kb", Path("kb.sqlite3"))
    assert (settings.embedder.spec, settings.embedder.folder) == (
        "bge-m3",
        Path("models/bge"),
    )
    assert (settings.chat.model, settings.chat.think, settings.chat.context_size) == (
        "qwen3:1.7b",
        True,
        8192,
    )
    assert (settings.answer.min_score, settings.answer.passages) == (0.5, 5)


def test_reranking_can_be_turned_off() -> None:
    assert parse_settings({"reranker": {"enabled": False}}).reranker is None
    assert parse_settings({"reranker": {"depth": 30}}).reranker is not None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"colection": "kb"}, "unknown settings key"),
        ({"chat": {"modle": "x"}}, r"unknown \[chat\] key"),
        ({"chat": "qwen3"}, r"\[chat\] must be a table"),
        ({"chunk_tokens": 0}, "chunk_tokens must be positive"),
        ({"answer": {"passages": 0}}, "passages must be positive"),
        ({"reranker": {"depth": 0}}, "depth must be positive"),
    ],
)
def test_mistakes_are_refused(data: dict[str, object], message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        parse_settings(data)


def test_readers_get_their_groups_and_everyone_else_the_defaults() -> None:
    settings = parse_settings(
        {"users": {"an@bank.vn": ["it", "public"]}, "default_groups": ["public"]}
    )
    assert settings.groups_of("an@bank.vn") == ("it", "public")
    assert settings.groups_of("someone@bank.vn") == ("public",)
    assert settings.groups_of(None) == ("public",)
