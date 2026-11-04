"""Everything the app is configured with, read from one TOML file.

Names refer to things defined in code (a profile, a model spec); the file only
chooses among them and sets their numbers. See knowledge_base.example.toml.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

type Device = Literal["auto", "cuda", "mps", "cpu"]


@dataclass(frozen=True, kw_only=True)
class EmbedderSettings:
    spec: str = "bge-m3"  # a name in HUGGINGFACE_MODEL_SPECS
    folder: Path = Path("huggingface/BAAI/bge_m3")
    device: Device = "auto"


@dataclass(frozen=True, kw_only=True)
class RerankerSettings:
    spec: str = "bge-reranker-v2-m3"  # a name in HUGGINGFACE_RERANKER_SPECS
    folder: Path = Path("huggingface/BAAI/bge_reranker_v2_m3")
    device: Device = "auto"
    depth: int = 20  # hits it sees per question


@dataclass(frozen=True, kw_only=True)
class ChatSettings:
    model: str = "qwen3:4b"  # an Ollama model tag
    url: str = "http://localhost:11434"
    context_size: int = 8192
    temperature: float = 0.0
    think: bool | None = False
    max_tokens: int = 1024  # longest answer
    timeout: float = 120.0  # seconds to wait for an answer


@dataclass(frozen=True, kw_only=True)
class AnswerSettings:
    template: str = "vn_regulation"  # a name in ANSWER_STYLES
    passages: int = 5  # shown to the model
    min_score: float | None = 0.3  # best passage below this: "not found"
    min_passage_score: float | None = 0.05  # passages below this are left out
    # when hits share a section, add the rest of it, up to this many characters; 0: off
    section_chars: int = 10_000
    # earlier questions (with their replies) used to understand a follow-up; 0: none
    history_turns: int = 2


@dataclass(frozen=True, kw_only=True)
class AppSettings:
    registry: Path = Path("data/knowledge_base.sqlite3")
    qdrant_url: str = "http://localhost:6333"
    collection: str = "knowledge-base"
    profile: str = "vn_regulation"  # a name in PROFILES
    chunk_tokens: int = 512
    embedder: EmbedderSettings = field(default_factory=EmbedderSettings)
    reranker: RerankerSettings | None = field(default_factory=RerankerSettings)
    chat: ChatSettings = field(default_factory=ChatSettings)
    answer: AnswerSettings = field(default_factory=AnswerSettings)
    # who may read what: a user's access groups, by user name (e.g. an email)
    users: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    default_groups: tuple[str, ...] = ("public",)  # for users not listed
    # lets a running server add and remove documents; None: only the CLI can
    admin_token: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {self.chunk_tokens}")
        if self.answer.history_turns < 0:
            raise ValueError(
                f"answer.history_turns must not be negative, got {self.answer.history_turns}"
            )
        if self.answer.section_chars < 0:
            raise ValueError(
                f"answer.section_chars must not be negative, got {self.answer.section_chars}"
            )
        if self.answer.passages <= 0:
            raise ValueError(
                f"answer.passages must be positive, got {self.answer.passages}"
            )
        if self.reranker is not None and self.reranker.depth <= 0:
            raise ValueError(
                f"reranker.depth must be positive, got {self.reranker.depth}"
            )
        if self.admin_token is not None and len(self.admin_token) < 16:
            raise ValueError("admin_token must be at least 16 characters")
        object.__setattr__(self, "users", MappingProxyType(dict(self.users)))

    def groups_of(self, user: str | None) -> tuple[str, ...]:
        """The access groups a user may read; unknown or anonymous users get the defaults."""
        if user is None:
            return self.default_groups
        return self.users.get(user, self.default_groups)


def load_settings(path: Path | str) -> AppSettings:
    """Settings from a TOML file; anything left out keeps its default."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return parse_settings(data)


def parse_settings(data: Mapping[str, Any]) -> AppSettings:
    known = {
        "registry",
        "qdrant_url",
        "collection",
        "profile",
        "chunk_tokens",
        "embedder",
        "reranker",
        "chat",
        "answer",
        "users",
        "default_groups",
        "admin_token",
    }
    _check_keys("settings", data, known)
    values = dict(data)
    for key in ("registry",):
        if key in values:
            values[key] = Path(values[key])
    if "embedder" in values:
        values["embedder"] = _section(EmbedderSettings, "embedder", values["embedder"])
    if "reranker" in values:
        reranker = values["reranker"]
        # [reranker] enabled = false turns reranking off
        enabled = reranker.pop("enabled", True) if isinstance(reranker, dict) else True
        values["reranker"] = (
            _section(RerankerSettings, "reranker", reranker) if enabled else None
        )
    if "chat" in values:
        values["chat"] = _section(ChatSettings, "chat", values["chat"])
    if "answer" in values:
        values["answer"] = _section(AnswerSettings, "answer", values["answer"])
    if "users" in values:
        values["users"] = {
            name: tuple(groups) for name, groups in values["users"].items()
        }
    if "default_groups" in values:
        values["default_groups"] = tuple(values["default_groups"])
    return AppSettings(**values)


def _section[T](kind: type[T], name: str, data: Any) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"[{name}] must be a table")
    known = set(kind.__dataclass_fields__)  # type: ignore[attr-defined]
    _check_keys(f"[{name}]", data, known)
    values = {
        key: Path(value) if key == "folder" else value for key, value in data.items()
    }
    return kind(**values)


def _check_keys(where: str, data: Mapping[str, Any], known: set[str]) -> None:
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {where} key(s): {', '.join(unknown)}")