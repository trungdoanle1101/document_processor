import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

from kb_server.domain.users import User

CredentialKind = Literal["session", "api_key"]
CREDENTIAL_KINDS = get_args(CredentialKind)

_CREDENTIAL_MARKERS: dict[CredentialKind, str] = {
    "session": "kb_s_",
    "api_key": "kb_k_",
}
_PREFIX_LENGTH = 12


@dataclass(frozen=True, kw_only=True)
class Credential:
    id: uuid.UUID
    user_id: uuid.UUID
    kind: CredentialKind
    name: str  # "login" for sessions; the user's label for keys
    prefix: str  # first characters of the token, safe to show
    act_for_users: bool
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None  # session always expires; key might not
    revoked_at: datetime | None


@dataclass(frozen=True, kw_only=True)
class Principal:
    """Who a request acts as, and through which credential."""

    user: User
    credential_id: uuid.UUID
    credential_kind: CredentialKind
    acting_for_others: bool  # True when an act_for_users key named another user

    def __post_init__(self) -> None:
        if self.acting_for_others and self.credential_kind != "api_key":
            raise ValueError("only an API key can act for other users")


@dataclass(frozen=True, kw_only=True)
class IssuedToken:
    """A new credential and its secret; the secret only exists here, once."""

    token: str
    credential: Credential


@dataclass(frozen=True, kw_only=True)
class NewToken:
    token: str  # the secret: give to the user once, not stored
    prefix: str  # safe to show, e.g. in the key list
    token_hash: str  # what to store


def new_token(kind: CredentialKind) -> NewToken:
    token = _CREDENTIAL_MARKERS[kind] + secrets.token_urlsafe(32)
    return NewToken(
        token=token, prefix=token[:_PREFIX_LENGTH], token_hash=hash_token(token)
    )


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def looks_like_token(token: str) -> bool:
    return token.startswith(tuple(_CREDENTIAL_MARKERS.values()))
