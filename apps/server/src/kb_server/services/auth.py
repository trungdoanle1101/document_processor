import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

from kb_server.domain.credentials import (
    Credential,
    CredentialKind,
    CredentialNotFound,
    IssuedToken,
    NotAllowed,
    Principal,
    hash_token,
    looks_like_token,
    new_token,
)
from kb_server.domain.users import User
from kb_server.services.passwords import Passwords
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.users import UserRepository


class AuthService:
    def __init__(
        self,
        *,
        users: UserRepository,
        credentials: CredentialRepository,
        passwords: Passwords,
        session_ttl: timedelta,
        now: Callable[[], datetime],
        touch_every: timedelta = timedelta(minutes=1),
    ) -> None:
        self._users = users
        self._credentials = credentials
        self._passwords = passwords
        self._session_ttl = session_ttl
        self._now = now
        self._touch_every = touch_every

    def authenticate(self, email: str, password: str) -> User | None:
        """Verify a user using email and password."""
        credentials = self._users.get_credentials_by_email(email)
        user, password_hash = credentials if credentials is not None else (None, None)
        if (
            not self._passwords.verify(password_hash, password)
            or user is None
            or not user.is_active
        ):
            return None
        return user

    def login(self, email: str, password: str) -> IssuedToken | None:
        """Log user in and issue a new token."""
        user = self.authenticate(email, password)
        if user is None:
            return None

        return self._issue(
            user_id=user.id,
            kind="session",
            name="login",
            act_for_users=False,
            expires_at=self._now() + self._session_ttl,
        )

    def logout(self, principal: Principal) -> None:
        """Log out of the session."""
        if principal.credential_kind != "session":
            raise NotAllowed("only a login session can log out")

        self._credentials.revoke(
            principal.credential_id,
            owner_id=principal.user.id,
            now=self._now(),
        )

    def resolve(self, token: str, *, acting_for: str | None = None) -> Principal | None:
        # Check potential junk key
        if not looks_like_token(token):
            return None
        now = self._now()

        found = self._credentials.find_usable(hash_token(token), now=now)
        if found is None:
            return None
        credential, owner = found

        # record use, at most once per touch_every
        if (
            credential.last_used_at is None
            or credential.last_used_at + self._touch_every <= now
        ):
            self._credentials.touch(credential.id, now=now)

        # Check if it's a front-end key (act_for_users == True)
        if credential.act_for_users:
            # A front-end key can't act as itself
            if acting_for is None:
                return None

            # Get the user information the key is acting for
            target = self._users.get_by_email(acting_for)
            if target is None or not target.is_active:
                return None
            return Principal(
                user=target,
                credential_id=credential.id,
                credential_kind=credential.kind,
                acting_for_others=True,
            )

        # an ordinary key (act_for_users == False) may not name another user
        if acting_for is not None:
            return None

        return Principal(
            user=owner,
            credential_id=credential.id,
            credential_kind=credential.kind,
            acting_for_others=False,
        )

    def create_key(
        self, owner: Principal, *, name: str, expires_at: datetime | None = None
    ) -> IssuedToken:

        if owner.acting_for_others:
            raise NotAllowed(
                "a front-end key can't create keys for the users it acts for"
            )

        if expires_at is not None and expires_at <= self._now():
            raise ValueError("expires_at must be in the future")

        if owner.credential_kind != "session":
            raise NotAllowed("API keys can only be created after logging in")

        return self._issue(
            user_id=owner.user.id,
            kind="api_key",
            name=name,
            act_for_users=False,
            expires_at=expires_at,
        )

    def create_front_end_key(
        self,
        admin: Principal,
        *,
        service_user_id: uuid.UUID,
        name: str,
        expires_at: datetime | None = None,
    ) -> IssuedToken:
        if admin.user.role != "admin":
            raise NotAllowed("front-end keys can only be created by admins")

        if admin.credential_kind != "session":
            raise NotAllowed("API keys can only be created after logging in")

        if expires_at is not None and expires_at <= self._now():
            raise ValueError("expires_at must be in the future")

        if self._users.get(service_user_id) is None:
            raise ValueError("unknown service_user_id")

        return self._issue(
            user_id=service_user_id,
            kind="api_key",
            name=name,
            act_for_users=True,
            expires_at=expires_at,
        )

    def list_keys(self, owner: Principal) -> list[Credential]:
        if owner.acting_for_others:
            raise NotAllowed("not allowed to list other users' keys")
        return self._credentials.list_for_user(owner.user.id, kind="api_key")

    def revoke_key(self, owner: Principal, key_id: uuid.UUID) -> None:
        if owner.acting_for_others:
            raise NotAllowed("not allowed to revoke other users' keys")
        revoked = self._credentials.revoke(
            credential_id=key_id,
            owner_id=owner.user.id,
            now=self._now(),
        )
        if not revoked:
            raise CredentialNotFound("key not found for the owner")

    def _issue(
        self,
        *,
        user_id: uuid.UUID,
        kind: CredentialKind,
        name: str,
        act_for_users: bool,
        expires_at: datetime | None,
    ) -> IssuedToken:
        generated_token = new_token(kind)
        credential = self._credentials.add(
            user_id=user_id,
            kind=kind,
            name=name,
            prefix=generated_token.prefix,
            token_hash=generated_token.token_hash,
            act_for_users=act_for_users,
            expires_at=expires_at,
        )
        return IssuedToken(
            token=generated_token.token,
            credential=credential,
        )
