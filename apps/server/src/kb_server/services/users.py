import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from kb_server.domain.credentials import Principal
from kb_server.domain.errors import (
    AdminExists,
    InvalidRequest,
    NotAllowed,
    UserNotFound,
)
from kb_server.domain.users import User, UserRole
from kb_server.services.passwords import Passwords
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.users import UserRepository


class UserService:
    def __init__(
        self,
        users: UserRepository,
        credentials: CredentialRepository,
        passwords: Passwords,
        now: Callable[[], datetime],
    ) -> None:
        self._users = users
        self._credentials = credentials
        self._passwords = passwords
        self._now = now

    def create_first_admin(
        self, *, email: str, display_name: str, password: str
    ) -> User:
        """For kb-server create-admin: only while there is no admin yet."""
        if self._users.has_admin():
            raise AdminExists("an admin already exists; can't create a new one")
        return self._create(
            email=email, display_name=display_name, role="admin", password=password
        )

    def create_user(
        self,
        admin: Principal,
        *,
        email: str,
        display_name: str,
        role: UserRole,
        password: str,
    ) -> User:
        """For the API: only an admin, with their own credential."""
        self._require_admin(admin)
        return self._create(
            email=email, display_name=display_name, role=role, password=password
        )

    def list_users(self, admin: Principal) -> list[User]:
        self._require_admin(admin)
        return self._users.list_all()

    def get_user(self, admin: Principal, user_id: uuid.UUID) -> User:
        self._require_admin(admin)
        user = self._users.get(user_id)
        if user is None:
            raise UserNotFound("can't find user with this id")
        return user

    def update_user(
        self,
        admin: Principal,
        user_id: uuid.UUID,
        *,
        display_name: str | None = None,
        role: UserRole | None = None,
        is_active: bool | None = None,
    ) -> User:
        self._require_admin(admin)
        user = self.get_user(admin, user_id)
        if user_id == admin.user.id and (role is not None or is_active is not None):
            raise NotAllowed("an admin can't change their own role or is_active status")

        payload = {
            "display_name": display_name,
            "role": role,
            "is_active": is_active,
        }

        update_fields = {
            key: value for key, value in payload.items() if value is not None
        }
        self._users.update(user_id, display_name=display_name, **update_fields)
        new = replace(user, **update_fields)

        # Revoke all credentials belonging to this user if they become inactive
        if not new.is_active:
            self._credentials.revoke_all(user_id, now=self._now())

        return new

    def change_password(self, principal: Principal, *, current: str, new: str) -> None:
        if principal.acting_for_others:
            raise NotAllowed("a front-end key can't chaange a user's password")
        if principal.credential_kind != "session":
            raise NotAllowed("log in to change your password")

        found = self._users.get_credentials_by_email(principal.user.email)
        assert found is not None
        _, password_hash = found

        if not self._passwords.verify(password_hash, current):
            raise InvalidRequest("the current password is wrong")
        self._users.set_password_hash(principal.user.id, self._passwords.hash(new))
        self._credentials.revoke_other_sessions(
            principal.user.id, keep=principal.credential_id, now=self._now()
        )

    def _create(
        self, *, email: str, display_name: str, role: UserRole, password: str
    ) -> User:
        """Create a new user."""
        return self._users.add(
            email=email,
            display_name=display_name,
            role=role,
            password_hash=self._passwords.hash(password),
        )

    def _require_admin(principal: Principal) -> None:
        if principal.acting_for_others:
            raise NotAllowed("a front-end key can't manage users")
        if principal.user.role != "admin":
            raise NotAllowed("only an admin can manage users")
