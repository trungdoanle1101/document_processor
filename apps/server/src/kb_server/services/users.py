from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from kb_server.domain.users import User, UserRole
from kb_server.storage.repositories.users import UserRepository


class UserService:
    def __init__(self, repo: UserRepository, hasher: PasswordHasher) -> None:
        self._repo = repo
        self._hasher = hasher
        self._dummy_hash = hasher.hash("not a real password")

    def create(
        self, *, email: str, display_name: str, role: UserRole, password: str
    ) -> User:
        return self._repo.add(
            email=email,
            display_name=display_name,
            role=role,
            password_hash=self._hasher.hash(password),
        )

    def authenticate(self, email: str, password: str) -> User | None:
        credentials = self._repo.get_credentials_by_email(email)
        user, password_hash = (
            credentials if credentials is not None else (None, self._dummy_hash)
        )
        try:
            self._hasher.verify(password_hash, password)
        except VerificationError:
            return None
        if user is None or not user.is_active:
            return None
        return user
