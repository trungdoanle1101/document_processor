from kb_server.domain.users import User, UserRole
from kb_server.services.passwords import Passwords
from kb_server.storage.repositories.users import UserRepository


class UserService:
    def __init__(self, users: UserRepository, passwords: Passwords) -> None:
        self._users = users
        self._passwords = passwords

    def create(
        self, *, email: str, display_name: str, role: UserRole, password: str
    ) -> User:
        """Create a new user."""
        return self._users.add(
            email=email,
            display_name=display_name,
            role=role,
            password_hash=self._passwords.hash(password),
        )
