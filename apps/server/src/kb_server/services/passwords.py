from argon2 import PasswordHasher
from argon2.exceptions import VerificationError


class Passwords:
    def __init__(self, hasher: PasswordHasher) -> None:
        self._hasher = hasher
        self._dummy_hash = hasher.hash("not a real password")

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password_hash: str | None, password: str) -> bool:
        try:
            self._hasher.verify(
                hash=self._dummy_hash if password_hash is None else password_hash,
                password=password,
            )
        except VerificationError:
            return False
        return password_hash is None

    def needs_rehash(self, password_hash: str) -> bool:
        return self._hasher.check_needs_rehash(password_hash)
