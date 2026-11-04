from sqlalchemy.orm import Session

from kb_server.services.passwords import Passwords
from kb_server.services.users import UserService
from kb_server.storage.models import UserRow
from kb_server.storage.repositories.users import UserRepository
from tests.support import PASSWORD


def test_a_new_users_password_is_stored_hashed(
    session: Session, users: UserRepository, passwords: Passwords
) -> None:
    user = UserService(users, passwords).create(
        email="an@example.com", display_name="An", role="reader", password=PASSWORD
    )

    row = session.get(UserRow, user.id)
    assert row is not None
    assert row.password_hash != PASSWORD
    assert passwords.verify(row.password_hash, PASSWORD)
