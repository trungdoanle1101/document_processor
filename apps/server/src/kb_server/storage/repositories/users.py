import uuid

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.errors import EmailTaken
from kb_server.domain.users import User, UserRole, normalize_email
from kb_server.storage.models import UserRow


class UserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        email: str,
        display_name: str,
        role: UserRole,
        password_hash: str,
    ) -> User:
        email = normalize_email(email)
        row = UserRow(
            email=email,
            display_name=display_name,
            role=role,
            password_hash=password_hash,
        )
        try:
            # a savepoint: a failed insert undoes only itself
            with self._session.begin_nested():
                self._session.add(row)
                self._session.flush()
        except IntegrityError as error:
            if self.get_by_email(email) is not None:
                raise EmailTaken(email) from error
            raise
        return to_user(row)

    def get(self, user_id: uuid.UUID) -> User | None:
        row = self._session.get(UserRow, user_id)
        return to_user(row) if row is not None else None

    def get_by_email(self, email: str) -> User | None:
        row = self._row_by_email(email)
        return to_user(row) if row is not None else None

    def get_credentials_by_email(self, email: str) -> tuple[User, str] | None:
        """Returns User and the password hash."""
        row = self._row_by_email(email)
        if row is None:
            return None
        return to_user(row), row.password_hash

    def list_all(self) -> list[User]:
        rows = self._session.scalars(select(UserRow).order_by(UserRow.email)).all()
        return [to_user(row) for row in rows]

    def has_admin(self) -> bool:
        rows = self._session.scalars(
            select(UserRow.id).where(UserRow.role == "admin").limit(1)
        ).all()
        return bool(rows)

    def update(
        self,
        user_id: uuid.UUID,
        *,
        display_name: str,
        role: UserRole,
        is_active: bool,
    ) -> None:
        payload = {
            "display_name": display_name,
            "role": role,
            "is_active": is_active,
        }
        self._session.execute(
            update(UserRow).where(UserRow.id == user_id).values(**payload)
        )

    def set_password_hash(self, user_id: uuid.UUID, password_hash: str) -> None:
        (
            self._session.execute(
                update(UserRow)
                .where(UserRow.id == user_id)
                .values(password_hash=password_hash)
            ),
        )

    def _row_by_email(self, email: str) -> UserRow | None:
        return self._session.scalar(
            select(UserRow).where(UserRow.email == normalize_email(email))
        )


def to_user(row: UserRow) -> User:
    return User(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        role=row.role,
        is_active=row.is_active,
        created_at=row.created_at,
    )
