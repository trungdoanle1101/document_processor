import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, or_, select, update
from sqlalchemy.orm import Session

from kb_server.domain.credentials import Credential, CredentialKind
from kb_server.domain.users import User
from kb_server.storage.models import CredentialRow, UserRow
from kb_server.storage.repositories.users import to_user


class CredentialRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        user_id: uuid.UUID,
        kind: CredentialKind,
        name: str,
        prefix: str,
        token_hash: str,
        act_for_users: bool,
        expires_at: datetime | None,
    ) -> Credential:
        row = CredentialRow(
            user_id=user_id,
            kind=kind,
            name=name,
            token_hash=token_hash,
            prefix=prefix,
            act_for_users=act_for_users,
            expires_at=expires_at,
        )

        self._session.add(row)
        self._session.flush()
        return to_credential(row)

    def find_usable(
        self, token_hash: str, *, now: datetime
    ) -> tuple[Credential, User] | None:
        found = self._session.execute(
            select(CredentialRow, UserRow)
            .join(UserRow, CredentialRow.user_id == UserRow.id)
            .where(
                CredentialRow.token_hash == token_hash,
                CredentialRow.revoked_at.is_(None),
                or_(CredentialRow.expires_at.is_(None), CredentialRow.expires_at > now),
                UserRow.is_active,
            )
        ).one_or_none()
        if found is None:
            return None
        credential_row, user_row = found
        return to_credential(credential_row), to_user(user_row)

    def touch(self, credential_id: uuid.UUID, *, now: datetime) -> None:
        self._session.execute(
            update(CredentialRow)
            .where(
                CredentialRow.id == credential_id,
                CredentialRow.revoked_at.is_(None),
                or_(CredentialRow.expires_at.is_(None), CredentialRow.expires_at > now),
            )
            .values(last_used_at=now)
        )

    def revoke(
        self, credential_id: uuid.UUID, *, owner_id: uuid.UUID, now: datetime
    ) -> bool:
        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(CredentialRow)
                .where(
                    CredentialRow.id == credential_id,
                    CredentialRow.user_id == owner_id,
                    CredentialRow.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            ),
        )
        return result.rowcount == 1

    def revoke_all(self, user_id: uuid.UUID, *, now: datetime) -> int:
        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(CredentialRow)
                .where(
                    CredentialRow.user_id == user_id,
                    CredentialRow.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            ),
        )
        return result.rowcount

    def list_for_user(
        self, user_id: uuid.UUID, *, kind: CredentialKind
    ) -> list[Credential]:
        rows = self._session.scalars(
            select(CredentialRow)
            .where(
                CredentialRow.user_id == user_id,
                CredentialRow.kind == kind,
            )
            .order_by(CredentialRow.created_at.desc(), CredentialRow.id)
        ).all()
        return [to_credential(row) for row in rows]


def to_credential(row: CredentialRow) -> Credential:
    return Credential(
        id=row.id,
        user_id=row.user_id,
        kind=row.kind,
        name=row.name,
        prefix=row.prefix,
        act_for_users=row.act_for_users,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
    )
