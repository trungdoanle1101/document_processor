import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    Uuid,
    false,
    func,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from kb_server.domain.credentials import CREDENTIAL_KINDS, CredentialKind
from kb_server.domain.documents import (
    DOCUMENT_STATUSES,
)
from kb_server.domain.jobs import (
    JOB_KINDS,
    JOB_STATUSES,
)
from kb_server.domain.users import USER_ROLES, UserRole
from kb_server.storage.database import Base

_CASCADE = "CASCADE"
_RESTRICT = "RESTRICT"


# --- users
class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    display_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        Enum(
            *USER_ROLES,
            name="user_role",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- credentials
class CredentialRow(Base):
    __tablename__ = "credentials"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete=_CASCADE), index=True
    )
    kind: Mapped[CredentialKind] = mapped_column(
        Enum(
            *CREDENTIAL_KINDS,
            name="credential_kind",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(20))
    token_hash: Mapped[str] = mapped_column(String(255), unique=True)
    act_for_users: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- groups
class GroupRow(Base):
    __tablename__ = "groups"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- user_groups
class UserGroupRow(Base):
    __tablename__ = "user_groups"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete=_CASCADE), primary_key=True
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("groups.id", ondelete=_CASCADE), primary_key=True, index=True
    )


# --- collections
class CollectionRow(Base):
    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(50), unique=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- collection_groups
class CollectionGroupRow(Base):
    __tablename__ = "collection_groups"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete=_CASCADE), primary_key=True
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("groups.id", ondelete=_RESTRICT), primary_key=True, index=True
    )


# --- collection_managers
class CollectionManagerRow(Base):
    __tablename__ = "collection_managers"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete=_CASCADE), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete=_CASCADE), primary_key=True
    )


# --- stored_files
class StoredFileRow(Base):
    __tablename__ = "stored_files"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    size: Mapped[int] = mapped_column(BigInteger)
    media_type: Mapped[str] = mapped_column(String(100))
    storage_key: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- documents
class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete=_RESTRICT), index=True
    )
    title: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(
        Enum(
            *DOCUMENT_STATUSES,
            name="document_status",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete=_RESTRICT)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# --- document_versions
class DocumentVersionRow(Base):
    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete=_CASCADE), primary_key=True
    )
    version: Mapped[uuid.UUID] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("stored_files.id", ondelete=_RESTRICT)
    )
    original_filename: Mapped[uuid.UUID] = mapped_column(String(255))
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete=_RESTRICT)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- document_groups
class DocumentGroupRow(Base):
    __tablename__ = "document_groups"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete=_CASCADE), primary_key=True
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("groups.id", ondelete=_RESTRICT), primary_key=True, index=True
    )


# --- indexing_jobs
class IndexingJobRow:
    __tablename__ = "indexing_jobs"
    __table_args__ = ForeignKeyConstraint(
        ["document_id", "version"],
        ["document_versions.document_id", "document_versions.version"],
        ondelete=_CASCADE,
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(
        Enum(
            *JOB_KINDS,
            name="job_kind",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete=_CASCADE), index=True
    )
    version: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        Enum(
            *JOB_STATUSES,
            name="job_status",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete=_RESTRICT)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
