from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from KB_* environment variables, or a .env file next to where the server runs."""

    model_config = SettingsConfigDict(env_prefix="KB_", env_file=".env", extra="ignore")
    database_url: str
    session_ttl_hours: int = 12
    act_for_headers: tuple[str, ...] = ("x-act-for-user",)
    max_upload_mb: int = 50

    storage_kind: Literal["local", "s3"] = "local"
    # local storage
    storage_path: Path
