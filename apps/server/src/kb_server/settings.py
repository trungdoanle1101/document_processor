from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The same values as the framework's huggingface Device. Defined here because importing
# that one loads torch, and every process reads Settings, not only the worker.
type EmbedderDevice = Literal["auto", "cuda", "mps", "cpu"]
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

type Positive = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    """Configuration from KB_* environment variables, or a .env file next to where it runs.

    Every process reads the same Settings. Worker-only fields have defaults, or are None
    until set, so the API server and create-admin start without them; the worker checks
    the ones it needs when it builds its tools.
    """

    model_config = SettingsConfigDict(env_prefix="KB_", env_file=".env", extra="ignore")

    # every process
    database_url: str  # holds the password: .env only, never committed
    log_level: LogLevel = "INFO"

    # the API: sessions and requests
    session_ttl_hours: Positive = 12
    act_for_headers: tuple[str, ...] = ("x-act-for-user",)
    max_upload_mb: Positive = 50

    # file storage: the API writes originals, the worker reads them, so both need the same
    storage_kind: Literal["local", "s3"] = "local"
    storage_path: Path

    # worker, preparation: candidate profiles, most specific first
    profiles: tuple[str, ...] = ("vn_regulation",)
    fallback_profile: str = "plain"
    chunk_tokens: Positive = 512  # counted with the embedder's tokenizer

    # worker, embedder: a name from the framework's HUGGINGFACE_MODEL_SPECS
    embedder_spec: str = "bge-m3"
    embedder_folder: Path | None = None  # required by the worker
    embedder_device: EmbedderDevice = "auto"
    # memory and speed only; the vectors don't change
    embedder_batch_size: Positive = 32

    # worker, index
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_chunks"
    qdrant_api_key: SecretStr | None = None

    # worker, the job loop
    worker_poll_seconds: Annotated[float, Field(gt=0)] = 2.0
    job_max_attempts: Annotated[int, Field(ge=1)] = 3
    job_timeout_minutes: Positive = 60  # longer than the longest real job
