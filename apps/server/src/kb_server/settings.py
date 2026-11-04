from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from KB_* environment variables, or a .env file next to where the server runs"""

    model_config = SettingsConfigDict(env_prefix="KB_", env_file=".env", extra="ignore")
    database_url: str
