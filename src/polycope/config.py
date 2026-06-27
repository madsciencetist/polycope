"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Override via env vars prefixed POLYCOPE_ or a .env file."""

    model_config = SettingsConfigDict(env_prefix="POLYCOPE_", env_file=".env", extra="ignore")

    data_api: str = "https://data-api.polymarket.com"
    gamma_api: str = "https://gamma-api.polymarket.com"
    data_dir: Path = Path("./data")

    max_concurrency: int = 8
    request_timeout: float = 30.0
    max_retries: int = 4

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.synthetic_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
