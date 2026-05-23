from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration. All values come from env vars (or .env in dev)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = Field("local", alias="MTS_ENV")
    log_level: str = Field("INFO", alias="MTS_LOG_LEVEL")
    host: str = Field("0.0.0.0", alias="MTS_HOST")
    port: int = Field(8000, alias="MTS_PORT")

    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field("claude-sonnet-4-6", alias="MTS_LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="MTS_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4096, alias="MTS_LLM_MAX_TOKENS")
    llm_timeout_s: int = Field(60, alias="MTS_LLM_TIMEOUT_S")

    repair_loop_cap: int = Field(3, alias="MTS_REPAIR_LOOP_CAP")
    battery_reserve_min_pct: float = Field(20.0, alias="MTS_BATTERY_RESERVE_MIN_PCT")
    min_agl_m: float = Field(20.0, alias="MTS_MIN_AGL_M")

    database_url: str = Field(
        "postgresql+psycopg://mts:mts@localhost:5432/mts", alias="MTS_DATABASE_URL"
    )
    checkpoint_database_url: str = Field(
        "postgresql://mts:mts@localhost:5432/mts", alias="MTS_CHECKPOINT_DATABASE_URL"
    )
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field("mts-local", alias="LANGSMITH_PROJECT")

    weather_provider: str = Field("synthetic", alias="MTS_WEATHER_PROVIDER")
    weather_seed: int = Field(42, alias="MTS_WEATHER_SEED")


@lru_cache
def get_settings() -> Settings:
    return Settings()
