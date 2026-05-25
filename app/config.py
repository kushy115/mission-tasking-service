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

    api_key: str = Field("", alias="API_KEY")
    llm_provider: str = Field("google_genai", alias="MTS_LLM_PROVIDER")
    llm_model: str = Field("gemini-2.5-flash", alias="MTS_LLM_MODEL")
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
    # Hard reject thresholds — see docs/DESIGN_DECISIONS.md §4.
    weather_max_wind_mps: float = Field(12.0, alias="MTS_WEATHER_MAX_WIND_MPS")
    weather_max_gust_mps: float = Field(16.0, alias="MTS_WEATHER_MAX_GUST_MPS")
    weather_min_visibility_m: float = Field(1500.0, alias="MTS_WEATHER_MIN_VISIBILITY_M")
    weather_max_precip_mmh: float = Field(2.5, alias="MTS_WEATHER_MAX_PRECIP_MMH")
    # Wind power penalty: per-m/s multiplier on cruise/hover power.
    weather_wind_power_coeff: float = Field(0.02, alias="MTS_WEATHER_WIND_COEFF")


@lru_cache
def get_settings() -> Settings:
    return Settings()
