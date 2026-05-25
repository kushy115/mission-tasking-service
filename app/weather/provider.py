"""Weather providers feeding the planner prompt, physics model, and kernel.

See docs/DESIGN_DECISIONS.md §4 for the rationale (10-min LRU TTL, fallback
to synthetic on failure, simple scalar wind penalty in the physics model).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeatherObservation:
    """Current conditions over an operating area. SI units, single observation.

    Conditions are reported AT THE GROUND, in nominal small-drone flight
    altitude band. Sufficient for go/no-go decisions in this demo; production
    would interpolate to flight altitude.
    """

    wind_mps: float           # sustained horizontal wind speed, ground level
    gust_mps: float           # peak gust over the past hour
    wind_dir_deg: float       # meteorological "from" direction, 0=N, 90=E
    visibility_m: float       # horizontal visibility in meters
    precipitation_mmh: float  # liquid precipitation rate, mm/hour
    temperature_c: float      # surface temperature in degrees Celsius
    source: str               # "synthetic" or "open-meteo"

    def summary_for_prompt(self) -> str:
        """One-line description an LLM can include in a plan's reasoning."""
        return (
            f"wind {self.wind_mps:.1f} m/s gusting {self.gust_mps:.1f} m/s "
            f"from {self.wind_dir_deg:.0f}°, visibility {self.visibility_m/1000:.1f} km, "
            f"precipitation {self.precipitation_mmh:.1f} mm/h, "
            f"{self.temperature_c:.0f}°C (source: {self.source})"
        )


class WeatherProvider(Protocol):
    def observe(self, lat: float, lon: float) -> WeatherObservation: ...


class SyntheticWeather:
    """Seeded-RNG weather. Reproducible: same (seed, lat, lon, hour) → same obs.

    Useful for tests and for offline demos. Generates plausible mild conditions
    skewed toward "fine for flight" so the default example missions don't get
    rejected on weather grounds.
    """

    def __init__(self, seed: int):
        self._seed = seed

    def observe(self, lat: float, lon: float) -> WeatherObservation:
        # Hour-stable so consecutive compiles see the same weather; varies day-
        # to-day so re-running tomorrow looks different.
        hour_bucket = int(time.time() // 3600)
        # random.Random only accepts hashable primitive seeds, not tuples;
        # mash the components into a single string then hash.
        seed_str = f"{self._seed}:{round(lat, 2)}:{round(lon, 2)}:{hour_bucket}"
        rng = random.Random(seed_str)
        wind = rng.uniform(0.5, 6.0)
        gust = wind + rng.uniform(0.5, 3.0)
        vis_km = rng.uniform(5.0, 15.0)
        precip = max(0.0, rng.gauss(0.0, 0.3))
        return WeatherObservation(
            wind_mps=wind,
            gust_mps=gust,
            wind_dir_deg=rng.uniform(0.0, 360.0),
            visibility_m=vis_km * 1000.0,
            precipitation_mmh=precip,
            temperature_c=rng.uniform(8.0, 24.0),
            source="synthetic",
        )


class OpenMeteoWeather:
    """Free HTTP weather from open-meteo.com. No API key required.

    https://api.open-meteo.com/v1/forecast?latitude=...&longitude=...&current=...
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    TIMEOUT_S = 5.0

    def observe(self, lat: float, lon: float) -> WeatherObservation:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": ",".join([
                "temperature_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "visibility",
            ]),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        with httpx.Client(timeout=self.TIMEOUT_S) as client:
            r = client.get(self.BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()
        cur = data.get("current") or {}
        # Open-Meteo returns visibility in meters; some endpoints return null
        # for it depending on the location — fall back to 10km if absent.
        vis = cur.get("visibility")
        return WeatherObservation(
            wind_mps=float(cur.get("wind_speed_10m") or 0.0),
            gust_mps=float(cur.get("wind_gusts_10m") or cur.get("wind_speed_10m") or 0.0),
            wind_dir_deg=float(cur.get("wind_direction_10m") or 0.0),
            visibility_m=float(vis) if vis is not None else 10_000.0,
            precipitation_mmh=float(cur.get("precipitation") or 0.0),
            temperature_c=float(cur.get("temperature_2m") or 15.0),
            source="open-meteo",
        )


WEATHER_PROVIDERS: dict[str, type] = {
    "synthetic": SyntheticWeather,
    "open-meteo": OpenMeteoWeather,
}


# ----- module-level cache -----
# We cache by (lat_bucket, lon_bucket, hour) so re-compiles of the same mission
# don't re-fetch. 10-minute TTL is shorter than the hour bucket — if a request
# falls inside the same hour bucket twice within 10 min, we serve from cache;
# beyond 10 min we re-validate. Open-Meteo's published quota is ~10k req/day per
# IP and we want to stay well under it.
_CACHE: dict[tuple, tuple[float, WeatherObservation]] = {}
_TTL_S = 600.0


def _cache_key(lat: float, lon: float, provider_name: str) -> tuple:
    return (provider_name, round(lat, 2), round(lon, 2), int(time.time() // 3600))


def _build_provider(name: str) -> WeatherProvider:
    settings = get_settings()
    cls = WEATHER_PROVIDERS.get(name.lower())
    if cls is None:
        log.warning("unknown weather provider %r, falling back to synthetic", name)
        return SyntheticWeather(settings.weather_seed)
    if cls is SyntheticWeather:
        return SyntheticWeather(settings.weather_seed)
    return cls()


def get_weather_for_area(lat: float, lon: float) -> WeatherObservation:
    """Public entry point used by the planner / kernel / physics layers.

    Always returns an observation (never raises): real-provider failure falls
    through to SyntheticWeather so a network blip can't block a compile.
    """
    settings = get_settings()
    name = (settings.weather_provider or "synthetic").lower()
    key = _cache_key(lat, lon, name)
    cached = _CACHE.get(key)
    if cached and (time.time() - cached[0]) < _TTL_S:
        return cached[1]
    try:
        provider = _build_provider(name)
        obs = provider.observe(lat, lon)
    except Exception as e:  # noqa: BLE001
        log.warning("weather provider %s failed (%s); falling back to synthetic", name, e)
        obs = SyntheticWeather(settings.weather_seed).observe(lat, lon)
    _CACHE[key] = (time.time(), obs)
    return obs


def clear_cache() -> None:
    """Mostly for tests."""
    _CACHE.clear()
