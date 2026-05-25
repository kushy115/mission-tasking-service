"""Weather provider abstraction.

See docs/DESIGN_DECISIONS.md §4. Two implementations: SyntheticWeather (seeded
RNG, reproducible for tests/demos) and OpenMeteoWeather (free HTTP API, no key).
Selected via the MTS_WEATHER_PROVIDER setting.

Failure mode: if the real provider raises, we fall back to synthetic so a
network blip never blocks a compile.
"""

from app.weather.provider import (
    WEATHER_PROVIDERS,
    WeatherObservation,
    WeatherProvider,
    get_weather_for_area,
)

__all__ = [
    "WEATHER_PROVIDERS",
    "WeatherObservation",
    "WeatherProvider",
    "get_weather_for_area",
]
