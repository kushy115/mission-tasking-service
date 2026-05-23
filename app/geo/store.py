"""PostGIS-backed spatial store.

The store owns three relations:
- `areas`(area_id PK, boundary geometry(Polygon,4326), ceiling_m, home_lon, home_lat)
- `nfzs`(id PK, area_id FK, geom geometry(Polygon,4326))
- `drones`(profile_id PK, profile JSONB)  -- canonical YAML loaded at seed time

Deterministic, no LLM. All geometry stored in WGS84 (SRID 4326).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from shapely import wkt
from shapely.geometry import Polygon, mapping, shape
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.schemas.enums import SensorMode
from app.validation.kernel import GeoContext
from app.validation.physics import DroneProfile, SensorSpec


@dataclass
class _AreaRow:
    area_id: str
    boundary_wkt: str
    ceiling_m: float
    home_lon: float
    home_lat: float


def get_engine() -> Engine:
    s = get_settings()
    return create_engine(s.database_url, pool_pre_ping=True)


DDL = [
    "CREATE EXTENSION IF NOT EXISTS postgis",
    """
    CREATE TABLE IF NOT EXISTS areas (
        area_id TEXT PRIMARY KEY,
        boundary geometry(Polygon, 4326) NOT NULL,
        ceiling_m DOUBLE PRECISION NOT NULL,
        home_lon DOUBLE PRECISION NOT NULL,
        home_lat DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS nfzs (
        id SERIAL PRIMARY KEY,
        area_id TEXT REFERENCES areas(area_id) ON DELETE CASCADE,
        geom geometry(Polygon, 4326) NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_nfzs_geom ON nfzs USING GIST(geom)",
    "CREATE INDEX IF NOT EXISTS idx_areas_boundary ON areas USING GIST(boundary)",
    """
    CREATE TABLE IF NOT EXISTS drones (
        profile_id TEXT PRIMARY KEY,
        profile JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS missions (
        mission_id TEXT PRIMARY KEY,
        area_id TEXT,
        plan JSONB NOT NULL,
        status TEXT NOT NULL,
        approved BOOLEAN DEFAULT FALSE,
        operator_note TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT now()
    )
    """,
]


def init_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))


def upsert_area(engine: Engine, area_id: str, geojson: dict, ceiling_m: float) -> None:
    """Persist an operating area + its NFZs from a GeoJSON FeatureCollection.

    Convention: the feature with properties.role == "boundary" is the area boundary;
    features with properties.role == "nfz" are no-fly-zones; properties.home contains
    [lon, lat] of the base/home point on the boundary feature.
    """
    boundary_feat = next(
        f for f in geojson["features"] if f["properties"].get("role") == "boundary"
    )
    boundary_poly: Polygon = shape(boundary_feat["geometry"])  # type: ignore[assignment]
    home = boundary_feat["properties"]["home"]
    nfz_feats = [f for f in geojson["features"] if f["properties"].get("role") == "nfz"]

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO areas(area_id, boundary, ceiling_m, home_lon, home_lat)
                VALUES (:aid, ST_GeomFromText(:wkt, 4326), :ceil, :hlon, :hlat)
                ON CONFLICT (area_id) DO UPDATE
                  SET boundary = EXCLUDED.boundary,
                      ceiling_m = EXCLUDED.ceiling_m,
                      home_lon = EXCLUDED.home_lon,
                      home_lat = EXCLUDED.home_lat
                """
            ),
            {
                "aid": area_id,
                "wkt": boundary_poly.wkt,
                "ceil": ceiling_m,
                "hlon": home[0],
                "hlat": home[1],
            },
        )
        conn.execute(text("DELETE FROM nfzs WHERE area_id = :aid"), {"aid": area_id})
        for f in nfz_feats:
            poly: Polygon = shape(f["geometry"])  # type: ignore[assignment]
            conn.execute(
                text(
                    "INSERT INTO nfzs(area_id, geom) VALUES (:aid, ST_GeomFromText(:wkt, 4326))"
                ),
                {"aid": area_id, "wkt": poly.wkt},
            )


def upsert_drone(engine: Engine, profile_id: str, profile_dict: dict) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO drones(profile_id, profile) VALUES (:pid, CAST(:p AS JSONB))
                ON CONFLICT (profile_id) DO UPDATE SET profile = EXCLUDED.profile
                """
            ),
            {"pid": profile_id, "p": json.dumps(profile_dict)},
        )


def load_geo_context(engine: Engine, area_id: str) -> GeoContext:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT ST_AsText(boundary), ceiling_m, home_lon, home_lat FROM areas WHERE area_id = :a"
            ),
            {"a": area_id},
        ).first()
        if row is None:
            raise KeyError(f"unknown area_id: {area_id}")
        boundary_wkt, ceiling_m, home_lon, home_lat = row
        nfz_rows = conn.execute(
            text("SELECT ST_AsText(geom) FROM nfzs WHERE area_id = :a"), {"a": area_id}
        ).all()
    boundary = wkt.loads(boundary_wkt)
    nfzs = tuple(wkt.loads(r[0]) for r in nfz_rows)
    return GeoContext(
        area_id=area_id,
        boundary=boundary,
        nfz_polygons=nfzs,
        altitude_ceiling_m=float(ceiling_m),
        home_point=(float(home_lon), float(home_lat)),
    )


def load_drone_profile(engine: Engine, profile_id: str) -> DroneProfile:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT profile FROM drones WHERE profile_id = :p"), {"p": profile_id}
        ).first()
    if row is None:
        raise KeyError(f"unknown drone profile: {profile_id}")
    raw = row[0]
    return _profile_from_dict(profile_id, raw)


def _profile_from_dict(profile_id: str, raw: dict) -> DroneProfile:
    sensors = tuple(
        SensorSpec(
            name=s["name"],
            mode=SensorMode(s["mode"]),
            power_w=float(s["power_w"]),
            half_angle_deg=float(s["half_angle_deg"]),
            ground_resolution_at_100m=float(s["ground_resolution_at_100m"]),
        )
        for s in raw.get("sensors", [])
    )
    return DroneProfile(
        profile_id=profile_id,
        rated_endurance_s=float(raw["rated_endurance_s"]),
        cruise_speed_mps=float(raw["cruise_speed_mps"]),
        climb_rate_mps=float(raw["climb_rate_mps"]),
        battery_wh=float(raw["battery_wh"]),
        cruise_power_w=float(raw["cruise_power_w"]),
        hover_power_w=float(raw["hover_power_w"]),
        sensors=sensors,
    )


def save_mission(engine: Engine, plan_dict: dict) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO missions(mission_id, area_id, plan, status)
                VALUES (:mid, :aid, CAST(:p AS JSONB), :s)
                ON CONFLICT (mission_id) DO UPDATE
                  SET plan = EXCLUDED.plan, status = EXCLUDED.status
                """
            ),
            {
                "mid": plan_dict["mission_id"],
                "aid": plan_dict.get("area_id"),
                "p": json.dumps(plan_dict),
                "s": plan_dict.get("status"),
            },
        )


def set_mission_approval(
    engine: Engine, mission_id: str, approved: bool, operator_note: str
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE missions SET approved = :ap, operator_note = :note WHERE mission_id = :m"
            ),
            {"ap": approved, "note": operator_note, "m": mission_id},
        )


def load_mission(engine: Engine, mission_id: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT plan FROM missions WHERE mission_id = :m"), {"m": mission_id}
        ).first()
    return row[0] if row else None


def geojson_from_polygon(poly: Polygon) -> dict:
    return mapping(poly)
