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
from typing import Any

from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import nearest_points
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
        home_lat DOUBLE PRECISION NOT NULL,
        -- LFU eviction columns (DD-008). protected=TRUE shields seeded areas.
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        protected BOOLEAN NOT NULL DEFAULT FALSE,
        notes TEXT NOT NULL DEFAULT ''
    )
    """,
    # Additive migrations for repos that pre-date the LFU columns.
    "ALTER TABLE areas ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE areas ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    "ALTER TABLE areas ADD COLUMN IF NOT EXISTS protected BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE areas ADD COLUMN IF NOT EXISTS notes TEXT NOT NULL DEFAULT ''",
    # Mark the two seeded areas as protected so LFU never evicts them.
    "UPDATE areas SET protected = TRUE WHERE area_id IN ('yard-simple', 'farmland-complex')",
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
        profile JSONB NOT NULL,
        protected BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    "ALTER TABLE drones ADD COLUMN IF NOT EXISTS protected BOOLEAN NOT NULL DEFAULT FALSE",
    "UPDATE drones SET protected = TRUE WHERE profile_id IN ('long-endurance-quad', 'short-endurance-quad')",
    """
    CREATE TABLE IF NOT EXISTS missions (
        mission_id TEXT PRIMARY KEY,
        area_id TEXT,
        plan JSONB NOT NULL,
        status TEXT NOT NULL,
        approved BOOLEAN DEFAULT FALSE,
        operator_note TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT now(),
        -- Mission history extensions: original command, drone used, and every
        -- repair-attempt draft so we can show a diff timeline for debugging.
        command TEXT DEFAULT '',
        drone_profile_id TEXT DEFAULT '',
        repair_drafts JSONB DEFAULT '[]'::jsonb
    )
    """,
    # Best-effort additive migrations for repos that pre-date the new columns.
    "ALTER TABLE missions ADD COLUMN IF NOT EXISTS command TEXT DEFAULT ''",
    "ALTER TABLE missions ADD COLUMN IF NOT EXISTS drone_profile_id TEXT DEFAULT ''",
    "ALTER TABLE missions ADD COLUMN IF NOT EXISTS repair_drafts JSONB DEFAULT '[]'::jsonb",
    "CREATE INDEX IF NOT EXISTS idx_missions_created_at ON missions(created_at DESC)",
]


def init_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))


# LFU cap on user-authored areas. Protected (seeded) areas don't count toward
# the cap. See DESIGN_DECISIONS §8.
AREA_LFU_CAP = 10


def evict_lfu_area_if_needed(engine: Engine, exclude_area_id: str | None = None) -> str | None:
    """If unprotected-area count >= cap, evict the LFU one. Returns evicted id."""
    with engine.begin() as conn:
        n = (
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM areas WHERE protected = FALSE"
                    + (" AND area_id <> :ex" if exclude_area_id else "")
                ),
                {"ex": exclude_area_id} if exclude_area_id else {},
            ).scalar()
            or 0
        )
        if n < AREA_LFU_CAP:
            return None
        # Eviction key: lowest access_count, oldest last_accessed_at as tiebreak.
        row = conn.execute(
            text(
                "SELECT area_id FROM areas WHERE protected = FALSE"
                + (" AND area_id <> :ex" if exclude_area_id else "")
                + " ORDER BY access_count ASC, last_accessed_at ASC LIMIT 1"
            ),
            {"ex": exclude_area_id} if exclude_area_id else {},
        ).first()
        if not row:
            return None
        victim: str = row[0]
        conn.execute(text("DELETE FROM areas WHERE area_id = :a"), {"a": victim})
        return victim


def touch_area_access(engine: Engine, area_id: str) -> None:
    """Bump access_count and last_accessed_at — drives the LFU eviction order."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE areas SET access_count = access_count + 1, "
                "last_accessed_at = now() WHERE area_id = :a"
            ),
            {"a": area_id},
        )


def set_area_notes(engine: Engine, area_id: str, notes: str) -> None:
    """Store the advisory note returned by the LLM-research step (DD-007)."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE areas SET notes = :n WHERE area_id = :a"), {"n": notes, "a": area_id}
        )


def upsert_area(engine: Engine, area_id: str, geojson: dict[str, Any], ceiling_m: float) -> None:
    """Persist an operating area + its NFZs from a GeoJSON FeatureCollection.

    Convention: the feature with properties.role == "boundary" is the area boundary;
    features with properties.role == "nfz" are no-fly-zones; properties.home contains
    [lon, lat] of the base/home point on the boundary feature.
    """
    boundary_feat = next(
        f for f in geojson["features"] if f["properties"].get("role") == "boundary"
    )
    boundary_poly: Polygon = shape(boundary_feat["geometry"])
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
            poly: Polygon = shape(f["geometry"])
            conn.execute(
                text("INSERT INTO nfzs(area_id, geom) VALUES (:aid, ST_GeomFromText(:wkt, 4326))"),
                {"aid": area_id, "wkt": poly.wkt},
            )


def snap_home_to_boundary(
    boundary: Polygon, home_lon: float, home_lat: float
) -> tuple[float, float]:
    """Project the user-picked home point onto the nearest point on the polygon
    exterior. Matches kernel's "home_point is on the boundary" semantics.
    See docs/DESIGN_DECISIONS.md §5.
    """
    p = Point(home_lon, home_lat)
    boundary_line = LineString(boundary.exterior.coords)
    snapped, _ = nearest_points(boundary_line, p)
    return float(snapped.x), float(snapped.y)


def delete_area(engine: Engine, area_id: str) -> bool:
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM areas WHERE area_id = :a"), {"a": area_id})
    return bool(res.rowcount)


def upsert_drone(engine: Engine, profile_id: str, profile_dict: dict[str, Any]) -> None:
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


def _profile_from_dict(profile_id: str, raw: dict[str, Any]) -> DroneProfile:
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


def save_mission(
    engine: Engine,
    plan_dict: dict[str, Any],
    command: str = "",
    drone_profile_id: str = "",
    repair_drafts: list[dict[str, Any]] | None = None,
) -> None:
    """Persist a final plan plus history metadata.

    `repair_drafts` is a list of {attempt, draft_plan, violations} captured
    during the graph's repair loop, so the UI can show a diff timeline.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO missions(mission_id, area_id, plan, status, command, drone_profile_id, repair_drafts)
                VALUES (:mid, :aid, CAST(:p AS JSONB), :s, :cmd, :dpid, CAST(:rd AS JSONB))
                ON CONFLICT (mission_id) DO UPDATE
                  SET plan = EXCLUDED.plan,
                      status = EXCLUDED.status,
                      command = EXCLUDED.command,
                      drone_profile_id = EXCLUDED.drone_profile_id,
                      repair_drafts = EXCLUDED.repair_drafts
                """
            ),
            {
                "mid": plan_dict["mission_id"],
                "aid": plan_dict.get("area_id"),
                "p": json.dumps(plan_dict),
                "s": plan_dict.get("status"),
                "cmd": command,
                "dpid": drone_profile_id,
                "rd": json.dumps(repair_drafts or []),
            },
        )


def list_missions(engine: Engine, limit: int = 100) -> list[dict[str, Any]]:
    """Return the N most-recent missions with summary fields. Used by the UI history."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT mission_id, area_id, drone_profile_id, status, command,
                       approved, operator_note, created_at,
                       jsonb_array_length(repair_drafts) AS repair_count,
                       (plan->>'total_duration_s')::float AS duration_s,
                       (plan->>'total_battery_pct')::float AS battery_pct,
                       jsonb_array_length(plan->'legs') AS leg_count
                FROM missions
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"lim": limit},
        ).all()
    return [
        {
            "mission_id": r[0],
            "area_id": r[1],
            "drone_profile_id": r[2],
            "status": r[3],
            "command": r[4] or "",
            "approved": bool(r[5]),
            "operator_note": r[6] or "",
            "created_at": r[7].isoformat() if r[7] else None,
            "repair_count": int(r[8] or 0),
            "duration_s": float(r[9]) if r[9] is not None else 0.0,
            "battery_pct": float(r[10]) if r[10] is not None else 0.0,
            "leg_count": int(r[11] or 0),
        }
        for r in rows
    ]


def load_mission_detail(engine: Engine, mission_id: str) -> dict[str, Any] | None:
    """Full mission record: final plan + repair-draft timeline + metadata."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT plan, status, command, drone_profile_id, area_id,
                       approved, operator_note, created_at, repair_drafts
                FROM missions WHERE mission_id = :m
                """
            ),
            {"m": mission_id},
        ).first()
    if not row:
        return None
    return {
        "plan": row[0],
        "status": row[1],
        "command": row[2] or "",
        "drone_profile_id": row[3] or "",
        "area_id": row[4],
        "approved": bool(row[5]),
        "operator_note": row[6] or "",
        "created_at": row[7].isoformat() if row[7] else None,
        "repair_drafts": row[8] or [],
    }


def set_mission_approval(
    engine: Engine, mission_id: str, approved: bool, operator_note: str
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE missions SET approved = :ap, operator_note = :note WHERE mission_id = :m"),
            {"ap": approved, "note": operator_note, "m": mission_id},
        )


def load_mission(engine: Engine, mission_id: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT plan FROM missions WHERE mission_id = :m"), {"m": mission_id}
        ).first()
    if not row:
        return None
    result: dict[str, Any] = row[0]
    return result


def geojson_from_polygon(poly: Polygon) -> dict[str, Any]:
    result: dict[str, Any] = mapping(poly)
    return result


def list_areas(engine: Engine) -> list[dict[str, Any]]:
    """All areas with boundary + NFZs as GeoJSON. Used by the UI map."""
    out: list[dict[str, Any]] = []
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT area_id, ST_AsGeoJSON(boundary), ceiling_m, home_lon, home_lat, "
                "access_count, last_accessed_at, protected, notes "
                "FROM areas ORDER BY protected DESC, access_count DESC, area_id"
            )
        ).all()
        for area_id, bnd_json, ceiling_m, hlon, hlat, count, last_acc, prot, notes in rows:
            nfz_rows = conn.execute(
                text("SELECT ST_AsGeoJSON(geom) FROM nfzs WHERE area_id = :a"),
                {"a": area_id},
            ).all()
            out.append(
                {
                    "area_id": area_id,
                    "boundary": json.loads(bnd_json),
                    "nfzs": [json.loads(r[0]) for r in nfz_rows],
                    "ceiling_m": float(ceiling_m),
                    "home_lon": float(hlon),
                    "home_lat": float(hlat),
                    "access_count": int(count or 0),
                    "last_accessed_at": last_acc.isoformat() if last_acc else None,
                    "protected": bool(prot),
                    "notes": notes or "",
                }
            )
    return out


def list_drones(engine: Engine) -> list[dict[str, Any]]:
    """All drone profiles. Used by the UI dropdown."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT profile_id, profile, protected FROM drones ORDER BY protected DESC, profile_id"
            )
        ).all()
    return [
        {"profile_id": pid, "protected": bool(p), **(profile or {})} for pid, profile, p in rows
    ]
