"""Seed Postgres/PostGIS with operating areas and drone profiles from data/.

Usage:
    uv run python scripts/seed_db.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.geo.store import (  # noqa: E402
    get_engine,
    init_schema,
    upsert_area,
    upsert_drone,
)


def seed() -> None:
    engine = get_engine()
    init_schema(engine)

    areas_dir = ROOT / "data" / "areas"
    for gj_path in sorted(areas_dir.glob("*.geojson")):
        with gj_path.open() as f:
            gj = json.load(f)
        area_id = gj["properties"]["area_id"]
        ceiling_m = float(gj["properties"]["ceiling_m"])
        print(f"seeding area {area_id} from {gj_path.name}")
        upsert_area(engine, area_id, gj, ceiling_m)

    drones_dir = ROOT / "data" / "drones"
    for yp in sorted(drones_dir.glob("*.yaml")):
        with yp.open() as f:
            d = yaml.safe_load(f)
        pid = d["profile_id"]
        print(f"seeding drone profile {pid} from {yp.name}")
        upsert_drone(engine, pid, d)

    print("done.")


if __name__ == "__main__":
    seed()
