"""Mission-plan export to KML, GPX, and DJI Waypoint v1 formats.

Pure stdlib. One waypoint per leg-vertex; leg type encoded in the waypoint name
(`L1-TRANSIT-WP3`). Altitudes are AGL (relative-to-ground) — documented in the
file header so an operator does not assume MSL and fly into terrain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from xml.sax.saxutils import escape as xml_escape

from app.schemas.plan import MissionPlan, Waypoint


@dataclass(frozen=True)
class ExportSpec:
    name: str
    extension: str
    media_type: str
    description: str


# Public registry of supported formats.
EXPORT_FORMATS: dict[str, ExportSpec] = {
    "kml": ExportSpec(
        "kml", "kml", "application/vnd.google-earth.kml+xml", "Google Earth / Mission Planner KML"
    ),
    "gpx": ExportSpec("gpx", "gpx", "application/gpx+xml", "GPS Exchange Format (universal)"),
    "dji": ExportSpec("dji", "csv", "text/csv", "DJI Waypoint v1 (Litchi/DJI Fly compatible)"),
}


def _iter_waypoints(plan: MissionPlan) -> Iterator[tuple[int, str, int, Waypoint]]:
    """Yield (leg_idx, leg_type_str, wp_idx_in_plan, waypoint) tuples."""
    global_idx = 0
    for li, leg in enumerate(plan.legs):
        for _wi, wp in enumerate(leg.geometry):
            global_idx += 1
            yield li + 1, leg.leg_type.value, global_idx, wp


def plan_to_kml(plan: MissionPlan) -> str:
    """KML: a Document with one Placemark per waypoint + one LineString per leg."""
    now = datetime.now(UTC).isoformat()
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"<name>MTS Mission {xml_escape(plan.mission_id)}</name>",
        f"<description>Generated {xml_escape(now)} by Mission Tasking Service. "
        f"Area={xml_escape(plan.area_id)}. Status={xml_escape(plan.status.value)}. "
        "Altitudes are AGL (relativeToGround).</description>",
        '<Style id="leg"><LineStyle><color>ff5b9bff</color><width>3</width></LineStyle></Style>',
    ]

    # One Placemark per waypoint
    for leg_num, leg_type, wp_idx, wp in _iter_waypoints(plan):
        parts.extend(
            [
                "<Placemark>",
                f"<name>L{leg_num}-{xml_escape(leg_type)}-WP{wp_idx}</name>",
                "<Point>",
                "<altitudeMode>relativeToGround</altitudeMode>",
                f"<coordinates>{wp.lon:.6f},{wp.lat:.6f},{wp.alt_m:.1f}</coordinates>",
                "</Point>",
                "</Placemark>",
            ]
        )

    # One LineString per leg with >=2 points
    for li, leg in enumerate(plan.legs):
        if len(leg.geometry) < 2:
            continue
        coords = " ".join(f"{wp.lon:.6f},{wp.lat:.6f},{wp.alt_m:.1f}" for wp in leg.geometry)
        parts.extend(
            [
                "<Placemark>",
                f"<name>Leg {li + 1} ({xml_escape(leg.leg_type.value)})</name>",
                "<styleUrl>#leg</styleUrl>",
                "<LineString>",
                "<altitudeMode>relativeToGround</altitudeMode>",
                f"<coordinates>{coords}</coordinates>",
                "</LineString>",
                "</Placemark>",
            ]
        )

    parts.append("</Document></kml>")
    return "\n".join(parts)


def plan_to_gpx(plan: MissionPlan) -> str:
    """GPX 1.1: each leg becomes a <rte> route; waypoints also emitted as <wpt>."""
    now = datetime.now(UTC).isoformat()
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="MTS" xmlns="http://www.topografix.com/GPX/1/1">',
        f"<metadata><name>MTS Mission {xml_escape(plan.mission_id)}</name>"
        f"<desc>Area={xml_escape(plan.area_id)}; status={xml_escape(plan.status.value)}; "
        "elevations are AGL (not MSL).</desc>"
        f"<time>{xml_escape(now)}</time></metadata>",
    ]

    for leg_num, leg_type, wp_idx, wp in _iter_waypoints(plan):
        parts.extend(
            [
                f'<wpt lat="{wp.lat:.6f}" lon="{wp.lon:.6f}">',
                f"<ele>{wp.alt_m:.1f}</ele>",
                f"<name>L{leg_num}-{xml_escape(leg_type)}-WP{wp_idx}</name>",
                "</wpt>",
            ]
        )

    for li, leg in enumerate(plan.legs):
        parts.append(f"<rte><name>Leg {li + 1} ({xml_escape(leg.leg_type.value)})</name>")
        for wi, wp in enumerate(leg.geometry):
            parts.extend(
                [
                    f'<rtept lat="{wp.lat:.6f}" lon="{wp.lon:.6f}">',
                    f"<ele>{wp.alt_m:.1f}</ele>",
                    f"<name>WP{wi + 1}</name>",
                    "</rtept>",
                ]
            )
        parts.append("</rte>")

    parts.append("</gpx>")
    return "\n".join(parts)


def plan_to_dji(plan: MissionPlan) -> str:
    """DJI Waypoint v1 CSV (Litchi/DJI Fly compatible).

    Columns (Litchi format):
      latitude, longitude, altitude(m), heading(deg), curvesize(m), rotationdir,
      gimbalmode, gimbalpitchangle, actiontype1, actionparam1, altitudemode,
      speed(m/s), poi_latitude, poi_longitude, poi_altitude, poi_altitudemode

    Altitudes are AGL (altitudemode = 0 means relative-to-takeoff).
    """
    lines = [
        "# MTS Mission Export — DJI Waypoint v1 / Litchi-compatible CSV",
        f"# mission_id={plan.mission_id} area={plan.area_id} status={plan.status.value}",
        "# Altitudes are AGL (altitudemode=0). Speed default 5 m/s.",
        "latitude,longitude,altitude(m),heading(deg),curvesize(m),rotationdir,"
        "gimbalmode,gimbalpitchangle,actiontype1,actionparam1,altitudemode,"
        "speed(m/s),poi_latitude,poi_longitude,poi_altitude,poi_altitudemode",
    ]
    for _leg_num, _leg_type, _wp_idx, wp in _iter_waypoints(plan):
        lines.append(f"{wp.lat:.6f},{wp.lon:.6f},{wp.alt_m:.1f},0,0.2,0,0,0,-1,0,0,5.0,0,0,0,0")
    return "\n".join(lines) + "\n"


_RENDERERS: dict[str, Callable[[MissionPlan], str]] = {
    "kml": plan_to_kml,
    "gpx": plan_to_gpx,
    "dji": plan_to_dji,
}


def render_export(plan: MissionPlan, fmt: str) -> tuple[str, ExportSpec]:
    """Render the plan in `fmt`. Raises KeyError on unknown format."""
    fmt = fmt.lower()
    if fmt not in _RENDERERS:
        raise KeyError(f"unsupported format: {fmt}")
    return _RENDERERS[fmt](plan), EXPORT_FORMATS[fmt]
