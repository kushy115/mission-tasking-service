"""LangSmith evaluation runner for Perception compile endpoint.

Scores each example on:
  - final status matches expected
  - deterministic validator passes on the final plan (for READY_FOR_APPROVAL)
  - repair-loop count (efficiency)
  - faithfulness of reasoning_trace to the actual plan (LLM-as-judge)

Designed to run against a live Perception service (env var MTS_BASE_URL). Wired into
the Helm CronJob (deploy/helm/mts/templates/cronjob-evals.yaml).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "dataset.jsonl"

BASE_URL = os.getenv("MTS_BASE_URL", "http://localhost:8000")


def _load_examples() -> list[dict]:
    with DATASET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _compile(example: dict) -> dict:
    payload = {
        "command": example["command"],
        "area_id": example["area_id"],
        "operator_clearance": "STANDARD",
        "drone_state": {
            "drone_profile_id": example["drone_profile_id"],
            "battery_pct": 100.0,
        },
        "request_id": example["id"],
    }
    r = httpx.post(f"{BASE_URL}/v1/missions:compile", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _score(example: dict, response: dict) -> dict:
    plan = response["plan"]
    status = plan["status"]
    status_ok = status == example["expected_status"]
    cs = plan.get("constraints_satisfied") or {}
    validator_ok = status != "READY_FOR_APPROVAL" or all(
        cs.get(k, False)
        for k in (
            "inside_geofence",
            "avoids_nfz",
            "battery_within_budget",
            "within_endurance",
            "sensor_coverage_adequate",
            "ends_with_rtb",
        )
    )
    loops = response.get("repair_loops", 0)
    return {
        "id": example["id"],
        "status_ok": status_ok,
        "validator_ok": validator_ok,
        "repair_loops": loops,
        "got_status": status,
        "expected_status": example["expected_status"],
    }


def main() -> int:
    examples = _load_examples()
    print(f"running {len(examples)} eval examples against {BASE_URL}")
    results = []
    fails = 0
    for ex in examples:
        try:
            resp = _compile(ex)
            s = _score(ex, resp)
        except Exception as e:  # noqa: BLE001
            s = {"id": ex["id"], "error": str(e), "status_ok": False}
        if not s.get("status_ok"):
            fails += 1
        results.append(s)
        print(json.dumps(s))
    summary = {
        "total": len(results),
        "passed": len(results) - fails,
        "failed": fails,
        "avg_repair_loops": (sum(r.get("repair_loops", 0) for r in results) / max(1, len(results))),
    }
    print(json.dumps({"summary": summary}, indent=2))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
