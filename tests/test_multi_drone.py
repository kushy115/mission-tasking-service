"""Tests for multi-drone slot assignment (pure-Python; no LLM/DB)."""

from __future__ import annotations

from app.graph.multi_drone import (
    DEFAULT_LAYERS_M,
    LAYER_HALF_BAND,
    MIN_VERTICAL_SEP_M,
    TAKEOFF_STAGGER_S,
    assign_slots,
    nearest_home_base,
)


def test_assign_slots_count_matches():
    slots = assign_slots(["a", "b", "c"])
    assert len(slots) == 3
    assert [s.drone_profile_id for s in slots] == ["a", "b", "c"]


def test_layers_are_separated():
    slots = assign_slots(["a", "b", "c"])
    centers = [(s.alt_min + s.alt_max) / 2 for s in slots]
    for i in range(len(centers) - 1):
        assert centers[i + 1] - centers[i] >= MIN_VERTICAL_SEP_M


def test_takeoff_stagger_grows_linearly():
    slots = assign_slots(["a", "b", "c"])
    assert slots[0].takeoff_offset_s == 0.0
    assert slots[1].takeoff_offset_s == TAKEOFF_STAGGER_S
    assert slots[2].takeoff_offset_s == 2 * TAKEOFF_STAGGER_S


def test_sibling_bands_excludes_self():
    slots = assign_slots(["a", "b", "c"])
    for s in slots:
        # Each sibling band is (lo, hi); count should be total-1.
        assert len(s.sibling_bands) == s.total - 1
        # Own band not in siblings
        own = (s.alt_min, s.alt_max)
        assert own not in s.sibling_bands


def test_default_layers_within_kernel_min_agl():
    # min AGL is 20m by default; smallest assigned layer should clear that.
    slots = assign_slots(["a"])
    assert slots[0].alt_min >= 20.0


def test_more_than_default_layers_extrapolates():
    slots = assign_slots(["a", "b", "c", "d", "e"])
    centers = [(s.alt_min + s.alt_max) / 2 for s in slots]
    # First three match DEFAULT_LAYERS_M
    assert centers[: len(DEFAULT_LAYERS_M)] == DEFAULT_LAYERS_M
    # Each subsequent layer continues the 30m progression
    assert centers[3] == DEFAULT_LAYERS_M[-1] + MIN_VERTICAL_SEP_M
    assert centers[4] == DEFAULT_LAYERS_M[-1] + 2 * MIN_VERTICAL_SEP_M


def test_slot_band_half_band():
    slots = assign_slots(["a"])
    s = slots[0]
    assert (s.alt_max - s.alt_min) / 2 == LAYER_HALF_BAND


def test_nearest_home_base_selects_closest_lonlat():
    bases = [(-74.195, 40.711), (-74.185, 40.710), (-74.175, 40.727)]
    assert nearest_home_base(bases, (-74.176, 40.726)) == (-74.175, 40.727)
