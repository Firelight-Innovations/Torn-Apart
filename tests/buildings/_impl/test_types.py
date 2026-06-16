"""Tests for buildings/_impl/types.py — the Foundation + RoofSlab slab value types."""

from __future__ import annotations

import numpy as np

from fire_engine.buildings._impl.types import Foundation, RoofSlab


def _square() -> np.ndarray:
    return np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], dtype=np.float64)


class TestFoundation:
    def test_fields(self) -> None:
        f = Foundation(polygon=_square(), depth_m=0.5)
        assert f.depth_m == 0.5
        assert f.polygon.shape == (4, 2)

    def test_round_trip(self) -> None:
        f = Foundation(polygon=_square(), depth_m=0.5)
        d = f.to_dict()
        # plain-primitive payload only (no numpy / live refs)
        assert isinstance(d["polygon"], list)
        assert d["depth_m"] == 0.5
        back = Foundation.from_dict(d)
        assert back.depth_m == f.depth_m
        assert np.array_equal(back.polygon, f.polygon)

    def test_to_dict_is_json_safe(self) -> None:
        import json

        d = Foundation(polygon=_square(), depth_m=0.5).to_dict()
        assert json.loads(json.dumps(d)) == d


class TestRoofSlab:
    def test_fields(self) -> None:
        r = RoofSlab(polygon=_square(), thickness_m=0.2)
        assert r.thickness_m == 0.2
        assert r.polygon.shape == (4, 2)

    def test_defaults_to_flat_roof(self) -> None:
        from fire_engine.buildings.enums import RoofKind

        r = RoofSlab(polygon=_square(), thickness_m=0.2)
        assert r.kind is RoofKind.FLAT
        assert r.pitch_deg == 30.0 and r.ridge_dir_rad == 0.0 and r.overhang_m == 0.0

    def test_round_trip(self) -> None:
        r = RoofSlab(polygon=_square(), thickness_m=0.2)
        back = RoofSlab.from_dict(r.to_dict())
        assert back.thickness_m == r.thickness_m
        assert np.array_equal(back.polygon, r.polygon)

    def test_round_trip_pitched_fields(self) -> None:
        from fire_engine.buildings.enums import RoofKind

        r = RoofSlab(
            polygon=_square(),
            thickness_m=0.25,
            kind=RoofKind.GABLE,
            pitch_deg=42.0,
            ridge_dir_rad=1.25,
            overhang_m=0.6,
        )
        back = RoofSlab.from_dict(r.to_dict())
        assert back.kind is RoofKind.GABLE
        assert back.pitch_deg == 42.0
        assert back.ridge_dir_rad == 1.25
        assert back.overhang_m == 0.6

    def test_legacy_dict_without_pitched_keys_loads_flat(self) -> None:
        from fire_engine.buildings.enums import RoofKind

        legacy = {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "thickness_m": 0.2}
        back = RoofSlab.from_dict(legacy)
        assert back.kind is RoofKind.FLAT  # back-compat with Iteration-1 saves

    def test_reexport_path_matches(self) -> None:
        # The historical import path must resolve to the same class object.
        from fire_engine.buildings.types import RoofSlab as ReexportedRoofSlab

        assert ReexportedRoofSlab is RoofSlab
